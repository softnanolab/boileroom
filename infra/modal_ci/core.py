"""Controller logic, independent of Modal and FastAPI so it can be tested with fakes.

`Core.handle_webhook` is the only entry for GitHub deliveries; `Core.reconcile` is the periodic
sweep that tears down stragglers and recovers jobs whose `queued` delivery was lost or whose
runner was killed by the guard. Both run in the single controller container behind `Core.lock`.
"""

from __future__ import annotations

import json
import logging
import secrets
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from infra.modal_ci import policy
from infra.modal_ci.github_api import GitHubError
from infra.modal_ci.ledger import PROFILES, STARTUP_ALLOWANCE_S, Ledger, Profile

log = logging.getLogger("modal_ci")

MAX_BODY_BYTES = 1_000_000
RECOVER_AFTER_S = 120  # a queued job this old with no sandbox lost its delivery (or its runner)
MAX_TRIES = 3
LAUNCH_GRACE_S = 600
GONE_GRACE_S = 120  # a launched sandbox missing from the listing this soon may just not be listed yet
DELIVERY_PREFIX = "delivery:"
# Facts about a run that cannot change, so one rejection stands for all of its jobs (a fork PR with
# a big matrix would otherwise cost two API calls per job, serialised ahead of real launches).
PERMANENT_RUN_REJECTIONS = ("event_not_allowed", "head_repository_not_this_repo", "api_run_repo_mismatch")
REJECTED_RUNS_MAX = 2000


class SpendUnknown(RuntimeError):
    """The measured spend cannot be trusted right now; nothing may launch until it can."""


@dataclass(frozen=True)
class Response:
    status: int
    body: str


@dataclass(frozen=True)
class SandboxState:
    sandbox_id: str
    finished: bool
    ephemeral: bool = False  # an operator's selftest sandbox: owns no ledger record, ends itself


class Sandboxes(Protocol):
    """What the controller needs from Modal Sandboxes."""

    def create(self, *, profile: Profile, name: str, env: Mapping[str, str], tags: Mapping[str, str]) -> str: ...
    def terminate(self, sandbox_id: str) -> None: ...
    def states(self) -> dict[str, SandboxState]: ...
    def running(self, sandbox_id: str) -> bool: ...


class GitHubClient(Protocol):
    """What the controller needs from the GitHub App client (`github_api.GitHubApp`)."""

    def get_job(self, repo: str, job_id: int) -> Mapping[str, Any]: ...
    def get_run(self, repo: str, run_id: int) -> Mapping[str, Any]: ...
    def queued_jobs(self, repo: str) -> list[Mapping[str, Any]]: ...
    def generate_jit(self, repo: str, name: str, labels: list[str]) -> tuple[str, int]: ...
    def delete_runner(self, repo: str, runner_id: int) -> None: ...


class Deliveries(Protocol):
    def put(self, key: str, value: Any, *, skip_if_exists: bool = False) -> bool: ...
    def pop(self, key: str) -> Any: ...


class Core:
    def __init__(
        self,
        *,
        cfg: policy.Config,
        webhook_secret: bytes,
        github: GitHubClient,
        ledger: Ledger,
        sandboxes: Sandboxes,
        deliveries: Deliveries,
        external_spend: Callable[[], float] = lambda: 0.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.cfg = cfg
        self.webhook_secret = webhook_secret
        self.github = github
        self.ledger = ledger
        self.sandboxes = sandboxes
        self.deliveries = deliveries
        self.external_spend = external_spend
        self.clock = clock
        self.lock = threading.Lock()
        self._rejected_runs: dict[tuple[str, int, int], str] = {}

    # -- webhook ---------------------------------------------------------------------------

    def handle_webhook(self, headers: Mapping[str, str], body: bytes) -> Response:
        hdrs = {k.lower(): v for k, v in headers.items()}
        if len(body) > MAX_BODY_BYTES:
            return Response(413, "too large")
        if not policy.verify_signature(self.webhook_secret, body, hdrs.get("x-hub-signature-256")):
            return Response(401, "bad signature")
        delivery = hdrs.get("x-github-delivery", "")
        if not delivery:
            return Response(400, "missing delivery id")
        event = hdrs.get("x-github-event", "")
        if event == "ping":
            return Response(200, "pong")
        if event != "workflow_job":
            return Response(204, f"ignored event {event}")
        try:
            payload = json.loads(body)
        except ValueError:
            return Response(400, "bad json")
        if not isinstance(payload, dict):
            return Response(400, "bad payload")

        decision = policy.parse_workflow_job(payload, self.cfg)
        if isinstance(decision, policy.Rejected):
            log.info("ignored delivery=%s reason=%s", delivery, decision.reason)
            return Response(204, decision.reason)

        # Claim the delivery before any side effect so a replay (or GitHub redelivery) is a no-op.
        if not self.deliveries.put(f"{DELIVERY_PREFIX}{delivery}", self.clock(), skip_if_exists=True):
            return Response(200, "duplicate delivery")
        try:
            with self.lock:
                resp = self._cleanup(decision) if isinstance(decision, policy.Cleanup) else self._launch(decision)
        except Exception:
            self.deliveries.pop(f"{DELIVERY_PREFIX}{delivery}")  # let a redelivery try again
            log.exception("webhook failed delivery=%s job=%s", delivery, decision)
            return Response(503, "retry later")
        if resp.status >= 500:
            self.deliveries.pop(f"{DELIVERY_PREFIX}{delivery}")  # nothing happened that a redelivery must not repeat
        return resp

    # -- launch / cleanup -------------------------------------------------------------------

    def expired(self) -> bool:
        return self.clock() >= self.cfg.active_until

    def _launch(self, req: policy.JobRequest) -> Response:
        if self.expired():
            # An admission stop, not a fallback: no new sandbox launches past the deadline, but GitHub has already
            # routed this job to a self-hosted label, so it stays queued with no runner (until GitHub's queue limit)
            # unless the repository variable that routes jobs here was unset beforehand.
            log.warning("not launched job=%s: activation expired", req.job_id)
            return Response(204, "activation_expired")
        try:
            reason = self._check_run(req)
            reason = reason or policy.check_job(req, self.github.get_job(req.repo, req.job_id))
        except GitHubError as e:
            log.warning("verification failed job=%s: %s", req.job_id, e)
            return Response(503, "verification unavailable")
        if reason:
            log.info("rejected job=%s repo=%s reason=%s", req.job_id, req.repo, reason)
            return Response(204, reason)

        profile_label = next((label for label in req.labels if label in policy.PROFILE_LABELS), policy.RUNNER_LABEL)
        profile = PROFILES[profile_label]
        job_label = next(label for label in req.labels if policy.JOB_LABEL_RE.match(label))
        runner_name = f"modal-{req.job_id}-{secrets.token_hex(16)}"
        meta = {"repo": req.repo, "run_id": req.run_id, "job_key": req.job_key, "runner_name": runner_name}
        try:
            spend = self.external_spend()
        except SpendUnknown as e:
            log.warning("not launched job=%s: %s", req.job_id, e)
            return Response(503, "spend unknown")
        denial = self.ledger.reserve(req.job_id, profile, meta, spend)
        if denial:
            log.warning("not launched job=%s reason=%s", req.job_id, denial)
            return Response(204 if denial == "already_launched" else 503, denial)

        runner_id: int | None = None
        try:
            labels = [*policy.RUNNER_LABELS, job_label]
            if profile_label != policy.RUNNER_LABEL:
                labels.append(profile_label)
            jit, runner_id = self.github.generate_jit(req.repo, runner_name, labels)
            self.ledger.update(req.job_id, runner_id=runner_id)
        except Exception:
            self._abandon(req, runner_id)
            raise
        binding = {
            "repo": req.repo,
            "job_id": req.job_id,
            "run_id": req.run_id,
            "run_attempt": req.run_attempt,
            "job_key": req.job_key,
            "head_sha": req.head_sha,
            "runner_name": runner_name,
        }
        # Persist before calling Modal: a timeout or controller crash can hide a
        # successful creation. Keep its capacity and full reservation until the
        # hard lifetime ends; the sweep also terminates any untracked sandbox.
        launched = self.clock()
        self.ledger.update(req.job_id, state="uncertain", launched=launched)
        sandbox_id = self.sandboxes.create(
            profile=profile,
            name=runner_name,
            env={
                "CI_BINDING": json.dumps(binding),
                "CI_JIT": jit,
                "CI_MAX_SECONDS": str(profile.max_seconds),
            },
            tags={"job_id": str(req.job_id), "repo": req.repo, "runner": runner_name},
        )
        self.ledger.update(req.job_id, state="running", sandbox_id=sandbox_id, launched=launched)
        log.info("launched job=%s repo=%s sandbox=%s runner=%s", req.job_id, req.repo, sandbox_id, runner_name)
        return Response(200, "launched")

    def _check_run(self, req: policy.JobRequest) -> str | None:
        """`policy.check_run`, remembering rejections that hold for every job of the run."""
        key = (req.repo, req.run_id, req.run_attempt)
        if key in self._rejected_runs:
            return self._rejected_runs[key]
        reason = policy.check_run(req, self.github.get_run(req.repo, req.run_id))
        if reason and reason.startswith(PERMANENT_RUN_REJECTIONS):
            if len(self._rejected_runs) >= REJECTED_RUNS_MAX:
                self._rejected_runs.clear()
            self._rejected_runs[key] = reason
        return reason

    def _abandon(self, req: policy.JobRequest, runner_id: int | None) -> None:
        """Undo a half-finished launch, retaining failed registration cleanup for the sweep."""
        self.ledger.update(req.job_id, state="cleanup", cleanup_seconds=0.0, runner_id=runner_id)
        self._teardown(req.job_id, self.ledger.store.get(f"job:{req.job_id}"), sandbox_alive=False)

    def _cleanup(self, done: policy.Cleanup) -> Response:
        rec = self._record_to_clean(done)
        if rec is None or rec["state"] == "settled" or rec["repo"] != done.repo:
            return Response(204, "nothing to clean up")
        if not self._teardown(rec["job_id"], rec):
            return Response(202, "teardown deferred to the next sweep")
        return Response(200, "cleaned up")

    def _record_to_clean(self, done: policy.Cleanup) -> Mapping[str, Any] | None:
        """The sandbox that ran this job. Matrix legs share a label, so a runner bound to one leg can
        end up running another: when GitHub names the runner, that runner's sandbox is the one to end."""
        if done.runner_name is None:
            return self.ledger.store.get(f"job:{done.job_id}")  # never picked up: its own sandbox is idle
        return next((r for r in self.ledger.records() if r.get("runner_name") == done.runner_name), None)

    def _teardown(self, job_id: int, rec: Mapping[str, Any], sandbox_alive: bool = True) -> bool:
        """End a job's sandbox and runner registration and settle its cost. `False` means it must be retried."""
        sandbox_id = rec.get("sandbox_id")
        seconds = rec.get("cleanup_seconds")
        if rec["state"] == "uncertain":
            # Even a completed webhook cannot prove an untracked sandbox has
            # exited. Do not release its slot or call an unknown launch free.
            seconds = PROFILES[rec["profile"]].hard_timeout_s + STARTUP_ALLOWANCE_S
            if self.clock() - rec["launched"] < seconds:
                return False
            self.ledger.update(job_id, state="cleanup", cleanup_seconds=seconds)
        if seconds is None and sandbox_id and sandbox_alive:
            try:
                self.sandboxes.terminate(sandbox_id)
            except Exception:
                log.exception("could not terminate sandbox %s", sandbox_id)
                return False  # keep the reservation; the next sweep retries
        if seconds is None:
            started = rec.get("launched") or rec["created"]
            seconds = self.clock() - started
            # Freeze compute time once the sandbox ends. A GitHub outage can delay deregistration,
            # but must neither lose that cleanup obligation nor bill its waiting time as compute.
            self.ledger.update(job_id, state="cleanup", cleanup_seconds=seconds)
        if rec.get("runner_id") is not None:
            try:
                self.github.delete_runner(rec["repo"], rec["runner_id"])
            except Exception:
                log.exception("could not delete runner %s", rec["runner_id"])
                return False
        self.ledger.settle(job_id, seconds)
        return True

    # -- periodic sweep -----------------------------------------------------------------------

    def reconcile(self) -> dict[str, int]:
        with self.lock:
            return self._reconcile()

    def _reconcile(self) -> dict[str, int]:
        now = self.clock()
        stats = {"settled": 0, "terminated": 0, "orphans": 0, "recovered": 0, "folded": 0}
        states = self.sandboxes.states()
        records = self.ledger.records()
        # A sandbox no live record points at (its launch died between create and bookkeeping)
        # has no budget behind it. End it; the job is picked up again by the recovery sweep.
        known = {r.get("sandbox_id") for r in records if r["state"] != "settled"}
        for sandbox_id, state in states.items():
            if not state.finished and not state.ephemeral and sandbox_id not in known:
                log.warning("terminating orphan sandbox %s", sandbox_id)
                try:
                    self.sandboxes.terminate(sandbox_id)
                    stats["orphans"] += 1
                except Exception:
                    log.exception("could not terminate orphan sandbox %s", sandbox_id)
        for rec in records:
            if rec["state"] == "settled":
                continue
            if rec["state"] in {"cleanup", "uncertain"}:
                stats["settled"] += self._teardown(rec["job_id"], rec, sandbox_alive=False)
                continue
            sandbox = states.get(rec.get("sandbox_id", ""))
            age = now - (rec.get("launched") or rec["created"])
            if rec["state"] == "reserved" and age > LAUNCH_GRACE_S:
                # The launch never completed (controller died mid-way): give the reservation back.
                stats["settled"] += self._teardown(rec["job_id"], rec, sandbox_alive=False)
            elif rec["state"] != "running":
                continue
            elif self._ended(rec, sandbox, age):
                stats["settled"] += self._teardown(rec["job_id"], rec, sandbox_alive=False)
            elif age > PROFILES[rec["profile"]].max_seconds + 120:
                stats["terminated"] += self._teardown(rec["job_id"], rec)
        stats["recovered"] = self._recover_queued()
        stats["folded"] = self.ledger.fold()
        return stats

    def _ended(self, rec: Mapping[str, Any], listed: SandboxState | None, age: float) -> bool:
        """Whether a running record's sandbox has really ended, not merely dropped out of the listing."""
        if listed is not None:
            return listed.finished
        if age <= GONE_GRACE_S:
            return False
        try:
            return not self.sandboxes.running(rec["sandbox_id"])
        except Exception:
            log.exception("could not look up sandbox %s", rec.get("sandbox_id"))
            return False  # unknown: keep the reservation; the deadline check ends it if it is overdue

    def _recover_queued(self) -> int:
        """Launch runners for queued jobs that have none (lost webhook, guard kill, capacity)."""
        recovered = 0
        if self.expired():
            return recovered
        for repo in sorted(self.cfg.repos):
            try:
                jobs = self.github.queued_jobs(repo)
            except Exception:  # noqa: BLE001 -- bad credentials must not stop the sweep's cleanup work
                log.exception("could not list queued jobs for %s", repo)
                continue
            for job in jobs:
                req = policy.request_from_job(repo, job)
                if isinstance(req, policy.Rejected):
                    continue
                rec = self.ledger.store.get(f"job:{req.job_id}")
                if rec is not None and (rec["state"] != "settled" or rec.get("tries", 0) >= MAX_TRIES):
                    continue
                created = job.get("created_at")
                if created and self.clock() - _parse_ts(created) < RECOVER_AFTER_S:
                    continue
                try:
                    resp = self._launch(req)
                except Exception:
                    log.exception("recovery launch failed job=%s", req.job_id)
                    continue
                recovered += resp.status == 200
        return recovered


def _parse_ts(value: str) -> float:
    from datetime import datetime

    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
