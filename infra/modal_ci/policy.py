"""Admission policy for the Modal CI controller.

Pure functions over webhook/API payloads: no network, no Modal, stdlib only. The
controller calls these before it spends money or mints a credential, and every
function fails closed: anything unexpected is a rejection with a reason.
"""

from __future__ import annotations

import hashlib
import hmac
import re
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

# `pull_request_target`, `issue_comment` and friends run with secrets and attacker-chosen text, so
# they are never admitted. `workflow_run` is left out too: its run's `head_repository` is the base
# repository even when a fork PR triggered it, so `check_run` could not tell the two apart. Add
# it here only together with a check that can.
ALLOWED_EVENTS = frozenset({"push", "pull_request", "workflow_dispatch", "schedule"})

RUNNER_LABEL = "modal-ci"
# GitHub's JIT endpoint registers exactly the labels it is given, so the defaults are spelled out.
RUNNER_LABELS = ("self-hosted", "Linux", "X64", RUNNER_LABEL)
JOB_LABEL_RE = re.compile(
    r"^job-(?P<run_id>[0-9]{1,18})-(?P<attempt>[0-9]{1,4})-(?P<key>[A-Za-z0-9_][A-Za-z0-9_-]{0,99})$"
)
REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
LIVE_RUN_STATUSES = frozenset({"queued", "in_progress", "waiting", "pending", "requested"})


@dataclass(frozen=True)
class Config:
    """Static controller configuration; everything the policy may trust."""

    repos: frozenset[str]
    installation_id: int
    # Epoch seconds from which nothing new launches. Required, so a controller can never be left on indefinitely by
    # omission: the saving this runner exists for depends on a GitHub allowance and Modal credits that both end.
    active_until: float


def parse_deadline(value: str) -> float:
    """Epoch seconds of an ISO-8601 instant that names its time zone; anything else is an error, not a guess."""
    when = datetime.fromisoformat(value)
    if when.tzinfo is None:
        raise ValueError(f"deadline {value!r} has no time zone")
    return when.astimezone(UTC).timestamp()


@dataclass(frozen=True)
class JobRequest:
    """A `workflow_job` that asks for a Modal runner, as claimed by the webhook payload."""

    repo: str
    job_id: int
    run_id: int
    run_attempt: int
    job_key: str
    head_sha: str
    labels: tuple[str, ...]


@dataclass(frozen=True)
class Rejected:
    reason: str


@dataclass(frozen=True)
class Cleanup:
    """A finished job whose sandbox (if any) should be torn down."""

    repo: str
    job_id: int
    runner_name: str | None  # the runner that ran it; `None` if it was never picked up


def verify_signature(secret: bytes, body: bytes, header: str | None) -> bool:
    """Check GitHub's `X-Hub-Signature-256` over the *raw* body in constant time."""
    if not secret or not header or not header.startswith("sha256="):
        return False
    expected = hmac.new(secret, body, hashlib.sha256).hexdigest().encode()
    # Bytes, not str: `compare_digest` raises on a non-ASCII str, which would turn a bad header into a 500.
    return hmac.compare_digest(expected, header.removeprefix("sha256=").encode())


def parse_job_label(labels: Collection[str]) -> tuple[int, int, str] | None:
    """Return `(run_id, attempt, job_key)` iff exactly one per-job label is present."""
    matches = [m for label in labels if (m := JOB_LABEL_RE.match(label))]
    if len(matches) != 1:
        return None
    m = matches[0]
    return int(m["run_id"]), int(m["attempt"]), m["key"]


def modal_labels(job: Mapping[str, Any]) -> list[str] | Rejected:
    """The job's labels, if it asks for a Modal runner at all."""
    labels = job.get("labels")
    if not isinstance(labels, list) or not all(isinstance(x, str) for x in labels):
        return Rejected("bad_labels")
    if RUNNER_LABEL not in labels or "self-hosted" not in labels:
        return Rejected("not_a_modal_ci_job")
    return labels


def request_from_job(repo: str, job: Mapping[str, Any]) -> JobRequest | Rejected:
    """Validate a job record that asks for a Modal runner into a `JobRequest`.

    The webhook payload's `workflow_job` and the REST API's job object share these fields, so both
    the webhook path and the periodic recovery sweep admit jobs through this one function.
    """
    labels = modal_labels(job)
    if isinstance(labels, Rejected):
        return labels
    job_id = job.get("id")
    if not isinstance(job_id, int) or isinstance(job_id, bool):
        return Rejected("bad_job_id")
    parsed = parse_job_label(labels)
    if parsed is None:
        return Rejected("missing_or_ambiguous_job_label")
    run_id, attempt, key = parsed
    if job.get("run_id") != run_id or job.get("run_attempt") != attempt:
        return Rejected("label_run_mismatch")
    sha = job.get("head_sha")
    if not isinstance(sha, str) or not SHA_RE.match(sha):
        return Rejected("bad_head_sha")
    return JobRequest(
        repo=repo,
        job_id=job_id,
        run_id=run_id,
        run_attempt=attempt,
        job_key=key,
        head_sha=sha,
        labels=tuple(labels),
    )


def parse_workflow_job(payload: Mapping[str, Any], cfg: Config) -> JobRequest | Cleanup | Rejected:
    """Interpret an already-authenticated `workflow_job` delivery."""
    repo = (payload.get("repository") or {}).get("full_name")
    if not isinstance(repo, str) or not REPO_RE.match(repo) or repo not in cfg.repos:
        return Rejected("repo_not_allowed")
    if (payload.get("installation") or {}).get("id") != cfg.installation_id:
        return Rejected("wrong_installation")

    job = payload.get("workflow_job")
    if not isinstance(job, dict):
        return Rejected("no_workflow_job")
    labels = modal_labels(job)
    if isinstance(labels, Rejected):
        return labels

    action = payload.get("action")
    if action == "completed":
        job_id = job.get("id")
        if not isinstance(job_id, int) or isinstance(job_id, bool):
            return Rejected("bad_job_id")
        runner_name = job.get("runner_name")
        return Cleanup(repo=repo, job_id=job_id, runner_name=runner_name if isinstance(runner_name, str) else None)
    if action != "queued":
        return Rejected(f"ignored_action:{action}")
    return request_from_job(repo, job)


def check_job(req: JobRequest, api_job: Mapping[str, Any]) -> str | None:
    """Compare the webhook claim against GitHub's own record of the job. `None` means admit."""
    if api_job.get("id") != req.job_id or api_job.get("run_id") != req.run_id:
        return "api_job_mismatch"
    if api_job.get("run_attempt") != req.run_attempt or api_job.get("head_sha") != req.head_sha:
        return "api_job_mismatch"
    if api_job.get("status") != "queued":
        return f"job_not_queued:{api_job.get('status')}"
    if sorted(api_job.get("labels") or []) != sorted(req.labels):
        return "api_labels_mismatch"
    return None


def check_run(req: JobRequest, api_run: Mapping[str, Any]) -> str | None:
    """Check the workflow run is same-repo code from an allowed event. `None` means admit."""
    if api_run.get("id") != req.run_id or api_run.get("run_attempt") != req.run_attempt:
        return "api_run_mismatch"
    if api_run.get("event") not in ALLOWED_EVENTS:
        return f"event_not_allowed:{api_run.get('event')}"
    if api_run.get("head_sha") != req.head_sha:
        return "api_run_sha_mismatch"
    if (api_run.get("repository") or {}).get("full_name") != req.repo:
        return "api_run_repo_mismatch"
    # A fork PR's workflow file comes from the PR itself, so it can ask for this runner. The head
    # repository is the one field the submitter cannot make equal to ours.
    head_repo = api_run.get("head_repository") or {}
    if head_repo.get("full_name") != req.repo or head_repo.get("fork"):
        return "head_repository_not_this_repo"
    if api_run.get("status") not in LIVE_RUN_STATUSES:
        return f"run_not_live:{api_run.get('status')}"
    return None
