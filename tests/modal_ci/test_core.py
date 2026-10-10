"""Controller flows against fakes: authentication, replay, admission, credential handoff, cleanup."""

import hashlib
import hmac
import json

import pytest

from infra.modal_ci import ledger as L
from infra.modal_ci import policy
from infra.modal_ci.core import GONE_GRACE_S, Core, SandboxState, SpendUnknown
from infra.modal_ci.github_api import GitHubError

REPO = "softnanolab/boileroom"
SHA = "a" * 40
SECRET = b"webhook-secret"
ACTIVE_UNTIL = 2_000_000_000.0  # 2033: past every time these tests visit
APP_KEY = "-----BEGIN PRIVATE KEY-----app-key-material"
JIT = "ENCODED-JIT-CONFIG"


class FakeStore(dict):
    def put(self, key, value, *, skip_if_exists=False):
        if skip_if_exists and key in self:
            return False
        self[key] = value
        return True


class FakeGitHub:
    def __init__(self) -> None:
        self.job = {
            "id": 7,
            "run_id": 100,
            "run_attempt": 1,
            "head_sha": SHA,
            "status": "queued",
            "labels": ["self-hosted", "modal-ci", "job-100-1-unit-tests"],
        }
        self.run = {
            "id": 100,
            "run_attempt": 1,
            "event": "pull_request",
            "head_sha": SHA,
            "status": "queued",
            "repository": {"full_name": REPO},
            "head_repository": {"full_name": REPO, "fork": False},
        }
        self.deleted: list[int] = []
        self.minted: list[tuple[str, str, list[str]]] = []
        self.queued: list[dict] = []
        self.fail_jit = False

    def get_job(self, repo, job_id):
        return self.job

    def get_run(self, repo, run_id):
        return self.run

    def queued_jobs(self, repo):
        return self.queued

    def generate_jit(self, repo, name, labels):
        if self.fail_jit:
            raise GitHubError(500, "generate jit config")
        self.minted.append((repo, name, labels))
        return JIT, 555

    def delete_runner(self, repo, runner_id):
        self.deleted.append(runner_id)


class FakeSandboxes:
    def __init__(self) -> None:
        self.created: list[dict] = []
        self.terminated: list[str] = []
        self.live: dict[str, SandboxState] = {}
        self.unlisted: set[str] = set()  # alive on Modal's side but missing from the listing
        self.fail_create = False
        self.lookup_fails = False

    def create(self, *, profile, name, env, tags):
        if self.fail_create:
            raise RuntimeError("modal down")
        sandbox_id = f"sb-{len(self.created)}"
        self.created.append({"profile": profile, "name": name, "env": dict(env), "tags": dict(tags)})
        self.live[sandbox_id] = SandboxState(sandbox_id, finished=False)
        return sandbox_id

    def terminate(self, sandbox_id):
        self.terminated.append(sandbox_id)
        self.live.pop(sandbox_id, None)

    def states(self):
        return dict(self.live)

    def running(self, sandbox_id):
        if self.lookup_fails:
            raise RuntimeError("modal down")
        return sandbox_id in self.live or sandbox_id in self.unlisted


class Clock:
    now = 1_000_000.0

    def __call__(self):
        return self.now


@pytest.fixture
def world():
    clock = Clock()
    gh, sandboxes = FakeGitHub(), FakeSandboxes()
    store, deliveries = FakeStore(), FakeStore()
    core = Core(
        cfg=policy.Config(repos=frozenset({REPO}), installation_id=42, active_until=ACTIVE_UNTIL),
        webhook_secret=SECRET,
        github=gh,
        ledger=L.Ledger(store, L.Limits(ceiling_usd=20.0, daily_cap_usd=20.0, max_concurrent=4), clock),
        sandboxes=sandboxes,
        deliveries=deliveries,
        clock=clock,
    )
    return core, gh, sandboxes, store, deliveries, clock


def delivery(action="queued", *, job=None, delivery_id="d1", secret=SECRET, event="workflow_job", **top):
    payload = {
        "action": action,
        "repository": {"full_name": REPO},
        "installation": {"id": 42},
        "workflow_job": {
            "id": 7,
            "run_id": 100,
            "run_attempt": 1,
            "head_sha": SHA,
            "labels": ["self-hosted", "modal-ci", "job-100-1-unit-tests"],
            **(job or {}),
        },
        **top,
    }
    body = json.dumps(payload).encode()
    headers = {
        "X-Hub-Signature-256": "sha256=" + hmac.new(secret, body, hashlib.sha256).hexdigest(),
        "X-GitHub-Delivery": delivery_id,
        "X-GitHub-Event": event,
    }
    return headers, body


def test_valid_delivery_launches_one_sandbox_bound_to_the_job(world) -> None:
    core, gh, sandboxes, store, *_ = world
    resp = core.handle_webhook(*delivery())
    assert (resp.status, resp.body) == (200, "launched")
    (created,) = sandboxes.created
    assert json.loads(created["env"]["CI_BINDING"]) == {
        "repo": REPO,
        "job_id": 7,
        "run_id": 100,
        "run_attempt": 1,
        "job_key": "unit-tests",
        "head_sha": SHA,
        "runner_name": created["name"],
    }
    assert gh.minted == [(REPO, created["name"], ["self-hosted", "Linux", "X64", "modal-ci", "job-100-1-unit-tests"])]
    assert store["job:7"]["state"] == "running" and store["job:7"]["sandbox_id"] == "sb-0"


def test_only_the_jit_config_and_binding_reach_the_sandbox(world) -> None:
    core, _, sandboxes, *_ = world
    core.handle_webhook(*delivery())
    env = sandboxes.created[0]["env"]
    assert set(env) == {"CI_BINDING", "CI_JIT", "CI_MAX_SECONDS"}
    assert env["CI_JIT"] == JIT
    blob = json.dumps({"env": env, "tags": sandboxes.created[0]["tags"]})
    assert APP_KEY not in blob and SECRET.decode() not in blob


def test_bad_or_missing_signature_is_401_and_does_nothing(world) -> None:
    core, gh, sandboxes, store, deliveries, _ = world
    headers, body = delivery(secret=b"wrong")
    assert core.handle_webhook(headers, body).status == 401
    headers.pop("X-Hub-Signature-256")
    assert core.handle_webhook(headers, body).status == 401
    assert not sandboxes.created and not gh.minted and not store and not deliveries


def test_signature_covers_the_exact_body(world) -> None:
    core, _, sandboxes, *_ = world
    headers, body = delivery()
    assert core.handle_webhook(headers, body.replace(b'"id": 7', b'"id": 8')).status == 401
    assert not sandboxes.created


def test_replayed_delivery_is_a_noop(world) -> None:
    core, gh, sandboxes, *_ = world
    request = delivery()
    assert core.handle_webhook(*request).status == 200
    again = core.handle_webhook(*request)
    assert (again.status, again.body) == (200, "duplicate delivery")
    assert len(sandboxes.created) == len(gh.minted) == 1


def test_second_delivery_for_the_same_job_does_not_launch_twice(world) -> None:
    core, _, sandboxes, *_ = world
    core.handle_webhook(*delivery(delivery_id="d1"))
    resp = core.handle_webhook(*delivery(delivery_id="d2"))
    assert resp.status == 204 and resp.body == "already_launched"
    assert len(sandboxes.created) == 1


def test_stale_replay_of_a_queued_event_for_a_started_job_is_rejected(world) -> None:
    core, gh, sandboxes, *_ = world
    gh.job["status"] = "in_progress"
    resp = core.handle_webhook(*delivery())
    assert resp.status == 204 and resp.body == "job_not_queued:in_progress"
    assert not sandboxes.created and not gh.minted


def test_fork_pull_request_gets_no_runner(world) -> None:
    core, gh, sandboxes, *_ = world
    gh.run["head_repository"] = {"full_name": "mallory/boileroom", "fork": True}
    resp = core.handle_webhook(*delivery())
    assert resp.status == 204 and resp.body == "head_repository_not_this_repo"
    assert not sandboxes.created and not gh.minted


@pytest.mark.parametrize("event", ["pull_request_target", "workflow_run", "issue_comment"])
def test_privileged_events_get_no_runner(world, event) -> None:
    core, gh, sandboxes, *_ = world
    gh.run["event"] = event
    assert core.handle_webhook(*delivery()).status == 204
    assert not sandboxes.created and not gh.minted


def test_webhook_claim_disagreeing_with_github_is_rejected(world) -> None:
    core, gh, sandboxes, *_ = world
    gh.job["head_sha"] = "b" * 40  # the webhook said SHA, GitHub says otherwise
    assert core.handle_webhook(*delivery()).status == 204
    assert not sandboxes.created


def test_other_repositories_and_installations_are_ignored(world) -> None:
    core, _, sandboxes, *_ = world
    assert core.handle_webhook(*delivery(repository={"full_name": "evil/repo"})).status == 204
    assert core.handle_webhook(*delivery(delivery_id="d2", installation={"id": 1})).status == 204
    assert not sandboxes.created


def test_ping_and_other_events_are_acknowledged_without_action(world) -> None:
    core, _, sandboxes, *_ = world
    assert core.handle_webhook(*delivery(event="ping")).body == "pong"
    assert core.handle_webhook(*delivery(event="push", delivery_id="d2")).status == 204
    assert not sandboxes.created


def test_ping_still_requires_a_valid_signature(world) -> None:
    core, *_ = world
    assert core.handle_webhook(*delivery(event="ping", secret=b"wrong")).status == 401


def test_github_outage_during_verification_asks_for_retry_without_spending(world) -> None:
    core, gh, sandboxes, store, *_ = world

    def boom(*_):
        raise GitHubError(502, "get job")

    gh.get_job = boom
    assert core.handle_webhook(*delivery()).status == 503
    assert not sandboxes.created and "job:7" not in store


def test_failed_sandbox_creation_keeps_reservation_until_hard_expiry(world) -> None:
    core, gh, sandboxes, store, deliveries, clock = world
    sandboxes.fail_create = True
    assert core.handle_webhook(*delivery()).status == 503
    assert gh.deleted == []
    assert store["job:7"]["state"] == "uncertain"
    assert core.ledger.totals()["active"] == 1
    assert not deliveries  # the claim is released so a redelivery can retry
    sandboxes.fail_create = False
    assert core.handle_webhook(*delivery()).body == "already_launched"
    clock.now += L.PROFILES["modal-ci"].hard_timeout_s + L.STARTUP_ALLOWANCE_S
    assert core.reconcile()["settled"] == 1
    assert gh.deleted == [555]
    assert store["job:7"]["actual_usd"] == store["job:7"]["reserved_usd"]
    # A separate delivery may retry only after the uncertain launch is settled.
    assert core.handle_webhook(*delivery(delivery_id="after-expiry")).status == 200
    assert store["job:7"]["tries"] == 2


@pytest.mark.parametrize("ceiling,denial", [(0.3, "budget_exhausted"), (20.0, "capacity")])
def test_lost_create_response_cannot_free_capacity_or_budget(world, monkeypatch, ceiling, denial) -> None:
    core, gh, sandboxes, store, _, clock = world
    core.ledger.limits = L.Limits(ceiling_usd=ceiling, daily_cap_usd=ceiling, max_concurrent=1)
    create = sandboxes.create

    def lost_response(**kwargs):
        create(**kwargs)
        raise TimeoutError("sandbox exists but its response was lost")

    monkeypatch.setattr(sandboxes, "create", lost_response)
    assert core.handle_webhook(*delivery()).status == 503
    assert len(sandboxes.live) == 1
    assert core.ledger.totals()["committed"] == L.worst_case_usd(L.PROFILES["modal-ci"])
    # Completion delivery cannot release an unknown sandbox either.
    assert core.handle_webhook(*delivery("completed", delivery_id="done")).status == 202
    gh.job["id"] = 8
    assert core.handle_webhook(*delivery(job={"id": 8}, delivery_id="other")).body == denial
    assert len(sandboxes.live) == 1
    # The orphan is terminated, but an unreliable listing is not evidence that
    # no other creation exists: keep the reservation through its hard lifetime.
    assert core.reconcile()["orphans"] == 1
    assert core.ledger.totals()["active"] == 1
    clock.now += L.PROFILES["modal-ci"].hard_timeout_s + L.STARTUP_ALLOWANCE_S
    assert core.reconcile()["settled"] == 1
    assert store["job:7"]["actual_usd"] == store["job:7"]["reserved_usd"]


def test_failed_jit_minting_leaves_nothing_running(world) -> None:
    core, gh, sandboxes, store, *_ = world
    gh.fail_jit = True
    assert core.handle_webhook(*delivery()).status == 503
    assert not sandboxes.created and gh.deleted == []
    assert store["job:7"]["state"] == "settled"


def test_budget_exhaustion_refuses_to_launch(world) -> None:
    core, _, sandboxes, *_ = world
    core.ledger.limits = L.Limits(ceiling_usd=0.1, daily_cap_usd=0.1, max_concurrent=4)  # below one job's worst case
    resp = core.handle_webhook(*delivery())
    assert (resp.status, resp.body) == (503, "budget_exhausted")
    assert not sandboxes.created


def test_measured_spend_feeds_the_ceiling(world) -> None:
    core, _, sandboxes, *_ = world
    core.external_spend = lambda: 19.9
    assert core.handle_webhook(*delivery()).body == "budget_exhausted"
    assert not sandboxes.created


def test_completed_event_tears_down_sandbox_and_runner(world) -> None:
    core, gh, sandboxes, store, *_ = world
    core.handle_webhook(*delivery())
    resp = core.handle_webhook(*delivery("completed", delivery_id="d2"))
    assert resp.status == 200
    assert sandboxes.terminated == ["sb-0"] and gh.deleted == [555]
    assert store["job:7"]["state"] == "settled"
    assert core.handle_webhook(*delivery("completed", delivery_id="d3")).status == 204


def test_runner_deletion_failure_retries_without_billing_cleanup_wait(world, monkeypatch) -> None:
    core, gh, sandboxes, store, _, clock = world
    core.handle_webhook(*delivery())
    delete_runner = gh.delete_runner

    def unavailable(*args):
        raise GitHubError(503, "delete runner")

    monkeypatch.setattr(gh, "delete_runner", unavailable)
    clock.now += 30
    assert core.handle_webhook(*delivery("completed", delivery_id="done")).status == 202
    assert store["job:7"]["state"] == "cleanup"
    assert store["job:7"]["runner_id"] == 555
    assert sandboxes.terminated == ["sb-0"]
    clock.now += 300
    assert core.reconcile()["settled"] == 0
    assert store["job:7"]["state"] == "cleanup"
    monkeypatch.setattr(gh, "delete_runner", delete_runner)
    assert core.reconcile()["settled"] == 1
    assert gh.deleted == [555] and sandboxes.terminated == ["sb-0"]
    assert store["job:7"]["actual_usd"] == pytest.approx(L.cost_usd(L.PROFILES["modal-ci"], 30))


def test_failed_launch_retains_registration_cleanup_before_retry(world, monkeypatch) -> None:
    core, gh, sandboxes, store, _, clock = world
    delete_runner = gh.delete_runner

    def unavailable(*args):
        raise GitHubError(503, "delete runner")

    sandboxes.fail_create = True
    monkeypatch.setattr(gh, "delete_runner", unavailable)
    assert core.handle_webhook(*delivery()).status == 503
    assert store["job:7"]["state"] == "uncertain"
    assert core.handle_webhook(*delivery(delivery_id="retry")).body == "already_launched"
    assert len(gh.minted) == 1
    clock.now += L.PROFILES["modal-ci"].hard_timeout_s + L.STARTUP_ALLOWANCE_S
    assert core.reconcile()["settled"] == 0
    assert store["job:7"]["state"] == "cleanup"
    monkeypatch.setattr(gh, "delete_runner", delete_runner)
    assert core.reconcile()["settled"] == 1
    assert gh.deleted == [555] and store["job:7"]["actual_usd"] == store["job:7"]["reserved_usd"]
    sandboxes.fail_create = False
    assert core.handle_webhook(*delivery(delivery_id="retry-after-cleanup")).status == 200


def test_settlement_includes_sandbox_creation_time(world, monkeypatch) -> None:
    core, _, sandboxes, store, _, clock = world
    create = sandboxes.create

    def slow_create(**kwargs):
        clock.now += 20
        return create(**kwargs)

    monkeypatch.setattr(sandboxes, "create", slow_create)
    started = clock.now
    core.handle_webhook(*delivery())
    assert store["job:7"]["launched"] == started
    clock.now += 10
    core.handle_webhook(*delivery("completed", delivery_id="done"))
    assert store["job:7"]["actual_usd"] == pytest.approx(L.cost_usd(L.PROFILES["modal-ci"], 30))


def test_completed_event_ends_the_runner_that_ran_the_job_not_the_one_bound_to_it(world) -> None:
    """Runner A (job 7) ended up running job 8; job 8 finishing must not kill job 7's own sibling B."""
    core, gh, sandboxes, store, *_ = world
    core.handle_webhook(*delivery(job={"id": 7}, delivery_id="a"))
    gh.job = {**gh.job, "id": 8}  # a second leg of the same job key: same label, different job id
    core.handle_webhook(*delivery(job={"id": 8}, delivery_id="b"))
    runner_of_7 = store["job:7"]["runner_name"]
    done = delivery("completed", job={"id": 8, "runner_name": runner_of_7}, delivery_id="c")
    assert core.handle_webhook(*done).status == 200
    assert sandboxes.terminated == [store["job:7"]["sandbox_id"]]
    assert store["job:7"]["state"] == "settled" and store["job:8"]["state"] == "running"


def test_completed_event_from_a_runner_we_do_not_own_is_ignored(world) -> None:
    core, _, sandboxes, store, *_ = world
    core.handle_webhook(*delivery())
    resp = core.handle_webhook(*delivery("completed", job={"runner_name": "someone-elses"}, delivery_id="d2"))
    assert resp.status == 204 and sandboxes.terminated == [] and store["job:7"]["state"] == "running"


def test_failed_termination_is_reported_as_deferred(world) -> None:
    core, _, sandboxes, store, *_ = world
    core.handle_webhook(*delivery())

    def boom(sandbox_id):
        raise RuntimeError("modal down")

    sandboxes.terminate = boom
    resp = core.handle_webhook(*delivery("completed", delivery_id="d2"))
    assert resp.status == 202 and store["job:7"]["state"] == "running"


def test_completed_event_for_a_job_in_another_repo_is_ignored(world) -> None:
    core, _, sandboxes, store, *_ = world
    core.handle_webhook(*delivery())
    store["job:7"]["repo"] = "softnanolab/bakeoff"
    assert core.handle_webhook(*delivery("completed", delivery_id="d2")).status == 204
    assert sandboxes.terminated == []


def test_reconcile_settles_jobs_whose_sandbox_is_gone(world) -> None:
    core, gh, sandboxes, store, _, clock = world
    core.handle_webhook(*delivery())
    clock.now += 300
    sandboxes.live.clear()  # the supervisor exited and Modal ended the sandbox
    assert core.reconcile()["settled"] == 1
    assert store["job:7"]["state"] == "settled" and gh.deleted == [555]
    assert 0 < store["job:7"]["actual_usd"] < store["job:7"]["reserved_usd"]


def test_reconcile_terminates_a_sandbox_that_outlives_its_deadline(world) -> None:
    core, _, sandboxes, store, _, clock = world
    core.handle_webhook(*delivery())
    clock.now += L.PROFILES["modal-ci"].max_seconds + 200
    assert core.reconcile()["terminated"] == 1
    assert sandboxes.terminated == ["sb-0"] and store["job:7"]["state"] == "settled"


def test_reconcile_still_cleans_up_when_github_credentials_are_broken(world) -> None:
    core, gh, sandboxes, store, _, clock = world
    core.handle_webhook(*delivery())
    clock.now += 300
    sandboxes.live.clear()

    def broken(repo):
        raise ValueError("could not parse the private key")

    gh.queued_jobs = broken
    stats = core.reconcile()  # must not raise: settling and folding do not depend on GitHub
    assert stats["settled"] == 1 and stats["recovered"] == 0
    assert store["job:7"]["state"] == "settled"


def test_reconcile_leaves_healthy_sandboxes_alone(world) -> None:
    core, _, sandboxes, store, _, clock = world
    core.handle_webhook(*delivery())
    clock.now += 600
    assert core.reconcile() == {"settled": 0, "terminated": 0, "orphans": 0, "recovered": 0, "folded": 0}
    assert sandboxes.terminated == [] and store["job:7"]["state"] == "running"


def test_reconcile_terminates_a_sandbox_no_record_points_at(world) -> None:
    core, _, sandboxes, store, *_ = world
    sandboxes.live["sb-orphan"] = SandboxState("sb-orphan", finished=False)  # launch died before bookkeeping
    sandboxes.live["sb-selftest"] = SandboxState("sb-selftest", finished=False, ephemeral=True)
    sandboxes.live["sb-done"] = SandboxState("sb-done", finished=True)
    assert core.reconcile()["orphans"] == 1
    assert sandboxes.terminated == ["sb-orphan"]


def test_reconcile_gives_back_a_launch_that_never_completed(world) -> None:
    core, _, _, store, _, clock = world
    core.ledger.reserve(9, L.PROFILES["modal-ci"], {"repo": REPO})  # controller died before create
    clock.now += 700
    assert core.reconcile()["settled"] == 1 and store["job:9"]["state"] == "settled"


def queued_api_job(job_id=7, created_at="2026-10-10T10:00:00Z"):
    return {
        "id": job_id,
        "run_id": 100,
        "run_attempt": 1,
        "head_sha": SHA,
        "status": "queued",
        "created_at": created_at,
        "labels": ["self-hosted", "modal-ci", "job-100-1-unit-tests"],
    }


def test_reconcile_recovers_a_queued_job_whose_delivery_was_lost(world) -> None:
    core, gh, sandboxes, _, _, clock = world
    clock.now = 1_791_626_400.0 + 600  # 2026-10-10T10:00:00Z + 10 min
    gh.queued = [queued_api_job()]
    assert core.reconcile()["recovered"] == 1
    assert len(sandboxes.created) == 1


def test_reconcile_waits_before_treating_a_fresh_queued_job_as_lost(world) -> None:
    core, gh, sandboxes, _, _, clock = world
    clock.now = 1_791_626_400.0 + 30
    gh.queued = [queued_api_job()]
    assert core.reconcile()["recovered"] == 0 and not sandboxes.created


def test_recovery_is_bounded_per_job(world) -> None:
    core, gh, sandboxes, _, _, clock = world
    clock.now = 1_791_626_400.0 + 600
    gh.queued = [queued_api_job()]
    for _ in range(6):
        core.reconcile()
        sandboxes.live.clear()  # each runner dies at once (e.g. the guard keeps killing it)
        clock.now += GONE_GRACE_S + 1  # long enough that its absence from the listing is believed
        core.reconcile()
    assert len(sandboxes.created) == 3  # MAX_TRIES


def test_recovery_ignores_labels_that_do_not_belong_to_the_run(world) -> None:
    """The recovery path enforces the same label/run binding as the webhook path."""
    core, gh, sandboxes, _, _, clock = world
    clock.now = 1_791_626_400.0 + 600
    gh.queued = [{**queued_api_job(), "labels": ["self-hosted", "modal-ci", "job-1-1-unit-tests"]}]
    assert core.reconcile()["recovered"] == 0 and not sandboxes.created


def test_recovery_ignores_jobs_that_did_not_ask_for_modal_ci(world) -> None:
    core, gh, sandboxes, _, _, clock = world
    clock.now = 1_791_626_400.0 + 600
    gh.queued = [{**queued_api_job(), "labels": ["self-hosted", "gpu"]}]
    assert core.reconcile()["recovered"] == 0 and not sandboxes.created


def test_a_sandbox_missing_from_the_listing_is_not_settled_while_it_is_new(world) -> None:
    core, _, sandboxes, store, _, clock = world
    core.handle_webhook(*delivery())
    sandboxes.live.clear()  # the listing lags: Modal has not caught up with the sandbox just created
    clock.now += GONE_GRACE_S - 10
    assert core.reconcile()["settled"] == 0
    assert store["job:7"]["state"] == "running" and sandboxes.terminated == []


def test_a_sandbox_missing_from_the_listing_but_still_running_is_left_alone(world) -> None:
    core, gh, sandboxes, store, _, clock = world
    core.handle_webhook(*delivery())
    sandboxes.unlisted.add("sb-0")
    sandboxes.live.clear()  # partial listing: it is still running, so settling it would free a live job's budget
    clock.now += 600
    assert core.reconcile()["settled"] == 0
    assert store["job:7"]["state"] == "running" and gh.deleted == []


def test_an_unanswerable_sandbox_lookup_keeps_the_reservation_until_the_deadline(world) -> None:
    core, _, sandboxes, store, _, clock = world
    core.handle_webhook(*delivery())
    sandboxes.live.clear()
    sandboxes.lookup_fails = True
    clock.now += 600
    assert core.reconcile()["settled"] == 0 and store["job:7"]["state"] == "running"
    clock.now += L.PROFILES["modal-ci"].max_seconds  # past the deadline it is ended regardless
    assert core.reconcile()["terminated"] == 1 and store["job:7"]["state"] == "settled"


def test_unknown_measured_spend_blocks_launches_and_a_redelivery_can_retry(world) -> None:
    core, _, sandboxes, store, deliveries, _ = world

    def unknown():
        raise SpendUnknown("billing report down")

    core.external_spend = unknown
    resp = core.handle_webhook(*delivery())
    assert (resp.status, resp.body) == (503, "spend unknown")
    assert not sandboxes.created and "job:7" not in store
    assert not deliveries  # the claim is released, so GitHub's redelivery is not dropped as a duplicate
    core.external_spend = lambda: 0.0
    assert core.handle_webhook(*delivery()).status == 200


def test_a_503_from_verification_releases_the_delivery_claim(world) -> None:
    core, gh, sandboxes, _, deliveries, _ = world
    real = gh.get_run

    def down(repo, run_id):
        raise GitHubError(502, "get run")

    gh.get_run = down
    assert core.handle_webhook(*delivery()).status == 503
    assert not deliveries
    gh.get_run = real
    assert core.handle_webhook(*delivery()).status == 200 and sandboxes.created


def test_a_rejection_that_holds_for_the_whole_run_is_looked_up_once(world) -> None:
    core, gh, sandboxes, *_ = world
    gh.run = {**gh.run, "head_repository": {"full_name": "someone/boileroom", "fork": True}}
    calls = {"run": 0, "job": 0}
    get_run, get_job = gh.get_run, gh.get_job

    def counted_run(repo, run_id):
        calls["run"] += 1
        return get_run(repo, run_id)

    def counted_job(repo, job_id):
        calls["job"] += 1
        return get_job(repo, job_id)

    gh.get_run, gh.get_job = counted_run, counted_job
    for i in range(5):  # five jobs of one fork run (a matrix)
        resp = core.handle_webhook(*delivery(job={"id": 100 + i}, delivery_id=f"d{i}"))
        assert (resp.status, resp.body) == (204, "head_repository_not_this_repo")
    assert calls == {"run": 1, "job": 0}
    assert not sandboxes.created


def test_a_transient_run_state_is_not_remembered(world) -> None:
    core, gh, sandboxes, *_ = world
    gh.run = {**gh.run, "status": "completed"}
    assert core.handle_webhook(*delivery()).body == "run_not_live:completed"
    gh.run = {**gh.run, "status": "queued"}
    assert core.handle_webhook(*delivery(delivery_id="d2")).status == 200 and sandboxes.created


def test_runner_names_are_unguessable(world) -> None:
    core, _, sandboxes, *_ = world
    core.handle_webhook(*delivery())
    suffix = sandboxes.created[0]["name"].rsplit("-", 1)[1]
    assert len(suffix) == 32 and int(suffix, 16) >= 0  # 128 bits of randomness


# -- activation deadline ---------------------------------------------------------------------


def test_nothing_launches_once_the_activation_deadline_has_passed(world) -> None:
    core, gh, sandboxes, store, _, clock = world
    clock.now = ACTIVE_UNTIL  # the deadline itself is already too late
    resp = core.handle_webhook(*delivery())
    assert (resp.status, resp.body) == (204, "activation_expired")
    assert not sandboxes.created and not store and not gh.minted


def test_the_deadline_does_not_cut_off_a_job_already_running(world) -> None:
    core, gh, sandboxes, store, _, clock = world
    assert core.handle_webhook(*delivery()).body == "launched"
    clock.now = ACTIVE_UNTIL + 10
    assert core.handle_webhook(*delivery("completed", delivery_id="d2")).status == 200
    assert store["job:7"]["state"] == "settled" and gh.deleted == [555]


def test_recovery_sweep_launches_nothing_after_the_deadline_but_still_cleans_up(world) -> None:
    core, gh, sandboxes, store, _, clock = world
    core.handle_webhook(*delivery())
    gh.queued = [queued_api_job(job_id=8)]
    clock.now = ACTIVE_UNTIL + 10
    sandboxes.live.clear()
    stats = core.reconcile()
    assert stats["recovered"] == 0 and stats["settled"] == 1 and len(sandboxes.created) == 1


@pytest.mark.parametrize("label", sorted(policy.PROFILE_LABELS))
def test_profile_is_reserved_registered_and_passed_to_worker(world, label) -> None:
    core, gh, sandboxes, store, *_ = world
    labels = ["self-hosted", "modal-ci", label, "job-100-1-unit-tests"]
    gh.job["labels"] = labels
    assert core.handle_webhook(*delivery(job={"labels": labels})).body == "launched"
    profile = L.PROFILES[label]
    assert sandboxes.created[0]["profile"] == profile
    assert sandboxes.created[0]["env"]["CI_MAX_SECONDS"] == str(profile.max_seconds)
    assert label in gh.minted[0][2]
    assert store["job:7"]["profile"] == label
    assert store["job:7"]["reserved_usd"] == pytest.approx(L.worst_case_usd(profile))


def test_uncertain_long_worker_keeps_full_reservation_past_default_lifetime(world) -> None:
    core, gh, sandboxes, store, _, clock = world
    labels = ["self-hosted", "modal-ci", "modal-ci-long", "job-100-1-unit-tests"]
    gh.job["labels"] = labels
    sandboxes.fail_create = True
    assert core.handle_webhook(*delivery(job={"labels": labels})).status == 503
    clock.now += L.PROFILES["modal-ci"].hard_timeout_s + L.STARTUP_ALLOWANCE_S
    core.reconcile()
    assert core.ledger.totals()["active"] == 1
    assert store["job:7"]["reserved_usd"] == L.worst_case_usd(L.PROFILES["modal-ci-long"])
    clock.now += L.PROFILES["modal-ci-long"].hard_timeout_s
    core.reconcile()
    assert core.ledger.totals()["active"] == 0
    assert store["job:7"]["actual_usd"] == store["job:7"]["reserved_usd"]
