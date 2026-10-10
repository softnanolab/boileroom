"""Admission policy: what the Modal CI controller will and will not spend money on."""

import hashlib
import hmac

import pytest

from infra.modal_ci import policy

REPO = "softnanolab/boileroom"
SHA = "a" * 40
CFG = policy.Config(repos=frozenset({REPO}), installation_id=42, active_until=2_000_000_000.0)


def payload(**overrides):
    job = {
        "id": 7,
        "run_id": 100,
        "run_attempt": 1,
        "head_sha": SHA,
        "labels": ["self-hosted", "modal-ci", "job-100-1-unit-tests"],
    }
    body = {
        "action": "queued",
        "repository": {"full_name": REPO},
        "installation": {"id": 42},
        "workflow_job": job,
    }
    job.update(overrides.pop("job", {}))
    body.update(overrides)
    return body


def sign(secret: bytes, body: bytes) -> str:
    return "sha256=" + hmac.new(secret, body, hashlib.sha256).hexdigest()


def test_signature_accepts_only_the_exact_body_and_secret() -> None:
    body = b'{"a": 1}'
    good = sign(b"s3cret", body)
    assert policy.verify_signature(b"s3cret", body, good)
    assert not policy.verify_signature(b"other", body, good)
    assert not policy.verify_signature(b"s3cret", body + b" ", good)
    assert not policy.verify_signature(b"s3cret", body, None)
    assert not policy.verify_signature(b"s3cret", body, good.removeprefix("sha256="))
    assert not policy.verify_signature(b"", body, sign(b"", body))


def test_a_non_ascii_signature_header_is_a_failed_check_not_an_exception() -> None:
    assert policy.verify_signature(b"s3cret", b"{}", "sha256=\u00e9\u4e2d") is False


def test_queued_job_is_admitted_with_its_binding() -> None:
    req = policy.parse_workflow_job(payload(), CFG)
    assert req == policy.JobRequest(
        repo=REPO,
        job_id=7,
        run_id=100,
        run_attempt=1,
        job_key="unit-tests",
        head_sha=SHA,
        labels=("self-hosted", "modal-ci", "job-100-1-unit-tests"),
    )


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"repository": {"full_name": "evil/boileroom"}}, "repo_not_allowed"),
        ({"repository": {}}, "repo_not_allowed"),
        ({"installation": {"id": 43}}, "wrong_installation"),
        ({"installation": None}, "wrong_installation"),
        ({"job": {"labels": ["self-hosted", "job-100-1-unit-tests"]}}, "not_a_modal_ci_job"),
        ({"job": {"labels": ["modal-ci", "job-100-1-unit-tests"]}}, "not_a_modal_ci_job"),
        ({"job": {"labels": ["self-hosted", "modal-ci"]}}, "missing_or_ambiguous_job_label"),
        (
            {"job": {"labels": ["self-hosted", "modal-ci", "job-100-1-a", "job-100-1-b"]}},
            "missing_or_ambiguous_job_label",
        ),
        ({"job": {"run_id": 101}}, "label_run_mismatch"),
        ({"job": {"run_attempt": 2}}, "label_run_mismatch"),
        ({"job": {"head_sha": "main"}}, "bad_head_sha"),
        ({"job": {"id": "7"}}, "bad_job_id"),
        ({"job": {"id": True}}, "bad_job_id"),
        ({"action": "in_progress"}, "ignored_action:in_progress"),
        ({"workflow_job": None}, "no_workflow_job"),
        ({"job": {"labels": "modal-ci"}}, "bad_labels"),
    ],
)
def test_rejections(overrides, reason) -> None:
    assert policy.parse_workflow_job(payload(**overrides), CFG) == policy.Rejected(reason)


def test_completed_job_becomes_cleanup() -> None:
    assert policy.parse_workflow_job(payload(action="completed"), CFG) == policy.Cleanup(
        repo=REPO, job_id=7, runner_name=None
    )


def test_completed_job_names_the_runner_that_ran_it() -> None:
    body = payload(action="completed", job={"runner_name": "modal-7-abc"})
    assert policy.parse_workflow_job(body, CFG) == policy.Cleanup(repo=REPO, job_id=7, runner_name="modal-7-abc")


def test_request_from_job_is_what_the_recovery_sweep_admits_through() -> None:
    api_job = {
        "id": 7,
        "run_id": 100,
        "run_attempt": 1,
        "head_sha": SHA,
        "labels": ["self-hosted", "modal-ci", "job-100-1-unit-tests"],
    }
    assert policy.request_from_job(REPO, api_job) == request()
    assert policy.request_from_job(
        REPO, {**api_job, "labels": ["modal-ci", "job-100-1-unit-tests"]}
    ) == policy.Rejected("not_a_modal_ci_job")
    # a static label that does not belong to this run is refused on every path
    stolen = {**api_job, "labels": ["self-hosted", "modal-ci", "job-1-1-unit-tests"]}
    assert policy.request_from_job(REPO, stolen) == policy.Rejected("label_run_mismatch")
    assert policy.request_from_job(REPO, {**api_job, "head_sha": "not-a-sha"}) == policy.Rejected("bad_head_sha")


def test_completed_job_of_another_repo_is_still_rejected() -> None:
    body = payload(action="completed", repository={"full_name": "evil/x"})
    assert policy.parse_workflow_job(body, CFG) == policy.Rejected("repo_not_allowed")


def request() -> policy.JobRequest:
    req = policy.parse_workflow_job(payload(), CFG)
    assert isinstance(req, policy.JobRequest)
    return req


def api_job(**overrides):
    job = {
        "id": 7,
        "run_id": 100,
        "run_attempt": 1,
        "head_sha": SHA,
        "status": "queued",
        "labels": ["modal-ci", "self-hosted", "job-100-1-unit-tests"],
    }
    return {**job, **overrides}


def api_run(**overrides):
    run = {
        "id": 100,
        "run_attempt": 1,
        "event": "pull_request",
        "head_sha": SHA,
        "status": "queued",
        "repository": {"full_name": REPO},
        "head_repository": {"full_name": REPO, "fork": False},
    }
    return {**run, **overrides}


def test_api_job_must_agree_with_the_webhook_claim() -> None:
    assert policy.check_job(request(), api_job()) is None
    assert policy.check_job(request(), api_job(status="in_progress")) == "job_not_queued:in_progress"
    assert policy.check_job(request(), api_job(status="completed")) == "job_not_queued:completed"
    assert policy.check_job(request(), api_job(head_sha="b" * 40)) == "api_job_mismatch"
    assert policy.check_job(request(), api_job(run_id=5)) == "api_job_mismatch"
    assert policy.check_job(request(), api_job(labels=["self-hosted"])) == "api_labels_mismatch"


def test_run_from_same_repo_is_admitted() -> None:
    for event in sorted(policy.ALLOWED_EVENTS):
        assert policy.check_run(request(), api_run(event=event)) is None


@pytest.mark.parametrize("event", ["pull_request_target", "workflow_run", "issue_comment", "release", "fork"])
def test_events_that_carry_secrets_or_forks_are_refused(event) -> None:
    assert policy.check_run(request(), api_run(event=event)) == f"event_not_allowed:{event}"


def test_fork_pull_request_is_refused() -> None:
    forked = api_run(head_repository={"full_name": "mallory/boileroom", "fork": True})
    assert policy.check_run(request(), forked) == "head_repository_not_this_repo"
    # A fork flag alone is enough, and so is a missing head repository (deleted fork).
    assert policy.check_run(request(), api_run(head_repository={"full_name": REPO, "fork": True}))
    assert policy.check_run(request(), api_run(head_repository=None))


def test_run_binding_mismatches_are_refused() -> None:
    assert policy.check_run(request(), api_run(head_sha="b" * 40)) == "api_run_sha_mismatch"
    assert policy.check_run(request(), api_run(run_attempt=2)) == "api_run_mismatch"
    assert policy.check_run(request(), api_run(repository={"full_name": "x/y"})) == "api_run_repo_mismatch"
    assert policy.check_run(request(), api_run(status="completed")) == "run_not_live:completed"


@pytest.mark.parametrize(
    ("value", "epoch"),
    [
        ("2026-10-31T00:00:00+00:00", 1_793_404_800.0),
        ("2026-10-31T00:00:00Z", 1_793_404_800.0),
        ("2026-10-31T01:00:00+01:00", 1_793_404_800.0),
    ],
)
def test_a_deadline_is_an_instant_with_a_time_zone(value, epoch) -> None:
    assert policy.parse_deadline(value) == epoch


@pytest.mark.parametrize("value", ["2026-10-31T00:00:00", "2026-10-31", "", "soon"])
def test_a_deadline_without_a_time_zone_or_not_a_date_is_an_error(value) -> None:
    with pytest.raises(ValueError):
        policy.parse_deadline(value)


@pytest.mark.parametrize("profile", sorted(policy.PROFILE_LABELS))
def test_resource_profile_is_admitted_as_one_label(profile) -> None:
    labels = ["self-hosted", "modal-ci", profile, "job-100-1-evaluate"]
    req = policy.parse_workflow_job(payload(job={"labels": labels}), CFG)
    assert isinstance(req, policy.JobRequest)
    assert req.labels == tuple(labels)


@pytest.mark.parametrize(
    "profiles", [["modal-ci-unknown"], ["modal-ci-long", "modal-ci-heavy"], ["modal-ci-long", "modal-ci-long"]]
)
def test_unknown_or_ambiguous_resource_profiles_are_rejected(profiles) -> None:
    labels = ["self-hosted", "modal-ci", *profiles, "job-100-1-evaluate"]
    assert policy.parse_workflow_job(payload(job={"labels": labels}), CFG) == policy.Rejected(
        "unknown_or_ambiguous_profile"
    )
