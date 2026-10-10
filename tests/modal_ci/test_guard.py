"""The job-started guard: a mismatched job must never get past it, whatever goes wrong."""

import json
import signal

import pytest

from infra.modal_ci.sandbox import guard

SHA = "a" * 40
BINDING = {
    "repo": "softnanolab/boileroom",
    "job_id": 7,
    "run_id": 100,
    "run_attempt": 1,
    "job_key": "unit-tests",
    "head_sha": SHA,
    "runner_name": "modal-7-abc123",
}


def job_env(**overrides):
    env = {
        "GITHUB_REPOSITORY": "softnanolab/boileroom",
        "GITHUB_RUN_ID": "100",
        "GITHUB_RUN_ATTEMPT": "1",
        "GITHUB_JOB": "unit-tests",
        "RUNNER_NAME": "modal-7-abc123",
        "GITHUB_EVENT_NAME": "push",
        "GITHUB_SHA": SHA,
    }
    env.update(overrides)
    return {k: v for k, v in env.items() if v is not None}


def pr_event(sha=SHA, repo="softnanolab/boileroom", fork=False):
    return {"pull_request": {"head": {"sha": sha, "repo": {"full_name": repo, "fork": fork}}}}


def test_the_bound_job_is_allowed() -> None:
    assert guard.check(BINDING, job_env(), None) is None
    assert guard.check(BINDING, job_env(GITHUB_EVENT_NAME="pull_request"), pr_event()) is None


@pytest.mark.parametrize(
    ("override", "reason"),
    [
        ({"GITHUB_REPOSITORY": "softnanolab/bakeoff"}, "repo_mismatch"),
        ({"GITHUB_RUN_ID": "101"}, "run_id_mismatch"),
        ({"GITHUB_RUN_ATTEMPT": "2"}, "run_attempt_mismatch"),
        ({"GITHUB_JOB": "lint-checks"}, "job_key_mismatch"),
        ({"RUNNER_NAME": "someone-else"}, "runner_name_mismatch"),
        ({"GITHUB_SHA": "b" * 40}, "sha_mismatch"),
        ({"GITHUB_EVENT_NAME": "pull_request_target"}, "event_not_allowed:pull_request_target"),
        ({"GITHUB_EVENT_NAME": "workflow_run"}, "event_not_allowed:workflow_run"),
    ],
)
def test_any_mismatch_is_denied(override, reason) -> None:
    assert guard.check(BINDING, job_env(**override), None) == reason


@pytest.mark.parametrize("missing", sorted(job_env()))
def test_a_missing_variable_is_an_error_not_an_allow(missing) -> None:
    with pytest.raises(ValueError):
        guard.check(BINDING, job_env(**{missing: None}), None)


def test_pull_request_head_must_match_and_come_from_this_repo() -> None:
    env = job_env(GITHUB_EVENT_NAME="pull_request")
    assert guard.check(BINDING, env, pr_event(sha="b" * 40)) == "head_sha_mismatch"
    assert guard.check(BINDING, env, pr_event(repo="mallory/boileroom")) == "fork_or_unknown_head_repo"
    assert guard.check(BINDING, env, pr_event(fork=True)) == "fork_or_unknown_head_repo"
    assert guard.check(BINDING, env, {"pull_request": {"head": {"sha": SHA, "repo": {}}}}) == (
        "fork_or_unknown_head_repo"
    )
    assert guard.check(BINDING, env, None) == "head_sha_mismatch"


def test_an_incomplete_binding_denies() -> None:
    for key in guard.BINDING_KEYS:
        assert guard.check({**BINDING, key: ""}, job_env(), None) == "binding_incomplete"


def write(tmp_path, name, content):
    path = tmp_path / name
    path.write_text(content if isinstance(content, str) else json.dumps(content))
    return str(path)


def test_decide_reads_the_binding_and_event_files(tmp_path) -> None:
    binding = write(tmp_path, "binding.json", BINDING)
    event = write(tmp_path, "event.json", pr_event())
    assert guard.decide(job_env(), binding) is None
    pr_env = job_env(GITHUB_EVENT_NAME="pull_request", GITHUB_EVENT_PATH=event)
    assert guard.decide(pr_env, binding) is None
    forked = write(tmp_path, "fork.json", pr_event(repo="mallory/boileroom", fork=True))
    assert guard.decide({**pr_env, "GITHUB_EVENT_PATH": forked}, binding) == "fork_or_unknown_head_repo"


@pytest.mark.parametrize("content", ["", "not json", "[]", "null", '{"repo": "x"}'])
def test_decide_denies_unreadable_or_malformed_bindings(tmp_path, content) -> None:
    assert guard.decide(job_env(), write(tmp_path, "binding.json", content)) is not None


def test_decide_denies_a_missing_binding_file(tmp_path) -> None:
    assert str(guard.decide(job_env(), str(tmp_path / "nope.json"))).startswith("guard_error:")


def test_decide_denies_a_pull_request_without_an_event_file(tmp_path) -> None:
    binding = write(tmp_path, "binding.json", BINDING)
    env = job_env(GITHUB_EVENT_NAME="pull_request", GITHUB_EVENT_PATH=str(tmp_path / "missing.json"))
    assert str(guard.decide(env, binding)).startswith("guard_error:")
    assert str(guard.decide(job_env(GITHUB_EVENT_NAME="pull_request"), binding)).startswith("guard_error:")


class Signals:
    def __init__(self) -> None:
        self.kills: list[tuple[int, int]] = []
        self.killpgs: list[tuple[int, int]] = []


@pytest.fixture
def signals(monkeypatch):
    """Record kill signals instead of sending them: the real `kill(-1)` would end this session."""
    rec = Signals()
    monkeypatch.setattr(guard.os, "kill", lambda pid, sig: rec.kills.append((pid, sig)))
    monkeypatch.setattr(guard.os, "killpg", lambda pgid, sig: rec.killpgs.append((pgid, sig)))
    return rec


def test_terminate_kills_everything_then_the_runner_group(tmp_path, signals) -> None:
    pgid = write(tmp_path, "runner.pgid", "4242")
    guard.terminate(pgid)
    assert signals.kills == [(-1, signal.SIGKILL)]
    assert signals.killpgs == [(4242, signal.SIGKILL)]


@pytest.mark.parametrize("content", ["1", "0", "-1", "", "abc"])
def test_terminate_never_signals_init_or_a_bad_group_but_still_kills_the_user(tmp_path, signals, content) -> None:
    guard.terminate(write(tmp_path, "runner.pgid", content))
    assert signals.kills == [(-1, signal.SIGKILL)]
    assert signals.killpgs == []


def test_terminate_without_a_pgid_file_still_kills_the_user(tmp_path, signals) -> None:
    guard.terminate(str(tmp_path / "missing"))
    assert signals.kills == [(-1, signal.SIGKILL)]


def test_main_denial_terminates_and_exits_non_zero(monkeypatch, signals, capsys) -> None:
    monkeypatch.setattr(guard, "decide", lambda env: "repo_mismatch")
    assert guard.main() == 1
    assert capsys.readouterr().out.strip() == "DENY repo_mismatch"
    assert signals.kills == [(-1, signal.SIGKILL)]


def test_main_allow_prints_the_only_line_the_wrapper_accepts(monkeypatch, signals, capsys) -> None:
    monkeypatch.setattr(guard, "decide", lambda env: None)
    assert guard.main() == 0
    assert capsys.readouterr().out == "ALLOW\n"
    assert signals.kills == []
