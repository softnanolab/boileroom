"""The sandbox supervisor: refuses a bad handoff and hands the runner an allowlisted environment."""

import json
import os
import stat

import pytest

from infra.modal_ci.sandbox import supervisor

BINDING = {
    "repo": "softnanolab/boileroom",
    "job_id": 7,
    "run_id": 100,
    "run_attempt": 1,
    "job_key": "unit-tests",
    "head_sha": "a" * 40,
    "runner_name": "modal-7-abc123",
}


def test_binding_must_be_complete() -> None:
    assert supervisor.validate_binding(json.dumps(BINDING)) == BINDING
    for key in supervisor.BINDING_KEYS:
        with pytest.raises(ValueError):
            supervisor.validate_binding(json.dumps({**BINDING, key: ""}))
    with pytest.raises(ValueError):
        supervisor.validate_binding("[]")
    with pytest.raises(ValueError):
        supervisor.validate_binding("{")


def test_runner_environment_is_an_allowlist() -> None:
    env = supervisor.runner_env("JITCONFIG")
    assert env["ACTIONS_RUNNER_HOOK_JOB_STARTED"] == supervisor.HOOK
    assert env["ACTIONS_RUNNER_INPUT_JITCONFIG"] == "JITCONFIG"
    assert not any(k.startswith(("CI_", "MODAL_", "GITHUB_")) for k in env)


def test_main_refuses_without_a_binding_or_jit(capsys) -> None:
    assert supervisor.main({"CI_JIT": "x"}) == 2
    assert supervisor.main({"CI_BINDING": json.dumps(BINDING)}) == 2
    assert supervisor.main({"CI_BINDING": "{}", "CI_JIT": "x"}) == 2
    assert "refusing to start" in capsys.readouterr().out


def test_main_refuses_when_the_hook_or_runner_is_missing(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr(supervisor, "HOOK", str(tmp_path / "missing-hook.sh"))
    monkeypatch.setattr(supervisor, "RUN_DIR", tmp_path / "run")
    assert supervisor.main({"CI_BINDING": json.dumps(BINDING), "CI_JIT": "x"}) == 2
    assert "missing-hook.sh" in capsys.readouterr().out


def test_strip_setuid_clears_setuid_and_setgid_bits(tmp_path) -> None:
    modes = {"plain": 0o755, "suid": 0o4755, "sgid": 0o2755, "both": 0o6755}
    for name, mode in modes.items():
        (tmp_path / name).write_text("#!/bin/sh\n")
        os.chmod(tmp_path / name, mode)
        if stat.S_IMODE((tmp_path / name).stat().st_mode) != mode:
            pytest.skip("this filesystem does not preserve setuid/setgid on the test files")
    assert supervisor.strip_setuid(str(tmp_path)) == []
    assert {stat.S_IMODE((tmp_path / name).stat().st_mode) for name in modes} == {0o755}


def test_strip_setuid_reports_what_it_could_not_clear(monkeypatch, tmp_path) -> None:
    """A `chmod` that does nothing (read-only mount, unsupported filesystem) must be visible, not assumed."""
    (tmp_path / "suid").write_text("#!/bin/sh\n")
    os.chmod(tmp_path / "suid", 0o4755)
    if not (tmp_path / "suid").stat().st_mode & stat.S_ISUID:
        pytest.skip("this filesystem does not preserve setuid on the test file")
    monkeypatch.setattr(supervisor, "_CHMOD", "/usr/bin/true")
    assert supervisor.strip_setuid(str(tmp_path)) == [str(tmp_path / "suid")]


def test_strip_setuid_fails_closed_when_the_scan_cannot_run(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="setuid scan failed"):
        supervisor.strip_setuid(str(tmp_path / "does-not-exist"))


def test_main_refuses_to_start_while_a_setuid_file_remains(monkeypatch, capsys) -> None:
    monkeypatch.setattr(supervisor, "prepare", lambda raw: BINDING)
    monkeypatch.setattr(supervisor, "strip_setuid", lambda root="/": ["/usr/bin/su"])
    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *a, **k: pytest.fail("runner must not start"))
    assert supervisor.main({"CI_BINDING": json.dumps(BINDING), "CI_JIT": "x"}) == 2
    assert "/usr/bin/su" in capsys.readouterr().out


def test_main_refuses_to_start_when_the_strip_itself_fails(monkeypatch, capsys) -> None:
    def broken(root: str = "/") -> list[str]:
        raise FileNotFoundError("/usr/bin/find")

    monkeypatch.setattr(supervisor, "prepare", lambda raw: BINDING)
    monkeypatch.setattr(supervisor, "strip_setuid", broken)
    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *a, **k: pytest.fail("runner must not start"))
    assert supervisor.main({"CI_BINDING": json.dumps(BINDING), "CI_JIT": "x"}) == 2
    assert "refusing to start" in capsys.readouterr().out
