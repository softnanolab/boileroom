"""Offline contracts for the explicit free-runner CPU SIF route."""

from __future__ import annotations

import json
import signal
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.testing import run_cpu_sif_validation as launcher

pytestmark = pytest.mark.contract
ROOT = Path(__file__).resolve().parents[2]
AVAILABLE_MODELS = [model for model in launcher.LIMITS if (ROOT / f"boileroom/models/{model}/Dockerfile").is_file()]


@pytest.fixture(autouse=True)
def owned_runner_temp(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))


def _plan(model: str = "profam") -> dict[str, Any]:
    return {
        "model": model,
        "model_size": "12L" if model == "minifold" else None,
        "dirty": False,
        "entrypoint_exists": True,
        "entrypoint": f"tests/{model}/{model}_public_validation.py",
        "git_sha": "a" * 40,
        "base_dockerfile": "boileroom/images/Dockerfile",
        "model_dockerfile": f"boileroom/models/{model}/Dockerfile",
        "limits": launcher.LIMITS[model],
    }


def _state(tmp_path: Path) -> dict[str, Any]:
    nonce = "a" * 32
    scratch = tmp_path / f"boileroom-cpu-sif-{nonce}"
    scratch.mkdir()
    (scratch / ".owner").write_text(nonce)
    return {
        "nonce": nonce,
        "scratch": str(scratch),
        "builder": f"boileroom-cpu-sif-{nonce}",
        "image_tag": f"boileroom-cpu-sif-{nonce}:local",
    }


def test_default_plan_cannot_build_or_launch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(launcher, "make_plan", lambda *args: _plan())
    monkeypatch.setattr(launcher, "execute", lambda *args: pytest.fail("Plan launched execution"))
    assert launcher.main(["--model", "profam", "--output-dir", str(tmp_path)]) == 0
    assert json.loads((tmp_path / "plan.json").read_text())["model"] == "profam"
    assert not (tmp_path / "run.json").exists()


@pytest.mark.parametrize("field,value", [("dirty", True), ("entrypoint_exists", False)])
def test_execute_requires_clean_full_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, field: str, value: bool
) -> None:
    monkeypatch.setattr(launcher, "make_plan", lambda *args: _plan() | {field: value})
    monkeypatch.setattr(launcher, "execute", lambda *args: pytest.fail("Invalid checkout launched"))
    with pytest.raises(SystemExit) as caught:
        launcher.main(
            [
                "--model",
                "profam",
                "--expected-sha",
                "a" * 40,
                "--output-dir",
                str(tmp_path),
                "--apptainer-package",
                "official.deb",
                "--execute",
            ]
        )
    assert caught.value.code == 2


def test_sha_mismatch_fails_before_source_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(launcher, "git", lambda *args: "a" * 40)
    with pytest.raises(ValueError, match="full HEAD SHA"):
        launcher.make_plan(ROOT, "profam", "12L", "b" * 40)


@pytest.mark.parametrize("model", AVAILABLE_MODELS)
def test_plan_records_canonical_sources_and_honest_scope(model: str) -> None:
    plan = launcher.make_plan(ROOT, model, "12L", None)
    assert plan["source_files_sha256"][f"boileroom/models/{model}/Dockerfile"] == launcher.sha256(
        ROOT / f"boileroom/models/{model}/Dockerfile"
    )
    assert plan["entrypoint"] == f"tests/{model}/{model}_public_validation.py"
    assert plan["paid_compute"] is plan["image_publication"] is plan["hermetic_build"] is False
    assert "GPU" in plan["scope"] and "parity" in plan["scope"]
    assert launcher.TOTAL_TIMEOUT + 60 < 55 * 60


def test_standard_16gib_host_explicitly_blocks_full_minifold() -> None:
    resources = {
        "available_memory_bytes": 16 * 2**30,
        "free_disk_bytes": {"/tmp": 50 * 2**30},
        "cpu_affinity": [0, 1, 2, 3],
    }
    blockers = launcher.resource_blockers(_plan("minifold"), resources)
    assert len(blockers) == 1 and "24 GiB available RAM" in blockers[0]
    assert not launcher.resource_blockers(_plan("profam"), resources)


def test_each_filesystem_is_checked_without_freeing_preexisting_data() -> None:
    resources = {
        "available_memory_bytes": 16 * 2**30,
        "free_disk_bytes": {"/tmp": 50 * 2**30, "/var/lib/docker": 14 * 2**30},
        "cpu_affinity": [0, 1],
    }
    assert "free disk" in launcher.resource_blockers(_plan(), resources)[0]


def test_host_rejects_unverified_public_host_and_package(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    with pytest.raises(ValueError, match="GitHub-hosted"):
        launcher.verify_host(tmp_path / "missing.deb")
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("RUNNER_ENVIRONMENT", "github-hosted")
    monkeypatch.setenv("GITHUB_REPOSITORY_VISIBILITY", "private")
    with pytest.raises(ValueError, match="public repository"):
        launcher.verify_host(tmp_path / "missing.deb")
    monkeypatch.setenv("GITHUB_REPOSITORY_VISIBILITY", "public")
    original = Path.read_text
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda path, *a, **k: (
            'ID=ubuntu\nVERSION_ID="22.04"\n' if str(path) == "/etc/os-release" else original(path, *a, **k)
        ),
    )
    monkeypatch.setattr(launcher.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(launcher.os, "geteuid", lambda: 1001)
    package = tmp_path / "official.deb"
    package.write_bytes(b"unverified bytes")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        launcher.verify_host(package)


def test_build_uses_unchanged_canonical_dockerfiles_and_local_oci_base(tmp_path: Path) -> None:
    state = _state(tmp_path)
    commands = launcher.build_commands(ROOT, state, _plan())
    create, base, model = [command for _, command in commands]
    assert "--driver" in create and "docker-container" in create
    assert "memory=6g" in create[-1] and "cpu-quota=200000" in create[-1]
    assert "type=oci," in " ".join(base)
    assert str(ROOT / "boileroom/images/Dockerfile") in base
    assert str(ROOT / "boileroom/models/profam/Dockerfile") in model
    assert "BASE_IMAGE=validation-base" in model and "--load" in model
    assert any(value.startswith("validation-base=oci-layout://") for value in model)
    assert f"{launcher.OWNER_LABEL}={state['nonce']}" in model
    for forbidden in ("--push", "--privileged", "--allow", "--network=host", "--gpus", "prune"):
        assert all(forbidden not in command for _, command in commands)


def test_cleanup_is_scoped_and_idempotent_without_host_data_deletion(tmp_path: Path) -> None:
    state = _state(tmp_path)
    unrelated = tmp_path / "existing-checkpoints"
    unrelated.mkdir()
    (unrelated / "keep").write_text("unchanged")
    result = launcher.cleanup(state, tmp_path)
    assert result["passed"] and unrelated.exists()
    state["cleanup"] = result
    assert launcher.cleanup(state, tmp_path) == result
    assert (unrelated / "keep").read_text() == "unchanged"


@pytest.mark.parametrize(
    "change", [{"nonce": "bad"}, {"scratch": "/tmp"}, {"builder": "default"}, {"image_tag": "existing:model"}]
)
def test_cleanup_rejects_unowned_state(tmp_path: Path, change: dict[str, str]) -> None:
    state = _state(tmp_path) | change
    with pytest.raises(ValueError, match="Invalid cleanup"):
        launcher.cleanup(state, tmp_path)


def test_cleanup_refuses_symlink_and_wrong_owner_marker(tmp_path: Path) -> None:
    state = _state(tmp_path)
    scratch = Path(state["scratch"])
    (scratch / ".owner").write_text("someone else")
    with pytest.raises(ValueError, match="ownership marker"):
        launcher.cleanup(state, tmp_path)
    (scratch / ".owner").unlink()
    scratch.rmdir()
    scratch.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="scratch path"):
        launcher.cleanup(state, tmp_path)


def test_cleanup_requires_image_ownership_and_never_prunes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    state = _state(tmp_path) | {"builder_attempted": True, "model_build_attempted": True}
    called: list[list[str]] = []

    def run(_self: Any, name: str, command: list[str], *args: Any, **kwargs: Any) -> str:
        called.append(command)
        if name == "cleanup-image-inspect":
            return json.dumps([{"Config": {"Labels": {launcher.OWNER_LABEL: "other-run"}}}])
        return "sha256:present" if name == "cleanup-image-list" else ""

    monkeypatch.setattr(launcher.Commands, "run", run)
    result = launcher.cleanup(state, tmp_path)
    assert not result["passed"] and "ownership label" in result["errors"][0]
    assert not any(command[:3] == ["docker", "image", "rm"] for command in called)
    assert called[0][-1] == state["builder"]
    assert all("prune" not in command for command in called)


def test_timeout_kills_only_created_process_group(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    events: list[Any] = []

    class Process:
        pid = 1234567

        def __init__(self, command: list[str], **kwargs: Any) -> None:
            assert kwargs["start_new_session"] is True
            events.append(command)
            self.calls = 0

        def wait(self, timeout: float | None = None) -> int:
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired("own-child", timeout or 0.0)
            return -15

        def poll(self) -> None:
            return None

    monkeypatch.setattr(launcher.subprocess, "Popen", Process)
    monkeypatch.setattr(launcher.os, "killpg", lambda pid, sig: events.append((pid, sig)))
    commands = launcher.Commands(tmp_path, launcher.time.monotonic() + 10)
    with pytest.raises(subprocess.TimeoutExpired):
        commands.run("own", ["owned-command"], 1)
    assert (1234567, signal.SIGTERM) in events and (1234567, signal.SIGKILL) in events
    assert commands.active is None


def test_resource_failure_records_unrun_and_cleans_without_build(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "reports"
    output.mkdir()
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
    monkeypatch.setattr(launcher, "verify_host", lambda package: None)
    monkeypatch.setattr(launcher, "sha256", lambda path: launcher.APPTAINER_PACKAGE_SHA256)
    monkeypatch.setattr(
        launcher,
        "resource_snapshot",
        lambda paths: {
            "available_memory_bytes": 16 * 2**30,
            "free_disk_bytes": {"/tmp": 50 * 2**30},
            "cpu_affinity": [0, 1],
        },
    )
    calls = []

    def run(_self: Any, name: str, command: list[str], *args: Any, **kwargs: Any) -> str:
        calls.append(name)
        if name == "apptainer-version":
            return "apptainer version 1.5.4\n"
        if name == "apptainer-buildcfg":
            return "APPTAINER_SUID_INSTALL=0\n"
        if name == "docker-info":
            return json.dumps({"DockerRootDir": str(tmp_path)})
        return "version recorded"

    monkeypatch.setattr(launcher.Commands, "run", run)
    assert not launcher.execute(ROOT, _plan("minifold"), output, tmp_path / "official.deb")
    report = json.loads((output / "run.json").read_text())
    assert report["validation_status"] == "unrun" and report["passed"] is False
    assert report["cleanup"]["passed"]
    assert "Resource preflight" in report["error"]
    assert "builder-create" not in calls and "public-wrapper" not in calls
    assert not list(tmp_path.glob("boileroom-cpu-sif-*"))


def test_48l_cpu_cannot_silently_substitute_12l(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(launcher, "make_plan", lambda *args: _plan("minifold") | {"model_size": "48L"})
    monkeypatch.setattr(launcher, "execute", lambda *args: pytest.fail("48L launched on standard CPU"))
    with pytest.raises(SystemExit) as caught:
        launcher.main(
            [
                "--model",
                "minifold",
                "--minifold-size",
                "48L",
                "--expected-sha",
                "a" * 40,
                "--apptainer-package",
                "official.deb",
                "--output-dir",
                str(tmp_path),
                "--execute",
            ]
        )
    assert caught.value.code == 2


def test_direct_child_cannot_bypass_execute_gates(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(launcher, "run_public_child", lambda *args: pytest.fail("Ungated child launched"))
    with pytest.raises(SystemExit) as caught:
        launcher.main(["--model", "profam", "--output-dir", str(tmp_path), "--child"])
    assert caught.value.code == 2


def test_missing_failed_build_image_is_cleanable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    state = _state(tmp_path) | {"model_build_attempted": True}
    calls = []

    def run(_self: Any, name: str, command: list[str], *args: Any, **kwargs: Any) -> str:
        calls.append(name)
        assert name == "cleanup-image-list"
        return ""

    monkeypatch.setattr(launcher.Commands, "run", run)
    report = launcher.cleanup(state, tmp_path)
    assert report["passed"] and report["model_image_absent"]
    assert calls == ["cleanup-image-list"]


@pytest.mark.parametrize("model,wrapper_name", [("profam", "ProFam"), ("minifold", "MiniFold")])
def test_child_uses_selected_public_wrapper_cpu_and_checks_retained_process_and_client(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, model: str, wrapper_name: str
) -> None:
    import boileroom
    from boileroom.backend import apptainer

    state = _state(tmp_path)
    cache = Path(state["scratch"]) / "models"
    monkeypatch.setenv("MODEL_DIR", str(cache))
    state["image_uri"] = f"docker://docker.io/task/boileroom-{model}:local"
    sif = apptainer._get_cached_sif_path(state["image_uri"], cache)
    sif.parent.mkdir(parents=True)
    sif.write_bytes(b"exact local sif bytes")
    state |= {"sif_path": str(sif), "sif_sha256": launcher.sha256(sif)}
    helper = tmp_path / "public_helper.py"
    helper.write_text(
        "def run_public_validation(factory, *, backend, device, model_size=None):\n"
        f"    assert model_size == {repr('12L' if model == 'minifold' else None)}\n"
        "    assert backend == 'apptainer' and device == 'cpu'\n"
        "    with factory(backend=backend, device=device, config={'num_samples': 2}):\n"
        "        pass\n"
        "    return {'passed': True}\n"
    )
    calls = []
    process = SimpleNamespace(pid=4242, code=None)
    process.poll = lambda: process.code
    client = SimpleNamespace(is_closed=False)
    worker = SimpleNamespace(_sif_path=sif, _device="cpu", _process=process, _client=client)

    def stop() -> None:
        process.code = 0
        client.is_closed = True
        worker._process = worker._client = None

    worker.stop = stop

    class Wrapper:
        def __init__(self, **kwargs: Any) -> None:
            calls.append(kwargs)
            self._backend = worker

        def __enter__(self) -> Wrapper:
            return self

        def __exit__(self, *_args: Any) -> None:
            stop()

    monkeypatch.setattr(boileroom, wrapper_name, Wrapper, raising=False)
    monkeypatch.setattr(launcher.os, "sched_getaffinity", lambda pid: {0, 1, 2, 3})
    affinity = []
    monkeypatch.setattr(launcher.os, "sched_setaffinity", lambda pid, cpus: affinity.append(cpus))
    plan = _plan(model) | {"entrypoint": str(helper)}
    report = launcher.run_public_child(ROOT, plan, state, tmp_path)
    assert calls == [{"backend": "apptainer", "device": "cpu", "config": {"num_samples": 2}}]
    assert affinity == [[0, 1]]
    assert report["passed"] and report["actual_processes_stopped"] and report["sif_unchanged"]
    assert report["processes"][0] == {
        "pid": 4242,
        "exit_code": 0,
        "client_closed": True,
        "backend_process_cleared": True,
        "backend_client_cleared": True,
    }


def test_child_refuses_changed_source_plan(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    plan = _plan()
    launcher.write_json(tmp_path / "plan.json", plan | {"git_sha": "b" * 40})
    state = _state(tmp_path) | {"validation_status": "running"}
    launcher.write_json(tmp_path / "run.json", state)
    monkeypatch.setattr(launcher, "verify_host", lambda package: None)
    monkeypatch.setattr(launcher, "make_plan", lambda *args: plan)
    monkeypatch.setattr(launcher, "run_public_child", lambda *args: pytest.fail("Changed plan launched"))
    with pytest.raises(SystemExit) as caught:
        launcher.main(
            [
                "--model",
                "profam",
                "--output-dir",
                str(tmp_path),
                "--child",
                "--execute",
                "--expected-sha",
                "a" * 40,
                "--apptainer-package",
                "official.deb",
            ]
        )
    assert caught.value.code == 2


@pytest.mark.parametrize("changed", ["BOILEROOM_DOCKER_REPOSITORY", "BOILEROOM_IMAGE_TAG", "image_uri"])
def test_child_refuses_alternate_registry_lookup_before_wrapper_start(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, changed: str
) -> None:
    plan = _plan()
    launcher.write_json(tmp_path / "plan.json", plan)
    state = _state(tmp_path) | {"validation_status": "running"}
    namespace = f"docker.io/boileroom-cpu-validation-{state['nonce']}"
    tag = f"sha-{plan['git_sha']}"
    state["image_uri"] = f"docker://{namespace}/boileroom-profam:{tag}"
    monkeypatch.setenv("MODEL_DIR", str(Path(state["scratch"]) / "models"))
    monkeypatch.setenv("BOILEROOM_DOCKER_REPOSITORY", namespace)
    monkeypatch.setenv("BOILEROOM_IMAGE_TAG", tag)
    if changed == "image_uri":
        state[changed] = "docker://docker.io/other/model:latest"
    else:
        monkeypatch.setenv(changed, "other")
    launcher.write_json(tmp_path / "run.json", state)
    monkeypatch.setattr(launcher, "verify_host", lambda package: None)
    monkeypatch.setattr(launcher, "make_plan", lambda *args: plan)
    monkeypatch.setattr(launcher, "run_public_child", lambda *args: pytest.fail("Alternate image lookup launched"))
    with pytest.raises(SystemExit) as caught:
        launcher.main(
            [
                "--model",
                "profam",
                "--output-dir",
                str(tmp_path),
                "--child",
                "--execute",
                "--expected-sha",
                "a" * 40,
                "--apptainer-package",
                "official.deb",
            ]
        )
    assert caught.value.code == 2
