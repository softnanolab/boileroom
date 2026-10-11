"""No-network contracts for the manual-only, bounded source-image launcher."""

from __future__ import annotations

import asyncio
import json
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from scripts.testing import run_model_validation as launcher

pytestmark = pytest.mark.contract
ROOT = Path(__file__).resolve().parents[2]
AVAILABLE_MODELS = [
    model for model in launcher.MODEL_LIMITS if (ROOT / f"boileroom/models/{model}/Dockerfile").is_file()
]


def test_dispatch_defaults_do_not_start_paid_work() -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/integration-tests.yaml").read_text())
    events = workflow.get("on", workflow.get(True))
    inputs = events["workflow_dispatch"]["inputs"]
    assert inputs["run-target"]["default"] == "none"
    assert inputs["run-target"]["options"] == ["none", "minifold", "profam", "full-suite"]
    assert inputs["minifold-size"]["options"] == ["12L", "48L"]
    assert "push" not in events
    assert events["schedule"] == [{"cron": "0 2 * * 0"}]
    old = workflow["jobs"]["integration-tests"]["if"]
    assert "inputs.run-target == 'full-suite'" in old
    assert "github.event.workflow_run.conclusion == 'success'" in old
    job = workflow["jobs"]["model-validation"]
    assert "github.event_name == 'workflow_dispatch'" in job["if"]
    assert "inputs.run-target == 'minifold'" in job["if"]
    assert "inputs.run-target == 'profam'" in job["if"]
    assert job["timeout-minutes"] <= 60
    assert all("runner." not in str(value) for value in job.get("env", {}).values())
    steps = job["steps"]
    cleanup = next(step for step in steps if "--cleanup-only" in step.get("run", ""))
    assert cleanup["if"] == "always()"
    upload = next(step for step in steps if step.get("uses", "").startswith("actions/upload-artifact@"))
    assert upload["if"] == "always()"
    source = (ROOT / ".github/workflows/integration-tests.yaml").read_text()
    assert "token set" not in source
    assert "MODAL_TOKEN_ID: ${{ secrets.MODAL_API_TOKEN_ID }}" in source
    assert "MODAL_TOKEN_SECRET: ${{ secrets.MODAL_API_TOKEN_SECRET }}" in source
    assert "MODAL_CONFIG_PATH: /dev/null" in source


@pytest.mark.parametrize("model", ["profam", "minifold"])
def test_fixed_resource_bounds(model: str) -> None:
    limits = launcher.MODEL_LIMITS[model]
    assert limits["memory_mib"] >= 16384
    assert limits["remote_timeout_seconds"] <= 30 * 60
    assert limits["gpu"] == ("L4" if model == "minifold" else "T4")
    assert launcher.BUILD_TIMEOUT <= 20 * 60
    assert launcher.TOTAL_TIMEOUT < 60 * 60


@pytest.mark.parametrize("model", AVAILABLE_MODELS)
def test_dockerfile_composition_preserves_every_runtime_instruction(model: str) -> None:
    text = (ROOT / f"boileroom/models/{model}/Dockerfile").read_text()
    assert launcher.model_dockerfile_body(text) == text.split("FROM ${BASE_IMAGE}\n", 1)[1]


@pytest.mark.parametrize(
    "text",
    [
        "FROM ubuntu:latest\nRUN true\n",
        "FROM ${BASE_IMAGE}\nFROM other\n",
        "RUN echo before\nFROM ${BASE_IMAGE}\n",
        "ARG NEW_KEY=bad\nFROM ${BASE_IMAGE}\n",
    ],
)
def test_dockerfile_composition_refuses_changed_base_semantics(text: str) -> None:
    with pytest.raises(ValueError):
        launcher.model_dockerfile_body(text)


def test_clean_exact_head_is_required_before_execute(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(launcher, "make_plan", lambda *args: {"dirty": True, "entrypoint_exists": True})
    monkeypatch.setattr(launcher, "execute", lambda *args: pytest.fail("Must not launch"))
    with pytest.raises(SystemExit) as exc:
        launcher.main(["--model", "profam", "--execute", "--expected-sha", "0" * 40, "--output-dir", str(tmp_path)])
    assert exc.value.code == 2


def test_default_is_offline_plan(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(launcher, "make_plan", lambda *args: {"dirty": True, "entrypoint_exists": False})
    monkeypatch.setattr(launcher, "execute", lambda *args: pytest.fail("Must not launch"))
    assert launcher.main(["--model", "profam", "--output-dir", str(tmp_path)]) == 0
    assert (tmp_path / "plan.json").is_file()
    assert not (tmp_path / "run.json").exists()


def test_sha_mismatch_is_refused_before_hashing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(launcher, "git", lambda *args: "0" * 40)
    with pytest.raises(ValueError, match="HEAD SHA"):
        launcher.make_plan(ROOT, "profam", "12L", "1" * 40)


@pytest.mark.parametrize("model", AVAILABLE_MODELS)
def test_actual_sdk_can_construct_image_offline(model: str) -> None:
    import modal

    # Image methods only describe an image; no build/app/client is started.
    plan = launcher.make_plan(ROOT, model, "12L", None)
    image = launcher.build_image(modal, ROOT, plan)
    assert not image.is_hydrated


def _fake_modal(monkeypatch: pytest.MonkeyPatch, *, passed: bool = True, stall: bool = False) -> tuple[Any, list[Any]]:
    events: list[Any] = []

    async def build(app: Any) -> None:
        events.append("build")
        if stall:
            await asyncio.sleep(60)

    image = SimpleNamespace(build=SimpleNamespace(aio=build), object_id="im-source")
    monkeypatch.setattr(launcher, "build_image", lambda *args: image)

    class App:
        def __init__(self, name: str, **kwargs: Any) -> None:
            self.app_id = "ap-" + ("b" if "build" in name else "v") * 22

        @asynccontextmanager
        async def run(self, *, detach: bool) -> Any:
            assert detach is False
            events.append(("enter", self.app_id))
            try:
                yield self
            finally:
                events.append(("exit", self.app_id))

        def function(self, **kwargs: Any) -> Any:
            events.append(("function", kwargs))

            async def remote(*args: Any) -> dict[str, Any]:
                events.append(("remote", args))
                return {"passed": passed, "phases": {"parity": "evidence"}}

            return lambda function: SimpleNamespace(remote=SimpleNamespace(aio=remote))

    return SimpleNamespace(App=App), events


@pytest.mark.parametrize("passed", [True, False])
def test_serial_build_then_single_attached_gpu_and_evidence_on_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, passed: bool
) -> None:
    modal, events = _fake_modal(monkeypatch, passed=passed)
    plan = launcher.make_plan(ROOT, "profam", "12L", None)
    state: dict[str, Any] = {"apps": [], "passed": False}
    if passed:
        asyncio.run(launcher.run_apps(modal, ROOT, plan, tmp_path, state))
    else:
        with pytest.raises(RuntimeError, match="validation failed"):
            asyncio.run(launcher.run_apps(modal, ROOT, plan, tmp_path, state))
    assert state["passed"] is passed
    assert json.loads((tmp_path / "validation.json").read_text())["passed"] is passed
    assert [record["phase"] for record in state["apps"]] == ["cpu_image_build", "gpu_validation"]
    kwargs = next(event[1] for event in events if isinstance(event, tuple) and event[0] == "function")
    assert kwargs["max_containers"] == 1
    assert kwargs["retries"] == 0
    assert kwargs["min_containers"] == kwargs["buffer_containers"] == 0
    assert kwargs["single_use_containers"] is True
    assert kwargs["memory"] == (16384, 16384)
    assert events.index(("exit", "ap-" + "b" * 22)) < events.index(("enter", "ap-" + "v" * 22))
    assert events[-1] == ("exit", "ap-" + "v" * 22)


def test_build_timeout_unwinds_without_starting_gpu(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    modal, events = _fake_modal(monkeypatch, stall=True)
    monkeypatch.setattr(launcher, "BUILD_TIMEOUT", 0.001)
    plan = launcher.make_plan(ROOT, "profam", "12L", None)
    state: dict[str, Any] = {"apps": [], "passed": False}
    with pytest.raises(TimeoutError):
        asyncio.run(launcher.run_apps(modal, ROOT, plan, tmp_path, state))
    assert events[-1] == ("exit", "ap-" + "b" * 22)
    assert not any(isinstance(event, tuple) and event[0] == "function" for event in events)
    assert len(state["apps"]) == 1


def test_cleanup_stops_only_recorded_ids_and_filters_workspace_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    own = "ap-" + "x" * 22
    launcher.write_json(tmp_path / "run.json", {"apps": [{"app_id": own}]})
    calls: list[list[str]] = []

    def run(command: list[str], **kwargs: Any) -> Any:
        calls.append(command)
        assert kwargs["env"]["MODAL_CONFIG_PATH"] == "/dev/null"
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps([{"App ID": own, "State": "stopped", "Tasks": "0"}, {"App ID": "other", "State": "deployed"}]),
        )

    monkeypatch.setattr(launcher.subprocess, "run", run)
    assert launcher.cleanup(tmp_path)
    assert calls[0][-2:] == ["stop", own]
    assert len(calls) == 2
    evidence = json.loads((tmp_path / "cleanup.json").read_text())
    assert evidence["apps"] == [{"App ID": own, "State": "stopped", "Tasks": "0"}]


def test_cleanup_never_treats_missing_state_as_verified(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(launcher, "CLEANUP_VERIFY_SECONDS", 0.01)
    launcher.write_json(tmp_path / "run.json", {"apps": [{"app_id": "ap-" + "x" * 22}]})
    monkeypatch.setattr(launcher.subprocess, "run", lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 0, "[]"))
    assert not launcher.cleanup(tmp_path)


def test_cleanup_rejects_untrusted_app_identifier(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    launcher.write_json(tmp_path / "run.json", {"apps": [{"app_id": "production-app"}]})
    monkeypatch.setattr(launcher.subprocess, "run", lambda *args, **kwargs: pytest.fail("Must not stop"))
    with pytest.raises(ValueError, match="invalid/unbounded"):
        launcher.cleanup(tmp_path)


@pytest.mark.parametrize(
    "model", [model for model in launcher.MODEL_LIMITS if (ROOT / f"boileroom/models/{model}/Dockerfile").exists()]
)
def test_source_build_sdk_context_contains_canonical_copy_inputs(monkeypatch: pytest.MonkeyPatch, model: str) -> None:
    """Reproduce Modal's real COPY discovery, without building or contacting its API."""
    import modal
    from modal.image import AUTO_DOCKERIGNORE, _create_context_mount_function
    from modal.mount import _select_files

    original = modal.Image.dockerfile_commands
    received: list[str] = []

    def commands(image: Any, lines: list[str], **kwargs: Any) -> Any:
        assert isinstance(lines, list), "Modal context discovery requires individual Dockerfile lines"
        received.extend(lines)
        return original(image, lines, **kwargs)

    monkeypatch.setattr(modal.Image, "dockerfile_commands", commands)
    plan = launcher.make_plan(ROOT, model, "12L", None)
    launcher.build_image(modal, ROOT, plan)
    dockerfile = ROOT / plan["model_dockerfile"]
    body = launcher.model_dockerfile_body(dockerfile.read_text())
    assert "\n".join(received) + "\n" == body
    mount = _create_context_mount_function(
        ignore=AUTO_DOCKERIGNORE, dockerfile_cmds=received, context_dir=dockerfile.parent
    )()
    assert mount is not None
    selected = {str(remote): local for local, remote in _select_files(mount._entries)}
    assert selected["/requirements.txt"] == dockerfile.parent / "requirements.txt"
    if model == "minifold":
        assert selected["/patch_upstream.py"] == dockerfile.parent / "patch_upstream.py"
    assert all(path.is_file() for path in selected.values())
    # The old one-string body failed to discover any COPY files, despite a
    # correctly constructed Image descriptor, and failed only during paid build.
    assert (
        _create_context_mount_function(
            ignore=AUTO_DOCKERIGNORE, dockerfile_cmds=[body], context_dir=dockerfile.parent
        )()
        is None
    )


def test_sdk_runtime_mounts_contain_every_tracked_source_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Enumerate real SDK directory mounts, which call ignore with relative paths."""
    import modal
    from modal.mount import _select_files

    model = next(model for model in launcher.MODEL_LIMITS if (ROOT / f"boileroom/models/{model}/Dockerfile").is_file())
    plan = launcher.make_plan(ROOT, model, "12L", None)
    # Source selection must not depend on the launching process's working directory.
    monkeypatch.chdir(tmp_path)
    current = launcher.build_image(modal, ROOT, plan)
    selected: dict[str, Path] = {}
    # Three directory mounts and pyproject.toml, in the actual lazy Image graph.
    for _ in range(4):
        image, mount = current.deps()
        selected.update({str(remote): local for local, remote in _select_files(mount.entries)})
        current = image
    expected = {f"/validation/{name}" for name in plan["source_files_sha256"]}
    assert set(selected) == expected
    assert selected[f"/validation/boileroom/models/{model}/core.py"] == ROOT / f"boileroom/models/{model}/core.py"
    runtime_modules = [
        name
        for name in expected
        if name.startswith(f"/validation/tests/{model}/") and name.endswith("runtime_validation.py")
    ]
    assert runtime_modules, "The actual validation entrypoint must be present in the source mount"
    assert all(selected[name].is_file() for name in runtime_modules)
    assert not any("__pycache__" in name or ".model_cache" in name or name.endswith(".env") for name in selected)


def test_cleanup_waits_for_stopped_app_tasks_to_reach_zero(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    own = "ap-" + "x" * 22
    launcher.write_json(tmp_path / "run.json", {"apps": [{"app_id": own}]})
    rows = iter(
        [
            [{"App ID": own, "State": "stopped", "Tasks": "1"}],
            [{"App ID": own, "State": "stopped", "Tasks": "0"}],
        ]
    )

    def run(command: list[str], **kwargs: Any) -> Any:
        if "stop" in command:
            return subprocess.CompletedProcess(command, 0, "")
        return subprocess.CompletedProcess(command, 0, json.dumps(next(rows)))

    monkeypatch.setattr(launcher.subprocess, "run", run)
    monkeypatch.setattr(launcher, "CLEANUP_POLL_SECONDS", 0)
    assert launcher.cleanup(tmp_path)
    evidence = json.loads((tmp_path / "cleanup.json").read_text())
    assert len(evidence["verification_checks"]) == 2
    assert evidence["verification_checks"][0]["apps"][0]["Tasks"] == "1"
    assert evidence["apps"][0]["Tasks"] == "0"


@pytest.mark.parametrize("tasks", ["1", None])
def test_cleanup_refuses_stopped_state_without_zero_tasks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tasks: str | None
) -> None:
    own = "ap-" + "x" * 22
    launcher.write_json(tmp_path / "run.json", {"apps": [{"app_id": own}]})
    monkeypatch.setattr(launcher, "CLEANUP_VERIFY_SECONDS", 0.01)
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 0, json.dumps([{"App ID": own, "State": "stopped", "Tasks": tasks}])
        ),
    )
    assert not launcher.cleanup(tmp_path)
