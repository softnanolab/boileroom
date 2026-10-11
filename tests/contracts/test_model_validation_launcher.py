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
    model for model in launcher.MODEL_LIMITS if (ROOT / f"boileroom/models/{model}/Dockerfile").exists()
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
    # GitHub does not allow the runner context in job-level env expressions.
    assert "runner." not in json.dumps(job.get("env", {}))
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


@pytest.mark.parametrize("model", AVAILABLE_MODELS)
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


def test_public_mode_is_explicit_and_uses_its_own_entrypoint() -> None:
    core = launcher.make_plan(ROOT, "profam", "12L", None)
    public = launcher.make_plan(ROOT, "profam", "12L", None, "public-modal")
    assert core["mode"] == "core"
    assert public["mode"] == "public-modal"
    assert public["entrypoint"] == "tests/profam/profam_public_validation.py"
    assert core["source_reference"] != public["source_reference"]
    workflow = yaml.safe_load((ROOT / ".github/workflows/integration-tests.yaml").read_text())
    events = workflow.get("on", workflow.get(True))
    assert events["workflow_dispatch"]["inputs"]["validation-mode"]["default"] == "core"
    job = workflow["jobs"]["model-validation"]
    command = next(step["run"] for step in job["steps"] if "--execute" in step.get("run", ""))
    assert '--mode "$VALIDATION_MODE"' in command
    assert "--with numpy" in command and "--with httpx" in command


@pytest.mark.parametrize("code", [0, 1])
def test_public_child_records_real_app_even_when_it_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, code: int
) -> None:
    own = "ap-" + "p" * 22
    launcher.write_json(tmp_path / "public-run.json", {"apps": [{"app_id": own, "phase": "public_modal_wrapper"}]})
    launcher.write_json(tmp_path / "validation.json", {"passed": code == 0})
    commands = []

    class Process:
        returncode = code

        async def wait(self) -> int:
            return code

    async def create(*command: str) -> Process:
        commands.append(command)
        return Process()

    monkeypatch.setattr(launcher.asyncio, "create_subprocess_exec", create)
    plan = launcher.make_plan(ROOT, "profam", "12L", None, "public-modal")
    state: dict[str, Any] = {"apps": [], "passed": False}
    if code == 0:
        asyncio.run(launcher.run_public_process(ROOT, plan, tmp_path, state))
    else:
        with pytest.raises(RuntimeError, match="exited 1"):
            asyncio.run(launcher.run_public_process(ROOT, plan, tmp_path, state))
    assert state["apps"][0]["app_id"] == own
    assert "--public-worker" in commands[0] and "--execute" in commands[0]
    assert state["passed"] is (code == 0)


def test_public_child_timeout_terminates_process_and_preserves_app_ids(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    own = "ap-" + "p" * 22
    launcher.write_json(tmp_path / "public-run.json", {"apps": [{"app_id": own, "phase": "public_modal_wrapper"}]})

    class Process:
        returncode: int | None = None

        async def wait(self) -> int:
            if self.returncode is None:
                await asyncio.sleep(60)
            return self.returncode or 0

        def terminate(self) -> None:
            self.returncode = -15

    process = Process()

    async def create(*command: str) -> Process:
        return process

    monkeypatch.setattr(launcher.asyncio, "create_subprocess_exec", create)
    plan = launcher.make_plan(ROOT, "profam", "12L", None, "public-modal")
    plan["limits"]["remote_timeout_seconds"] = 0.001
    state: dict[str, Any] = {"apps": [], "passed": False}
    with pytest.raises(TimeoutError):
        asyncio.run(launcher.run_public_process(ROOT, plan, tmp_path, state))
    assert process.returncode == -15
    assert state["apps"][0]["app_id"] == own


def test_public_mode_does_not_launch_core_gpu_function(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    modal, events = _fake_modal(monkeypatch)

    async def public(*args: Any) -> None:
        events.append("public")

    monkeypatch.setattr(launcher, "run_public_process", public)
    plan = launcher.make_plan(ROOT, "profam", "12L", None, "public-modal")
    state: dict[str, Any] = {"apps": [], "passed": False}
    asyncio.run(launcher.run_apps(modal, ROOT, plan, tmp_path, state))
    assert events[-1] == "public"
    assert not any(isinstance(event, tuple) and event[0] == "function" for event in events)


def test_public_worker_cannot_bypass_explicit_execute(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(launcher, "make_plan", lambda *args: {"dirty": False})
    monkeypatch.setattr(launcher, "public_worker", lambda *args: pytest.fail("Must not launch"))
    with pytest.raises(SystemExit) as exc:
        launcher.main(["--public-worker", "--model", "profam", "--mode", "public-modal", "--output-dir", str(tmp_path)])
    assert exc.value.code == 2


@pytest.mark.parametrize("model", AVAILABLE_MODELS)
def test_public_descriptor_keeps_image_volume_dependency_order(model: str) -> None:
    # Fresh process proves host-only overrides preserve the dependency positions
    # used by the pinned SDK when it imports the unmodified module remotely.
    import sys
    import textwrap

    script = textwrap.dedent("""
        import importlib
        import modal
        from scripts.testing import run_model_validation as launcher
        plan = launcher.make_plan(launcher.ROOT, MODEL, "12L", None, "public-modal")
        image = launcher.build_image(modal, launcher.ROOT, plan)
        helpers = importlib.import_module("boileroom.images.modal")
        original_lookup = helpers.get_modal_image
        helpers.get_modal_image = lambda name: image if name == MODEL else original_lookup(name)
        volumes = importlib.import_module("boileroom.images.volumes")
        volume = modal.Volume.from_name("offline-never-hydrated", create_if_missing=False)
        volumes.model_weights = volume
        module = importlib.import_module(f"boileroom.models.{MODEL}.{MODEL}")
        cls = getattr(module, {"profam": "ModalProFam", "minifold": "ModalMiniFold"}[MODEL])
        dependencies = cls.deps()[0].deps(only_explicit_mounts=True)
        assert dependencies == [image, volume]
        bounded = cls.with_options(gpu=plan["limits"]["gpu"], cpu=(2, 2),
            memory=(plan["limits"]["memory_mib"], plan["limits"]["memory_mib"]),
            max_containers=1, buffer_containers=0, retries=0, scaledown_window=2,
            timeout=plan["limits"]["remote_timeout_seconds"])
        assert bounded._get_app() is module.app
        assert module.app.app_id is None
    """)
    subprocess.run([sys.executable, "-c", f"MODEL={model!r}\n" + script], cwd=ROOT, check=True, timeout=30)


def test_cpu_sif_is_an_isolated_manual_non_setuid_job() -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/integration-tests.yaml").read_text())
    job = workflow["jobs"]["cpu-sif-validation"]
    assert "github.event_name == 'workflow_dispatch'" in job["if"]
    assert "inputs.validation-mode == 'cpu-sif'" in job["if"]
    assert "inputs.validation-mode != 'cpu-sif'" in workflow["jobs"]["model-validation"]["if"]
    assert job["runs-on"] == "ubuntu-22.04"
    assert job["env"]["GITHUB_REPOSITORY_VISIBILITY"] == "${{ github.event.repository.visibility }}"
    assert job["timeout-minutes"] <= 60
    assert "runner." not in json.dumps(job.get("env", {}))
    assert "secrets." not in json.dumps(job)
    install = next(step["run"] for step in job["steps"] if "sha256sum" in step.get("run", ""))
    assert "https://github.com/apptainer/apptainer/releases/download/v1.5.4/apptainer_1.5.4_amd64.deb" in install
    assert "281cf9bd68f16b6f7c34a3a94d77668bc327b9c09db0412ce7890162734a97f7" in install
    assert install.index("sha256sum") < install.index("sudo apt-get install")
    cleanup = next(step for step in job["steps"] if "--cleanup-only" in step.get("run", ""))
    assert cleanup["if"] == "always()"
    assert '[ -f "$ARTIFACT_DIR/run.json" ]' in cleanup["run"]
    assert "apptainer-suid" not in install
    assert "sysctl" not in install and "apparmor" not in install.lower()
    upload = next(step for step in job["steps"] if step.get("uses", "").startswith("actions/upload-artifact@"))
    assert upload["if"] == "always()"


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


@pytest.mark.parametrize("gpu", ["T4", "L4"])
def test_profam_gpu_choice_changes_only_selected_gpu(gpu: str) -> None:
    default = launcher.make_plan(ROOT, "profam", "12L", None)
    chosen = launcher.make_plan(ROOT, "profam", "12L", None, profam_gpu=gpu)
    assert default["limits"]["gpu"] == "T4"
    assert chosen["limits"] == default["limits"] | {"gpu": gpu}
    # Hardware selection does not rebuild an otherwise identical source image.
    assert chosen["source_reference"] == default["source_reference"]
    minifold = launcher.make_plan(ROOT, "minifold", "12L", None, profam_gpu=gpu)
    assert minifold["limits"]["gpu"] == "L4"
    assert minifold["limits"]["memory_mib"] == 32768


def test_profam_gpu_choice_refuses_unapproved_hardware() -> None:
    with pytest.raises(ValueError, match="T4 or L4"):
        launcher.make_plan(ROOT, "profam", "12L", None, profam_gpu="H100")
    with pytest.raises(SystemExit) as exc:
        launcher.main(["--model", "profam", "--profam-gpu", "H100"])
    assert exc.value.code == 2


def test_workflow_gpu_choice_is_passed_only_to_profam() -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/integration-tests.yaml").read_text())
    events = workflow.get("on", workflow.get(True))
    selector = events["workflow_dispatch"]["inputs"]["profam-gpu"]
    assert selector["default"] == "T4"
    assert selector["options"] == ["T4", "L4"]
    steps = workflow["jobs"]["model-validation"]["steps"]
    for step in steps:
        if "--expected-sha" in step.get("run", ""):
            assert 'if [ "$VALIDATION_MODEL" = "profam" ]; then' in step["run"]
            assert 'gpu_args=(--profam-gpu "$PROFAM_GPU")' in step["run"]
            assert '"${gpu_args[@]}"' in step["run"]
    assert "profam-gpu" not in json.dumps(workflow["jobs"]["cpu-sif-validation"])


def test_public_child_keeps_selected_profam_gpu(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    launcher.write_json(tmp_path / "validation.json", {"passed": True})
    commands = []

    class Process:
        returncode = 0

        async def wait(self) -> int:
            return 0

    async def create(*command: str) -> Process:
        commands.append(command)
        return Process()

    monkeypatch.setattr(launcher.asyncio, "create_subprocess_exec", create)
    for model in ("profam", "minifold"):
        plan = launcher.make_plan(ROOT, model, "12L", None, "public-modal", profam_gpu="L4")
        asyncio.run(launcher.run_public_process(ROOT, plan, tmp_path, {"apps": [], "passed": False}))
    assert commands[0][-2:] == ("--profam-gpu", "L4")
    assert "--profam-gpu" not in commands[1]


def test_profam_l4_reaches_bounded_function_and_saved_report(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    modal, events = _fake_modal(monkeypatch)
    plan = launcher.make_plan(ROOT, "profam", "12L", None, profam_gpu="L4")
    asyncio.run(launcher.run_apps(modal, ROOT, plan, tmp_path, {"apps": [], "passed": False}))
    options = next(event[1] for event in events if isinstance(event, tuple) and event[0] == "function")
    assert options["gpu"] == "L4"
    assert options["memory"] == (16384, 16384)
    assert options["max_containers"] == 1 and options["timeout"] == 1800
    report = json.loads((tmp_path / "validation.json").read_text())
    assert report["launcher_requested_gpu"] == "L4"
