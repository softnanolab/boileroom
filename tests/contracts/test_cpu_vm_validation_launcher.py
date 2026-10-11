"""Offline control-plane and source-pack contracts for the CPU VM SIF route."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.testing import run_cpu_vm_validation as vm

pytestmark = pytest.mark.contract


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def helpers(monkeypatch: pytest.MonkeyPatch) -> Any:
    def git(root: Path, *args: str) -> str:
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()

    cpu = SimpleNamespace(
        git=git,
        sha256=lambda path: hashlib.sha256(path.read_bytes()).hexdigest(),
        write_json=_write,
        utc_now=lambda: "2026-10-11T07:00:00+00:00",
        NONCE=re.compile(r"[0-9a-f]{32}\Z"),
        APPTAINER_PACKAGE_URL="https://github.com/apptainer/apptainer/releases/download/v1.5.4/apptainer_1.5.4_amd64.deb",
        APPTAINER_PACKAGE_SHA256="281cf9bd68f16b6f7c34a3a94d77668bc327b9c09db0412ce7890162734a97f7",
        APPTAINER_VERSION="1.5.4",
    )
    monkeypatch.setattr(vm, "cpu_helpers", lambda: cpu)
    return cpu


def _plan(run_model: bool = False) -> dict[str, Any]:
    return {
        "git_sha": "a" * 40,
        "source_sha256": "b" * 64,
        "dirty": False,
        "entrypoint_exists": True,
        "run_model_after_probe": run_model,
    }


def test_helpers_resolve_reviewed_sibling_without_package_path(monkeypatch: pytest.MonkeyPatch) -> None:
    vm.cpu_helpers.cache_clear()
    monkeypatch.setitem(sys.modules, "scripts", SimpleNamespace())
    try:
        helper = vm.cpu_helpers()
        assert Path(helper.__file__).resolve() == vm.ROOT / "scripts/testing/run_cpu_sif_validation.py"
        assert helper.APPTAINER_PACKAGE_SHA256 == "281cf9bd68f16b6f7c34a3a94d77668bc327b9c09db0412ce7890162734a97f7"
        assert vm.cpu_helpers() is helper
    finally:
        vm.cpu_helpers.cache_clear()


@pytest.mark.parametrize("sif_runtime", ["mounted", "unsquash"])
def test_direct_isolated_cli_writes_offline_plan_without_pythonpath(tmp_path: Path, sif_runtime: str) -> None:
    head = subprocess.check_output(["git", "-C", str(vm.ROOT), "rev-parse", "HEAD"], text=True).strip()
    output = tmp_path / "reports"
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            str(vm.ROOT / "scripts/testing/run_cpu_vm_validation.py"),
            "--expected-sha",
            head,
            "--output-dir",
            str(output),
            "--sif-runtime",
            sif_runtime,
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    plan = json.loads(completed.stdout)
    assert json.loads((output / "plan.json").read_text()) == plan
    assert plan["git_sha"] == head
    assert plan["model"] == "minifold" and plan["device"] == "cpu"
    assert plan["run_model_after_probe"] is False and plan["probe"]["model_weights"] is False
    assert plan["entrypoint_exists"] and not (output / "vm-run.json").exists()
    assert plan["sif_runtime"] == sif_runtime
    assert plan["limits"]["free_disk_gib"] == (64 if sif_runtime == "unsquash" else 40)
    assert plan["prior_mounted_probe_observation"]["status"] == "failed"
    assert "apptainer_source_url" not in plan and "apptainer_source_sha256" not in plan
    assert plan["apptainer_install_method"] == "official-vendor-deb"
    assert plan["apptainer_package_sha256"] == vm.cpu_helpers().APPTAINER_PACKAGE_SHA256
    assert plan["vm_base_image"] == "python:3.12-slim-bookworm"
    assert plan["vm_guest_distribution"] == "Debian 12 (bookworm)"
    assert plan["limits"]["launcher_timeout_seconds"] == plan["launcher_timeout_seconds"] == 53 * 60
    assert plan["full"]["command_timeout_seconds"] == vm.FULL_COMMAND_TIMEOUT == 53 * 60


def test_vm_provenance_and_timeout_are_independent_of_free_source_installer(helpers: Any, tmp_path: Path) -> None:
    helpers.make_plan = lambda *args: _plan() | {
        "apptainer_source_url": "free-runner-source.tar.gz",
        "apptainer_source_sha256": "d" * 64,
        "limits": {"free_disk_gib": 40, "launcher_timeout_seconds": 40 * 60},
    }
    helpers.TOTAL_TIMEOUT = 40 * 60
    plan = vm.make_plan(tmp_path, "a" * 40, True)
    assert not any(field.startswith("apptainer_source_") for field in plan)
    assert plan["apptainer_package_url"] == helpers.APPTAINER_PACKAGE_URL
    assert plan["apptainer_package_sha256"] == helpers.APPTAINER_PACKAGE_SHA256
    assert plan["apptainer_version"] == "1.5.4"
    assert plan["full"]["timeout_seconds"] == 55 * 60
    assert plan["full"]["command_timeout_seconds"] == 53 * 60
    assert plan["limits"]["launcher_timeout_seconds"] == 53 * 60


def test_vm_full_command_bound_preserves_remote_cleanup_and_controller_remainder() -> None:
    assert vm.full_command_timeout(55 * 60) == 53 * 60
    assert vm.full_command_timeout(47 * 60) == 46 * 60
    assert vm.full_command_timeout(90) == 30
    with pytest.raises(TimeoutError, match="remote cleanup"):
        vm.full_command_timeout(60)


# Exact successful command output from the real extracted capability probe:
# https://github.com/softnanolab/boileroom/actions/runs/38134571057
# Log SHA256: 281adf4e004c2a6fbc6aa85ef94ff2a8a6f8c7d9d531477d0a32981197e928b2
EXTRACTED_PROBE_OUTPUT = (
    "INFO:    Converting SIF file to temporary sandbox...\n"
    "WARNING: passwd file doesn't exist in container, not updating\n"
    "WARNING: group file doesn't exist in container, not updating\n"
    "cpu-sif-probe-passed\n"
    "INFO:    Cleaning up image...\n"
)


def test_real_extracted_output_allows_cleanup_after_exact_single_sentinel() -> None:
    assert hashlib.sha256(EXTRACTED_PROBE_OUTPUT.encode()).hexdigest() == (
        "281adf4e004c2a6fbc6aa85ef94ff2a8a6f8c7d9d531477d0a32981197e928b2"
    )
    vm.verify_probe_output(EXTRACTED_PROBE_OUTPUT, "unsquash")
    vm.verify_probe_output("cpu-sif-probe-passed\n", "mounted")
    with pytest.raises(ValueError, match="temporary conversion"):
        vm.verify_probe_output("cpu-sif-probe-passed\n", "unsquash")
    with pytest.raises(ValueError, match="Mounted-SIF probe refused"):
        vm.verify_probe_output(EXTRACTED_PROBE_OUTPUT, "mounted")
    with pytest.raises(ValueError, match="Unknown"):
        vm.verify_probe_output(EXTRACTED_PROBE_OUTPUT, "automatic-fallback")


@pytest.mark.parametrize(
    "output",
    [
        "",
        "INFO: cpu-sif-probe-passed\n",
        "prefix-cpu-sif-probe-passed\n",
        "cpu-sif-probe-passed-suffix\n",
        EXTRACTED_PROBE_OUTPUT + "cpu-sif-probe-passed\n",
    ],
)
def test_probe_sentinel_rejects_missing_partial_and_duplicate_output(output: str) -> None:
    with pytest.raises(ValueError, match="exactly one standalone"):
        vm.verify_probe_output(output, "unsquash")


@pytest.mark.parametrize("command_failed,leftover", [(False, False), (True, False), (False, True)])
def test_real_probe_path_requires_exit_success_conversion_and_observed_rootfs_cleanup(
    helpers: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, command_failed: bool, leftover: bool
) -> None:
    plan = _plan() | {"sif_runtime": "unsquash"}
    scratch, output = tmp_path / "scratch", tmp_path / "reports"
    scratch.mkdir()
    output.mkdir()
    state = {
        "scratch": str(scratch),
        "image_tag": "task-owned:local",
        "nonce": "a" * 32,
        "passed": False,
    }
    monkeypatch.setattr(vm, "verify_remote_source", lambda sha: plan)
    monkeypatch.setattr(vm, "own_scratch", lambda: state)
    monkeypatch.setattr(vm, "verify_toolchain", lambda *args: {"non_root_user_mount_namespace": True})
    monkeypatch.setattr(vm, "give_own_artifacts_to_client", lambda *args: None)
    monkeypatch.setattr(vm.shutil, "copyfile", lambda source, target: target.write_bytes(b"static test binary"))
    helpers.resource_snapshot = lambda paths: {"free_disk_bytes": {str(paths[0]): 4 * 2**30}}
    helpers.OWNER_LABEL = "org.boileroom.cpu-sif-validation"
    cleanup_called: list[bool] = []

    def cleanup(*args: Any) -> dict[str, bool]:
        cleanup_called.append(True)
        return {"passed": True}

    helpers.cleanup = cleanup

    class Commands:
        def __init__(self, *args: Any) -> None:
            pass

        def run(self, name: str, command: list[str], *args: Any, **kwargs: Any) -> str:
            if name == "probe-image-inspect":
                return json.dumps([{"Id": "sha256:verified-image", "Size": 2**20}])
            if name == "probe-sif-build":
                Path(command[4]).write_bytes(b"built local SIF")
            if name == "probe-real-nonroot-sif":
                assert kwargs["env"]["APPTAINER_UNSQUASH"] == "true"
                if command_failed:
                    raise subprocess.CalledProcessError(1, command, output=EXTRACTED_PROBE_OUTPUT)
                if leftover:
                    (Path(kwargs["env"]["APPTAINER_TMPDIR"]) / "rootfs-leftover").mkdir()
                return EXTRACTED_PROBE_OUTPUT
            return ""

    helpers.Commands = Commands
    report = vm.probe_remote("a" * 40, output)
    assert cleanup_called == [True]
    assert report["passed"] is (not command_failed and not leftover)
    assert report["sif_sha256"] == helpers.sha256(scratch / "probe.sif")
    if not command_failed:
        assert report["temporary_extracted_rootfs"]["cleaned"] is (not leftover)
    if command_failed:
        assert "CalledProcessError" in report["error"]
    elif leftover:
        assert "retained" in report["error"] and (scratch / "apptainer-tmp/rootfs-leftover").exists()


@pytest.mark.parametrize("selected", ["mounted", "unsquash"])
def test_runtime_environment_is_explicit_without_automatic_fallback(selected: str) -> None:
    environment = {"APPTAINER_UNSQUASH": "inherited", "SINGULARITY_UNSQUASH": "true", "KEEP": "value"}
    original = dict(environment)
    actual = vm.sif_runtime_environment({"sif_runtime": selected}, environment)
    assert actual["APPTAINER_UNSQUASH"] == ("true" if selected == "unsquash" else "false")
    assert "SINGULARITY_UNSQUASH" not in actual and actual["KEEP"] == "value"
    assert environment == original
    with pytest.raises(ValueError, match="Unknown"):
        vm.sif_runtime_environment({"sif_runtime": "automatic-fallback"}, environment)


@pytest.mark.parametrize("probe,margin", [(True, 2), (False, 42)])
def test_extraction_requires_measured_extra_disk_and_retains_failure_evidence(
    helpers: Any, tmp_path: Path, probe: bool, margin: int
) -> None:
    helpers.LIMITS = {"minifold": {"free_disk_gib": 40}}
    required = (margin + 10) * 2**30  # 1.25 times an 8-GiB image, plus the unchanged model margin.
    helpers.resource_snapshot = lambda paths: {"free_disk_bytes": {str(paths[0]): required - 1}}
    state: dict[str, Any] = {}
    with pytest.raises(RuntimeError, match="Insufficient"):
        vm.extraction_preflight(state, {"sif_runtime": "unsquash"}, {"Size": 8 * 2**30}, tmp_path, probe=probe)
    assert state["extraction_disk_preflight"]["required_free_bytes"] == required
    helpers.resource_snapshot = lambda paths: {"free_disk_bytes": {str(paths[0]): required}}
    vm.extraction_preflight(state, {"sif_runtime": "unsquash"}, {"Size": 8 * 2**30}, tmp_path, probe=probe)
    with pytest.raises(ValueError, match="real Docker image size"):
        vm.extraction_preflight({}, {"sif_runtime": "unsquash"}, {}, tmp_path, probe=probe)
    vm.extraction_preflight({}, {"sif_runtime": "mounted"}, {}, tmp_path, probe=probe)


def test_extraction_cleanup_is_observed_before_owned_scratch_deletion(tmp_path: Path) -> None:
    state: dict[str, Any] = {}
    leftover = tmp_path / "rootfs-observed"
    leftover.mkdir()
    with pytest.raises(RuntimeError, match="retained"):
        vm.extraction_cleanup_evidence(state, {"sif_runtime": "unsquash"}, tmp_path)
    assert state["temporary_extracted_rootfs"] == {"remaining": [leftover.name], "cleaned": False}
    assert leftover.exists()
    leftover.rmdir()
    vm.extraction_cleanup_evidence(state, {"sif_runtime": "unsquash"}, tmp_path)
    assert state["temporary_extracted_rootfs"]["cleaned"] is True


def test_extracted_toolchain_records_fuse_denial_without_permission_changes(
    helpers: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(vm.Path, "is_char_device", lambda self: False)
    calls: list[Any] = []

    def run(name: str, command: list[str], *args: Any) -> str:
        calls.append((name, command))
        return {
            "apptainer-version": "apptainer version 1.5.4",
            "apptainer-buildcfg": "APPTAINER_SUID_INSTALL=0\n",
            "buildx-version": "github.com/docker/buildx v0.38.0",
            "non-root-fuse-access": json.dumps({"uid": 998, "can_open": False, "error_type": "PermissionError"}),
        }.get(name, "")

    commands = SimpleNamespace(run=run)
    observed = vm.verify_toolchain(commands, {"sif_runtime": "unsquash"})
    assert observed["non_root_fuse_access"]["can_open"] is False
    assert observed["fuse_character_device"] is False
    encoded = json.dumps(calls)
    for forbidden in ("chmod", "mknod", "usermod", "setfacl", "sysctl", "--privileged"):
        assert forbidden not in encoded
    with pytest.raises(ValueError, match="no security workaround"):
        vm.verify_toolchain(commands, {"sif_runtime": "mounted"})


def test_safe_default_is_offline_probe_plan(helpers: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(vm, "make_plan", lambda *args: _plan())
    monkeypatch.setattr(vm, "execute", lambda *args: pytest.fail("Offline plan launched"))
    assert vm.main(["--output-dir", str(tmp_path)]) == 0
    assert json.loads((tmp_path / "plan.json").read_text())["run_model_after_probe"] is False
    assert not (tmp_path / "vm-run.json").exists()


@pytest.mark.parametrize("change", [{"dirty": True}, {"entrypoint_exists": False}])
def test_paid_execution_requires_clean_exact_source(
    helpers: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, change: dict[str, Any]
) -> None:
    monkeypatch.setattr(vm, "make_plan", lambda *args: _plan() | change)
    monkeypatch.setattr(vm, "execute", lambda *args: pytest.fail("Invalid source launched"))
    with pytest.raises(SystemExit) as caught:
        vm.main(["--execute", "--expected-sha", "a" * 40, "--output-dir", str(tmp_path)])
    assert caught.value.code == 2


def test_internal_modes_cannot_bypass_execute_or_artifact_scope(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as caught:
        vm.main(["--remote-full", "--expected-sha", "a" * 40, "--output-dir", str(tmp_path)])
    assert caught.value.code == 2


def test_source_pack_reconstructs_exact_head_without_config_credentials(helpers: Any, tmp_path: Path) -> None:
    root, restored = tmp_path / "source", tmp_path / "restored"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / "public.txt").write_text("reviewed source\n")
    subprocess.run(["git", "-C", str(root), "add", "public.txt"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    head = helpers.git(root, "rev-parse", "HEAD")
    with (root / ".git/config").open("a") as handle:
        handle.write("[private]\n token = never-transmit-this-config\n")
    (root / "untracked-secret").write_text("never transmit this untracked file")
    pack = tmp_path / "source.pack"
    assert vm.source_pack(root, pack, head) == helpers.sha256(pack)
    subprocess.run(["git", "init", "-q", str(restored)], check=True)
    subprocess.run(["git", "-C", str(restored), "unpack-objects", "-q"], input=pack.read_bytes(), check=True)
    (restored / ".git/shallow").write_text(head + "\n")
    subprocess.run(["git", "-C", str(restored), "update-ref", "refs/heads/validation", head], check=True)
    subprocess.run(["git", "-C", str(restored), "checkout", "-q", "--detach", head], check=True)
    assert helpers.git(restored, "rev-parse", "HEAD") == head
    assert (restored / "public.txt").read_text() == "reviewed source\n"
    assert not (restored / "untracked-secret").exists()
    assert "never-transmit" not in (restored / ".git/config").read_text()
    assert helpers.git(restored, "status", "--porcelain") == ""


def test_image_has_official_hashed_tools_and_production_client_sdk(helpers: Any, tmp_path: Path) -> None:
    events = []

    class Image:
        def __getattr__(self, name: str) -> Any:
            def call(*args: Any, **kwargs: Any) -> Image:
                events.append((name, args, kwargs))
                return self

            return call

    image = Image()
    modal = SimpleNamespace(Image=SimpleNamespace(from_registry=lambda value: image))
    vm.build_image(modal, tmp_path / "pack", tmp_path / "plan", _plan() | {"source_pack_sha256": "c" * 64})
    encoded = json.dumps(events, default=str)
    assert vm.BUILDX_SHA256 in encoded and helpers.APPTAINER_PACKAGE_SHA256 in encoded
    assert f"modal=={vm.CLIENT_MODAL_VERSION}" in encoded
    assert f"modal=={vm.MODAL_VERSION}" not in encoded
    assert "chown -R validation:validation /validation" in encoded
    for forbidden in ("seccomp=unconfined", "apparmor=unconfined", "--privileged", "sysctl", "MODAL_TOKEN", "HF_TOKEN"):
        assert forbidden not in encoded


@pytest.mark.parametrize(
    "identifier", ["sb-" + "a" * 22, "01JMRXY3V2H8B1N4F9KTZP0C7A", "f69eb0aa-798c-46e8-9b4e-10b0c67304bb"]
)
def test_v2_identifiers_are_accepted_without_forcing_legacy_backend(identifier: str) -> None:
    assert vm.SDK_ID.fullmatch(identifier)


@pytest.mark.parametrize("identifier", ["../other", "abc\n--flag", "a" * 129, ""])
def test_unsafe_identifier_syntax_is_rejected(identifier: str) -> None:
    assert not vm.SDK_ID.fullmatch(identifier)


def _fake_sdk(
    monkeypatch: pytest.MonkeyPatch, *, probe_passed: bool = True, full_passed: bool = True, malformed: bool = False
) -> tuple[Any, list[Any]]:
    events: list[Any] = []

    class App:
        def __init__(self, name: str) -> None:
            self.app_id = "ap-" + "a" * 22
            events.append(("app", name))

        @asynccontextmanager
        async def run(self, *, detach: bool) -> Any:
            assert detach is False
            events.append("app-enter")
            yield self
            events.append("app-exit")

    async def image_build(**kwargs: Any) -> None:
        events.append("image-build")

    image = SimpleNamespace(object_id="im-reviewed", build=SimpleNamespace(aio=image_build))
    monkeypatch.setattr(vm, "build_image", lambda *args: image)

    def pack(root: Path, path: Path, head: str) -> str:
        path.write_bytes(b"source objects")
        return "c" * 64

    monkeypatch.setattr(vm, "source_pack", pack)
    count = 0

    async def create(*args: Any, **kwargs: Any) -> Any:
        nonlocal count
        count += 1
        phase = "probe" if count == 1 else "full"
        events.append(("create", phase, args, kwargs))
        sandbox = SimpleNamespace(object_id=f"opaque-v2-sandbox-{count}", ended=False)

        async def ready() -> None:
            events.append(("ready", phase))

        async def exec_(*args: Any, **kwargs: Any) -> Any:
            events.append(("exec", phase, args, kwargs))
            passed = probe_passed if phase == "probe" else full_passed
            text = (
                "not-json"
                if malformed
                else json.dumps(
                    {"report": {"passed": passed}, "evidence": {"run.json": json.dumps({"passed": passed})}}
                )
            )

            async def stdout() -> str:
                return text

            async def stderr() -> str:
                return ""

            async def wait() -> None:
                return None

            return SimpleNamespace(
                stdout=SimpleNamespace(read=SimpleNamespace(aio=stdout)),
                stderr=SimpleNamespace(read=SimpleNamespace(aio=stderr)),
                wait=SimpleNamespace(aio=wait),
                returncode=0 if passed else 1,
            )

        async def terminate(*, wait: bool) -> int:
            assert wait is True
            sandbox.ended = True
            events.append(("terminate", phase))
            return 137

        async def poll() -> int | None:
            return 137 if sandbox.ended else None

        sandbox.wait_until_ready = SimpleNamespace(aio=ready)
        sandbox.exec = SimpleNamespace(aio=exec_)
        sandbox.terminate = SimpleNamespace(aio=terminate)
        sandbox.poll = SimpleNamespace(aio=poll)
        return sandbox

    return SimpleNamespace(
        __version__=vm.MODAL_VERSION,
        App=App,
        Sandbox=SimpleNamespace(create=SimpleNamespace(aio=create)),
        Probe=SimpleNamespace(with_exec=lambda *args, **kwargs: (args, kwargs)),
    ), events


def test_default_execute_runs_only_small_no_weights_probe_and_stops_it(
    helpers: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    modal, events = _fake_sdk(monkeypatch)
    assert asyncio.run(vm.execute(modal, tmp_path, _plan(), tmp_path))
    calls = [event for event in events if isinstance(event, tuple) and event[0] == "create"]
    assert len(calls) == 1
    kwargs = calls[0][3]
    assert kwargs["runtime"] == "vm" and kwargs["cpu"] == (1, 1) and kwargs["memory"] == (2048, 2048)
    assert kwargs["timeout"] <= 300
    assert kwargs["secrets"] == [] and kwargs["volumes"] == {}
    assert kwargs["encrypted_ports"] == kwargs["unencrypted_ports"] == kwargs["h2_ports"] == []
    assert ("ready", "probe") in events and ("terminate", "probe") in events
    report = json.loads((tmp_path / "vm-run.json").read_text())
    assert report["passed"] and report["model_validation_status"] == "unrun"
    assert report["assigned_budget_usd"] == 10 and report["monetary_cap_enforced"] is False


def test_full_vm_requires_passing_probe_and_uses_fixed_bounds(
    helpers: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    modal, events = _fake_sdk(monkeypatch)
    assert asyncio.run(vm.execute(modal, tmp_path, _plan(True), tmp_path))
    calls = [event for event in events if isinstance(event, tuple) and event[0] == "create"]
    assert [call[1] for call in calls] == ["probe", "full"]
    assert calls[1][3]["cpu"] == (4, 4) and calls[1][3]["memory"] == (32768, 32768)
    assert calls[1][3]["timeout"] <= 55 * 60
    assert events.index(("terminate", "probe")) < events.index(calls[1])
    assert json.loads((tmp_path / "vm-run.json").read_text())["model_validation_status"] == "passed"


def test_failed_probe_prevents_large_vm_and_weights(
    helpers: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    modal, events = _fake_sdk(monkeypatch, probe_passed=False)
    assert not asyncio.run(vm.execute(modal, tmp_path, _plan(True), tmp_path))
    assert len([event for event in events if isinstance(event, tuple) and event[0] == "create"]) == 1
    assert ("terminate", "probe") in events
    assert json.loads((tmp_path / "vm-run.json").read_text())["model_validation_status"] == "unrun"


def test_remote_json_failure_still_terminates_recorded_vm(
    helpers: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    modal, events = _fake_sdk(monkeypatch, malformed=True)
    assert not asyncio.run(vm.execute(modal, tmp_path, _plan(), tmp_path))
    report = json.loads((tmp_path / "vm-run.json").read_text())
    assert report["sandboxes"][0]["terminated"]
    assert ("terminate", "probe") in events


def test_controller_rejects_production_sdk_without_launch(helpers: Any, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="modal==1.6.1"):
        asyncio.run(vm.execute(SimpleNamespace(__version__="1.3.0.post1"), tmp_path, _plan(), tmp_path))


def test_cleanup_only_requires_matching_nonce_tags_before_stop(
    helpers: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    nonce = "a" * 32
    state = {
        "controller_nonce": nonce,
        "app_name": f"boileroom-cpu-vm-{nonce[:12]}",
        "app_id": "ap-" + "a" * 22,
        "sandboxes": [{"sandbox_id": "opaque-v2-owned-sandbox", "terminated": False}],
    }
    _write(tmp_path / "vm-run.json", state)
    _write(tmp_path / "plan.json", {"controller_nonce": nonce})
    calls = []

    async def list_(**kwargs: Any) -> Any:
        assert kwargs["app_id"] == state["app_id"]
        assert kwargs["tags"] == {"boileroom_validation_nonce": nonce}
        if False:
            yield None

    async def from_id(identifier: str) -> Any:
        calls.append(identifier)

        async def tags() -> dict[str, str]:
            return {"boileroom_validation_nonce": "other-owner"}

        async def terminate(**kwargs: Any) -> None:
            pytest.fail("Unowned sandbox was stopped")

        return SimpleNamespace(get_tags=SimpleNamespace(aio=tags), terminate=SimpleNamespace(aio=terminate))

    modal = SimpleNamespace(
        __version__=vm.MODAL_VERSION,
        Sandbox=SimpleNamespace(list=SimpleNamespace(aio=list_), from_id=SimpleNamespace(aio=from_id)),
    )
    assert not asyncio.run(vm.cleanup_only(modal, tmp_path))
    assert calls == ["opaque-v2-owned-sandbox"]
    report = json.loads((tmp_path / "vm-cleanup.json").read_text())
    assert not report["app_stop_attempted"] and "ownership tag" in report["errors"][0]


def test_cleanup_confirms_only_owned_app_stopped_tasks_zero(
    helpers: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    nonce = "a" * 32
    app_id = "ap-" + "a" * 22
    state = {
        "controller_nonce": nonce,
        "app_name": f"boileroom-cpu-vm-{nonce[:12]}",
        "app_id": app_id,
        "sandboxes": [{"sandbox_id": "opaque-v2-owned-sandbox", "terminated": True}],
    }
    _write(tmp_path / "vm-run.json", state)
    _write(tmp_path / "plan.json", {"controller_nonce": nonce})

    async def list_(**kwargs: Any) -> Any:
        if False:
            yield None

    calls = []
    count = 0

    async def cli(arguments: list[str], timeout: float = 25) -> tuple[int, bytes]:
        nonlocal count
        calls.append(arguments)
        if "stop" in arguments:
            return 0, b"Stopped owned App"
        count += 1
        return 0, json.dumps(
            [
                {
                    "app_id": app_id,
                    "description": state["app_name"],
                    "state": "ephemeral" if count == 1 else "stopped",
                    "tasks": "1" if count == 1 else "0",
                },
                {"app_id": "ap-unrelated", "description": "private unrelated app", "state": "deployed", "tasks": "7"},
            ]
        ).encode()

    monkeypatch.setattr(vm, "modal_cli", cli)
    modal = SimpleNamespace(__version__=vm.MODAL_VERSION, Sandbox=SimpleNamespace(list=SimpleNamespace(aio=list_)))
    assert asyncio.run(vm.cleanup_only(modal, tmp_path))
    assert ["app", "stop", "--yes", app_id] in calls
    report = json.loads((tmp_path / "vm-cleanup.json").read_text())
    assert report["app_stopped_tasks_zero"] and len(report["app_verification"]) == 1
    assert "private unrelated app" not in (tmp_path / "vm-cleanup.json").read_text()


def test_cli_timeout_reaps_only_owned_new_session(monkeypatch: pytest.MonkeyPatch) -> None:
    events = []

    class Process:
        pid = 7654321
        returncode = None

        async def communicate(self) -> Any:
            await asyncio.sleep(10)

        async def wait(self) -> int:
            self.returncode = -15
            return self.returncode

    async def create(*args: Any, **kwargs: Any) -> Any:
        assert kwargs["start_new_session"] is True
        return Process()

    monkeypatch.setattr(vm.asyncio, "create_subprocess_exec", create)
    monkeypatch.setattr(vm.os, "killpg", lambda pid, sig: events.append((pid, sig)))
    with pytest.raises(TimeoutError):
        asyncio.run(vm.modal_cli(["app", "list", "--json"], 0.001))
    assert events == [(7654321, vm.signal.SIGTERM)]


def test_vm_workflow_is_explicit_isolated_and_probe_defaults_to_no_weights() -> None:
    import yaml

    root = Path(__file__).resolve().parents[2]
    workflow = yaml.safe_load((root / ".github/workflows/integration-tests.yaml").read_text())
    events = workflow.get("on", workflow.get(True))
    assert events["workflow_dispatch"]["inputs"]["validation-mode"]["default"] == "core"
    job = workflow["jobs"]["cpu-vm-validation"]
    assert "github.event_name == 'workflow_dispatch'" in job["if"]
    assert "inputs.validation-mode == 'cpu-vm-probe'" in job["if"]
    assert "inputs.validation-mode == 'cpu-vm-sif'" in job["if"]
    assert "inputs.validation-mode == 'cpu-vm-extract-probe'" in job["if"]
    assert "inputs.validation-mode == 'cpu-vm-extract-sif'" in job["if"]
    assert job["timeout-minutes"] == 60
    assert job["env"]["MODAL_CONFIG_PATH"] == "/dev/null"
    assert "runner." not in json.dumps(job["env"])
    assert "secrets." not in json.dumps(job["env"])
    checkout = next(step for step in job["steps"] if step.get("uses", "").startswith("actions/checkout@"))
    assert checkout["with"] == {"ref": "${{ github.sha }}", "persist-credentials": False}
    plan = next(step for step in job["steps"] if step["name"].startswith("Plan bounded"))
    assert 'test "$VALIDATION_MODEL" = minifold' in plan["run"]
    assert 'test "$MINIFOLD_SIZE" = 12L' in plan["run"]
    assert "secrets." not in json.dumps(plan)
    execute = next(step for step in job["steps"] if step["name"].startswith("Execute approved"))
    assert execute["timeout-minutes"] == 55
    for step in (plan, execute):
        assert (
            'if [[ "$VALIDATION_MODE" = cpu-vm-sif || "$VALIDATION_MODE" = cpu-vm-extract-sif ]]; then' in step["run"]
        )
        assert "mode_args=(--run-model-after-probe)" in step["run"]
        assert (
            'if [[ "$VALIDATION_MODE" = cpu-vm-extract-probe || "$VALIDATION_MODE" = cpu-vm-extract-sif ]]; then'
            in step["run"]
        )
        assert "mode_args+=(--sif-runtime unsquash)" in step["run"]
        assert '"${mode_args[@]}"' in step["run"]
    assert "--with modal==1.6.1" in execute["run"]
    assert "--with modal==1.6.1" not in json.dumps(workflow["jobs"]["model-validation"])
    cleanup = next(step for step in job["steps"] if "--cleanup-only" in step.get("run", ""))
    assert cleanup["if"] == "always()" and cleanup["timeout-minutes"] == 3
    assert '[ -f "$ARTIFACT_DIR/vm-run.json" ]' in cleanup["run"]
    upload = next(step for step in job["steps"] if step.get("uses", "").startswith("actions/upload-artifact@"))
    assert upload["if"] == "always()"
