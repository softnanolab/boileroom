"""Test-only Modal CPU VM probe and canonical local-SIF validation.

Offline plan by default. A paid execution needs --execute and a clean exact SHA.
The default execution stops after a <=5-minute, 2-GiB no-weights probe. Explicit
--run-model-after-probe additionally allows one 32-GiB/4-core VM if the real
unprivileged Apptainer probe passes. The assigned budget is $10 from MiniFold's
$80 allocation; resource/time bounds are not a monetary cap.

The control SDK is separately pinned to modal==1.6.1, which supports runtime="vm".
The VM's public-wrapper client keeps production modal==1.3.0.post1. No secrets,
volumes, external ports, security workarounds or registry publication are used.

https://modal.com/docs/sdk/py/releases#160-2026-09-28
https://modal.com/docs/guide/sandboxes#runtimes
https://modal.com/docs/guide/sandbox-resources
https://github.com/docker/buildx/releases/tag/v0.38.0
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import importlib.util
import json
import os
import pwd
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from functools import cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
MODAL_VERSION = "1.6.1"
CLIENT_MODAL_VERSION = "1.3.0.post1"
BUILDX_VERSION = "0.38.0"
BUILDX_URL = "https://github.com/docker/buildx/releases/download/v0.38.0/buildx-v0.38.0.linux-amd64"
BUILDX_SHA256 = "4fe4cc38adf48169132749b6ca22a990928db0118e3407584ee553723115d287"
TOTAL_TIMEOUT = 53 * 60
IMAGE_TIMEOUT = 15 * 60
PROBE_TIMEOUT = 5 * 60
FULL_TIMEOUT = 55 * 60
REMOTE_OUTPUT = Path("/tmp/cpu-sif-reports")
REMOTE_RUNNER_TEMP = Path("/tmp/cpu-sif-runner")
# V2 Sandbox identifiers are opaque and are not legacy sb-22-character IDs.
SDK_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{5,127}\Z")
APP_ID = re.compile(r"ap-[A-Za-z0-9]{22}\Z")


@cache
def cpu_helpers() -> Any:
    """Load the reviewed sibling even when this file is invoked directly."""
    path = ROOT / "scripts/testing/run_cpu_sif_validation.py"
    spec = importlib.util.spec_from_file_location("_boileroom_cpu_sif_validation_helpers", path)
    if spec is None or spec.loader is None:
        raise ValueError("CPU SIF validation helper is unavailable")
    helper = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = helper
    try:
        spec.loader.exec_module(helper)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    return helper


def make_plan(root: Path, expected_sha: str | None, run_model: bool) -> dict[str, Any]:
    cpu = cpu_helpers()
    plan = cpu.make_plan(root, "minifold", "12L", expected_sha)
    plan.update(
        {
            "mode": "cpu-vm-sif",
            "control_modal_sdk": MODAL_VERSION,
            "client_modal_sdk": CLIENT_MODAL_VERSION,
            "buildx_version": BUILDX_VERSION,
            "buildx_sha256": BUILDX_SHA256,
            "run_model_after_probe": run_model,
            "paid_compute": True,
            "assigned_budget_usd": 10,
            "monetary_cap_enforced": False,
            "probe": {
                "cpu": [1, 1],
                "memory_mib": [2048, 2048],
                "timeout_seconds": PROBE_TIMEOUT,
                "model_weights": False,
            },
            "full": {"cpu": [4, 4], "memory_mib": [32768, 32768], "timeout_seconds": FULL_TIMEOUT, "model_size": "12L"},
            "launcher_timeout_seconds": TOTAL_TIMEOUT,
            "scope": "Real CPU non-setuid SIF/public wrapper/transport/lifecycle. Independent FP32 CPU stock parity, GPU and compile remain unrun.",
        }
    )
    return plan


def source_pack(root: Path, destination: Path, head: str) -> str:
    """Pack only the exact HEAD commit/current tree, never Git config or credentials."""
    if not re.fullmatch(r"[0-9a-f]{40}", head):
        raise ValueError("Source pack requires a full SHA")
    cpu = cpu_helpers()
    objects = {head, cpu.git(root, "rev-parse", "HEAD^{tree}")}
    for line in cpu.git(root, "ls-tree", "-r", "-t", "HEAD").splitlines():
        mode, kind, object_id = line.split("\t", 1)[0].split()
        if kind == "commit" or mode == "160000":
            raise ValueError("Review source packaging before introducing submodules")
        objects.add(object_id)
    with destination.open("wb") as handle:
        subprocess.run(
            ["git", "-C", str(root), "pack-objects", "--stdout"],
            input="".join(value + "\n" for value in sorted(objects)).encode(),
            stdout=handle,
            check=True,
            timeout=60,
        )
    return cpu.sha256(destination)


def build_image(modal: Any, pack: Path, plan_file: Path, plan: dict[str, Any]) -> Any:
    cpu = cpu_helpers()
    image = modal.Image.from_registry("python:3.12-slim-bookworm").apt_install(
        "docker.io", "ca-certificates", "curl", "git", "fuse3", "util-linux", "busybox-static"
    )
    image = image.run_commands(
        f"curl --fail --location --proto '=https' --tlsv1.2 --max-time 180 {cpu.APPTAINER_PACKAGE_URL} -o /tmp/apptainer.deb && echo '{cpu.APPTAINER_PACKAGE_SHA256}  /tmp/apptainer.deb' | sha256sum --check - && apt-get update && apt-get install -y /tmp/apptainer.deb && rm -rf /var/lib/apt/lists/*",
        f"mkdir -p /usr/local/lib/docker/cli-plugins && curl --fail --location --proto '=https' --tlsv1.2 --max-time 180 {BUILDX_URL} -o /tmp/docker-buildx && echo '{BUILDX_SHA256}  /tmp/docker-buildx' | sha256sum --check - && install -m 755 /tmp/docker-buildx /usr/local/lib/docker/cli-plugins/docker-buildx && rm /tmp/docker-buildx",
        "useradd --system --create-home --shell /usr/sbin/nologin validation",
    ).pip_install(f"modal=={CLIENT_MODAL_VERSION}", "numpy==2.2.6", "httpx", "pyyaml", "biotite", "scipy")
    image = image.add_local_file(pack, "/tmp/source.pack", copy=True)
    image = image.add_local_file(plan_file, "/tmp/source-plan.json", copy=True)
    return image.run_commands(
        f"echo '{plan['source_pack_sha256']}  /tmp/source.pack' | sha256sum --check - && git init /validation && git -C /validation unpack-objects < /tmp/source.pack && printf '%s\\n' '{plan['git_sha']}' > /validation/.git/shallow && git -C /validation update-ref refs/heads/validation {plan['git_sha']} && git -C /validation checkout --detach {plan['git_sha']} && chown -R validation:validation /validation && rm /tmp/source.pack"
    ).env({"PYTHONPATH": "/validation:/validation/tests", "PYTHONUNBUFFERED": "1"})


def verify_remote_source(expected_sha: str) -> dict[str, Any]:
    cpu = cpu_helpers()
    plan = json.loads(Path("/tmp/source-plan.json").read_text())

    # The checkout belongs to the unprivileged public client. Root build-control
    # processes ask its owner to inspect Git, preserving Git's ownership safety.
    def source_git(*args: str) -> str:
        prefix = ["runuser", "-u", "validation", "--"] if os.geteuid() == 0 else []
        return subprocess.check_output(prefix + ["git", "-C", str(ROOT), *args], text=True).strip()

    head = source_git("rev-parse", "HEAD")
    dirty = source_git("status", "--porcelain", "--untracked-files=normal")
    names = source_git("ls-files", "-z", "boileroom", "tests", "scripts", "pyproject.toml").split("\0")
    hashes = {name: cpu.sha256(ROOT / name) for name in sorted(names) if name and (ROOT / name).is_file()}
    source_hash = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    if dirty or source_hash != plan["source_sha256"] or head != plan["git_sha"] or expected_sha != head:
        raise ValueError("VM source does not match the reviewed clean source plan")
    if not SDK_ID.fullmatch(os.environ.get("BOILEROOM_VALIDATION_VM_ID", "")):
        raise ValueError("VM must be invoked by its recorded Sandbox controller")
    if cpu.sha256(Path("/tmp/apptainer.deb")) != cpu.APPTAINER_PACKAGE_SHA256:
        raise ValueError("VM's official Apptainer package hash mismatch")
    return plan


def verify_toolchain(commands: Any) -> dict[str, Any]:
    cpu = cpu_helpers()
    version = commands.run("apptainer-version", ["apptainer", "--version"]).strip()
    cfg = commands.run("apptainer-buildcfg", ["apptainer", "buildcfg"])
    if version != f"apptainer version {cpu.APPTAINER_VERSION}" or not re.search(
        r"^APPTAINER_SUID_INSTALL\s*=\s*0\s*$", cfg, re.MULTILINE
    ):
        raise ValueError("VM must use the verified non-setuid Apptainer version")
    buildx = commands.run("buildx-version", ["docker", "buildx", "version"])
    if f"v{BUILDX_VERSION}" not in buildx:
        raise ValueError("VM Docker Buildx version mismatch")
    if not Path("/dev/fuse").is_char_device():
        raise ValueError("VM's existing /dev/fuse is unavailable; no security workaround is allowed")
    commands.run(
        "non-root-namespace-probe",
        ["runuser", "-u", "validation", "--", "unshare", "--user", "--map-root-user", "--mount", "true"],
        30,
    )
    commands.run("non-root-uid", ["runuser", "-u", "validation", "--", "sh", "-c", "test $(id -u) -ne 0"], 30)
    return {
        "apptainer": version,
        "buildx": buildx.strip(),
        "fuse_character_device": True,
        "non_root_user_mount_namespace": True,
    }


def own_scratch() -> dict[str, Any]:
    cpu = cpu_helpers()
    nonce = uuid.uuid4().hex
    REMOTE_RUNNER_TEMP.mkdir(exist_ok=True)
    os.environ["RUNNER_TEMP"] = str(REMOTE_RUNNER_TEMP)
    scratch = REMOTE_RUNNER_TEMP / f"boileroom-cpu-sif-{nonce}"
    scratch.mkdir()
    (scratch / ".owner").write_text(nonce)
    return {
        "nonce": nonce,
        "scratch": str(scratch),
        "builder": f"boileroom-cpu-sif-{nonce}",
        "image_tag": f"boileroom-cpu-sif-{nonce}:local",
        "passed": False,
        "validation_status": "unrun",
        "started_at": cpu.utc_now(),
    }


def give_own_artifacts_to_client(scratch: Path, output: Path) -> None:
    user = pwd.getpwnam("validation")
    for root in (scratch, output):
        for path in [root, *root.rglob("*")]:
            if path.is_symlink():
                raise ValueError("Unexpected symlink in task-owned execution artifacts")
            os.chown(path, user.pw_uid, user.pw_gid)


def probe_remote(expected_sha: str, output: Path) -> dict[str, Any]:
    cpu = cpu_helpers()
    plan = verify_remote_source(expected_sha)
    state = own_scratch()
    state["scope"] = (
        "Docker and real non-root static-binary SIF capability only; model weights and MiniFold inference remain unrun."
    )
    scratch = Path(state["scratch"])
    commands = cpu.Commands(output, time.monotonic() + PROBE_TIMEOUT - 45)
    try:
        state["toolchain"] = verify_toolchain(commands)
        commands.run("docker-info", ["docker", "info"], 30)
        context = scratch / "probe-context"
        context.mkdir()
        shutil.copyfile("/bin/busybox", context / "busybox")
        (context / "busybox").chmod(0o755)
        (context / "Dockerfile").write_text("FROM scratch\nCOPY busybox /bin/busybox\n")
        state["model_build_attempted"] = True
        cpu.write_json(output / "run.json", state)
        commands.run(
            "probe-image-build",
            [
                "docker",
                "buildx",
                "build",
                "--load",
                "--platform",
                "linux/amd64",
                "--tag",
                state["image_tag"],
                "--label",
                f"{cpu.OWNER_LABEL}={state['nonce']}",
                str(context),
            ],
            45,
        )
        inspected = json.loads(commands.run("probe-image-inspect", ["docker", "image", "inspect", state["image_tag"]]))[
            0
        ]
        state["docker_image_id"] = inspected["Id"]
        sif = scratch / "probe.sif"
        env = os.environ.copy() | {
            "APPTAINER_TMPDIR": str(scratch / "apptainer-tmp"),
            "APPTAINER_CACHEDIR": str(scratch / "apptainer-cache"),
        }
        Path(env["APPTAINER_TMPDIR"]).mkdir()
        commands.run(
            "probe-sif-build",
            [
                "apptainer",
                "build",
                "--mksquashfs-args",
                "-processors 1 -mem 256M",
                str(sif),
                f"docker-daemon:{state['image_tag']}",
            ],
            60,
            env=env,
        )
        state["sif_sha256"] = cpu.sha256(sif)
        give_own_artifacts_to_client(scratch, output)
        value = commands.run(
            "probe-real-nonroot-sif",
            [
                "runuser",
                "-u",
                "validation",
                "--",
                "apptainer",
                "exec",
                "--cleanenv",
                "--home",
                "/tmp",
                str(sif),
                "/bin/busybox",
                "echo",
                "cpu-sif-probe-passed",
            ],
            30,
            env=env,
        )
        if value.strip() != "cpu-sif-probe-passed":
            raise ValueError("Real non-root SIF smoke did not return the expected output")
        state["passed"] = True
        state["validation_status"] = "passed"
    except BaseException as error:
        state["error"] = f"{type(error).__name__}: {error}"
        state["validation_status"] = "failed"
    finally:
        state["cleanup"] = cpu.cleanup(state, output)
        state["passed"] = state["passed"] and state["cleanup"]["passed"]
        state["source_sha256"] = plan["source_sha256"]
        state["finished_at"] = cpu.utc_now()
        cpu.write_json(output / "run.json", state)
    return state


def public_child(expected_sha: str, output: Path) -> dict[str, Any]:
    cpu = cpu_helpers()
    plan = verify_remote_source(expected_sha)
    if os.geteuid() == 0:
        raise ValueError("Real public SIF client must run as the non-root validation user")
    state = json.loads((output / "run.json").read_text())
    nonce = state.get("nonce", "")
    scratch = Path(state["scratch"])
    namespace, tag = f"docker.io/boileroom-cpu-validation-{nonce}", f"sha-{plan['git_sha']}"
    if (
        not cpu.NONCE.fullmatch(nonce)
        or scratch.parent != REMOTE_RUNNER_TEMP
        or scratch.name != f"boileroom-cpu-sif-{nonce}"
        or scratch.is_symlink()
        or (scratch / ".owner").read_text() != nonce
    ):
        raise ValueError("VM child scratch ownership mismatch")
    if (
        state.get("validation_status") != "running"
        or os.environ.get("MODEL_DIR") != str(scratch / "models")
        or os.environ.get("BOILEROOM_DOCKER_REPOSITORY") != namespace
        or os.environ.get("BOILEROOM_IMAGE_TAG") != tag
        or state["image_uri"] != f"docker://{namespace}/boileroom-minifold:{tag}"
    ):
        raise ValueError("VM public child cache/image/source provenance mismatch")
    return cpu.run_public_child(ROOT, plan, state, output)


def full_remote(expected_sha: str, output: Path) -> dict[str, Any]:
    cpu = cpu_helpers()
    plan = verify_remote_source(expected_sha)
    if not plan["run_model_after_probe"]:
        raise ValueError("Model execution was not explicitly selected")
    state = own_scratch()
    scratch = Path(state["scratch"])
    duration = min(cpu.TOTAL_TIMEOUT, int(os.environ["BOILEROOM_VALIDATION_REMOTE_TIMEOUT"]) - 60)
    commands = cpu.Commands(output, time.monotonic() + duration)
    previous = signal.getsignal(signal.SIGTERM)

    def interrupted(_signal: int, _frame: Any) -> None:
        raise KeyboardInterrupt("CPU VM interrupted")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        cpu.write_json(output / "run.json", state)
        state["toolchain"] = verify_toolchain(commands)
        info = json.loads(commands.run("docker-info", ["docker", "info", "--format", "{{json .}}"]))
        state["resources"] = cpu.resource_snapshot([REMOTE_RUNNER_TEMP, Path(info["DockerRootDir"])])
        blockers = cpu.resource_blockers(plan, state["resources"])
        if blockers:
            state["resource_blockers"] = blockers
            raise RuntimeError("VM resource preflight blocked unchanged model: " + "; ".join(blockers))
        state["builder_attempted"] = True
        cpu.write_json(output / "run.json", state)
        commands.run(*cpu.build_commands(ROOT, state, plan)[0])
        commands.run("builder-bootstrap", ["docker", "buildx", "inspect", "--bootstrap", state["builder"]])
        builder = json.loads(
            commands.run(
                "builder-resources",
                [
                    "docker",
                    "inspect",
                    "--format",
                    '{"image_id":{{json .Image}},"memory":{{.HostConfig.Memory}},"cpu_quota":{{.HostConfig.CpuQuota}},"cpu_period":{{.HostConfig.CpuPeriod}}}',
                    f"buildx_buildkit_{state['builder']}0",
                ],
            )
        )
        if (builder["memory"], builder["cpu_quota"], builder["cpu_period"]) != (6 * 2**30, 200000, 100000):
            raise ValueError("VM BuildKit resource limits did not match the reviewed bounds")
        state["toolchain"]["buildkit"] = builder
        commands.run(*cpu.build_commands(ROOT, state, plan)[1], cpu.BUILD_TIMEOUT)
        state["base_oci_manifest"] = cpu.unpack_base(scratch)
        state["model_build_attempted"] = True
        cpu.write_json(output / "run.json", state)
        commands.run(*cpu.build_commands(ROOT, state, plan)[2], cpu.BUILD_TIMEOUT)
        inspected = json.loads(commands.run("image-inspect", ["docker", "image", "inspect", state["image_tag"]]))[0]
        if inspected["Architecture"] != "amd64" or inspected["Config"]["Labels"].get(cpu.OWNER_LABEL) != state["nonce"]:
            raise ValueError("Canonical VM model image architecture/ownership mismatch")
        state["docker_image_id"], state["docker_rootfs"] = inspected["Id"], inspected["RootFS"]
        namespace = f"docker.io/boileroom-cpu-validation-{state['nonce']}"
        tag = f"sha-{plan['git_sha']}"
        state["image_uri"] = f"docker://{namespace}/boileroom-minifold:{tag}"
        env = os.environ.copy() | {
            "MODEL_DIR": str(scratch / "models"),
            "BOILEROOM_DOCKER_REPOSITORY": namespace,
            "BOILEROOM_IMAGE_TAG": tag,
            "APPTAINER_CACHEDIR": str(scratch / "apptainer-cache"),
            "APPTAINER_TMPDIR": str(scratch / "apptainer-tmp"),
            "BOILEROOM_APPTAINER_STARTUP_TIMEOUT": str(cpu.PUBLIC_TIMEOUT - 60),
        }
        from boileroom.backend.apptainer import _get_cached_sif_path

        sif = _get_cached_sif_path(state["image_uri"], Path(env["MODEL_DIR"]))
        sif.parent.mkdir(parents=True)
        Path(env["APPTAINER_TMPDIR"]).mkdir()
        commands.run(
            "sif-build",
            [
                "apptainer",
                "build",
                "--mksquashfs-args",
                "-processors 2 -mem 512M",
                str(sif),
                f"docker-daemon:{state['image_tag']}",
            ],
            cpu.SIF_TIMEOUT,
            env=env,
        )
        state["sif_path"], state["sif_sha256"] = str(sif), cpu.sha256(sif)
        state["image_reference_kind"] = (
            "Task-local cache alias; immutable Docker image ID and local SIF SHA256 are recorded separately from registry digests."
        )
        state["validation_status"] = "running"
        cpu.write_json(output / "run.json", state)
        give_own_artifacts_to_client(scratch, output)
        commands.run(
            "sif-smoke",
            [
                "runuser",
                "-u",
                "validation",
                "--",
                "apptainer",
                "exec",
                "--cleanenv",
                "--home",
                "/tmp",
                str(sif),
                "/usr/local/bin/python",
                "-c",
                "import torch; assert not torch.cuda.is_available(); print(torch.__version__)",
            ],
            120,
            env=env,
        )
        commands.run(
            "public-wrapper",
            [
                "runuser",
                "-u",
                "validation",
                "--",
                sys.executable,
                str(ROOT / "scripts/testing/run_cpu_vm_validation.py"),
                "--remote-child",
                "--execute",
                "--expected-sha",
                expected_sha,
                "--output-dir",
                str(output),
            ],
            cpu.PUBLIC_TIMEOUT,
            env=env,
        )
        report = json.loads((output / "public-validation.json").read_text())
        state["passed"] = report.get("passed") is True
        state["validation_status"] = "passed" if state["passed"] else "failed"
    except BaseException as error:
        state["error"] = f"{type(error).__name__}: {error}"
        if state["validation_status"] == "running":
            state["validation_status"] = "failed"
    finally:
        signal.signal(signal.SIGTERM, previous)
        if commands.active is not None:
            cpu.stop_process_group(commands.active)
        for path in sorted((scratch / "models/logs").glob("*.log")):
            if path.is_file() and not path.is_symlink():
                with path.open("rb") as handle:
                    handle.seek(max(0, path.stat().st_size - 2**20))
                    (output / f"server-{path.name}").write_bytes(handle.read(2**20))
        state["cleanup"] = cpu.cleanup(state, output)
        state["passed"] = state["passed"] and state["cleanup"]["passed"]
        state["finished_at"] = cpu.utc_now()
        cpu.write_json(output / "run.json", state)
    return state


def remote_evidence() -> dict[str, str]:
    files = {}
    for path in sorted(REMOTE_OUTPUT.iterdir()):
        if path.is_file() and path.suffix in {".json", ".log"} and not path.is_symlink():
            if path.stat().st_size > 4 * 2**20:
                raise ValueError("Remote evidence exceeded the bounded text-file limit")
            files[path.name] = path.read_text(errors="replace")
    if sum(len(value.encode()) for value in files.values()) > 32 * 2**20:
        raise ValueError("Remote evidence exceeded its bounded total limit")
    return files


async def execute(modal: Any, root: Path, plan: dict[str, Any], output: Path) -> bool:
    cpu = cpu_helpers()
    if modal.__version__ != MODAL_VERSION:
        raise ValueError(f"This test-only controller requires modal=={MODAL_VERSION}")
    nonce = uuid.uuid4().hex
    plan["controller_nonce"] = nonce
    app_name = f"boileroom-cpu-vm-{nonce[:12]}"
    app = modal.App(app_name)
    state: dict[str, Any] = {
        "started_at": cpu.utc_now(),
        "passed": False,
        "sandboxes": [],
        "assigned_budget_usd": 10,
        "monetary_cap_enforced": False,
        "controller_nonce": nonce,
        "app_name": app_name,
        "model_validation_status": "unrun",
    }
    current = None
    deadline = time.monotonic() + TOTAL_TIMEOUT
    try:
        with tempfile.TemporaryDirectory(prefix="boileroom-vm-source-") as temporary:
            pack, metadata = Path(temporary) / "source.pack", Path(temporary) / "plan.json"
            plan["source_pack_sha256"] = source_pack(root, pack, plan["git_sha"])
            cpu.write_json(metadata, plan)
            cpu.write_json(output / "plan.json", plan)
            image = build_image(modal, pack, metadata, plan)
            async with app.run(detach=False):
                state["app_id"] = app.app_id
                cpu.write_json(output / "vm-run.json", state)
                await asyncio.wait_for(image.build.aio(app=app), IMAGE_TIMEOUT)
                state["vm_image_id"] = image.object_id
                phases = ["probe", "full"] if plan["run_model_after_probe"] else ["probe"]
                for phase in phases:
                    timeout = min(
                        PROBE_TIMEOUT if phase == "probe" else FULL_TIMEOUT, int(deadline - time.monotonic()) - 45
                    )
                    if timeout < 90:
                        raise TimeoutError("Not enough bounded time remains to create the next CPU VM")
                    current = await modal.Sandbox.create.aio(
                        "dockerd",
                        app=app,
                        image=image,
                        runtime="vm",
                        cpu=(1, 1) if phase == "probe" else (4, 4),
                        memory=(2048, 2048) if phase == "probe" else (32768, 32768),
                        timeout=timeout,
                        idle_timeout=120,
                        secrets=[],
                        volumes={},
                        encrypted_ports=[],
                        unencrypted_ports=[],
                        h2_ports=[],
                        readiness_probe=modal.Probe.with_exec("docker", "info", interval_ms=500),
                        tags={"boileroom_validation_nonce": nonce, "boileroom_validation_phase": phase},
                    )
                    entry = {
                        "phase": phase,
                        "sandbox_id": current.object_id,
                        "created_at": cpu.utc_now(),
                        "timeout_seconds": timeout,
                        "terminated": False,
                    }
                    state["sandboxes"].append(entry)
                    cpu.write_json(output / "vm-run.json", state)
                    await asyncio.wait_for(current.wait_until_ready.aio(), min(60, timeout - 60))
                    env = {
                        "BOILEROOM_VALIDATION_VM_ID": current.object_id,
                        "BOILEROOM_VALIDATION_REMOTE_TIMEOUT": str(timeout),
                    }
                    process = await current.exec.aio(
                        "/usr/local/bin/python",
                        "/validation/scripts/testing/run_cpu_vm_validation.py",
                        "--remote-probe" if phase == "probe" else "--remote-full",
                        "--execute",
                        "--expected-sha",
                        plan["git_sha"],
                        "--output-dir",
                        str(REMOTE_OUTPUT),
                        timeout=timeout - 30,
                        env=env,
                        workdir="/validation",
                    )
                    stdout, stderr = await asyncio.gather(process.stdout.read.aio(), process.stderr.read.aio())
                    await process.wait.aio()
                    (output / f"{phase}-stdout.log").write_text(stdout)
                    (output / f"{phase}-stderr.log").write_text(stderr)
                    result = json.loads(stdout)
                    for name, text in result["evidence"].items():
                        if Path(name).name != name or not name.endswith((".json", ".log")):
                            raise ValueError("Invalid remote evidence filename")
                        (output / f"{phase}-{name}").write_text(text)
                    entry["passed"] = result["report"].get("passed") is True and process.returncode == 0
                    if phase == "full":
                        state["model_validation_status"] = "passed" if entry["passed"] else "failed"
                    entry["process_exit_code"] = process.returncode
                    code = await current.terminate.aio(wait=True)
                    entry["termination_exit_code"] = code
                    entry["terminated"] = await current.poll.aio() is not None
                    if entry["terminated"]:
                        current = None
                    cpu.write_json(output / "vm-run.json", state)
                    if not entry["passed"] or not entry["terminated"]:
                        raise RuntimeError(f"CPU VM {phase} failed; inspect retained reports")
                state["passed"] = all(entry["passed"] and entry["terminated"] for entry in state["sandboxes"])
    except BaseException as error:
        state["error"] = f"{type(error).__name__}: {error}"
    finally:
        if current is not None:
            try:
                code = await asyncio.wait_for(current.terminate.aio(wait=True), 30)
                entry = state["sandboxes"][-1]
                entry["termination_exit_code"] = code
                entry["terminated"] = await asyncio.wait_for(current.poll.aio(), 5) is not None
                # The daemon is now stopped, so its streams can finish. These
                # task-local logs explain a readiness/kernel capability failure.
                for stream_name in ("stdout", "stderr"):
                    stream = getattr(current, stream_name, None)
                    if stream is not None:
                        with contextlib.suppress(Exception):
                            value = await asyncio.wait_for(stream.read.aio(), 5)
                            (output / f"failed-vm-{stream_name}.log").write_text(value[-(2**20) :])
            except BaseException as error:
                state["cleanup_error"] = f"{type(error).__name__}: {error}"
        state["finished_at"] = cpu.utc_now()
        cpu.write_json(output / "vm-run.json", state)
    return state["passed"]


async def modal_cli(arguments: list[str], timeout: float = 25) -> tuple[int, bytes]:
    """Bound and reap only the local CLI process group created by this controller."""
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "modal",
        *arguments,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout)
        status = process.returncode
        if status is None:
            raise RuntimeError("Owned Modal CLI process was not reaped")
        return status, stdout
    finally:
        if process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(process.wait(), 5)
            if process.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                await asyncio.wait_for(process.wait(), 5)


async def cleanup_only(modal: Any, output: Path) -> bool:
    """Recover and terminate only this controller's nonce-tagged temporary VMs."""
    cpu = cpu_helpers()
    if modal.__version__ != MODAL_VERSION:
        raise ValueError(f"Cleanup requires the test-only modal=={MODAL_VERSION} controller")
    state = json.loads((output / "vm-run.json").read_text())
    plan = json.loads((output / "plan.json").read_text())
    nonce = state.get("controller_nonce", "")
    if (
        not cpu.NONCE.fullmatch(nonce)
        or nonce != plan.get("controller_nonce")
        or state.get("app_name") != f"boileroom-cpu-vm-{nonce[:12]}"
    ):
        raise ValueError("Refusing cleanup of a controller state without matching task ownership")
    app_id = state.get("app_id")
    if app_id is not None and not APP_ID.fullmatch(app_id):
        raise ValueError("Refusing invalid task App ID")
    result: dict[str, Any] = {"passed": False, "sandbox_ids": [], "errors": [], "app_stop_attempted": False}
    entries = {entry["sandbox_id"]: entry for entry in state["sandboxes"]}
    if any(not SDK_ID.fullmatch(identifier) for identifier in entries):
        raise ValueError("Refusing invalid task Sandbox ID")
    ownership_error = False
    try:
        recovered = []
        if app_id:
            try:

                async def recover() -> None:
                    async for sandbox in modal.Sandbox.list.aio(
                        app_id=app_id, tags={"boileroom_validation_nonce": nonce}
                    ):
                        if not SDK_ID.fullmatch(sandbox.object_id):
                            raise ValueError("Unexpected recovered Sandbox ID syntax")
                        recovered.append(sandbox.object_id)
                        entries.setdefault(sandbox.object_id, {"sandbox_id": sandbox.object_id, "terminated": False})
                        if len(entries) > 4:
                            raise ValueError("Unexpected number of task-owned VMs; use exact App fallback")

                await asyncio.wait_for(recover(), 10)
            except Exception as error:
                result["errors"].append(f"Sandbox recovery: {type(error).__name__}: {error}")
        result["recovered_sandbox_ids"] = recovered
        for identifier, entry in entries.items():
            try:
                if entry.get("terminated") is True:
                    result["sandbox_ids"].append(identifier)
                    continue

                async def stop_one(identifier: str, entry: dict[str, Any]) -> None:
                    nonlocal ownership_error
                    sandbox = await modal.Sandbox.from_id.aio(identifier)
                    tags = await sandbox.get_tags.aio()
                    if tags.get("boileroom_validation_nonce") != nonce:
                        ownership_error = True
                        raise ValueError("Refusing to terminate a Sandbox without this run's ownership tag")
                    entry["termination_exit_code"] = await sandbox.terminate.aio(wait=True)
                    entry["terminated"] = await sandbox.poll.aio() is not None
                    result["sandbox_ids"].append(identifier)
                    if not entry["terminated"]:
                        raise RuntimeError("Task Sandbox termination was not confirmed")

                await asyncio.wait_for(stop_one(identifier, entry), 10)
            except Exception as error:
                result["errors"].append(f"{identifier}: {type(error).__name__}: {error}")
        state["sandboxes"] = list(entries.values())
        if app_id and not ownership_error:
            try:
                status, raw = await modal_cli(["app", "list", "--json"], 10)
                rows = [row for row in json.loads(raw) if row.get("App ID") == app_id]
                if status or len(rows) != 1 or rows[0].get("Description") != state["app_name"]:
                    raise ValueError("Recorded App ownership could not be verified before stop")
                result["app_stop_attempted"] = True
                status, stdout = await modal_cli(["app", "stop", "--yes", app_id], 15)
                (output / "cleanup-app.log").write_bytes(stdout)
                if status and b"already stopped" not in stdout.lower():
                    raise RuntimeError(f"Task App stop exited {status}")
                verification_deadline = time.monotonic() + 25
                while True:
                    remaining = verification_deadline - time.monotonic()
                    if remaining <= 0:
                        raise RuntimeError("Task App stopped/Tasks0 was not observed")
                    status, raw = await modal_cli(["app", "list", "--json"], min(10, remaining))
                    rows = [row for row in json.loads(raw) if row.get("App ID") == app_id]
                    result["app_verification"] = rows  # Never persist unrelated workspace App rows.
                    if (
                        not status
                        and len(rows) == 1
                        and rows[0].get("State") == "stopped"
                        and str(rows[0].get("Tasks")) == "0"
                    ):
                        result["app_stopped_tasks_zero"] = True
                        break
                    if time.monotonic() >= verification_deadline:
                        raise RuntimeError("Task App stopped/Tasks0 was not observed")
                    await asyncio.sleep(1)
            except Exception as error:
                result["errors"].append(f"App cleanup: {type(error).__name__}: {error}")
        result["passed"] = (
            not result["errors"]
            and all(entry.get("terminated") is True for entry in entries.values())
            and (not app_id or result.get("app_stopped_tasks_zero") is True)
        )
    except BaseException as error:
        result["errors"].append(f"{type(error).__name__}: {error}")
    finally:
        state["cleanup"] = result
        cpu.write_json(output / "vm-cleanup.json", result)
        cpu.write_json(output / "vm-run.json", state)
    return result["passed"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-sha")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--run-model-after-probe", action="store_true")
    parser.add_argument("--cleanup-only", action="store_true")
    for flag in ("--remote-probe", "--remote-full", "--remote-child"):
        parser.add_argument(flag, action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.cleanup_only:
        import modal

        return 0 if asyncio.run(asyncio.wait_for(cleanup_only(modal, output), 120)) else 1
    remote_modes = [args.remote_probe, args.remote_full, args.remote_child]
    if any(remote_modes):
        if not args.execute or sum(remote_modes) != 1 or args.expected_sha is None or output != REMOTE_OUTPUT:
            parser.error(
                "Remote execution requires --execute, exactly one internal mode, exact SHA and task-owned reports path"
            )
        report = (probe_remote if args.remote_probe else full_remote if args.remote_full else public_child)(
            args.expected_sha, output
        )
        if args.remote_child:
            return 0 if report["passed"] else 1
        print(json.dumps({"report": report, "evidence": remote_evidence()}, allow_nan=False))
        return 0 if report["passed"] else 1
    if (output / "vm-run.json").exists():
        parser.error("Use a fresh output directory; existing VM evidence must not be overwritten")
    plan = make_plan(ROOT, args.expected_sha, args.run_model_after_probe)
    cpu_helpers().write_json(output / "plan.json", plan)
    if not args.execute:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    if args.expected_sha is None or plan["dirty"] or not plan["entrypoint_exists"]:
        parser.error("Paid execution requires a clean exact source SHA and public helper")
    import modal

    async def bounded() -> bool:
        task = asyncio.current_task()
        if task is None:
            raise RuntimeError("CPU VM controller needs an active async task")
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGTERM, task.cancel)
        try:
            return await asyncio.wait_for(execute(modal, ROOT, plan, output), TOTAL_TIMEOUT)
        finally:
            loop.remove_signal_handler(signal.SIGTERM)

    return 0 if asyncio.run(bounded()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
