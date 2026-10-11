"""Manual, CPU-only canonical Docker -> local SIF -> public-wrapper validation.

The default is an offline plan. Execution is restricted to a standard public
GitHub Ubuntu 22.04 runner and requires a clean, exact Git SHA. The workflow
installs the official non-setuid Apptainer package after verifying the hash below;
this script verifies it again. No registry image, credentials, GPU or host
security configuration is used. Insufficient resources are an explicit unrun
failure, never a model substitution or a claim of scientific parity.

Official toolchain references:
https://github.com/apptainer/apptainer/releases/tag/v1.5.4
https://github.com/moby/buildkit/releases/tag/v0.33.0
https://docs.docker.com/build/builders/drivers/docker-container/
https://docs.docker.com/reference/cli/docker/buildx/build/#build-context
https://apptainer.org/docs/user/1.5/build_a_container.html
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
APPTAINER_VERSION = "1.5.4"
APPTAINER_PACKAGE_URL = "https://github.com/apptainer/apptainer/releases/download/v1.5.4/apptainer_1.5.4_amd64.deb"
APPTAINER_PACKAGE_SHA256 = "281cf9bd68f16b6f7c34a3a94d77668bc327b9c09db0412ce7890162734a97f7"
BUILDKIT_IMAGE = "moby/buildkit:v0.33.0"
TOTAL_TIMEOUT = 53 * 60  # Cleanup and artifact upload still fit the 60-minute workflow.
BUILD_TIMEOUT = 25 * 60
SIF_TIMEOUT = 10 * 60
PUBLIC_TIMEOUT = 15 * 60
OWNER_LABEL = "org.boileroom.cpu-sif-validation"
LIMITS = {
    "profam": {"available_memory_gib": 8, "free_disk_gib": 28},
    # FP32 ESM-2 3B resident weights plus both source checkpoints peak at 20-24 GiB.
    "minifold": {"available_memory_gib": 24, "free_disk_gib": 40},
}
NONCE = re.compile(r"[0-9a-f]{32}\Z")


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def make_plan(root: Path, model: str, model_size: str, expected_sha: str | None) -> dict[str, Any]:
    head = git(root, "rev-parse", "HEAD")
    if expected_sha is not None and (not re.fullmatch(r"[0-9a-f]{40}", expected_sha) or expected_sha != head):
        raise ValueError("--expected-sha must match this checkout's full HEAD SHA")
    names = git(root, "ls-files", "-z", "boileroom", "tests", "scripts", "pyproject.toml").split("\0")
    hashes = {name: sha256(root / name) for name in sorted(names) if name and (root / name).is_file()}
    return {
        "model": model,
        "model_size": model_size if model == "minifold" else None,
        "mode": "cpu-sif",
        "device": "cpu",
        "git_sha": head,
        "dirty": bool(git(root, "status", "--porcelain", "--untracked-files=normal")),
        "source_files_sha256": hashes,
        "source_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
        "entrypoint": f"tests/{model}/{model}_public_validation.py",
        "entrypoint_exists": (root / f"tests/{model}/{model}_public_validation.py").is_file(),
        "base_dockerfile": "boileroom/images/Dockerfile",
        "model_dockerfile": f"boileroom/models/{model}/Dockerfile",
        "apptainer_package_url": APPTAINER_PACKAGE_URL,
        "apptainer_package_sha256": APPTAINER_PACKAGE_SHA256,
        "buildkit_image_version_pin": BUILDKIT_IMAGE,
        "limits": LIMITS[model] | {"cpu": 2, "build_memory_gib": 6, "launcher_timeout_seconds": TOTAL_TIMEOUT},
        "scope": "CPU public wrapper, HTTP/HMAC transport and process lifecycle. GPU, compile, independent upstream parity and biological accuracy remain unrun.",
        "hermetic_build": False,
        "hermetic_build_note": "Canonical Dockerfiles retain their existing external base/frontend/package resolution. Resolved local image and SIF identities are recorded; this is not a reproducible-build claim.",
        "image_publication": False,
        "paid_compute": False,
    }


def resource_snapshot(paths: list[Path]) -> dict[str, Any]:
    fields = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    memory = int(fields["MemAvailable"].split()[0]) * 1024
    # Docker storage and scratch can be different filesystems; require space in each.
    return {
        "available_memory_bytes": memory,
        "free_disk_bytes": {str(path): shutil.disk_usage(path).free for path in paths},
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
    }


def resource_blockers(plan: dict[str, Any], resources: dict[str, Any]) -> list[str]:
    limits = plan["limits"]
    blockers = []
    if resources["available_memory_bytes"] < limits["available_memory_gib"] * 2**30:
        blockers.append(f"Needs {limits['available_memory_gib']} GiB available RAM for the unchanged model")
    if any(free < limits["free_disk_gib"] * 2**30 for free in resources["free_disk_bytes"].values()):
        blockers.append(f"Needs {limits['free_disk_gib']} GiB free disk for build cache, Docker image, SIF and weights")
    if len(resources["cpu_affinity"]) < 2:
        blockers.append("Needs at least two available CPU cores")
    return blockers


def verify_host(package: Path) -> None:
    if os.environ.get("GITHUB_ACTIONS") != "true" or os.environ.get("RUNNER_ENVIRONMENT") != "github-hosted":
        raise ValueError("Execution requires a GitHub-hosted runner")
    if os.environ.get("GITHUB_REPOSITORY_VISIBILITY") != "public":
        raise ValueError("Execution requires the workflow's verified public repository visibility")
    release = dict(line.split("=", 1) for line in Path("/etc/os-release").read_text().splitlines() if "=" in line)
    if release.get("ID", "").strip('"') != "ubuntu" or release.get("VERSION_ID", "").strip('"') != "22.04":
        raise ValueError("Execution requires Ubuntu 22.04")
    if platform.machine() != "x86_64" or os.geteuid() == 0:
        raise ValueError("Execution requires a non-root amd64 host client")
    if sha256(package) != APPTAINER_PACKAGE_SHA256:
        raise ValueError("Official non-setuid Apptainer package SHA256 mismatch")


def stop_process_group(process: subprocess.Popen[Any]) -> None:
    """Reap only a process group we created with start_new_session=True."""
    # A failed client may have left its server child alive in the same group,
    # even if the leader has already exited.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=5)
    # Reap grandchildren even if the group leader exited immediately on SIGTERM.
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    process.wait(timeout=5)


class Commands:
    def __init__(self, output: Path, deadline: float) -> None:
        self.output, self.deadline = output, deadline
        self.active: subprocess.Popen[Any] | None = None

    def run(self, name: str, command: list[str], timeout: int = 120, *, env: dict[str, str] | None = None) -> str:
        remaining = min(timeout, self.deadline - time.monotonic())
        if remaining <= 0:
            raise TimeoutError("CPU-SIF launcher wall deadline reached")
        log = self.output / f"{name}.log"
        with log.open("w") as handle:
            self.active = subprocess.Popen(
                command, stdout=handle, stderr=subprocess.STDOUT, env=env, start_new_session=True
            )
            try:
                status = self.active.wait(timeout=remaining)
                if status:
                    raise RuntimeError(f"{name} exited {status}; see {log.name}")
            finally:
                stop_process_group(self.active)
                self.active = None
        return log.read_text()


def build_commands(root: Path, state: dict[str, Any], plan: dict[str, Any]) -> list[tuple[str, list[str]]]:
    scratch = Path(state["scratch"])
    common = [
        "docker",
        "buildx",
        "build",
        "--builder",
        state["builder"],
        "--platform",
        "linux/amd64",
        "--provenance=false",
        "--progress=plain",
    ]
    # A container-driver builder cannot see daemon-local FROM images. Export the
    # canonical base to an OCI layout and feed it as a named context, without a registry.
    return [
        (
            "builder-create",
            [
                "docker",
                "buildx",
                "create",
                "--name",
                state["builder"],
                "--driver",
                "docker-container",
                "--driver-opt",
                f"image={BUILDKIT_IMAGE},memory=6g,cpu-quota=200000,cpu-period=100000",
            ],
        ),
        (
            "base-build",
            common
            + [
                "--output",
                f"type=oci,dest={scratch / 'base.oci.tar'}",
                "--file",
                str(root / plan["base_dockerfile"]),
                str(root / "boileroom/images"),
            ],
        ),
        (
            "model-build",
            common
            + [
                "--load",
                "--tag",
                state["image_tag"],
                "--label",
                f"{OWNER_LABEL}={state['nonce']}",
                "--build-arg",
                "BASE_IMAGE=validation-base",
                "--build-arg",
                f"UV_CACHE_ID=cpu-sif-{state['nonce']}",
                "--build-context",
                f"validation-base=oci-layout://{scratch / 'base.oci'}@{state.get('base_oci_manifest', 'sha256:UNSET')}",
                "--file",
                str(root / plan["model_dockerfile"]),
                str(root / f"boileroom/models/{plan['model']}"),
            ],
        ),
    ]


def unpack_base(scratch: Path) -> str:
    layout = scratch / "base.oci"
    layout.mkdir()
    with tarfile.open(scratch / "base.oci.tar") as archive:
        archive.extractall(layout, filter="data")
    manifests = json.loads((layout / "index.json").read_text())["manifests"]
    if len(manifests) != 1 or not re.fullmatch(r"sha256:[0-9a-f]{64}", manifests[0]["digest"]):
        raise ValueError("Expected one canonical base image in the OCI layout")
    digest = manifests[0]["digest"]
    if sha256(layout / "blobs/sha256" / digest.split(":")[1]) != digest.split(":")[1]:
        raise ValueError("Base OCI manifest digest mismatch")
    return digest


def cleanup(state: dict[str, Any], output: Path) -> dict[str, Any]:
    """Delete only nonce-bound artifacts, never daemon-wide caches or host data."""
    nonce = state.get("nonce", "")
    scratch = Path(state.get("scratch", "/"))
    if not NONCE.fullmatch(nonce) or state.get("builder") != f"boileroom-cpu-sif-{nonce}":
        raise ValueError("Invalid cleanup ownership state")
    if scratch.name != f"boileroom-cpu-sif-{nonce}" or scratch.is_symlink():
        raise ValueError("Invalid cleanup scratch path")
    if scratch.parent != Path(os.environ["RUNNER_TEMP"]).resolve():
        raise ValueError("Invalid cleanup scratch parent")
    if not scratch.exists() and state.get("cleanup", {}).get("passed") is True:
        return state["cleanup"]
    result: dict[str, Any] = {"errors": [], "scratch_removed": False}
    commands = Commands(output, time.monotonic() + 60)
    tag = state.get("image_tag", "")
    if tag != f"boileroom-cpu-sif-{nonce}:local":
        raise ValueError("Invalid cleanup image tag")
    if state.get("builder_attempted"):
        try:
            commands.run("cleanup-builder", ["docker", "buildx", "rm", "--force", state["builder"]], 25)
        except Exception as error:
            result["errors"].append(str(error))
    if state.get("model_build_attempted"):
        try:
            # A failed build can legitimately leave no model image. Query only
            # this exact task-owned tag before inspecting/removing it.
            images = commands.run(
                "cleanup-image-list", ["docker", "image", "ls", "--quiet", "--filter", f"reference={tag}"]
            ).strip()
            inspected = (
                json.loads(commands.run("cleanup-image-inspect", ["docker", "image", "inspect", tag]))[0]
                if images
                else None
            )
            if inspected is None:
                result["model_image_absent"] = True
            elif inspected.get("Config", {}).get("Labels", {}).get(OWNER_LABEL) != nonce:
                raise ValueError("Refusing to delete an image without this run's ownership label")
            else:
                commands.run("cleanup-image", ["docker", "image", "rm", tag], 20)
        except Exception as error:
            result["errors"].append(str(error))
    if scratch.exists():
        if (scratch / ".owner").read_text().strip() != nonce:
            raise ValueError("Refusing to delete scratch without this run's ownership marker")
        shutil.rmtree(scratch)
    result["scratch_removed"] = not scratch.exists()
    result["passed"] = not result["errors"] and result["scratch_removed"]
    write_json(output / "cleanup.json", result)
    return result


def run_public_child(root: Path, plan: dict[str, Any], state: dict[str, Any], output: Path) -> dict[str, Any]:
    """Use the real wrapper/backend, changing only normal image/cache environment."""
    sys.path[:0] = [str(root), str(root / "tests")]
    cpus = sorted(os.sched_getaffinity(0))[:2]
    os.sched_setaffinity(0, cpus)
    import boileroom
    from boileroom.backend.apptainer import _get_cached_sif_path

    cached = _get_cached_sif_path(state["image_uri"], Path(os.environ["MODEL_DIR"]))
    if cached.resolve() != Path(state["sif_path"]).resolve() or sha256(cached) != state["sif_sha256"]:
        raise ValueError("Public wrapper cache does not match the built local SIF")
    path = root / plan["entrypoint"]
    spec = importlib.util.spec_from_file_location("cpu_sif_public_validation", path)
    if spec is None or spec.loader is None:
        raise ValueError("Public validation helper is unavailable")
    helper = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = helper
    spec.loader.exec_module(helper)
    wrapper = getattr(boileroom, "MiniFold" if plan["model"] == "minifold" else "ProFam")
    observed: list[tuple[Any, Any, Any]] = []

    def factory(**kwargs: Any) -> Any:
        model = wrapper(**kwargs)
        worker = model._backend
        observed.append((worker, worker._process, worker._client))
        if worker._sif_path.resolve() != cached.resolve() or worker._device != "cpu":
            worker.stop()
            raise AssertionError("Public wrapper did not launch the exact local SIF on CPU")
        return model

    try:
        kwargs: dict[str, Any] = {"backend": "apptainer", "device": "cpu"}
        if plan["model"] == "minifold":
            kwargs["model_size"] = plan["model_size"]
        report = helper.run_public_validation(factory, **kwargs)
    finally:
        for worker, _, _ in observed:
            worker.stop()
    report["processes"] = [
        {
            "pid": process.pid if process else None,
            "exit_code": process.poll() if process else None,
            "client_closed": client.is_closed if client else False,
            "backend_process_cleared": worker._process is None,
            "backend_client_cleared": worker._client is None,
        }
        for worker, process, client in observed
    ]
    report["actual_processes_stopped"] = bool(observed) and all(
        row["pid"] is not None
        and row["exit_code"] is not None
        and row["client_closed"]
        and row["backend_process_cleared"]
        and row["backend_client_cleared"]
        for row in report["processes"]
    )
    report["sif_unchanged"] = sha256(cached) == state["sif_sha256"]
    report["passed"] = bool(report.get("passed")) and report["actual_processes_stopped"] and report["sif_unchanged"]
    write_json(output / "public-validation.json", report)
    return report


def execute(root: Path, plan: dict[str, Any], output: Path, package: Path) -> bool:
    verify_host(package)
    nonce = uuid.uuid4().hex
    parent = Path(os.environ["RUNNER_TEMP"]).resolve()
    scratch = parent / f"boileroom-cpu-sif-{nonce}"
    scratch.mkdir()
    (scratch / ".owner").write_text(nonce)
    state: dict[str, Any] = {
        "nonce": nonce,
        "scratch": str(scratch),
        "builder": f"boileroom-cpu-sif-{nonce}",
        "image_tag": f"boileroom-cpu-sif-{nonce}:local",
        "started_at": utc_now(),
        "passed": False,
        "validation_status": "unrun",
    }
    commands = Commands(output, time.monotonic() + TOTAL_TIMEOUT)
    previous = signal.getsignal(signal.SIGTERM)

    def interrupted(_signal: int, _frame: Any) -> None:
        raise KeyboardInterrupt("CPU-SIF job interrupted")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        write_json(output / "run.json", state)
        version = commands.run("apptainer-version", ["apptainer", "--version"]).strip()
        if version != f"apptainer version {APPTAINER_VERSION}":
            raise ValueError("Installed Apptainer version does not match the verified package")
        buildcfg = commands.run("apptainer-buildcfg", ["apptainer", "buildcfg"])
        if not re.search(r"^APPTAINER_SUID_INSTALL\s*=\s*0\s*$", buildcfg, re.MULTILINE):
            raise ValueError("Execution requires a non-setuid Apptainer installation")
        state["toolchain"] = {
            "apptainer": version,
            "docker": commands.run("docker-version", ["docker", "version"]),
            "buildx": commands.run("buildx-version", ["docker", "buildx", "version"]),
            "package_sha256": sha256(package),
        }
        info = json.loads(commands.run("docker-info", ["docker", "info", "--format", "{{json .}}"]))
        docker_root = Path(info["DockerRootDir"])
        state["resources"] = resource_snapshot([parent, docker_root])
        blockers = resource_blockers(plan, state["resources"])
        if blockers:
            state["resource_blockers"] = blockers
            raise RuntimeError("Resource preflight blocked unchanged model: " + "; ".join(blockers))
        state["non_setuid_capabilities"] = {"fuse_character_device": Path("/dev/fuse").is_char_device()}
        if not state["non_setuid_capabilities"]["fuse_character_device"]:
            raise ValueError("Non-setuid SIF execution requires the runner's existing /dev/fuse device")
        commands.run("user-namespace-probe", ["unshare", "--user", "--map-root-user", "--mount", "true"], 30)
        state["non_setuid_capabilities"]["user_mount_namespace"] = True
        builds = build_commands(root, state, plan)
        builders = commands.run("builder-name-preflight", ["docker", "buildx", "ls", "--format", "{{.Name}}"])
        if state["builder"] in builders.splitlines():
            raise ValueError("Refusing to replace an existing Docker builder")
        state["builder_attempted"] = True
        write_json(output / "run.json", state)
        commands.run(*builds[0])
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
            raise ValueError("BuildKit did not apply this run's memory/CPU resource bounds")
        state["toolchain"]["buildkit"] = builder | {"version_pin": BUILDKIT_IMAGE}
        commands.run(*builds[1], BUILD_TIMEOUT)
        state["base_oci_manifest"] = unpack_base(scratch)
        existing = commands.run(
            "model-tag-preflight",
            ["docker", "image", "ls", "--quiet", "--filter", f"reference={state['image_tag']}"],
        )
        if existing.strip():
            raise ValueError("Refusing to overwrite an existing Docker image tag")
        state["model_build_attempted"] = True
        write_json(output / "run.json", state)
        commands.run(*build_commands(root, state, plan)[2], BUILD_TIMEOUT)
        inspected = json.loads(commands.run("image-inspect", ["docker", "image", "inspect", state["image_tag"]]))[0]
        if inspected["Architecture"] != "amd64" or inspected["Config"]["Labels"].get(OWNER_LABEL) != nonce:
            raise ValueError("Canonical model image architecture/ownership mismatch")
        state["docker_image_id"] = inspected["Id"]
        state["docker_rootfs"] = inspected["RootFS"]
        namespace = f"boileroom-cpu-validation-{nonce}"
        image_tag = f"sha-{plan['git_sha']}"
        state["image_uri"] = f"docker://docker.io/{namespace}/boileroom-{plan['model']}:{image_tag}"
        env = os.environ.copy()
        env.update(
            {
                "MODEL_DIR": str(scratch / "models"),
                "BOILEROOM_DOCKER_REPOSITORY": f"docker.io/{namespace}",
                "BOILEROOM_IMAGE_TAG": image_tag,
                "APPTAINER_CACHEDIR": str(scratch / "apptainer-cache"),
                "APPTAINER_TMPDIR": str(scratch / "apptainer-tmp"),
                "BOILEROOM_APPTAINER_STARTUP_TIMEOUT": str(PUBLIC_TIMEOUT - 60),
                "PYTHONUNBUFFERED": "1",
            }
        )
        for name in (
            "MODAL_TOKEN_ID",
            "MODAL_TOKEN_SECRET",
            "HF_TOKEN",
            "HUGGING_FACE_HUB_TOKEN",
            "BOILEROOM_KIT_IMAGE_TAG",
            "BOILEROOM_IMAGE_REF",
        ):
            env.pop(name, None)
        sys.path.insert(0, str(root))
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
            SIF_TIMEOUT,
            env=env,
        )
        state["sif_path"], state["sif_sha256"] = str(sif), sha256(sif)
        state["image_reference_kind"] = (
            "Task-local cache alias, not a published registry manifest. Docker image ID and SIF SHA256 above identify the exact build."
        )
        write_json(output / "run.json", state)
        # No user namespace/AppArmor/firewall changes if this fails on the runner.
        commands.run(
            "sif-smoke",
            [
                "apptainer",
                "exec",
                "--cleanenv",
                str(sif),
                "/usr/local/bin/python",
                "-c",
                "import torch; print(torch.__version__); assert not torch.cuda.is_available()",
            ],
            120,
            env=env,
        )
        state["validation_status"] = "running"
        write_json(output / "run.json", state)
        commands.run(
            "public-wrapper",
            [
                sys.executable,
                str(root / "scripts/testing/run_cpu_sif_validation.py"),
                "--model",
                plan["model"],
                "--minifold-size",
                plan["model_size"] or "12L",
                "--output-dir",
                str(output),
                "--expected-sha",
                plan["git_sha"],
                "--apptainer-package",
                str(package.resolve()),
                "--execute",
                "--child",
            ],
            PUBLIC_TIMEOUT,
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
            stop_process_group(commands.active)
        # Preserve only bounded text logs before removing this run's checkpoints/SIF.
        for path in sorted((scratch / "models/logs").glob("*.log")):
            if path.is_file() and not path.is_symlink():
                with path.open("rb") as handle:
                    handle.seek(max(0, path.stat().st_size - 2**20))
                    (output / f"server-{path.name}").write_bytes(handle.read(2**20))
        state["finished_at"] = utc_now()
        write_json(output / "run.json", state)
        state["cleanup"] = cleanup(state, output)
        state["passed"] = state["passed"] and state["cleanup"]["passed"]
        write_json(output / "run.json", state)
    return state["passed"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=LIMITS, required=True)
    parser.add_argument("--minifold-size", choices=["12L", "48L"], default="12L")
    parser.add_argument("--expected-sha")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--apptainer-package", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--cleanup-only", action="store_true")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.cleanup_only:
        return 0 if cleanup(json.loads((output / "run.json").read_text()), output)["passed"] else 1
    if args.child:
        if not args.execute or args.expected_sha is None or args.apptainer_package is None:
            parser.error("Internal child execution requires the exact source SHA and verified package")
        verify_host(args.apptainer_package)
        plan = make_plan(ROOT, args.model, args.minifold_size, args.expected_sha)
        approved = json.loads((output / "plan.json").read_text())
        state = json.loads((output / "run.json").read_text())
        scratch = Path(state["scratch"])
        if plan != approved or plan["dirty"] or state.get("validation_status") != "running":
            parser.error("Child source/provenance does not match the approved execution plan")
        nonce = state.get("nonce", "")
        if (
            not NONCE.fullmatch(nonce)
            or scratch.parent != Path(os.environ["RUNNER_TEMP"]).resolve()
            or scratch.name != f"boileroom-cpu-sif-{nonce}"
            or scratch.is_symlink()
            or (scratch / ".owner").read_text() != nonce
        ):
            parser.error("Child scratch ownership does not match the launcher")
        namespace = f"docker.io/boileroom-cpu-validation-{nonce}"
        image_tag = f"sha-{plan['git_sha']}"
        expected_uri = f"docker://{namespace}/boileroom-{plan['model']}:{image_tag}"
        if (
            os.environ.get("MODEL_DIR") != str(scratch / "models")
            or os.environ.get("BOILEROOM_DOCKER_REPOSITORY") != namespace
            or os.environ.get("BOILEROOM_IMAGE_TAG") != image_tag
            or state["image_uri"] != expected_uri
        ):
            parser.error("Child model cache/image alias does not match the launcher")
        report = run_public_child(ROOT, plan, state, output)
        return 0 if report["passed"] else 1
    plan = make_plan(ROOT, args.model, args.minifold_size, args.expected_sha)
    write_json(output / "plan.json", plan)
    if not args.execute:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    if args.expected_sha is None or plan["dirty"] or not plan["entrypoint_exists"] or args.apptainer_package is None:
        parser.error("Execution requires --expected-sha, a clean checkout, the public helper and --apptainer-package")
    if args.model == "minifold" and args.minifold_size != "12L":
        parser.error("Standard CPU runner gate is scoped to the unchanged 12L model; 48L remains unrun")
    return 0 if execute(ROOT, plan, output, args.apptainer_package) else 1


if __name__ == "__main__":
    raise SystemExit(main())
