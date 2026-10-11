"""Manually validate one model in a temporary, source-built Modal app.

Safe default: print/write a plan without contacting Modal. After approval, pass
``--execute --expected-sha <full Git SHA>`` from a clean checkout. Credentials are
accepted by the Modal SDK from MODAL_TOKEN_ID / MODAL_TOKEN_SECRET only; this
launcher never retrieves, prints, saves, or forwards them to a model container.

The canonical base Dockerfile is built first and the selected model Dockerfile's
body is appended using Image.dockerfile_commands. No registry image is published
and production image lookup is unchanged. Image.build has no public timeout
parameter, so an async wall deadline encloses the attached build app. Both apps
use run(detach=False), and cleanup explicitly stops only their recorded IDs.

API references:
https://modal.com/docs/sdk/py/latest/Image
https://modal.com/docs/sdk/py/latest/App
https://modal.com/docs/guide/timeouts
https://modal.com/docs/cli/latest/app
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
BUILD_TIMEOUT = 20 * 60
TOTAL_TIMEOUT = 53 * 60  # Leave room for cleanup and artifact upload in the 60-minute job.
MODAL_VERSION = "1.3.0.post1"
MODEL_LIMITS: dict[str, dict[str, Any]] = {
    "minifold": {"gpu": "L4", "memory_mib": 32768, "remote_timeout_seconds": 20 * 60},
    "profam": {"gpu": "T4", "memory_mib": 16384, "remote_timeout_seconds": 30 * 60},
}
APP_ID = re.compile(r"ap-[A-Za-z0-9]{22}\Z")


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def write_json(path: Path, value: Any) -> None:
    """Atomically preserve evidence, including on a failed run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def model_dockerfile_body(text: str) -> str:
    """Drop only the canonical single-stage BASE_IMAGE preamble; fail closed."""
    lines = text.splitlines(keepends=True)
    from_lines = [index for index, line in enumerate(lines) if re.match(r"\s*FROM\s", line, re.IGNORECASE)]
    if len(from_lines) != 1 or lines[from_lines[0]].strip() != "FROM ${BASE_IMAGE}":
        raise ValueError("Validation requires one canonical FROM ${BASE_IMAGE} stage")
    for line in lines[: from_lines[0]]:
        if line.strip() and not line.lstrip().startswith(("#", "ARG ")):
            raise ValueError("Unexpected instruction before model FROM")
        if line.startswith("ARG ") and line.split("=", 1)[0].strip() not in {
            "ARG BASE_IMAGE",
            "ARG TARGETPLATFORM",
            "ARG TARGETARCH",
            "ARG TARGETOS",
            "ARG BUILDPLATFORM",
        }:
            raise ValueError("Model has a new pre-FROM argument; review source-build handling")
    return "".join(lines[from_lines[0] + 1 :])


def make_plan(root: Path, model: str, model_size: str, expected_sha: str | None) -> dict[str, Any]:
    """Describe exact source and fixed resource bounds without creating an app."""
    sha = git(root, "rev-parse", "HEAD")
    if expected_sha is not None and (not re.fullmatch(r"[0-9a-f]{40}", expected_sha) or expected_sha != sha):
        raise ValueError("--expected-sha must match this checkout's full HEAD SHA")
    paths = git(root, "ls-files", "-z", "boileroom", "tests", "scripts", "pyproject.toml").split("\0")
    hashes = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in sorted(paths)
        if name and (root / name).is_file()
    }
    source_hash = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    entrypoint = f"tests/{model}/runtime_validation.py"
    return {
        "model": model,
        "model_size": model_size if model == "minifold" else None,
        "other_minifold_variant": "unrun" if model == "minifold" else "not_applicable",
        "git_sha": sha,
        "dirty": bool(git(root, "status", "--porcelain", "--untracked-files=normal")),
        "source_sha256": source_hash,
        "source_files_sha256": hashes,
        "source_reference": f"source:{sha}:{model}:{source_hash}",
        "image_kind": "Modal source composition of canonical Dockerfiles; not a Docker/Apptainer image build",
        "base_dockerfile": "boileroom/images/Dockerfile",
        "model_dockerfile": f"boileroom/models/{model}/Dockerfile",
        "entrypoint": entrypoint,
        "entrypoint_exists": (root / entrypoint).is_file(),
        "modal_sdk": MODAL_VERSION,
        "limits": {
            **MODEL_LIMITS[model],
            "cpu": 2,
            "max_containers": 1,
            "gpu_count": 1,
            "retries": 0,
            "build_gpu_count": 0,
            "build_wall_timeout_seconds": BUILD_TIMEOUT,
            "launcher_wall_timeout_seconds": TOTAL_TIMEOUT,
            "workflow_timeout_minutes": 60,
        },
        "cost_evidence": {
            "actual_billed_usd": None,
            "monetary_cap_enforced": False,
            "note": "Resource/time bounds are not a monetary cap. Reconcile app IDs and timestamps with Modal billing; build, startup, CPU, memory, egress and platform retries may be charged.",
        },
    }


def build_image(modal: Any, root: Path, plan: dict[str, Any]) -> Any:
    """Compose canonical Dockerfiles and mount only Git-tracked source files."""
    base = root / plan["base_dockerfile"]
    model_file = root / plan["model_dockerfile"]
    body = model_dockerfile_body(model_file.read_text())
    image = modal.Image.from_dockerfile(base, context_dir=base.parent, gpu=None)
    image = image.dockerfile_commands(body, context_dir=model_file.parent, gpu=None)
    image = image.env(
        {
            "PYTHONPATH": "/validation",
            "MODEL_DIR": "/tmp/validation-models",
            "BOILEROOM_IMAGE_REF": plan["source_reference"],
            "BOILEROOM_VALIDATION_GIT_SHA": plan["git_sha"],
            "PYTHONUNBUFFERED": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "DO_NOT_TRACK": "1",
        }
    )
    tracked = {str((root / name).resolve()) for name in plan["source_files_sha256"]}

    def ignore(path: Path) -> bool:
        return not path.is_dir() and str(path.resolve()) not in tracked

    for directory in ("boileroom", "tests", "scripts"):
        image = image.add_local_dir(root / directory, f"/validation/{directory}", copy=False, ignore=ignore)
    return image.add_local_file(root / "pyproject.toml", "/validation/pyproject.toml", copy=False)


def remote_validation(model: str, model_size: str | None) -> dict[str, Any]:
    """Load only the selected runner in the actual canonical model image."""
    import importlib.util
    import json
    import os
    import sys
    import time
    from pathlib import Path

    started = time.monotonic()
    root = Path("/validation")
    os.chdir(root)
    sys.path.insert(0, str(root))
    path = root / "tests" / model / "runtime_validation.py"
    spec = importlib.util.spec_from_file_location(f"{model}_runtime_validation", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    report = module.run_validation(model_size=model_size) if model == "minifold" else module.run_validation()
    if not isinstance(report, dict):
        raise TypeError("run_validation must return a JSON-serializable dict")
    report["launcher_remote_seconds"] = time.monotonic() - started
    # Reject tensors/NaNs and other unserializable results before returning.
    json.dumps(report, allow_nan=False)
    return report


def remember_app(state: dict[str, Any], app: Any, phase: str, output: Path) -> None:
    app_id = app.app_id
    if app_id and not any(record["app_id"] == app_id for record in state["apps"]):
        state["apps"].append({"app_id": app_id, "phase": phase, "recorded_at": utc_now()})
        write_json(output / "run.json", state)


async def run_apps(modal: Any, root: Path, plan: dict[str, Any], output: Path, state: dict[str, Any]) -> None:
    image = build_image(modal, root, plan)
    build_app = modal.App(f"boileroom-validate-{plan['model']}-build-{plan['git_sha'][:12]}", include_source=False)
    started = time.monotonic()
    try:
        # Deadline covers app startup plus all CPU image layers, not each layer separately.
        async with asyncio.timeout(BUILD_TIMEOUT), build_app.run(detach=False):
            remember_app(state, build_app, "cpu_image_build", output)
            await image.build.aio(build_app)
            state["image_id"] = image.object_id
    finally:
        state["build_wall_seconds"] = time.monotonic() - started
        remember_app(state, build_app, "cpu_image_build", output)
        write_json(output / "run.json", state)

    limits = plan["limits"]
    app = modal.App(f"boileroom-validate-{plan['model']}-{plan['git_sha'][:12]}", include_source=False)
    function = app.function(
        image=image,
        serialized=True,
        include_source=False,
        gpu=limits["gpu"],
        cpu=(2, 2),
        memory=(limits["memory_mib"], limits["memory_mib"]),
        timeout=limits["remote_timeout_seconds"],
        startup_timeout=300,
        min_containers=0,
        max_containers=1,
        buffer_containers=0,
        scaledown_window=2,
        retries=0,
        single_use_containers=True,
    )(remote_validation)
    started = time.monotonic()
    try:
        # Includes scheduling/startup and any platform retry; not just a per-attempt timeout.
        async with asyncio.timeout(limits["remote_timeout_seconds"]), app.run(detach=False):
            remember_app(state, app, "gpu_validation", output)
            report = await function.remote.aio(plan["model"], plan["model_size"])
            write_json(output / "validation.json", report)
            state["passed"] = report.get("passed") is True
            if not state["passed"]:
                raise RuntimeError("Model validation failed; see validation.json")
    finally:
        state["validation_wall_seconds"] = time.monotonic() - started
        remember_app(state, app, "gpu_validation", output)
        write_json(output / "run.json", state)


async def execute(root: Path, plan: dict[str, Any], output: Path, state: dict[str, Any]) -> None:
    # Avoid any user credential/profile config; SDK uses the provided environment.
    os.environ["MODAL_CONFIG_PATH"] = "/dev/null"
    import modal

    if modal.__version__ != MODAL_VERSION:
        raise RuntimeError(f"Use modal=={MODAL_VERSION} for this reviewed launcher")
    loop = asyncio.get_running_loop()
    task = asyncio.current_task()
    assert task is not None
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, task.cancel)
    try:
        with modal.enable_output():
            async with asyncio.timeout(TOTAL_TIMEOUT):
                await run_apps(modal, root, plan, output, state)
    finally:
        for signum in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(signum)


def cleanup(output: Path) -> bool:
    """Stop only this invocation's recorded apps and retain scoped lifecycle evidence."""
    path = output / "run.json"
    if not path.exists():
        return True
    state = json.loads(path.read_text())
    records = state.get("apps", [])
    ids = [record["app_id"] for record in records]
    if any(not APP_ID.fullmatch(app_id) for app_id in ids) or len(ids) > 2:
        raise ValueError("Refusing cleanup of invalid/unbounded app IDs")
    evidence: dict[str, Any] = {"at": utc_now(), "stop_results": [], "verified_stopped": not ids}
    env = {**os.environ, "MODAL_CONFIG_PATH": "/dev/null"}
    for app_id in ids:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "modal", "app", "stop", app_id],
                env=env,
                capture_output=True,
                text=True,
                timeout=45,
            )
            evidence["stop_results"].append({"app_id": app_id, "returncode": result.returncode})
        except subprocess.TimeoutExpired:
            evidence["stop_results"].append({"app_id": app_id, "error": "stop request timed out"})
    if ids:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "modal", "app", "list", "--json"],
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            )
            rows = [row for row in json.loads(result.stdout) if row.get("App ID") in ids]
            evidence["apps"] = rows  # Never save other apps from the workspace.
            evidence["verified_stopped"] = len(rows) == len(ids) and all(row.get("State") == "stopped" for row in rows)
        except (subprocess.SubprocessError, ValueError) as exc:
            evidence["verification_error"] = type(exc).__name__
    write_json(output / "cleanup.json", evidence)
    return bool(evidence["verified_stopped"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(MODEL_LIMITS))
    parser.add_argument("--minifold-size", choices=("12L", "48L"), default="12L")
    parser.add_argument("--expected-sha")
    parser.add_argument(
        "--output-dir", type=Path, help="Evidence directory; defaults to a fresh directory outside the checkout"
    )
    parser.add_argument("--execute", action="store_true", help="Opt into paid work after explicit approval")
    parser.add_argument("--cleanup-only", action="store_true", help="Stop this run's recorded apps; create nothing")
    args = parser.parse_args(argv)
    if args.cleanup_only and args.output_dir is None:
        parser.error("--cleanup-only requires the exact original --output-dir")
    output = args.output_dir.resolve() if args.output_dir else Path(tempfile.mkdtemp(prefix="boileroom-validation-"))
    print(f"Validation evidence directory: {output}")
    if args.cleanup_only:
        return 0 if cleanup(output) else 1
    if not args.model:
        parser.error("--model is required unless --cleanup-only")
    plan = make_plan(ROOT, args.model, args.minifold_size, args.expected_sha)
    write_json(output / "plan.json", plan)
    if not args.execute:
        print(json.dumps({key: value for key, value in plan.items() if key != "source_files_sha256"}, indent=2))
        return 0
    if not args.expected_sha or plan["dirty"] or not plan["entrypoint_exists"]:
        parser.error("Paid execution requires --expected-sha, a clean checkout, and the selected validation runner")
    state: dict[str, Any] = {
        "started_at": utc_now(),
        "git_sha": plan["git_sha"],
        "model": args.model,
        "apps": [],
        "passed": False,
    }
    write_json(output / "run.json", state)
    exit_code = 1
    started = time.monotonic()
    try:
        asyncio.run(execute(ROOT, plan, output, state))
        exit_code = 0 if state["passed"] else 1
    except (Exception, KeyboardInterrupt, asyncio.CancelledError) as exc:
        state["error"] = {"type": type(exc).__name__, "message": str(exc)}
        print(f"Validation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    finally:
        state["finished_at"] = utc_now()
        state["total_wall_seconds"] = time.monotonic() - started
        state["cost_evidence"] = {
            **plan["cost_evidence"],
            "requested_gpu": plan["limits"]["gpu"],
            "requested_memory_mib": plan["limits"]["memory_mib"],
            "validation_wall_seconds": state.get("validation_wall_seconds"),
            "build_wall_seconds": state.get("build_wall_seconds"),
        }
        write_json(output / "run.json", state)
        if not cleanup(output):
            print(
                "App termination is unverified; see cleanup.json and check the recorded IDs immediately.",
                file=sys.stderr,
            )
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
