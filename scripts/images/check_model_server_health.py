#!/usr/bin/env python3
from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib import error, request

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from boileroom.backend.transport import TRANSPORT_HMAC_KEY_ENV  # noqa: E402
from boileroom.images.import_checks import compute_cuda_versions, iter_image_targets  # noqa: E402
from boileroom.images.metadata import normalize_requested_tag  # noqa: E402

SERVER_PATH = REPO_ROOT / "boileroom/backend/server.py"
FAKE_MODEL_CLASS = "boileroom.testing.fake_core.HealthcheckCore"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run /health smoke checks inside boileroom model images.")
    parser.add_argument(
        "--tag",
        default=None,
        help="Tag to check. Defaults to the installed boileroom version; explicit examples include 0.3.0 or cuda12.6-0.3.0.",
    )
    parser.add_argument(
        "--cuda-version",
        action="append",
        dest="cuda_versions",
        help="CUDA version to validate canonically (repeatable).",
    )
    parser.add_argument("--all-cuda", action="store_true", help="Validate all supported CUDA variants canonically.")
    parser.add_argument("--pull", action="store_true", help="Pull images before running checks.")
    parser.add_argument("--timeout", type=float, default=30.0, help="Seconds to wait for each container health check.")
    return parser.parse_args()


def ensure_docker() -> None:
    """Ensure Docker is available."""
    try:
        subprocess.run(
            ["docker", "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("Docker is required but was not found on PATH.") from exc


def _allocate_port() -> int:
    """Allocate an ephemeral localhost port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def _wait_for_health(port: int, timeout: float) -> None:
    """Poll the local /health endpoint until it responds successfully."""
    deadline = time.time() + timeout
    last_error: Exception | None = None
    url = f"http://127.0.0.1:{port}/health"
    while time.time() < deadline:
        try:
            with request.urlopen(url, timeout=2.0) as response:
                if response.status == 200 and b"ready" in response.read():
                    return
        except (error.URLError, TimeoutError, ConnectionError) as exc:
            last_error = exc
            time.sleep(1.0)
            continue
    raise RuntimeError(f"Timed out waiting for {url}: {last_error}")


def check_server_health(image_key: str, image_reference: str, pull: bool, timeout: float) -> None:
    """Run the server /health smoke test for one image."""
    print(f"Checking server health for {image_key} ({image_reference})")

    if pull:
        subprocess.run(["docker", "pull", image_reference], check=True)

    port = _allocate_port()
    container_name = f"boileroom-{image_key}-health-{port}"
    run_cmd = [
        "docker",
        "run",
        "--rm",
        "--detach",
        "--name",
        container_name,
        "--publish",
        f"127.0.0.1:{port}:8000",
        "--volume",
        f"{REPO_ROOT}:{REPO_ROOT}:ro",
        "--env",
        f"PYTHONPATH={REPO_ROOT}",
        "--env",
        f"MODEL_CLASS={FAKE_MODEL_CLASS}",
        "--env",
        "MODEL_CONFIG={}",
        "--env",
        "DEVICE=cpu",
        "--env",
        f"{TRANSPORT_HMAC_KEY_ENV}=healthcheck-secret",
        image_reference,
        "micromamba",
        "run",
        "-n",
        "base",
        "python",
        str(SERVER_PATH),
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
    ]
    subprocess.run(run_cmd, check=True, stdout=subprocess.DEVNULL)

    try:
        _wait_for_health(port, timeout)
    except Exception:
        logs = subprocess.run(
            ["docker", "logs", container_name],
            check=False,
            capture_output=True,
            text=True,
        )
        if logs.stdout:
            print(logs.stdout, file=sys.stderr)
        if logs.stderr:
            print(logs.stderr, file=sys.stderr)
        raise
    finally:
        subprocess.run(
            ["docker", "rm", "--force", container_name],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def main() -> None:
    """Run the server health-smoke workflow."""
    args = parse_args()
    ensure_docker()
    cuda_versions = compute_cuda_versions(args.cuda_versions, args.all_cuda)
    targets = iter_image_targets(args.tag, cuda_versions)
    if not targets:
        raise SystemExit("No image targets matched the requested CUDA selection.")

    for image_key, image_reference, _display_tag, _env_path, _core_path in targets:
        check_server_health(image_key, image_reference, args.pull, args.timeout)

    print(f"All server health checks succeeded for tag selection: {normalize_requested_tag(args.tag)}")


if __name__ == "__main__":
    main()
