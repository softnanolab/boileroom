#!/usr/bin/env python3
from __future__ import annotations

import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib import error, request

import click

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from boileroom.backend.transport import TRANSPORT_HMAC_KEY_ENV  # noqa: E402
from boileroom.images.import_checks import compute_cuda_versions, iter_image_targets  # noqa: E402
from boileroom.images.metadata import (  # noqa: E402
    DEFAULT_DOCKER_REPOSITORY,
    normalize_docker_repository,
    normalize_requested_tag,
)
from scripts.cli_utils import (  # noqa: E402
    CONTEXT_SETTINGS,
    all_cuda_option,
    cleanup_option,
    cuda_version_option,
    none_if_empty,
    pull_option,
    tag_option,
)

SERVER_PATH = REPO_ROOT / "boileroom/backend/server.py"
FAKE_MODEL_CLASS = "boileroom.testing.fake_core.HealthcheckCore"


@dataclass(frozen=True)
class HealthCheckOptions:
    """CLI options for image server health checks."""

    tag: str | None
    docker_user: str
    cuda_versions: list[str] | None
    all_cuda: bool
    pull: bool
    cleanup: bool
    timeout: float


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


def _remove_image(image_reference: str) -> None:
    """Remove a Docker image to free disk space."""
    result = subprocess.run(
        ["docker", "rmi", "--force", image_reference],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        err = (result.stderr or "").strip()
        print(f"WARNING: failed to remove image {image_reference}: {err}", file=sys.stderr)


def check_server_health(
    image_key: str,
    image_reference: str,
    pull: bool,
    timeout: float,
    cleanup: bool = False,
) -> None:
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

    if cleanup:
        _remove_image(image_reference)


def run_server_health_checks(options: HealthCheckOptions) -> None:
    """Run the server health-smoke workflow."""

    ensure_docker()
    docker_repository = normalize_docker_repository(options.docker_user)
    cuda_versions = compute_cuda_versions(options.cuda_versions, options.all_cuda)
    targets = iter_image_targets(options.tag, cuda_versions, docker_repository=docker_repository)
    if not targets:
        raise SystemExit("No image targets matched the requested CUDA selection.")

    for image_key, image_reference, _display_tag, _requirements_path, _core_path in targets:
        check_server_health(image_key, image_reference, options.pull, options.timeout, options.cleanup)

    print(f"All server health checks succeeded for tag selection: {normalize_requested_tag(options.tag)}")


@click.command(context_settings=CONTEXT_SETTINGS, help="Run /health smoke checks inside boileroom model images.")
@tag_option
@click.option("--docker-user", default=DEFAULT_DOCKER_REPOSITORY, help="Docker Hub user or namespace to check.")
@cuda_version_option("CUDA version to validate canonically (repeatable).")
@all_cuda_option("Validate all supported CUDA variants canonically.")
@pull_option
@cleanup_option
@click.option("--timeout", type=float, default=30.0, help="Seconds to wait for each container health check.")
def cli(
    tag: str | None,
    docker_user: str,
    cuda_versions: tuple[str, ...],
    all_cuda: bool,
    pull: bool,
    cleanup: bool,
    timeout: float,
) -> None:
    """Run the server-health Click command."""

    run_server_health_checks(
        HealthCheckOptions(
            tag=tag,
            docker_user=docker_user,
            cuda_versions=none_if_empty(cuda_versions),
            all_cuda=all_cuda,
            pull=pull,
            cleanup=cleanup,
            timeout=timeout,
        )
    )


if __name__ == "__main__":
    cli()
