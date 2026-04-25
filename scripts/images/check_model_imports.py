#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import click

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from boileroom.images.import_checks import (  # noqa: E402
    IMPORT_NAME_OVERRIDES,
    compute_cuda_versions,
    iter_image_targets,
)
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


@dataclass(frozen=True)
class ImportCheckOptions:
    """CLI options for image import smoke checks."""

    tag: str | None
    docker_user: str
    cuda_versions: list[str] | None
    all_cuda: bool
    pull: bool
    cleanup: bool


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


def check_image(
    image_key: str,
    image_reference: str,
    requirements_path: Path,
    core_path: Path,
    pull: bool,
    cleanup: bool = False,
) -> None:
    """Run the import smoke test for one image."""
    print(f"Checking imports for {image_key} ({image_reference})")

    if pull:
        subprocess.run(["docker", "pull", image_reference], check=True)

    if not requirements_path.exists():
        raise FileNotFoundError(f"Missing requirements file: {requirements_path}")
    if not core_path.exists():
        raise FileNotFoundError(f"Missing core module: {core_path}")

    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            image_reference,
            "/bin/sh",
            "-lc",
            "python --version && pip --version",
        ],
        check=True,
    )

    script = f"""
import ast
import importlib
import re
import sys
from pathlib import Path

requirements_txt = Path({str(requirements_path)!r})
core_file = Path({str(core_path)!r})
import_name_overrides = {IMPORT_NAME_OVERRIDES!r}

deps = []
with requirements_txt.open(encoding="utf-8") as handle:
    for line in handle:
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        pkg_name = re.split(r'[>=<!=;\\[]', stripped)[0].strip()
        import_name = import_name_overrides.get(pkg_name, pkg_name.replace('-', '_'))
        if import_name:
            deps.append(import_name)

deps.append('numpy')

try:
    ast.parse(core_file.read_text(encoding='utf-8'), filename=str(core_file))
    print(f'OK: {{core_file.name}} (syntax valid)')
except SyntaxError as exc:
    print(f'FAILED: {{core_file.name}} has syntax errors: {{exc}}', file=sys.stderr)
    sys.exit(1)

for dep in deps:
    try:
        importlib.import_module(dep)
        print(f'OK: {{dep}}')
    except Exception as exc:
        print(f'FAILED: {{dep}} - {{exc}}', file=sys.stderr)
        sys.exit(1)
"""

    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{REPO_ROOT}:{REPO_ROOT}:ro",
            "-e",
            f"PYTHONPATH={REPO_ROOT}",
            image_reference,
            "python",
            "-c",
            script,
        ],
        check=True,
    )

    if cleanup:
        _remove_image(image_reference)


def run_import_checks(options: ImportCheckOptions) -> None:
    """Run the import smoke workflow."""

    ensure_docker()
    docker_repository = normalize_docker_repository(options.docker_user)
    cuda_versions = compute_cuda_versions(options.cuda_versions, options.all_cuda)
    targets = iter_image_targets(options.tag, cuda_versions, docker_repository=docker_repository)
    if not targets:
        raise SystemExit("No image targets matched the requested CUDA selection.")

    for image_key, image_reference, _display_tag, requirements_path, core_path in targets:
        check_image(image_key, image_reference, requirements_path, core_path, options.pull, options.cleanup)

    print(f"All module imports succeeded for tag selection: {normalize_requested_tag(options.tag)}")


@click.command(context_settings=CONTEXT_SETTINGS, help="Run import smoke checks inside boileroom model images.")
@tag_option
@click.option("--docker-user", default=DEFAULT_DOCKER_REPOSITORY, help="Docker Hub user or namespace to check.")
@cuda_version_option("CUDA version to validate canonically (repeatable).")
@all_cuda_option("Validate all supported CUDA variants canonically.")
@pull_option
@cleanup_option
def cli(
    tag: str | None,
    docker_user: str,
    cuda_versions: tuple[str, ...],
    all_cuda: bool,
    pull: bool,
    cleanup: bool,
) -> None:
    """Run the image import-check Click command."""

    run_import_checks(
        ImportCheckOptions(
            tag=tag,
            docker_user=docker_user,
            cuda_versions=none_if_empty(cuda_versions),
            all_cuda=all_cuda,
            pull=pull,
            cleanup=cleanup,
        )
    )


if __name__ == "__main__":
    cli()
