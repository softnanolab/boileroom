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

from boileroom.images.metadata import (  # noqa: E402
    BASE_IMAGE_SPEC,
    CUDA_MICROMAMBA_BASE,
    DEFAULT_DOCKER_REPOSITORY,
    MODEL_IMAGE_SPECS,
    RuntimeImageSpec,
    get_supported_cuda,
    normalize_docker_repository,
    normalize_cuda_version,
    normalize_requested_tag,
    published_image_references,
)
from scripts.cli_utils import CONTEXT_SETTINGS, all_cuda_option, cuda_version_option, none_if_empty  # noqa: E402


@dataclass(frozen=True)
class PromoteOptions:
    """CLI options for runtime image promotion."""

    source_tag: str
    target_tag: str
    docker_user: str
    cuda_versions: list[str] | None
    all_cuda: bool


def compute_cuda_versions(requested: list[str] | None, all_cuda: bool) -> list[str]:
    """Resolve requested CUDA versions."""
    if all_cuda:
        return sorted({cuda for spec in (BASE_IMAGE_SPEC, *MODEL_IMAGE_SPECS) for cuda in get_supported_cuda(spec)})
    if not requested:
        raise ValueError("Specify at least one --cuda-version or use --all-cuda.")
    return [normalize_cuda_version(cuda_version) for cuda_version in requested]


def ensure_buildx() -> None:
    """Ensure Docker buildx is available."""
    try:
        subprocess.run(
            ["docker", "buildx", "version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("Docker buildx is required but was not found.") from exc


def promote_one(
    spec: RuntimeImageSpec,
    cuda_version: str,
    source_tag: str,
    target_tag: str,
    docker_repository: str,
) -> None:
    """Promote one canonical image manifest to its public target tags."""
    source_reference = published_image_references(spec.image_name, cuda_version, source_tag, docker_repository)[0]
    target_references = published_image_references(spec.image_name, cuda_version, target_tag, docker_repository)
    cmd = ["docker", "buildx", "imagetools", "create"]
    for target_reference in target_references:
        cmd.extend(["-t", target_reference])
    cmd.append(source_reference)
    print(f"Promoting {source_reference} -> {', '.join(target_references)}")
    subprocess.run(cmd, check=True)


def run_promote_images(options: PromoteOptions) -> None:
    """Promote validated runtime images to public tags."""

    docker_repository = normalize_docker_repository(options.docker_user)
    source_tag = normalize_requested_tag(options.source_tag)
    target_tag = normalize_requested_tag(options.target_tag)
    cuda_versions = compute_cuda_versions(options.cuda_versions, options.all_cuda)
    ensure_buildx()
    image_specs = (BASE_IMAGE_SPEC, *MODEL_IMAGE_SPECS)

    for cuda_version in cuda_versions:
        for spec in image_specs:
            if cuda_version not in get_supported_cuda(spec):
                continue
            promote_one(spec, cuda_version, source_tag, target_tag, docker_repository)


@click.command(context_settings=CONTEXT_SETTINGS, help="Promote validated boileroom image tags without rebuilding.")
@click.option("--source-tag", required=True, help="Validated source tag, for example sha-abcd1234.")
@click.option("--target-tag", required=True, help="Public target tag, for example 0.3.0.")
@click.option("--docker-user", default=DEFAULT_DOCKER_REPOSITORY, help="Docker Hub user or namespace to promote.")
@cuda_version_option("CUDA version to promote (repeatable). Supported values: 11.8, 12.6.")
@all_cuda_option("Promote all supported CUDA variants.")
def cli(source_tag: str, target_tag: str, docker_user: str, cuda_versions: tuple[str, ...], all_cuda: bool) -> None:
    """Run the image promotion Click command."""

    try:
        run_promote_images(
            PromoteOptions(
                source_tag=source_tag,
                target_tag=target_tag,
                docker_user=docker_user,
                cuda_versions=none_if_empty(cuda_versions),
                all_cuda=all_cuda,
            )
        )
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc


if __name__ == "__main__":
    cli()
