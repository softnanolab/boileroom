#!/usr/bin/env python3
from __future__ import annotations

import json
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
    DEFAULT_DOCKER_REPOSITORY,
    KIT_IMAGE_SPECS,
    MODEL_IMAGE_SPECS,
    RuntimeImageSpec,
    get_supported_cuda,
    kit_image_reference,
    normalize_docker_repository,
    normalize_cuda_version,
    normalize_requested_tag,
    published_image_references,
)
from scripts.cli_utils import CONTEXT_SETTINGS, all_cuda_option, cuda_version_option, none_if_empty  # noqa: E402


@dataclass(frozen=True)
class PromoteOptions:
    """CLI options for runtime image promotion."""

    source_tag: str | None
    target_tag: str
    docker_user: str
    cuda_versions: list[str] | None
    all_cuda: bool
    skip_kit_images: bool = False
    kit_images_only: bool = False
    force_kit_tags: bool = False


class PromotionError(RuntimeError):
    """A promotion that would move an existing tag, or whose target does not hold the source manifest afterwards."""


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


def reports_missing_manifest(output: str, reference: str) -> bool:
    """Return whether a failed ``imagetools inspect`` of ``reference`` says that the reference itself does not exist.

    buildx reports a missing tag or digest as ``ERROR: <reference>: not found`` (the registry's 404 for that manifest),
    and some registries answer ``manifest unknown``; both name the reference. Any other failure, including one whose
    text merely contains "not found" (a credential helper that is not on PATH, a token endpoint's 404, a repository the
    registry hides behind an authorization error), is not an absent tag, so the caller fails instead of pushing.

    Parameters
    ----------
    output : str
        The combined stderr and stdout of the failed inspect.
    reference : str
        The reference that was inspected, as passed to ``imagetools inspect``.

    Returns
    -------
    bool
        True only when a line of ``output`` names ``reference`` and reports it as not found or as an unknown manifest.
    """
    for line in output.splitlines():
        message = line.strip().removeprefix("ERROR:").strip()
        if message == f"{reference}: not found" or message.endswith(f" {reference}: not found"):
            return True
        if reference in message and "manifest unknown" in message.lower():
            return True
    return False


def manifest_digests(reference: str) -> tuple[str, ...] | None:
    """Return the digests a registry reference resolves to, or None when it does not exist.

    Parameters
    ----------
    reference : str
        A ``repository/name:tag`` or ``repository/name@digest`` reference.

    Returns
    -------
    tuple[str, ...] | None
        The top-level manifest digest first, then, for an image index, each child manifest's digest.

    Raises
    ------
    PromotionError
        If the registry cannot be read for another reason, or the manifest carries no digest.
    """
    result = subprocess.run(
        ["docker", "buildx", "imagetools", "inspect", "--format", "{{json .Manifest}}", reference],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        output = f"{result.stderr}\n{result.stdout}".strip()
        if reports_missing_manifest(output, reference):
            return None
        raise PromotionError(f"Could not inspect {reference}: {output}")
    manifest = json.loads(result.stdout)
    digest = manifest.get("digest")
    if not digest:
        raise PromotionError(f"The manifest of {reference} carries no digest: {result.stdout.strip()}")
    children = [str(child["digest"]) for child in manifest.get("manifests") or [] if child.get("digest")]
    return (str(digest), *children)


def holds(digests: tuple[str, ...] | None, digest: str) -> bool:
    """Return whether a reference resolving to ``digests`` serves the manifest ``digest``.

    ``imagetools create`` copies an image index as is but wraps a single-platform manifest in a new index, so the
    manifest is either the top-level digest or one of the index's children.
    """
    return digests is not None and digest in digests


def create_tags(source_reference: str, target_references: tuple[str, ...], digest: str) -> None:
    """Point every target at the manifest ``source_reference`` names, then check that each one serves ``digest``.

    Raises
    ------
    PromotionError
        If a target does not serve ``digest`` after the push.
    """
    cmd = ["docker", "buildx", "imagetools", "create"]
    for target_reference in target_references:
        cmd.extend(["-t", target_reference])
    cmd.append(source_reference)
    print(f"Promoting {source_reference} -> {', '.join(target_references)}")
    subprocess.run(cmd, check=True)
    for target_reference in target_references:
        resolved = manifest_digests(target_reference)
        if not holds(resolved, digest):
            raise PromotionError(
                f"{target_reference} resolves to {resolved[0] if resolved else 'nothing'} after the promotion, "
                f"not to {digest} from {source_reference}."
            )


def promote_one(
    spec: RuntimeImageSpec,
    cuda_version: str,
    source_tag: str,
    target_tag: str,
    docker_repository: str,
) -> None:
    """Promote one canonical image manifest to its public target tags.

    The source tag is resolved to its digest first and the targets are created from that digest, so a source tag
    re-pushed meanwhile cannot slip a different image under the target tags; each target is then checked to serve it.

    Raises
    ------
    PromotionError
        If the source does not exist, or a target does not serve the source's manifest afterwards.
    """
    source_reference = published_image_references(spec.image_name, cuda_version, source_tag, docker_repository)[0]
    target_references = published_image_references(spec.image_name, cuda_version, target_tag, docker_repository)
    source_digests = manifest_digests(source_reference)
    if source_digests is None:
        raise PromotionError(f"The source image {source_reference} does not exist.")
    digest = source_digests[0]
    print(f"{source_reference} is {digest}")
    create_tags(f"{source_reference.rsplit(':', 1)[0]}@{digest}", target_references, digest)


def kit_promotion(spec: RuntimeImageSpec, target_tag: str, docker_repository: str) -> tuple[str, str]:
    """Return the ``(source, target)`` references that promote one kit image.

    The kit images are not rebuilt per release: each release tag names the digest pinned in ``KIT_IMAGE_DIGESTS``, so
    the source is always that digest (never ``--source-tag``), and the tag carries no CUDA qualifier (the kit stack is
    CUDA 13.0, outside the CUDA-qualified scheme of the stock images).

    Raises
    ------
    ValueError
        If ``BOILEROOM_KIT_IMAGE_TAG`` is set, which would make the source a tag instead of the pinned digest.
    """
    source_reference = kit_image_reference(spec, None, docker_repository)
    if "@" not in source_reference:
        raise ValueError(
            f"Kit images are promoted from their pinned digests; unset BOILEROOM_KIT_IMAGE_TAG (got {source_reference})."
        )
    return source_reference, kit_image_reference(spec, target_tag, docker_repository)


def kit_target_needs_push(source_reference: str, target_reference: str, force: bool = False) -> bool:
    """Return whether ``target_reference`` must be pushed to name the pinned digest of ``source_reference``.

    A kit release tag names the digest that release pins; moving it silently would point the tag at a digest the
    release never pinned and could leave the old pin untagged (and so open to Docker Hub's garbage collection).

    Raises
    ------
    PromotionError
        If the target already exists at another digest and ``force`` is not set.
    """
    pinned = source_reference.rsplit("@", 1)[1]
    current = manifest_digests(target_reference)
    if current is None:
        return True
    if holds(current, pinned):
        print(f"{target_reference} already points at {pinned}")
        return False
    if not force:
        raise PromotionError(
            f"{target_reference} already exists at {current[0]}, not at the pinned {pinned}. Moving it could leave the "
            "digest an earlier release pins untagged. Pass --force-kit-tags to move it anyway."
        )
    print(f"Moving {target_reference} from {current[0]} to {pinned} (--force-kit-tags)")
    return True


def promote_kit_one(source_reference: str, target_reference: str) -> None:
    """Point ``target_reference`` at the pinned manifest of ``source_reference`` without rebuilding, and check it."""
    create_tags(source_reference, (target_reference,), source_reference.rsplit("@", 1)[1])


def run_promote_images(options: PromoteOptions) -> None:
    """Promote validated runtime images to public tags.

    Every stock image is promoted from ``source_tag`` per CUDA version, unless ``kit_images_only`` is set. Unless
    ``skip_kit_images`` is set, each kit image is then promoted once, from its pinned digest. A kit target that already
    names another digest is refused before anything is pushed, unless ``force_kit_tags`` is set.
    """

    if options.skip_kit_images and options.kit_images_only:
        raise ValueError("--skip-kit-images and --kit-images-only exclude each other.")
    docker_repository = normalize_docker_repository(options.docker_user)
    target_tag = normalize_requested_tag(options.target_tag)
    source_tag: str | None = None
    cuda_versions: list[str] = []
    if not options.kit_images_only:
        if not options.source_tag:
            raise ValueError("--source-tag is required unless --kit-images-only is set.")
        source_tag = normalize_requested_tag(options.source_tag)
        cuda_versions = compute_cuda_versions(options.cuda_versions, options.all_cuda)
    # Resolve the kit promotions before anything is pushed, so a bad kit setup fails without a partial promotion.
    kit_promotions = (
        [] if options.skip_kit_images else [kit_promotion(spec, target_tag, docker_repository) for spec in KIT_IMAGE_SPECS]
    )
    ensure_buildx()
    kit_pushes = [
        (source_reference, target_reference)
        for source_reference, target_reference in kit_promotions
        if kit_target_needs_push(source_reference, target_reference, options.force_kit_tags)
    ]
    image_specs = (BASE_IMAGE_SPEC, *MODEL_IMAGE_SPECS)

    for cuda_version in cuda_versions:
        for spec in image_specs:
            if source_tag is None or cuda_version not in get_supported_cuda(spec):
                continue
            promote_one(spec, cuda_version, source_tag, target_tag, docker_repository)

    if options.skip_kit_images:
        print("Skipping the kit images (--skip-kit-images).")
    for source_reference, target_reference in kit_pushes:
        promote_kit_one(source_reference, target_reference)


@click.command(context_settings=CONTEXT_SETTINGS, help="Promote validated boileroom image tags without rebuilding.")
@click.option(
    "--source-tag",
    default=None,
    help="Validated source tag, for example sha-abcd1234. Required unless --kit-images-only is set.",
)
@click.option("--target-tag", required=True, help="Public target tag, for example 0.3.0.")
@click.option("--docker-user", default=DEFAULT_DOCKER_REPOSITORY, help="Docker Hub user or namespace to promote.")
@cuda_version_option("CUDA version to promote (repeatable). Supported values: 11.8, 12.6.")
@all_cuda_option("Promote all supported CUDA variants.")
@click.option(
    "--skip-kit-images",
    is_flag=True,
    help="Do not tag the kit images. By default each kit image's pinned digest also gets the target tag.",
)
@click.option(
    "--kit-images-only",
    is_flag=True,
    help="Tag only the kit images' pinned digests, as the full-release workflow does.",
)
@click.option(
    "--force-kit-tags",
    is_flag=True,
    help="Move a kit target tag that already names another digest. Without it such a promotion is refused.",
)
def cli(
    source_tag: str | None,
    target_tag: str,
    docker_user: str,
    cuda_versions: tuple[str, ...],
    all_cuda: bool,
    skip_kit_images: bool,
    kit_images_only: bool,
    force_kit_tags: bool,
) -> None:
    """Run the image promotion Click command."""

    try:
        run_promote_images(
            PromoteOptions(
                source_tag=source_tag,
                target_tag=target_tag,
                docker_user=docker_user,
                cuda_versions=none_if_empty(cuda_versions),
                all_cuda=all_cuda,
                skip_kit_images=skip_kit_images,
                kit_images_only=kit_images_only,
                force_kit_tags=force_kit_tags,
            )
        )
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc
    except PromotionError as exc:
        raise click.ClickException(str(exc)) from exc


if __name__ == "__main__":
    cli()
