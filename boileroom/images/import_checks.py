"""Helpers shared by image import smoke checks."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from .metadata import (
    DEFAULT_DOCKER_REPOSITORY,
    MODEL_IMAGE_SPECS,
    SUPPORTED_CUDA_VERSIONS,
    RuntimeImageSpec,
    format_image_reference,
    get_supported_cuda,
    get_supported_platforms,
    normalize_cuda_version,
    normalize_requested_tag,
    published_image_references,
    split_platforms,
)

_CUDA_TAG_PATTERN = re.compile(r"^cuda\d+\.\d+(?:-.+)?$")

IMPORT_NAME_OVERRIDES: Final[dict[str, str | None]] = {
    "absl-py": "absl",
    "biopython": "Bio",
    "dm-haiku": "haiku",
    "ml-collections": "ml_collections",
    "tensorflow-cpu": "tensorflow",
    "pytorch-lightning": "pytorch_lightning",
    "torch-tensorrt": None,
    "hf-transfer": None,
    "hf_transfer": None,
}
"""Map package names to importable module names for smoke tests.

Add entries here when a package name does not map cleanly to
``package_name.replace("-", "_")`` or should be skipped entirely.
"""


def compute_cuda_versions(requested: list[str] | None, all_cuda: bool) -> list[str]:
    """Resolve CUDA versions requested by the caller."""
    if all_cuda:
        return sorted(normalize_cuda_version(cuda_version) for cuda_version in SUPPORTED_CUDA_VERSIONS)
    if not requested:
        return []
    return [normalize_cuda_version(cuda_version) for cuda_version in requested]


def package_name_to_import_name(package_name: str) -> str | None:
    """Return the importable module name for a dependency string."""
    return IMPORT_NAME_OVERRIDES.get(package_name, package_name.replace("-", "_"))


def requirement_line_to_package_name(line: str) -> str | None:
    """Return the package name from one requirements.txt line, or None for pip options."""
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or stripped.startswith("-"):
        return None
    return re.split(r"[>=<!=;\[]", stripped, maxsplit=1)[0].strip() or None


def requirement_import_names(requirements_path: Path) -> list[str]:
    """Return import names represented by a requirements.txt file."""
    import_names = []
    with requirements_path.open(encoding="utf-8") as handle:
        for line in handle:
            package_name = requirement_line_to_package_name(line)
            if package_name is None:
                continue
            import_name = package_name_to_import_name(package_name)
            if import_name:
                import_names.append(import_name)
    return import_names


def iter_image_targets(
    tag: str | None,
    cuda_versions: list[str],
    *,
    docker_repository: str = DEFAULT_DOCKER_REPOSITORY,
    image_specs: Sequence[RuntimeImageSpec] | None = None,
    platform: str | None = None,
) -> list[tuple[str, str, str, Path, Path]]:
    """Return model-image targets for smoke checks.

    Returns tuples of ``(image_key, image_reference, display_tag, requirements_path, core_path)``.
    """
    normalized_tag = normalize_requested_tag(tag)
    if cuda_versions and _CUDA_TAG_PATTERN.fullmatch(normalized_tag):
        raise ValueError("Do not combine --all-cuda/--cuda-version with an already CUDA-qualified --tag.")

    specs = MODEL_IMAGE_SPECS if image_specs is None else image_specs
    requested_platforms = set(split_platforms(platform)) if platform is not None else None
    targets: list[tuple[str, str, str, Path, Path]] = []
    for spec in specs:
        if requested_platforms is not None and not requested_platforms.issubset(get_supported_platforms(spec)):
            continue

        requirements_path = spec.context_path / "requirements.txt"
        core_path = spec.context_path / "core.py"
        if cuda_versions:
            for cuda_version in cuda_versions:
                if cuda_version not in get_supported_cuda(spec):
                    continue
                canonical_ref = published_image_references(
                    spec.image_name, cuda_version, normalized_tag, docker_repository
                )[0]
                display_tag = canonical_ref.rsplit(":", 1)[1]
                targets.append((spec.key, canonical_ref, display_tag, requirements_path, core_path))
            continue

        image_reference = format_image_reference(spec.image_name, normalized_tag, docker_repository)
        targets.append((spec.key, image_reference, normalized_tag, requirements_path, core_path))
    return targets
