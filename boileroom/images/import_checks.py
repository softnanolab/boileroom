"""Helpers shared by image import smoke checks."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

from .metadata import (
    CUDA_MICROMAMBA_BASE,
    MODEL_IMAGE_SPECS,
    format_image_reference,
    get_supported_cuda,
    normalize_cuda_version,
    normalize_requested_tag,
    published_image_references,
)

_CUDA_TAG_PATTERN = re.compile(r"^cuda\d+\.\d+(?:-.+)?$")

IMPORT_NAME_OVERRIDES: Final[dict[str, str | None]] = {
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
        return sorted(normalize_cuda_version(cuda_version) for cuda_version in CUDA_MICROMAMBA_BASE)
    if not requested:
        return []
    return [normalize_cuda_version(cuda_version) for cuda_version in requested]


def package_name_to_import_name(package_name: str) -> str | None:
    """Return the importable module name for a dependency string."""
    return IMPORT_NAME_OVERRIDES.get(package_name, package_name.replace("-", "_"))


def iter_image_targets(tag: str, cuda_versions: list[str]) -> list[tuple[str, str, str, Path, Path]]:
    """Return model-image targets for smoke checks.

    Returns tuples of ``(image_key, image_reference, display_tag, env_path, core_path)``.
    """
    normalized_tag = normalize_requested_tag(tag)
    if cuda_versions and _CUDA_TAG_PATTERN.fullmatch(normalized_tag):
        raise ValueError("Do not combine --all-cuda/--cuda-version with an already CUDA-qualified --tag.")

    targets: list[tuple[str, str, str, Path, Path]] = []
    for spec in MODEL_IMAGE_SPECS:
        env_path = spec.context_path / "environment.yml"
        core_path = spec.context_path / "core.py"
        if cuda_versions:
            for cuda_version in cuda_versions:
                if cuda_version not in get_supported_cuda(spec):
                    continue
                canonical_ref = published_image_references(spec.image_name, cuda_version, normalized_tag)[0]
                display_tag = canonical_ref.rsplit(":", 1)[1]
                targets.append((spec.key, canonical_ref, display_tag, env_path, core_path))
            continue

        image_reference = format_image_reference(spec.image_name, normalized_tag)
        targets.append((spec.key, image_reference, normalized_tag, env_path, core_path))
    return targets
