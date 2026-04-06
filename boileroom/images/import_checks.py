"""Helpers shared by image import smoke checks."""

from __future__ import annotations

from typing import Final

from .metadata import CUDA_MICROMAMBA_BASE, normalize_cuda_version

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
