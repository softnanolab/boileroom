"""Shared image metadata and tag helpers for Docker, Modal, and Apptainer."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Final

import yaml

DOCKER_REGISTRY: Final = "docker.io/jakublala"
DEFAULT_IMAGE_TAG: Final = "latest"
DEFAULT_CUDA_VERSION: Final = "12.6"
MODAL_IMAGE_TAG_ENV: Final = "BOILEROOM_MODAL_IMAGE_TAG"

CUDA_MICROMAMBA_BASE: Final[dict[str, str]] = {
    "11.8": "mambaorg/micromamba:2.4-cuda11.8.0-ubuntu22.04",
    "12.6": "mambaorg/micromamba:2.4-cuda12.6.3-ubuntu22.04",
}

CUDA_TORCH_WHEEL_INDEX: Final[dict[str, str]] = {
    "11.8": "https://download.pytorch.org/whl/cu118",
    "12.6": "https://download.pytorch.org/whl/cu126",
}

_CUDA_TAG_PATTERN = re.compile(r"^cuda(?P<cuda>\d+\.\d+)(?:-(?P<tag>.+))?$")


@dataclass(frozen=True)
class RuntimeImageSpec:
    """Container-image metadata shared across runtimes."""

    key: str
    image_name: str
    dockerfile_relative_path: str
    context_relative_path: str
    config_relative_path: str | None = None
    modal_runtime_env: tuple[tuple[str, str], ...] = ()

    @property
    def dockerfile_path(self) -> Path:
        """Return the absolute path to the Dockerfile."""
        return get_repo_root() / self.dockerfile_relative_path

    @property
    def context_path(self) -> Path:
        """Return the absolute path to the Docker build context."""
        return get_repo_root() / self.context_relative_path


BASE_IMAGE_SPEC: Final = RuntimeImageSpec(
    key="base",
    image_name="boileroom-base",
    dockerfile_relative_path="boileroom/images/Dockerfile",
    context_relative_path="boileroom/images",
)

MODEL_IMAGE_SPECS: Final[tuple[RuntimeImageSpec, ...]] = (
    RuntimeImageSpec(
        key="boltz",
        image_name="boileroom-boltz",
        dockerfile_relative_path="boileroom/models/boltz/Dockerfile",
        context_relative_path="boileroom/models/boltz",
        config_relative_path="boileroom/models/boltz/config.yaml",
    ),
    RuntimeImageSpec(
        key="chai",
        image_name="boileroom-chai1",
        dockerfile_relative_path="boileroom/models/chai/Dockerfile",
        context_relative_path="boileroom/models/chai",
        config_relative_path="boileroom/models/chai/config.yaml",
        modal_runtime_env=(("CHAI_DOWNLOADS_DIR", "{MODEL_DIR}/chai"),),
    ),
    RuntimeImageSpec(
        key="esm",
        image_name="boileroom-esm",
        dockerfile_relative_path="boileroom/models/esm/Dockerfile",
        context_relative_path="boileroom/models/esm",
        config_relative_path="boileroom/models/esm/config.yaml",
    ),
)

MODEL_IMAGE_SPECS_BY_KEY: Final = {spec.key: spec for spec in MODEL_IMAGE_SPECS}
MODEL_IMAGE_SPECS_BY_NAME: Final = {spec.image_name: spec for spec in MODEL_IMAGE_SPECS}


def get_repo_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parents[2]


def normalize_requested_tag(tag: str | None) -> str:
    """Return a normalized non-empty tag string."""
    normalized = (tag or DEFAULT_IMAGE_TAG).strip()
    return normalized or DEFAULT_IMAGE_TAG


def normalize_cuda_version(cuda_version: str) -> str:
    """Validate and normalize a CUDA version string."""
    normalized = cuda_version.strip()
    if normalized not in CUDA_MICROMAMBA_BASE:
        supported = ", ".join(sorted(CUDA_MICROMAMBA_BASE))
        raise ValueError(f"Unsupported CUDA version: {cuda_version}. Supported values: {supported}")
    return normalized


def canonical_image_tag(cuda_version: str, tag: str | None) -> str:
    """Return the canonical CUDA-qualified tag."""
    normalized_cuda = normalize_cuda_version(cuda_version)
    normalized_tag = normalize_requested_tag(tag)
    return f"cuda{normalized_cuda}-{normalized_tag}"


def resolve_registry_tag(tag: str | None) -> str:
    """Resolve a tag for runtime image lookup.

    Unqualified tags like ``latest`` or ``0.3.0`` are preserved so they resolve
    through the published default-CUDA aliases. Explicit CUDA-qualified tags are
    normalized to the canonical ``cuda<version>-<tag>`` form.
    """
    normalized_tag = normalize_requested_tag(tag)
    if match := _CUDA_TAG_PATTERN.fullmatch(normalized_tag):
        cuda_version = normalize_cuda_version(match.group("cuda"))
        tag_suffix = match.group("tag") or DEFAULT_IMAGE_TAG
        return canonical_image_tag(cuda_version, tag_suffix)
    return normalized_tag


def published_tags(cuda_version: str, tag: str | None) -> tuple[str, ...]:
    """Return the canonical published tags for a build output.

    The default CUDA line also gets an unqualified alias such as ``latest`` or
    ``0.3.0`` for convenience.
    """
    normalized_cuda = normalize_cuda_version(cuda_version)
    normalized_tag = normalize_requested_tag(tag)
    tags = [canonical_image_tag(normalized_cuda, normalized_tag)]
    if normalized_cuda == DEFAULT_CUDA_VERSION:
        tags.append(normalized_tag)
    return tuple(tags)


def format_image_reference(image_name: str, tag: str | None = None) -> str:
    """Return a fully qualified Docker image reference."""
    resolved_tag = resolve_registry_tag(tag)
    return f"{DOCKER_REGISTRY}/{image_name}:{resolved_tag}"


def published_image_references(image_name: str, cuda_version: str, tag: str | None) -> tuple[str, ...]:
    """Return all published references for a built image."""
    return tuple(
        f"{DOCKER_REGISTRY}/{image_name}:{published_tag}" for published_tag in published_tags(cuda_version, tag)
    )


def get_modal_image_tag() -> str:
    """Return the tag Modal should pull from the registry."""
    return resolve_registry_tag(os.environ.get(MODAL_IMAGE_TAG_ENV))


def get_model_image_spec(identifier: str) -> RuntimeImageSpec:
    """Return a model image spec by image name or short key."""
    if identifier in MODEL_IMAGE_SPECS_BY_KEY:
        return MODEL_IMAGE_SPECS_BY_KEY[identifier]
    if identifier in MODEL_IMAGE_SPECS_BY_NAME:
        return MODEL_IMAGE_SPECS_BY_NAME[identifier]
    raise KeyError(f"Unknown image spec: {identifier}")


@cache
def get_supported_cuda(spec: RuntimeImageSpec) -> tuple[str, ...]:
    """Return supported CUDA versions for a runtime image spec."""
    if spec.config_relative_path is None:
        return tuple(sorted(CUDA_MICROMAMBA_BASE))

    config_path = get_repo_root() / spec.config_relative_path
    if not config_path.exists():
        return tuple(sorted(CUDA_MICROMAMBA_BASE))

    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    raw_supported_cuda = config.get("supported_cuda", [])
    if isinstance(raw_supported_cuda, list):
        supported_cuda = [normalize_cuda_version(str(value)) for value in raw_supported_cuda]
    elif raw_supported_cuda:
        supported_cuda = [normalize_cuda_version(str(raw_supported_cuda))]
    else:
        supported_cuda = []

    if not supported_cuda:
        return tuple(sorted(CUDA_MICROMAMBA_BASE))
    return tuple(supported_cuda)


def render_modal_runtime_env(spec: RuntimeImageSpec, model_dir: str) -> dict[str, str]:
    """Render runtime environment overrides for Modal."""
    env = {"MODEL_DIR": model_dir}
    for key, value in spec.modal_runtime_env:
        env[key] = value.replace("{MODEL_DIR}", model_dir)
    return env
