"""Shared image metadata and tag helpers for Docker, Modal, and Apptainer."""

from __future__ import annotations

import importlib.metadata
import os
import platform as platform_module
import re
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Final

DEFAULT_DOCKER_REPOSITORY: Final = "docker.io/jakublala"
PACKAGE_NAME: Final = "boileroom"
DEFAULT_CUDA_VERSION: Final = "12.6"
DEFAULT_PYTHON_VERSION: Final = "3.12"
DOCKER_REPOSITORY_ENV: Final = "BOILEROOM_DOCKER_REPOSITORY"
IMAGE_TAG_ENV: Final = "BOILEROOM_IMAGE_TAG"
KIT_IMAGE_SOURCE_ENV: Final = "BOILEROOM_KIT_IMAGE_SOURCE"
KIT_IMAGE_SOURCES: Final[tuple[str, ...]] = ("build", "registry")
DEFAULT_KIT_IMAGE_SOURCE: Final = "build"

# The optimization kit behind ``optimization="exact"`` and ``"fast"``: github.com/anthropics/uplifting-biomolecular-modeling
# (Apache-2.0). The kit Dockerfiles under ``boileroom/models/<family>/kit/`` fetch exactly this commit; their ``KIT_COMMIT``
# build arg must stay equal to the constant below (tests/contracts/test_kit_images.py checks it).
KIT_REPOSITORY: Final = "https://github.com/anthropics/uplifting-biomolecular-modeling.git"
KIT_COMMIT: Final = "f4f62fa6592ae4938d49b1757bea0cfeff9f468e"

SUPPORTED_CUDA_VERSIONS: Final[tuple[str, ...]] = ("11.8", "12.6")
SUPPORTED_PLATFORMS: Final[tuple[str, ...]] = ("linux/amd64", "linux/arm64")

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
    # Extra model-family keys served by this same image (no separate image is
    # built for them). Used when families share one runtime image, e.g. ESM-C /
    # ESM3 run on the ESMFold2 Biohub image.
    shared_family_keys: tuple[str, ...] = ()
    # The interpreter the Apptainer service runs under, /usr/local/bin/python<version> in the image.
    python_version: str = DEFAULT_PYTHON_VERSION

    @property
    def family_keys(self) -> tuple[str, ...]:
        """Return every model-family key this image serves (primary + shared)."""
        return (self.key, *self.shared_family_keys)

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
        key="alphafold",
        image_name="boileroom-alphafold2-multimer",
        dockerfile_relative_path="boileroom/models/alphafold/Dockerfile",
        context_relative_path="boileroom/models/alphafold",
        config_relative_path="boileroom/models/alphafold/config.yaml",
    ),
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
    RuntimeImageSpec(
        key="esmfold2",
        image_name="boileroom-esmfold2",
        dockerfile_relative_path="boileroom/models/esmfold2/Dockerfile",
        context_relative_path="boileroom/models/esmfold2",
        config_relative_path="boileroom/models/esmfold2/config.yaml",
        # ESM-C and ESM3 (the ``esm3`` family) use the same Biohub ``esm``
        # package, so they share this image instead of building their own. The
        # SAE feature model reuses ESM-C hidden states and shares it too.
        shared_family_keys=("esm3", "sae"),
    ),
    RuntimeImageSpec(
        key="protenix",
        image_name="boileroom-protenix",
        dockerfile_relative_path="boileroom/models/protenix/Dockerfile",
        context_relative_path="boileroom/models/protenix",
        config_relative_path="boileroom/models/protenix/config.yaml",
        modal_runtime_env=(("LAYERNORM_TYPE", "openfold"),),
    ),
    RuntimeImageSpec(
        key="opendde",
        image_name="boileroom-opendde",
        dockerfile_relative_path="boileroom/models/opendde/Dockerfile",
        context_relative_path="boileroom/models/opendde",
        config_relative_path="boileroom/models/opendde/config.yaml",
        modal_runtime_env=(("LAYERNORM_TYPE", "fast_layernorm"),),
    ),
)

# Kit runtime images: opt-in variants that only ``optimization != "vanilla"`` uses, never part of MODEL_IMAGE_SPECS (so the
# CI image matrix, ``scripts/images`` and the published tags are untouched). They do not declare ``supported_cuda``: the kit
# stack is CUDA 13.0 (an NVIDIA driver 580 or newer), outside SUPPORTED_CUDA_VERSIONS, and nothing is published from them.
KIT_IMAGE_SPECS: Final[tuple[RuntimeImageSpec, ...]] = (
    RuntimeImageSpec(
        key="esmfold2",
        image_name="boileroom-esmfold2-kit",
        dockerfile_relative_path="boileroom/models/esmfold2/kit/Dockerfile",
        context_relative_path="boileroom/models/esmfold2/kit",
        config_relative_path="boileroom/models/esmfold2/kit/config.yaml",
    ),
    RuntimeImageSpec(
        key="protenix",
        image_name="boileroom-protenix-kit",
        dockerfile_relative_path="boileroom/models/protenix/kit/Dockerfile",
        context_relative_path="boileroom/models/protenix/kit",
        config_relative_path="boileroom/models/protenix/kit/config.yaml",
        python_version="3.11",
        # The kit's layernorm lever and its pinned environment need stock's fast LayerNorm (the vanilla image uses openfold).
        modal_runtime_env=(("LAYERNORM_TYPE", "fast_layernorm"),),
    ),
)
KIT_IMAGE_SPECS_BY_KEY: Final = {spec.key: spec for spec in KIT_IMAGE_SPECS}

MODEL_IMAGE_SPECS_BY_KEY: Final = {family_key: spec for spec in MODEL_IMAGE_SPECS for family_key in spec.family_keys}
MODEL_IMAGE_SPECS_BY_NAME: Final = {spec.image_name: spec for spec in MODEL_IMAGE_SPECS}
MODEL_IMAGE_SELECTOR_KEYS: Final = tuple(family_key for spec in MODEL_IMAGE_SPECS for family_key in spec.family_keys)


def get_repo_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parents[2]


@cache
def get_default_image_tag() -> str:
    """Return the default runtime image tag for this installed package."""
    pyproject_path = get_repo_root() / "pyproject.toml"
    if pyproject_path.exists():
        return tomllib.loads(pyproject_path.read_text(encoding="utf-8"))["project"]["version"].strip()
    try:
        return importlib.metadata.version(PACKAGE_NAME).strip()
    except importlib.metadata.PackageNotFoundError:
        raise RuntimeError(
            "Unable to determine the default boileroom image tag. "
            f"Install {PACKAGE_NAME} or set {IMAGE_TAG_ENV} explicitly."
        ) from None


def normalize_requested_tag(tag: str | None) -> str:
    """Return a normalized non-empty tag string."""
    normalized = (tag or get_default_image_tag()).strip()
    if not normalized:
        normalized = get_default_image_tag()
    if normalized == "latest":
        raise ValueError(
            "The 'latest' image tag is no longer published. "
            "Use a concrete package version such as '0.3.0', an alpha tag such as "
            "'0.3.1-alpha.1', or a temporary validation tag such as 'sha-<commit>'."
        )
    return normalized


def normalize_cuda_version(cuda_version: str) -> str:
    """Validate and normalize a CUDA version string."""
    normalized = cuda_version.strip()
    if normalized not in SUPPORTED_CUDA_VERSIONS:
        supported = ", ".join(SUPPORTED_CUDA_VERSIONS)
        raise ValueError(f"Unsupported CUDA version: {cuda_version}. Supported values: {supported}")
    return normalized


def normalize_platform(platform: str) -> str:
    """Return a normalized Docker platform string."""
    normalized = platform.strip().lower()
    if not normalized:
        raise ValueError("Platform must not be empty.")
    aliases = {
        "amd64": "linux/amd64",
        "x86_64": "linux/amd64",
        "arm64": "linux/arm64",
        "aarch64": "linux/arm64",
    }
    return aliases.get(normalized, normalized)


def split_platforms(platforms: str) -> tuple[str, ...]:
    """Return normalized Docker platforms from a comma-separated buildx value."""
    normalized = tuple(normalize_platform(platform) for platform in platforms.split(",") if platform.strip())
    if not normalized:
        raise ValueError("Platform selection must not be empty.")
    return normalized


def current_docker_platform() -> str:
    """Return the Docker Linux platform matching the current host architecture."""
    machine = platform_module.machine().lower()
    normalized = normalize_platform(machine)
    if "/" in normalized:
        return normalized
    return f"linux/{normalized}"


def canonical_image_tag(cuda_version: str, tag: str | None) -> str:
    """Return the canonical CUDA-qualified tag."""
    normalized_cuda = normalize_cuda_version(cuda_version)
    normalized_tag = normalize_requested_tag(tag)
    return f"cuda{normalized_cuda}-{normalized_tag}"


def resolve_registry_tag(tag: str | None) -> str:
    """Resolve a tag for runtime image lookup.

    Unqualified tags such as ``0.3.0``, ``0.3.1-alpha.1``, or ``sha-<commit>`` are
    preserved so they resolve through the published default-CUDA aliases. Explicit
    CUDA-qualified tags are normalized to the canonical ``cuda<version>-<tag>`` form.
    """
    normalized_tag = normalize_requested_tag(tag)
    if match := _CUDA_TAG_PATTERN.fullmatch(normalized_tag):
        cuda_version = normalize_cuda_version(match.group("cuda"))
        tag_suffix = match.group("tag") or get_default_image_tag()
        return canonical_image_tag(cuda_version, tag_suffix)
    return normalized_tag


def published_tags(cuda_version: str, tag: str | None) -> tuple[str, ...]:
    """Return the canonical published tags for a build output.

    The default CUDA line also gets an unqualified alias such as ``0.3.0``,
    ``0.3.1-alpha.1``, or ``sha-<commit>`` for convenience.
    """
    normalized_cuda = normalize_cuda_version(cuda_version)
    normalized_tag = normalize_requested_tag(tag)
    tags = [canonical_image_tag(normalized_cuda, normalized_tag)]
    if normalized_cuda == DEFAULT_CUDA_VERSION:
        tags.append(normalized_tag)
    return tuple(tags)


def get_docker_repository() -> str:
    """Return the Docker repository namespace for published runtime images."""
    override = os.environ.get(DOCKER_REPOSITORY_ENV)
    if override is None or not override.strip():
        return DEFAULT_DOCKER_REPOSITORY

    return normalize_docker_repository(override)


def normalize_docker_repository(repository: str) -> str:
    """Return a normalized Docker Hub repository namespace."""
    normalized = repository.strip().removesuffix("/")
    if not normalized:
        raise ValueError("Docker repository must not be empty.")
    if normalized == "docker.io" or ("." in normalized and "/" not in normalized):
        raise ValueError("Docker repository must use the form 'docker.io/<repository>'.")
    if "/" in normalized and "." in normalized.split("/", 1)[0] and not normalized.startswith("docker.io/"):
        raise ValueError("Docker repository must use the form 'docker.io/<repository>'.")
    if not normalized.startswith("docker.io/"):
        normalized = f"docker.io/{normalized}"
    if (
        re.fullmatch(
            r"docker\.io/(?:[a-z0-9]+(?:[._-][a-z0-9]+)*)(?:/(?:[a-z0-9]+(?:[._-][a-z0-9]+)*))*",
            normalized,
        )
        is None
    ):
        raise ValueError("Docker repository must use the form 'docker.io/<repository>'.")
    return normalized


def format_image_reference(image_name: str, tag: str | None = None, docker_repository: str | None = None) -> str:
    """Return a fully qualified Docker image reference."""
    resolved_tag = resolve_registry_tag(tag)
    repository = (
        get_docker_repository() if docker_repository is None else normalize_docker_repository(docker_repository)
    )
    return f"{repository}/{image_name}:{resolved_tag}"


def published_image_references(
    image_name: str,
    cuda_version: str,
    tag: str | None,
    docker_repository: str | None = None,
) -> tuple[str, ...]:
    """Return all published references for a built image."""
    repository = (
        get_docker_repository() if docker_repository is None else normalize_docker_repository(docker_repository)
    )
    return tuple(f"{repository}/{image_name}:{published_tag}" for published_tag in published_tags(cuda_version, tag))


def get_image_tag() -> str:
    """Return the shared runtime Docker image tag."""
    return resolve_registry_tag(os.environ.get(IMAGE_TAG_ENV))


def get_modal_base_image_reference() -> str:
    """Return the base image Modal should use for the shared runtime layer."""
    return format_image_reference(BASE_IMAGE_SPEC.image_name, get_image_tag())


def get_model_image_spec(identifier: str) -> RuntimeImageSpec:
    """Return a model image spec by image name or short key."""
    if identifier in MODEL_IMAGE_SPECS_BY_KEY:
        return MODEL_IMAGE_SPECS_BY_KEY[identifier]
    if identifier in MODEL_IMAGE_SPECS_BY_NAME:
        return MODEL_IMAGE_SPECS_BY_NAME[identifier]
    raise KeyError(f"Unknown image spec: {identifier}")


def get_kit_image_spec(identifier: str) -> RuntimeImageSpec:
    """Return the kit image spec by family key or image name."""
    if identifier in KIT_IMAGE_SPECS_BY_KEY:
        return KIT_IMAGE_SPECS_BY_KEY[identifier]
    for spec in KIT_IMAGE_SPECS:
        if spec.image_name == identifier:
            return spec
    raise KeyError(f"Unknown kit image spec: {identifier}")


def get_kit_image_source() -> str:
    """Return where Modal gets a kit image: ``build`` (from the Dockerfile) or ``registry`` (a published image)."""
    source = (os.environ.get(KIT_IMAGE_SOURCE_ENV) or DEFAULT_KIT_IMAGE_SOURCE).strip().lower()
    if source not in KIT_IMAGE_SOURCES:
        raise ValueError(f"{KIT_IMAGE_SOURCE_ENV} must be one of {list(KIT_IMAGE_SOURCES)}, got {source!r}")
    return source


def select_model_image_specs(identifiers: Sequence[str] | None) -> tuple[RuntimeImageSpec, ...]:
    """Return the requested model image specs in stable, de-duplicated order."""
    if not identifiers:
        return MODEL_IMAGE_SPECS

    selected: list[RuntimeImageSpec] = []
    seen_image_names: set[str] = set()
    for identifier in identifiers:
        spec = get_model_image_spec(identifier)
        if spec.image_name in seen_image_names:
            continue
        selected.append(spec)
        seen_image_names.add(spec.image_name)
    return tuple(selected)


@cache
def _load_image_config(config_relative_path: str | None) -> dict[str, object]:
    """Return the YAML config for a runtime image spec."""
    if config_relative_path is None:
        return {}

    config_path = get_repo_root() / config_relative_path
    if not config_path.exists():
        return {}

    import yaml

    return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}


@cache
def get_supported_cuda(spec: RuntimeImageSpec) -> tuple[str, ...]:
    """Return supported CUDA versions for a runtime image spec."""
    config = _load_image_config(spec.config_relative_path)
    raw_supported_cuda = config.get("supported_cuda", [])
    if isinstance(raw_supported_cuda, list):
        supported_cuda = [normalize_cuda_version(str(value)) for value in raw_supported_cuda]
    elif raw_supported_cuda:
        supported_cuda = [normalize_cuda_version(str(raw_supported_cuda))]
    else:
        supported_cuda = []

    if not supported_cuda:
        return SUPPORTED_CUDA_VERSIONS
    return tuple(supported_cuda)


@cache
def get_supported_platforms(spec: RuntimeImageSpec) -> tuple[str, ...]:
    """Return supported Docker platforms for a runtime image spec."""
    config = _load_image_config(spec.config_relative_path)
    raw_supported_platforms = config.get("supported_platforms", [])
    if isinstance(raw_supported_platforms, list):
        supported_platforms = [normalize_platform(str(value)) for value in raw_supported_platforms]
    elif raw_supported_platforms:
        supported_platforms = [normalize_platform(str(raw_supported_platforms))]
    else:
        supported_platforms = []

    if not supported_platforms:
        return SUPPORTED_PLATFORMS
    return tuple(supported_platforms)


def render_modal_runtime_env(spec: RuntimeImageSpec, model_dir: str) -> dict[str, str]:
    """Render runtime environment overrides for Modal."""
    env = {
        "MODEL_DIR": model_dir,
        DOCKER_REPOSITORY_ENV: get_docker_repository(),
        IMAGE_TAG_ENV: get_image_tag(),
    }
    for key, value in spec.modal_runtime_env:
        env[key] = value.replace("{MODEL_DIR}", model_dir)
    return env
