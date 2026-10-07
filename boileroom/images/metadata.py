"""Shared image metadata and tag helpers for Docker, Modal, and Apptainer."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform as platform_module
import re
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from types import MappingProxyType
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
# Names a kit image by tag instead of the pinned digest below. BOILEROOM_IMAGE_TAG never applies to kit images: their
# tags are not part of the CUDA-qualified release scheme of the stock images.
KIT_IMAGE_TAG_ENV: Final = "BOILEROOM_KIT_IMAGE_TAG"

# The optimization kit behind ``optimization="exact"`` and ``"fast"``: github.com/anthropics/uplifting-biomolecular-modeling
# (Apache-2.0). The kit Dockerfiles under ``boileroom/models/<family>/kit/`` and the OpenDDE Dockerfile fetch exactly
# this commit; their ``KIT_COMMIT`` build arg must stay equal to the constant below (tests/contracts/test_kit_images.py
# checks it).
KIT_COMMIT: Final = "f4f62fa6592ae4938d49b1757bea0cfeff9f468e"

# Content digests of the published kit images under DEFAULT_DOCKER_REPOSITORY, keyed by image name. Pulling by digest
# makes a kit run reproducible: a re-pushed tag cannot change what ``optimization="exact"`` / ``"fast"`` executes.
# Both images were built from KIT_COMMIT f4f62fa. Refresh these digests whenever the kit images are re-pushed (and
# re-run the GPU validation for both kit modes). They predate the in-image REQUIRE_FAST environment variable and smoke
# step of the kit Dockerfiles; the code-level guards (boileroom.optimization) still refuse a run whose fast path is
# missing. BOILEROOM_KIT_IMAGE_TAG overrides the digest with a tag.
KIT_IMAGE_DIGESTS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "boileroom-esmfold2-kit": "sha256:b6111c000142945e83ae2ceaef90d12857a315ed037286d6e3a1ad23d2518e3b",
        "boileroom-protenix-kit": "sha256:779a2040a2ffa3c4c05c801b44a776896073a0633d72a9a578beece694638436",
    }
)

# Every digest KIT_IMAGE_DIGESTS has ever pinned, per image, oldest first. Append-only: when a pin changes, add the new
# digest here and keep the old ones. A released boileroom still pulls the digest it pinned, so the Docker Hub cleanup
# (scripts/images/cleanup_dockerhub_tags.py) keeps every tag that names one of these digests and never lets an old pin
# become untagged. tests/contracts/test_cleanup_dockerhub_tags.py checks the history against the current pins and the
# git history of this file.
KIT_IMAGE_DIGEST_HISTORY: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "boileroom-esmfold2-kit": ("sha256:b6111c000142945e83ae2ceaef90d12857a315ed037286d6e3a1ad23d2518e3b",),
        "boileroom-protenix-kit": ("sha256:779a2040a2ffa3c4c05c801b44a776896073a0633d72a9a578beece694638436",),
    }
)

SUPPORTED_CUDA_VERSIONS: Final[tuple[str, ...]] = ("11.8", "12.6")
SUPPORTED_PLATFORMS: Final[tuple[str, ...]] = ("linux/amd64", "linux/arm64")

CUDA_TORCH_WHEEL_INDEX: Final[dict[str, str]] = {
    "11.8": "https://download.pytorch.org/whl/cu118",
    "12.6": "https://download.pytorch.org/whl/cu126",
}

_CUDA_TAG_PATTERN = re.compile(r"^cuda(?P<cuda>\d+\.\d+)(?:-(?P<tag>.+))?$")
# Docker's tag grammar: up to 128 characters of [A-Za-z0-9_.-], not starting with '.' or '-'.
_DOCKER_TAG_PATTERN = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$")


@dataclass(frozen=True)
class InterpreterSmokeCheck:
    """A CPU-only import smoke check for a second interpreter inside a runtime image.

    Some images run the model in an isolated virtualenv next to the boileroom interpreter (OpenDDE runs its stack under
    ``/opt/opendde/bin/python``). The image smoke check runs these imports through that interpreter, with
    ``library_path`` prepended to ``LD_LIBRARY_PATH`` the way the core's worker environment does.

    Attributes
    ----------
    python : str
        Absolute path of the interpreter inside the image.
    imports : tuple[str, ...]
        Modules that must import on a CPU-only host.
    library_path : str | None
        Directory prepended to ``LD_LIBRARY_PATH`` for the interpreter, or None.
    driver_linked_libraries : tuple[tuple[str, str], ...]
        ``(package, filename)`` pairs for shared libraries that link against the NVIDIA driver and so cannot be imported
        without a GPU. The check finds ``filename`` under the installed ``package`` without importing it and requires
        that ``ldd`` leaves only ``allowed_unresolved`` libraries unresolved.
    allowed_unresolved : tuple[str, ...]
        Libraries a driver-linked library may leave unresolved on a host without the NVIDIA driver.
    required_symbol_versions : tuple[tuple[str, str], ...]
        ``(library file, version string)`` pairs, such as a GLIBCXX version a bundled libstdc++ must provide.
    """

    python: str
    imports: tuple[str, ...]
    library_path: str | None = None
    driver_linked_libraries: tuple[tuple[str, str], ...] = ()
    allowed_unresolved: tuple[str, ...] = ("libcuda.so.1",)
    required_symbol_versions: tuple[tuple[str, str], ...] = ()


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
    # Extra interpreters inside the image whose imports the image smoke check runs too.
    interpreter_smoke_checks: tuple[InterpreterSmokeCheck, ...] = ()

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
        # No LAYERNORM_TYPE: the core's worker sets it per optimization mode (OPENDDE_LAYERNORM in opendde/core.py).
        # OpenDDE and the kit live in a standalone Python 3.11 install the system interpreter never imports.
        interpreter_smoke_checks=(
            InterpreterSmokeCheck(
                python="/opt/opendde/bin/python",
                library_path="/opt/opendde/lib",
                imports=(
                    "torch",
                    "cuequivariance_torch",
                    "opendde_opt",
                    "opendde_opt.lnstream",
                    "opendde_opt.lncensus",
                    "runner.batch_inference",
                ),
                # cuequivariance_ops_torch needs libcuda.so.1 to import; check its kernel library links instead.
                driver_linked_libraries=(("cuequivariance_ops", "libcue_ops.so"),),
                # The kit's bit-exact ``exact`` kernels need GCC 13's libstdc++ (see the Dockerfile).
                required_symbol_versions=(("/opt/opendde/lib/libstdc++.so.6", "GLIBCXX_3.4.32"),),
            ),
        ),
    ),
)

# Kit runtime images: opt-in variants that only ``optimization != "vanilla"`` uses, never part of MODEL_IMAGE_SPECS, so
# the CI build matrix does not build them. They are built by hand (or by Modal, from the Dockerfile) and pushed once per
# KIT_COMMIT; runtimes pull them by the digest in KIT_IMAGE_DIGESTS. The CUDA 13.0 kit stack (an NVIDIA driver 580 or
# newer) is outside SUPPORTED_CUDA_VERSIONS, so their tags carry no CUDA qualifier, and they are amd64 only.
KIT_IMAGE_SPECS: Final[tuple[RuntimeImageSpec, ...]] = (
    RuntimeImageSpec(
        key="esmfold2",
        image_name="boileroom-esmfold2-kit",
        dockerfile_relative_path="boileroom/models/esmfold2/kit/Dockerfile",
        context_relative_path="boileroom/models/esmfold2/kit",
    ),
    RuntimeImageSpec(
        key="protenix",
        image_name="boileroom-protenix-kit",
        dockerfile_relative_path="boileroom/models/protenix/kit/Dockerfile",
        context_relative_path="boileroom/models/protenix/kit",
        python_version="3.11",
        # The kit's layernorm lever and its pinned environment need stock's fast LayerNorm (the vanilla image uses openfold).
        modal_runtime_env=(("LAYERNORM_TYPE", "fast_layernorm"),),
    ),
)
KIT_IMAGE_SPECS_BY_KEY: Final = {spec.key: spec for spec in KIT_IMAGE_SPECS}
KIT_IMAGE_NAMES: Final[frozenset[str]] = frozenset(spec.image_name for spec in KIT_IMAGE_SPECS)

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


def get_kit_image_spec(key: str) -> RuntimeImageSpec:
    """Return the kit image spec of a model family.

    Parameters
    ----------
    key : str
        Model family key, such as ``"esmfold2"`` or ``"protenix"``.

    Returns
    -------
    RuntimeImageSpec
        The kit image spec.

    Raises
    ------
    KeyError
        If the family has no kit image.
    """
    try:
        return KIT_IMAGE_SPECS_BY_KEY[key]
    except KeyError:
        raise KeyError(
            f"Unknown kit image spec: {key!r}; kit images exist for {sorted(KIT_IMAGE_SPECS_BY_KEY)}"
        ) from None


def is_kit_image_spec(spec: RuntimeImageSpec) -> bool:
    """Return whether ``spec`` describes a kit image rather than a stock model image.

    Decided by image name, so a modified copy of a kit spec (``dataclasses.replace``) is still a kit image.
    """
    return spec.image_name in KIT_IMAGE_NAMES


def normalize_kit_image_tag(tag: str | None) -> str | None:
    """Return a validated kit image tag, or None when ``tag`` is empty.

    Parameters
    ----------
    tag : str | None
        A Docker tag such as ``sha-dc652b0`` or ``kit-f4f62fa``. Kit tags are used verbatim: no CUDA qualifier.

    Returns
    -------
    str | None
        The stripped tag, or None for None or a blank string.

    Raises
    ------
    ValueError
        If the tag is ``latest`` or not a valid Docker tag.
    """
    normalized = (tag or "").strip()
    if not normalized:
        return None
    if normalized == "latest":
        raise ValueError("The 'latest' tag is not used for kit images; name a concrete tag or use the pinned digest.")
    if _DOCKER_TAG_PATTERN.fullmatch(normalized) is None:
        raise ValueError(
            f"Invalid kit image tag {normalized!r}: a Docker tag is up to 128 characters of [A-Za-z0-9_.-]."
        )
    return normalized


def get_kit_image_tag() -> str | None:
    """Return the kit image tag from BOILEROOM_KIT_IMAGE_TAG, or None to use the pinned digest."""
    return normalize_kit_image_tag(os.environ.get(KIT_IMAGE_TAG_ENV))


def kit_image_reference(spec: RuntimeImageSpec, tag: str | None = None, docker_repository: str | None = None) -> str:
    """Return the registry reference of a kit image.

    Parameters
    ----------
    spec : RuntimeImageSpec
        A kit image spec from KIT_IMAGE_SPECS.
    tag : str | None
        An explicit tag. When None, BOILEROOM_KIT_IMAGE_TAG applies; when that is unset too, the image is named by its
        pinned digest from KIT_IMAGE_DIGESTS.
    docker_repository : str | None
        Repository namespace; defaults to BOILEROOM_DOCKER_REPOSITORY or DEFAULT_DOCKER_REPOSITORY.

    Returns
    -------
    str
        ``<repository>/<image>:<tag>`` or ``<repository>/<image>@sha256:<digest>``.

    Raises
    ------
    ValueError
        If ``spec`` is not a kit image spec, or the tag is invalid.
    """
    if not is_kit_image_spec(spec):
        raise ValueError(f"{spec.image_name} is not a kit image")
    repository = (
        get_docker_repository() if docker_repository is None else normalize_docker_repository(docker_repository)
    )
    resolved_tag = normalize_kit_image_tag(tag) or get_kit_image_tag()
    if resolved_tag is not None:
        return f"{repository}/{spec.image_name}:{resolved_tag}"
    return f"{repository}/{spec.image_name}@{KIT_IMAGE_DIGESTS[spec.image_name]}"


def _is_build_input(path: Path) -> bool:
    """Return whether a file under a Docker context is a build input (not a bytecode cache)."""
    return path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"


def kit_build_reference(spec: RuntimeImageSpec, build_inputs: Mapping[str, str] | None = None) -> str:
    """Return a deterministic reference for a kit image built from its Dockerfile in this repository.

    Parameters
    ----------
    spec : RuntimeImageSpec
        The kit image spec.
    build_inputs : Mapping[str, str] | None
        Every other input of the build, by name: build args, the environment and source of a build step run on top of
        the Dockerfile, and so on (``boileroom.images.modal`` lists them for each kit).

    Returns
    -------
    str
        ``build:<dockerfile path>@<hash>``, where the 12-hex hash covers the Dockerfile, every file of the build
        context (relative path and bytes) and the build inputs, so it changes whenever a build input in this repository
        does.
    """
    repo_root = get_repo_root()
    files = {spec.dockerfile_path, *(path for path in spec.context_path.rglob("*") if _is_build_input(path))}
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda path: path.relative_to(repo_root).as_posix()):
        digest.update(path.relative_to(repo_root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    for name, value in sorted((build_inputs or {}).items()):
        digest.update(f"{name}={value}\0".encode())
    return f"build:{spec.dockerfile_relative_path}@{digest.hexdigest()[:12]}"


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
    """Render runtime environment overrides for Modal.

    The image lookup overrides are baked in so an import inside the container resolves the same image. A stock image
    carries BOILEROOM_IMAGE_TAG; a kit image carries BOILEROOM_KIT_IMAGE_SOURCE and, when set, BOILEROOM_KIT_IMAGE_TAG.
    """
    env = {"MODEL_DIR": model_dir, DOCKER_REPOSITORY_ENV: get_docker_repository()}
    if is_kit_image_spec(spec):
        env[KIT_IMAGE_SOURCE_ENV] = get_kit_image_source()
        if (kit_tag := get_kit_image_tag()) is not None:
            env[KIT_IMAGE_TAG_ENV] = kit_tag
    else:
        env[IMAGE_TAG_ENV] = get_image_tag()
    for key, value in spec.modal_runtime_env:
        env[key] = value.replace("{MODEL_DIR}", model_dir)
    return env
