"""Fast contract tests for shared image metadata and tag behavior."""

import importlib.metadata
from pathlib import Path

import pytest

from boileroom.images.import_checks import (
    compute_cuda_versions,
    iter_image_targets,
    package_name_to_import_name,
    requirement_import_names,
    requirement_line_to_package_name,
)
from boileroom.images.metadata import (
    IMAGE_TAG_ENV,
    format_image_reference,
    get_default_image_tag,
    get_docker_repository,
    get_image_tag,
    get_modal_base_image_reference,
    get_model_image_spec,
    get_supported_cuda,
    get_supported_platforms,
    normalize_docker_repository,
    published_tags,
    render_modal_runtime_env,
    resolve_registry_tag,
    split_platforms,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_published_tags_include_default_cuda_aliases() -> None:
    """Default CUDA images should publish both canonical and alias tags."""
    assert published_tags("12.6", "0.3.0") == ("cuda12.6-0.3.0", "0.3.0")
    assert published_tags("12.6", "sha-abcd1234") == ("cuda12.6-sha-abcd1234", "sha-abcd1234")
    assert published_tags("11.8", "0.3.0") == ("cuda11.8-0.3.0",)


def test_registry_tag_resolution_preserves_aliases_and_normalizes_explicit_cuda() -> None:
    """Runtime tag resolution should preserve aliases while normalizing explicit CUDA tags."""
    assert resolve_registry_tag(None) == get_default_image_tag()
    assert resolve_registry_tag("0.3.0") == "0.3.0"
    assert resolve_registry_tag("cuda12.6") == f"cuda12.6-{get_default_image_tag()}"
    assert resolve_registry_tag("cuda11.8-0.3.0") == "cuda11.8-0.3.0"
    with pytest.raises(ValueError, match="latest"):
        resolve_registry_tag("latest")


def test_default_image_tag_matches_installed_package_version() -> None:
    """The runtime default image tag should track the installed package version."""
    assert get_default_image_tag() == importlib.metadata.version("boileroom")


def test_model_specs_report_supported_cuda_from_config() -> None:
    """Model image specs should report the CUDA variants advertised in config.yaml."""
    assert get_supported_cuda(get_model_image_spec("alphafold")) == ("12.6",)
    assert get_supported_cuda(get_model_image_spec("boltz")) == ("12.6",)
    assert get_supported_cuda(get_model_image_spec("chai")) == ("11.8", "12.6")
    assert get_supported_cuda(get_model_image_spec("esm")) == ("11.8", "12.6")
    assert get_supported_cuda(get_model_image_spec("protenix")) == ("12.6",)


def test_model_specs_report_supported_platforms_from_config() -> None:
    """Model image specs should report the Docker platforms advertised in config.yaml."""
    assert get_supported_platforms(get_model_image_spec("alphafold")) == ("linux/amd64",)
    assert get_supported_platforms(get_model_image_spec("boltz")) == ("linux/amd64", "linux/arm64")
    assert get_supported_platforms(get_model_image_spec("chai")) == ("linux/amd64", "linux/arm64")
    assert get_supported_platforms(get_model_image_spec("esm")) == ("linux/amd64", "linux/arm64")
    assert get_supported_platforms(get_model_image_spec("protenix")) == ("linux/amd64",)


def test_alphafold_biopython_pin_supports_python312_image_builds() -> None:
    """AlphaFold image requirements should avoid Biopython releases broken on Python 3.12."""
    requirements = (REPO_ROOT / "boileroom" / "models" / "alphafold" / "requirements.txt").read_text(encoding="utf-8")
    pin = next(line for line in requirements.splitlines() if line.startswith("biopython=="))
    major, minor, *_ = (int(part) for part in pin.split("==", 1)[1].split("."))

    assert (major, minor) >= (1, 83)


def test_protenix_image_uses_non_jit_layernorm_by_default() -> None:
    """Protenix images should avoid runtime CUDA-extension JIT compilation."""
    dockerfile = (REPO_ROOT / "boileroom" / "models" / "protenix" / "Dockerfile").read_text(encoding="utf-8")

    assert "LAYERNORM_TYPE=openfold" in dockerfile


def test_split_platforms_normalizes_arch_aliases() -> None:
    """Build and smoke helpers should normalize common Docker platform shorthands."""
    assert split_platforms("amd64,arm64") == ("linux/amd64", "linux/arm64")
    assert split_platforms(" linux/amd64 , aarch64 ") == ("linux/amd64", "linux/arm64")


def test_image_tag_uses_env_override(monkeypatch) -> None:
    """Runtime image lookups should respect an optional tag override."""
    monkeypatch.delenv(IMAGE_TAG_ENV, raising=False)
    assert get_image_tag() == get_default_image_tag()

    monkeypatch.setenv(IMAGE_TAG_ENV, "0.3.0")
    assert get_image_tag() == "0.3.0"

    monkeypatch.setenv(IMAGE_TAG_ENV, "cuda12.6")
    assert get_image_tag() == f"cuda12.6-{get_default_image_tag()}"


def test_modal_base_image_reference_uses_repository_and_tag(monkeypatch) -> None:
    """Modal base image lookups should use the same repository and tag as model images."""
    monkeypatch.setenv("BOILEROOM_DOCKER_REPOSITORY", "docker.io/example")
    monkeypatch.setenv(IMAGE_TAG_ENV, "0.3.0.1")

    assert get_modal_base_image_reference() == "docker.io/example/boileroom-base:0.3.0.1"


def test_modal_runtime_env_carries_image_lookup_overrides(monkeypatch) -> None:
    """Modal containers should re-import wrappers with the same resolved image lookup settings."""
    monkeypatch.setenv("BOILEROOM_DOCKER_REPOSITORY", "docker.io/example")
    monkeypatch.setenv(IMAGE_TAG_ENV, "0.3.0.1")

    env = render_modal_runtime_env(get_model_image_spec("boltz"), "/mnt/models")

    assert env["BOILEROOM_DOCKER_REPOSITORY"] == "docker.io/example"
    assert env[IMAGE_TAG_ENV] == "0.3.0.1"
    assert set(env) == {"MODEL_DIR", "BOILEROOM_DOCKER_REPOSITORY", IMAGE_TAG_ENV}


def test_protenix_modal_runtime_env_uses_non_jit_layernorm(monkeypatch) -> None:
    """Modal Protenix should avoid runtime CUDA-extension JIT compilation."""
    monkeypatch.setenv("BOILEROOM_DOCKER_REPOSITORY", "docker.io/example")
    monkeypatch.setenv(IMAGE_TAG_ENV, "0.3.0.1")

    env = render_modal_runtime_env(get_model_image_spec("protenix"), "/mnt/models")

    assert env["LAYERNORM_TYPE"] == "openfold"


def test_docker_repository_uses_env_override(monkeypatch) -> None:
    """Image references should support a shared Docker repository override."""
    monkeypatch.setenv("BOILEROOM_DOCKER_REPOSITORY", "example")
    assert get_docker_repository() == "docker.io/example"
    assert format_image_reference("boileroom-boltz", "0.3.0") == "docker.io/example/boileroom-boltz:0.3.0"


def test_normalize_docker_repository_accepts_user_or_full_namespace() -> None:
    """Docker repository normalization should match CLI shorthand behavior."""
    assert normalize_docker_repository("example") == "docker.io/example"
    assert normalize_docker_repository("docker.io/example/team") == "docker.io/example/team"
    assert normalize_docker_repository("docker.io/example/") == "docker.io/example"


@pytest.mark.parametrize("repository", ["docker.io", "docker.io/", "ghcr.io"])
def test_normalize_docker_repository_rejects_bare_registries(repository: str) -> None:
    """Bare registry hosts should not be treated as Docker Hub users."""
    with pytest.raises(ValueError, match="Docker repository"):
        normalize_docker_repository(repository)


def test_docker_repository_rejects_non_docker_io_override(monkeypatch) -> None:
    """Repository overrides should remain Docker Hub namespaces."""
    monkeypatch.setenv("BOILEROOM_DOCKER_REPOSITORY", "ghcr.io/example")
    with pytest.raises(ValueError, match="Docker repository"):
        get_docker_repository()


@pytest.mark.parametrize(
    ("override", "label"),
    [
        ("docker.io/Example", "uppercase"),
        ("docker.io/example repo", "spaces"),
        ("docker.io/example//repo", "double slash"),
    ],
)
def test_docker_repository_rejects_malformed_docker_io_override(monkeypatch, override: str, label: str) -> None:
    """Repository overrides should reject malformed Docker Hub namespaces."""
    monkeypatch.setenv("BOILEROOM_DOCKER_REPOSITORY", override)
    with pytest.raises(ValueError, match="Docker repository"):
        get_docker_repository()


def test_format_image_reference_uses_central_namespace() -> None:
    """All runtime image references should use the central Docker namespace."""
    assert format_image_reference("boileroom-boltz", None) == (
        f"docker.io/jakublala/boileroom-boltz:{get_default_image_tag()}"
    )


def test_compute_cuda_versions_uses_metadata_defaults() -> None:
    """Smoke checks should derive the all-CUDA list from shared metadata."""
    assert compute_cuda_versions(None, True) == ["11.8", "12.6"]
    assert compute_cuda_versions(["12.6"], False) == ["12.6"]


def test_package_name_to_import_name_handles_overrides_and_hyphens() -> None:
    """Import-name resolution should keep overrides centralized and predictable."""
    assert package_name_to_import_name("absl-py") == "absl"
    assert package_name_to_import_name("biopython") == "Bio"
    assert package_name_to_import_name("dm-haiku") == "haiku"
    assert package_name_to_import_name("ml-collections") == "ml_collections"
    assert package_name_to_import_name("tensorflow-cpu") == "tensorflow"
    assert package_name_to_import_name("pytorch-lightning") == "pytorch_lightning"
    assert package_name_to_import_name("torch-tensorrt") is None
    assert package_name_to_import_name("my-package") == "my_package"


def test_requirement_line_to_package_name_skips_pip_options() -> None:
    """Requirements parser should ignore index/option lines in image smoke checks."""
    assert requirement_line_to_package_name("-f https://example.test/simple") is None
    assert requirement_line_to_package_name("--extra-index-url https://example.test/simple") is None
    assert requirement_line_to_package_name("jaxlib==0.4.26+cuda12.cudnn89") == "jaxlib"
    assert requirement_line_to_package_name("openmm[cuda12]==8.2.0") == "openmm"


def test_requirement_import_names_uses_central_parser(tmp_path) -> None:
    """Image import checks should not treat pip option lines as packages."""
    requirements_path = tmp_path / "requirements.txt"
    requirements_path.write_text(
        "\n".join(
            [
                "-f https://example.test/simple",
                "absl-py==1.0.0",
                "openmm[cuda12]==8.2.0",
                "torch-tensorrt==2.0.0",
                "",
            ]
        ),
        encoding="utf-8",
    )

    assert requirement_import_names(requirements_path) == ["absl", "openmm"]


def test_iter_image_targets_uses_canonical_cuda_tags() -> None:
    """Image smoke targets should honor CUDA-qualified tag selection."""
    targets = iter_image_targets("0.3.0", ["12.6"], docker_repository="example")
    references = {image_key: image_reference for image_key, image_reference, *_ in targets}
    assert references["alphafold"].startswith("docker.io/example/")
    assert references["boltz"].startswith("docker.io/example/")
    assert references["chai"].startswith("docker.io/example/")
    assert references["esm"].startswith("docker.io/example/")
    assert references["protenix"].startswith("docker.io/example/")
    assert references["alphafold"].endswith(":cuda12.6-0.3.0")
    assert references["boltz"].endswith(":cuda12.6-0.3.0")
    assert references["chai"].endswith(":cuda12.6-0.3.0")
    assert references["esm"].endswith(":cuda12.6-0.3.0")
    assert references["protenix"].endswith(":cuda12.6-0.3.0")


def test_iter_image_targets_filters_by_platform() -> None:
    """ARM64 smoke checks should skip images that only advertise AMD64 support."""
    targets = iter_image_targets("0.3.0", ["12.6"], docker_repository="example", platform="linux/arm64")
    image_keys = [image_key for image_key, *_ in targets]

    assert image_keys == ["boltz", "chai", "esm"]
