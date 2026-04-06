"""Fast contract tests for shared image metadata and tag behavior."""

from boileroom.images.import_checks import compute_cuda_versions, package_name_to_import_name
from boileroom.images.metadata import (
    DEFAULT_IMAGE_TAG,
    format_image_reference,
    get_modal_image_tag,
    get_model_image_spec,
    get_supported_cuda,
    published_tags,
    resolve_registry_tag,
)


def test_published_tags_include_default_cuda_aliases() -> None:
    """Default CUDA images should publish both canonical and alias tags."""
    assert published_tags("12.6", "latest") == ("cuda12.6-latest", "latest")
    assert published_tags("12.6", "0.3.0") == ("cuda12.6-0.3.0", "0.3.0")
    assert published_tags("11.8", "latest") == ("cuda11.8-latest",)


def test_registry_tag_resolution_preserves_aliases_and_normalizes_explicit_cuda() -> None:
    """Runtime tag resolution should preserve aliases while normalizing explicit CUDA tags."""
    assert resolve_registry_tag(None) == DEFAULT_IMAGE_TAG
    assert resolve_registry_tag("latest") == "latest"
    assert resolve_registry_tag("0.3.0") == "0.3.0"
    assert resolve_registry_tag("cuda12.6") == "cuda12.6-latest"
    assert resolve_registry_tag("cuda11.8-0.3.0") == "cuda11.8-0.3.0"


def test_model_specs_report_supported_cuda_from_config() -> None:
    """Model image specs should report the CUDA variants advertised in config.yaml."""
    assert get_supported_cuda(get_model_image_spec("boltz")) == ("12.6",)
    assert get_supported_cuda(get_model_image_spec("chai")) == ("11.8", "12.6")
    assert get_supported_cuda(get_model_image_spec("esm")) == ("11.8", "12.6")


def test_modal_tag_uses_env_override(monkeypatch) -> None:
    """Modal image lookups should respect an optional tag override."""
    monkeypatch.delenv("BOILEROOM_MODAL_IMAGE_TAG", raising=False)
    assert get_modal_image_tag() == "latest"

    monkeypatch.setenv("BOILEROOM_MODAL_IMAGE_TAG", "0.3.0")
    assert get_modal_image_tag() == "0.3.0"

    monkeypatch.setenv("BOILEROOM_MODAL_IMAGE_TAG", "cuda12.6")
    assert get_modal_image_tag() == "cuda12.6-latest"


def test_format_image_reference_uses_central_namespace() -> None:
    """All runtime image references should use the central Docker namespace."""
    assert format_image_reference("boileroom-boltz", "latest") == "docker.io/jakublala/boileroom-boltz:latest"


def test_compute_cuda_versions_uses_metadata_defaults() -> None:
    """Smoke checks should derive the all-CUDA list from shared metadata."""
    assert compute_cuda_versions(None, True) == ["11.8", "12.6"]
    assert compute_cuda_versions(["12.6"], False) == ["12.6"]


def test_package_name_to_import_name_handles_overrides_and_hyphens() -> None:
    """Import-name resolution should keep overrides centralized and predictable."""
    assert package_name_to_import_name("pytorch-lightning") == "pytorch_lightning"
    assert package_name_to_import_name("torch-tensorrt") is None
    assert package_name_to_import_name("my-package") == "my_package"
