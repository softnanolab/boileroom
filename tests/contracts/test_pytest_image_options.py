"""Contract tests for pytest image lookup options."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

import pytest

from boileroom.images.metadata import (
    DOCKER_REPOSITORY_ENV,
    IMAGE_TAG_ENV,
    KIT_IMAGE_DIGESTS,
    KIT_IMAGE_SOURCE_ENV,
    KIT_IMAGE_TAG_ENV,
    format_image_reference,
)


def _load_test_conftest() -> Any:
    path = Path(__file__).parents[1] / "conftest.py"
    spec = importlib.util.spec_from_file_location("boileroom_test_conftest", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_BACKEND_DEFAULTS: dict[str, str | None] = {
    "--backend": "modal",
    "--gpu": None,
    "--device": None,
    "--image-tag": None,
    "--kit-image-tag": None,
}


class FakeConfig:
    def __init__(self, options: dict[str, str | None]) -> None:
        self.options = {**_BACKEND_DEFAULTS, **options}

    def getoption(self, name: str) -> str | None:
        return self.options.get(name)


def test_docker_user_sets_repository_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--docker-user`` should write the normalized repository to the env var."""
    monkeypatch.delenv(DOCKER_REPOSITORY_ENV, raising=False)
    conftest = _load_test_conftest()

    try:
        conftest.pytest_configure(FakeConfig({"--docker-user": "phauglin"}))
        assert os.environ[DOCKER_REPOSITORY_ENV] == "docker.io/phauglin"
    finally:
        os.environ.pop(DOCKER_REPOSITORY_ENV, None)


def test_docker_user_absent_leaves_repository_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without ``--docker-user``, the repository env var should not be touched."""
    monkeypatch.delenv(DOCKER_REPOSITORY_ENV, raising=False)
    conftest = _load_test_conftest()

    conftest.pytest_configure(FakeConfig({"--docker-user": None}))

    assert DOCKER_REPOSITORY_ENV not in os.environ


def test_image_tag_sets_runtime_lookup_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--image-tag`` should write the shared runtime image tag env var."""
    monkeypatch.delenv(IMAGE_TAG_ENV, raising=False)
    conftest = _load_test_conftest()

    try:
        conftest.pytest_configure(FakeConfig({"--image-tag": "sha-test"}))
        assert os.environ[IMAGE_TAG_ENV] == "sha-test"
    finally:
        os.environ.pop(IMAGE_TAG_ENV, None)


class FakeRequest:
    def __init__(self, config: FakeConfig) -> None:
        self.config = config


def test_image_tag_reaches_apptainer_through_the_env_not_the_backend_string(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--image-tag`` names stock images only, so it must not become an inline ``apptainer:<tag>``.

    An inline tag also names the kit image, so inlining the stock tag would point kit modes at a stock tag.
    """
    monkeypatch.delenv(IMAGE_TAG_ENV, raising=False)
    conftest = _load_test_conftest()
    config = FakeConfig({"--backend": "apptainer", "--image-tag": "sha-test"})

    try:
        conftest.pytest_configure(config)
        assert conftest.backend_option.__wrapped__(FakeRequest(config)) == "apptainer"
        assert os.environ[IMAGE_TAG_ENV] == "sha-test"
    finally:
        os.environ.pop(IMAGE_TAG_ENV, None)


@pytest.mark.parametrize(
    ("backend", "kit_source"), [("apptainer", None), ("modal", "registry")], ids=["apptainer", "modal-registry"]
)
def test_kit_image_tag_sets_the_kit_lookup_env(
    monkeypatch: pytest.MonkeyPatch, backend: str, kit_source: str | None
) -> None:
    monkeypatch.delenv(KIT_IMAGE_TAG_ENV, raising=False)
    monkeypatch.delenv(IMAGE_TAG_ENV, raising=False)
    if kit_source is None:
        monkeypatch.delenv(KIT_IMAGE_SOURCE_ENV, raising=False)
    else:
        monkeypatch.setenv(KIT_IMAGE_SOURCE_ENV, kit_source)
    conftest = _load_test_conftest()

    try:
        conftest.pytest_configure(FakeConfig({"--backend": backend, "--kit-image-tag": " sha-dc652b0 "}))
        assert os.environ[KIT_IMAGE_TAG_ENV] == "sha-dc652b0"
        assert IMAGE_TAG_ENV not in os.environ
    finally:
        os.environ.pop(KIT_IMAGE_TAG_ENV, None)


@pytest.mark.parametrize("kit_source", [None, "build", " Build "])
def test_kit_image_tag_on_modal_building_the_kits_is_a_usage_error(
    monkeypatch: pytest.MonkeyPatch, kit_source: str | None
) -> None:
    """Modal builds the kit images by default and never reads a kit tag, so the tag used to be silently ignored."""
    monkeypatch.delenv(KIT_IMAGE_TAG_ENV, raising=False)
    if kit_source is None:
        monkeypatch.delenv(KIT_IMAGE_SOURCE_ENV, raising=False)
    else:
        monkeypatch.setenv(KIT_IMAGE_SOURCE_ENV, kit_source)
    conftest = _load_test_conftest()

    with pytest.raises(pytest.UsageError, match=f"{KIT_IMAGE_SOURCE_ENV}=registry"):
        conftest.pytest_configure(FakeConfig({"--kit-image-tag": "sha-dc652b0"}))
    assert KIT_IMAGE_TAG_ENV not in os.environ


def test_an_invalid_kit_image_source_on_modal_is_a_usage_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(KIT_IMAGE_SOURCE_ENV, "dockerfile")
    conftest = _load_test_conftest()

    with pytest.raises(pytest.UsageError, match=KIT_IMAGE_SOURCE_ENV):
        conftest.pytest_configure(FakeConfig({}))
    conftest.pytest_configure(FakeConfig({"--backend": "apptainer"}))  # Apptainer never builds, so never reads it


def _header(conftest: Any, options: dict[str, str | None]) -> str:
    (kit_line,) = [line for line in conftest.pytest_report_header(FakeConfig(options)) if "kit image" in line]
    return kit_line


@pytest.mark.parametrize("kit_tag", [None, "kit-env"])
def test_header_on_modal_says_the_kit_images_are_built(monkeypatch: pytest.MonkeyPatch, kit_tag: str | None) -> None:
    """The header used to name the pinned registry digest (or the kit tag) though Modal builds the kits by default."""
    monkeypatch.delenv(KIT_IMAGE_SOURCE_ENV, raising=False)
    if kit_tag is None:
        monkeypatch.delenv(KIT_IMAGE_TAG_ENV, raising=False)
    else:
        monkeypatch.setenv(KIT_IMAGE_TAG_ENV, kit_tag)
    line = _header(_load_test_conftest(), {})

    assert "built on Modal from boileroom/models/<family>/kit/Dockerfile" in line
    assert "@<pinned digest>" not in line
    assert ("kit-env is ignored" in line) is (kit_tag is not None)


@pytest.mark.parametrize(
    ("backend", "kit_source", "kit_tag", "expected"),
    [
        ("modal", "registry", None, f"-kit@<pinned digest> ({len(KIT_IMAGE_DIGESTS)} pinned)"),
        ("modal", "registry", "kit-env", f"-kit:kit-env ({KIT_IMAGE_TAG_ENV})"),
        ("apptainer", "build", None, "-kit@<pinned digest>"),  # Apptainer always pulls; the source is Modal's
        ("apptainer", None, "kit-env", "-kit:kit-env"),
        ("apptainer:inline", None, "kit-env", "-kit:inline (from --backend tag)"),
    ],
)
def test_header_names_the_registry_kit_image(
    monkeypatch: pytest.MonkeyPatch, backend: str, kit_source: str | None, kit_tag: str | None, expected: str
) -> None:
    for name, value in ((KIT_IMAGE_SOURCE_ENV, kit_source), (KIT_IMAGE_TAG_ENV, kit_tag)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    line = _header(_load_test_conftest(), {"--backend": backend})

    assert expected in line
    assert "built on Modal" not in line


def test_blank_kit_image_tag_leaves_the_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(KIT_IMAGE_TAG_ENV, raising=False)
    conftest = _load_test_conftest()

    conftest.pytest_configure(FakeConfig({"--kit-image-tag": "  "}))

    assert KIT_IMAGE_TAG_ENV not in os.environ


@pytest.mark.parametrize("tag", ["latest", "not a tag"])
def test_invalid_kit_image_tag_is_a_usage_error(monkeypatch: pytest.MonkeyPatch, tag: str) -> None:
    monkeypatch.delenv(KIT_IMAGE_TAG_ENV, raising=False)
    conftest = _load_test_conftest()

    with pytest.raises(pytest.UsageError, match="--kit-image-tag"):
        conftest.pytest_configure(FakeConfig({"--kit-image-tag": tag}))
    assert KIT_IMAGE_TAG_ENV not in os.environ


@pytest.mark.parametrize(("optimization", "kit"), [("vanilla", False), ("exact", True)])
def test_apptainer_inline_tag_wins_over_image_tag(
    monkeypatch: pytest.MonkeyPatch, optimization: str, kit: bool
) -> None:
    """``--backend apptainer:<tag>`` names the image a wrapper pulls even when ``--image-tag`` is set too.

    Runs the options through ``pytest_configure`` and the ``backend_option`` fixture into the wrapper's image
    resolution, with the Apptainer backend replaced so that nothing is pulled or started.
    """
    import boileroom.backend.apptainer as apptainer_module
    from boileroom.base import ModelWrapper
    from boileroom.models.registry import PROTENIX_SPEC

    image_uris: list[str] = []

    class FakeApptainer:
        def __init__(self, core_class_path: str, image_uri: str, config: dict | None = None, **kwargs: Any) -> None:
            image_uris.append(image_uri)

        def start(self) -> None:
            pass

    monkeypatch.setattr(apptainer_module, "ApptainerBackend", FakeApptainer)
    monkeypatch.delenv(IMAGE_TAG_ENV, raising=False)
    monkeypatch.delenv(KIT_IMAGE_TAG_ENV, raising=False)
    conftest = _load_test_conftest()
    config = FakeConfig({"--backend": "apptainer:inline", "--image-tag": "sha-test"})

    try:
        conftest.pytest_configure(config)
        backend = conftest.backend_option.__wrapped__(FakeRequest(config))
        ModelWrapper()._initialize_backend_from_spec(
            PROTENIX_SPEC, backend=backend, config={"optimization": optimization}
        )
    finally:
        os.environ.pop(IMAGE_TAG_ENV, None)

    (image_uri,) = image_uris
    assert "sha-test" not in image_uri
    if kit:
        assert image_uri == "docker://docker.io/jakublala/boileroom-protenix-kit:inline"
    else:
        assert PROTENIX_SPEC.apptainer_image_name is not None
        assert image_uri == f"docker://{format_image_reference(PROTENIX_SPEC.apptainer_image_name, 'inline')}"
