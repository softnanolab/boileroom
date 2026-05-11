"""Contract tests for pytest image lookup options."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

from boileroom.images.metadata import DOCKER_REPOSITORY_ENV, MODAL_IMAGE_TAG_ENV


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
}


class FakeConfig:
    def __init__(self, options: dict[str, str | None]) -> None:
        self.options = {**_BACKEND_DEFAULTS, **options}

    def getoption(self, name: str) -> str | None:
        return self.options.get(name)


def test_docker_user_sets_repository_env(monkeypatch) -> None:
    """``--docker-user`` should write the normalized repository to the env var."""
    monkeypatch.delenv(DOCKER_REPOSITORY_ENV, raising=False)
    conftest = _load_test_conftest()

    try:
        conftest.pytest_configure(FakeConfig({"--docker-user": "phauglin"}))
        assert os.environ[DOCKER_REPOSITORY_ENV] == "docker.io/phauglin"
    finally:
        os.environ.pop(DOCKER_REPOSITORY_ENV, None)


def test_docker_user_absent_leaves_repository_env_unset(monkeypatch) -> None:
    """Without ``--docker-user``, the repository env var should not be touched."""
    monkeypatch.delenv(DOCKER_REPOSITORY_ENV, raising=False)
    conftest = _load_test_conftest()

    conftest.pytest_configure(FakeConfig({"--docker-user": None}))

    assert DOCKER_REPOSITORY_ENV not in os.environ


def test_image_tag_sets_modal_lookup_env(monkeypatch) -> None:
    """``--image-tag`` should write the Modal image tag env var."""
    monkeypatch.delenv(MODAL_IMAGE_TAG_ENV, raising=False)
    conftest = _load_test_conftest()

    try:
        conftest.pytest_configure(FakeConfig({"--image-tag": "sha-test"}))
        assert os.environ[MODAL_IMAGE_TAG_ENV] == "sha-test"
    finally:
        os.environ.pop(MODAL_IMAGE_TAG_ENV, None)


class FakeRequest:
    def __init__(self, config: FakeConfig) -> None:
        self.config = config


def test_image_tag_applies_to_apptainer_backend_option(monkeypatch) -> None:
    """``--image-tag`` should make ``--backend apptainer`` use that tag."""
    monkeypatch.delenv(MODAL_IMAGE_TAG_ENV, raising=False)
    conftest = _load_test_conftest()
    config = FakeConfig({"--backend": "apptainer", "--image-tag": "sha-test"})

    try:
        conftest.pytest_configure(config)
        assert conftest.backend_option.__wrapped__(FakeRequest(config)) == "apptainer:sha-test"
    finally:
        os.environ.pop(MODAL_IMAGE_TAG_ENV, None)


def test_apptainer_inline_tag_wins_over_image_tag() -> None:
    """An explicit Apptainer backend suffix should take precedence over ``--image-tag``."""
    conftest = _load_test_conftest()
    config = FakeConfig({"--backend": "apptainer:inline", "--image-tag": "sha-test"})

    assert conftest.backend_option.__wrapped__(FakeRequest(config)) == "apptainer:inline"
