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


_BACKEND_DEFAULTS: dict[str, str | None] = {"--backend": "modal", "--gpu": None, "--device": None}


class FakeConfig:
    def __init__(self, options: dict[str, str | None]) -> None:
        self.options = {**_BACKEND_DEFAULTS, **options}

    def getoption(self, name: str) -> str | None:
        return self.options.get(name)


def test_pytest_image_options_set_modal_lookup_env(monkeypatch) -> None:
    monkeypatch.delenv(DOCKER_REPOSITORY_ENV, raising=False)
    monkeypatch.delenv(MODAL_IMAGE_TAG_ENV, raising=False)
    conftest = _load_test_conftest()

    try:
        conftest.pytest_configure(FakeConfig({"--docker-user": "phauglin", "--image-tag": "sha-test"}))

        assert os.environ[DOCKER_REPOSITORY_ENV] == "docker.io/phauglin"
        assert os.environ[MODAL_IMAGE_TAG_ENV] == "sha-test"
    finally:
        os.environ.pop(DOCKER_REPOSITORY_ENV, None)
        os.environ.pop(MODAL_IMAGE_TAG_ENV, None)


def test_pytest_image_options_leave_env_unset_when_absent(monkeypatch) -> None:
    monkeypatch.delenv(DOCKER_REPOSITORY_ENV, raising=False)
    monkeypatch.delenv(MODAL_IMAGE_TAG_ENV, raising=False)
    conftest = _load_test_conftest()

    conftest.pytest_configure(FakeConfig({"--docker-user": None, "--image-tag": None}))

    assert DOCKER_REPOSITORY_ENV not in os.environ
    assert MODAL_IMAGE_TAG_ENV not in os.environ


def test_pytest_image_options_can_set_only_one_env(monkeypatch) -> None:
    monkeypatch.delenv(DOCKER_REPOSITORY_ENV, raising=False)
    monkeypatch.delenv(MODAL_IMAGE_TAG_ENV, raising=False)
    conftest = _load_test_conftest()

    try:
        conftest.pytest_configure(FakeConfig({"--docker-user": "phauglin", "--image-tag": None}))
        assert os.environ[DOCKER_REPOSITORY_ENV] == "docker.io/phauglin"
        assert MODAL_IMAGE_TAG_ENV not in os.environ

        os.environ.pop(DOCKER_REPOSITORY_ENV, None)
        conftest.pytest_configure(FakeConfig({"--docker-user": None, "--image-tag": "sha-test"}))
        assert DOCKER_REPOSITORY_ENV not in os.environ
        assert os.environ[MODAL_IMAGE_TAG_ENV] == "sha-test"
    finally:
        os.environ.pop(DOCKER_REPOSITORY_ENV, None)
        os.environ.pop(MODAL_IMAGE_TAG_ENV, None)
