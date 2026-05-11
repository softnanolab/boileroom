"""Contract tests for the ``--docker-user`` pytest option."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

from boileroom.images.metadata import DOCKER_REPOSITORY_ENV


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
