"""Contract tests of the ``kit`` opt-in: the paid GPU kit tests skip unless ``--run-kit`` or ``BOILEROOM_RUN_KIT``."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest


def _conftest() -> ModuleType:
    module = sys.modules.get("conftest")
    assert module is not None and hasattr(module, "_run_kit_requested"), "tests/conftest.py is not loaded as 'conftest'"
    return module


class _Config:
    def __init__(self, run_kit: bool) -> None:
        self._run_kit = run_kit

    def getoption(self, name: str) -> Any:
        assert name == "--run-kit", name
        return self._run_kit


class _Item:
    def __init__(self, *markers: str) -> None:
        self._markers = set(markers)
        self.added: list[Any] = []

    def get_closest_marker(self, name: str) -> object | None:
        return SimpleNamespace(name=name) if name in self._markers else None

    def add_marker(self, marker: Any) -> None:
        self.added.append(marker)


@pytest.mark.parametrize(
    ("flag", "env", "expected"),
    [
        (False, None, False),
        (False, "", False),
        (False, "0", False),
        (False, "no", False),
        (True, None, True),
        (True, "0", True),
        (False, "1", True),
        (False, " TRUE ", True),
        (False, "yes", True),
    ],
)
def test_run_kit_requested(monkeypatch: pytest.MonkeyPatch, flag: bool, env: str | None, expected: bool) -> None:
    """``--run-kit`` or a truthy ``BOILEROOM_RUN_KIT`` opts in; nothing, or a falsy value, does not."""
    conftest = _conftest()
    if env is None:
        monkeypatch.delenv(conftest.RUN_KIT_ENV, raising=False)
    else:
        monkeypatch.setenv(conftest.RUN_KIT_ENV, env)
    assert conftest._run_kit_requested(_Config(flag)) is expected


def test_run_kit_refuses_an_unknown_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """A typo in ``BOILEROOM_RUN_KIT`` is a usage error, not a silent skip."""
    conftest = _conftest()
    monkeypatch.setenv(conftest.RUN_KIT_ENV, "ture")
    with pytest.raises(pytest.UsageError, match="BOILEROOM_RUN_KIT"):
        conftest._run_kit_requested(_Config(False))


def test_kit_items_skip_unless_opted_in(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only items marked ``kit`` get the skip marker, and only without the opt-in."""
    conftest = _conftest()
    monkeypatch.delenv(conftest.RUN_KIT_ENV, raising=False)
    kit, other = _Item("kit", "gpu"), _Item("gpu", "integration")
    conftest.pytest_collection_modifyitems(_Config(False), [kit, other])
    assert [marker.name for marker in kit.added] == ["skip"]
    assert "--run-kit" in kit.added[0].kwargs["reason"]
    assert other.added == []

    opted = _Item("kit")
    conftest.pytest_collection_modifyitems(_Config(True), [opted])
    assert opted.added == []
