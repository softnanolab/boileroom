"""Chai-1 core: which MSA sources are used, without chai-lab or a GPU (the sources and the cache are stubbed)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from boileroom.models.chai import core as chai_core
from boileroom.models.chai.core import Chai1Core

TARGET = "MKVWPQGHSTNRYLAEFCID"
BINDER = "ACDEFGHIKLMNPQRSTVWY"


class StubCache:
    def __init__(self) -> None:
        self.entries: dict[str, Path] = {}

    def get(self, key: str) -> Path | None:
        return self.entries.get(key)

    def put(self, key: str, path: Path) -> None:
        self.entries[key] = path


@pytest.fixture
def core(monkeypatch: pytest.MonkeyPatch) -> Chai1Core:
    monkeypatch.setattr(chai_core, "expected_basename", lambda sequence: f"{sequence[:6]}.aligned.pqt")
    return Chai1Core({"device": "cpu"})


def test_a_default_fold_never_builds_the_msa_cache(
    core: Chai1Core, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The cache constructor creates a directory in the model folder: no server, no cache, no writes there."""

    def forbidden() -> Any:
        raise AssertionError("the MSA cache must not be built when it is not used")

    monkeypatch.setattr(core, "_msa_cache", forbidden)
    directory, chai_server = core._prepare_msas(f"{TARGET}:{BINDER}", core.config, tmp_path)
    assert directory is None and chai_server is False  # single-sequence run, chai-lab does not call the server


def test_server_mode_fetches_each_sequence_once_and_then_uses_the_cache(
    core: Chai1Core, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache = StubCache()
    fetches: list[str] = []

    def fetch(sequence: str, directory: Path, config: dict[str, Any]) -> Path:
        fetches.append(sequence)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{sequence[:6]}.aligned.pqt"
        path.write_text(f"pqt of {sequence}")
        return path

    monkeypatch.setattr(core, "_msa_cache", lambda: cache)
    monkeypatch.setattr(core, "_fetch_chain_msa", fetch)
    config = {**core.config, "use_msa_server": True}

    first_dir, first_server = core._prepare_msas(f"{TARGET}:{BINDER}", config, tmp_path / "one")
    assert fetches == [TARGET, BINDER] and len(cache.entries) == 2
    assert first_server is False  # chai-lab reads the files; it never calls the server itself
    assert first_dir is not None and sorted(p.name for p in first_dir.iterdir()) == sorted(
        [f"{TARGET[:6]}.aligned.pqt", f"{BINDER[:6]}.aligned.pqt"]
    )

    core._prepare_msas(f"{TARGET}:{BINDER}", config, tmp_path / "two")  # nothing changed: nothing is asked again
    assert fetches == [TARGET, BINDER]


def test_the_cache_off_and_nothing_provided_leaves_the_msas_to_chai_lab(core: Chai1Core, tmp_path: Path) -> None:
    config = {**core.config, "use_msa_server": True, "msa_cache_enabled": False}
    assert core._prepare_msas(f"{TARGET}:{BINDER}", config, tmp_path) == (None, True)
