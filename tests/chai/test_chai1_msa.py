"""Unit tests for the Chai-1 side of MSA handling (no chai-lab or GPU needed)."""

from __future__ import annotations

from pathlib import Path

from boileroom.models.chai.msa import assemble_msa_directory, cache_key, stage_provided_alignments
from boileroom.msa_sources import EMPTY

TARGET = "MKVWPQGHSTNRYLAEFCID"
TARGET_2 = "MKVWPQGHSTNRYLAEFCIE"
BINDER = "ACDEFGHIKLMNPQRSTVWY"


def a3m(sequence: str) -> str:
    return f">query\n{sequence}\n>hit\n{sequence}\n"


def basename(sequence: str) -> str:
    return f"{sequence[:6]}.aligned.pqt"


class FakeWriter:
    """Stands in for the chai-lab conversion: records what it was asked to convert."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def __call__(self, a3m_path: Path, sequence: str, directory: Path) -> Path:
        self.calls.append((a3m_path.read_text(encoding="utf-8"), sequence))
        out = directory / basename(sequence)
        out.write_text(f"pqt of {sequence}")
        return out


def test_only_alignments_of_sequences_that_occur_are_converted(tmp_path: Path) -> None:
    writer = FakeWriter()
    alignments = {TARGET: a3m(TARGET), BINDER: a3m(BINDER)}
    staged = stage_provided_alignments(alignments, [TARGET_2, BINDER], tmp_path / "provided", writer)  # target changed
    assert list(staged) == [BINDER] and writer.calls == [(a3m(BINDER), BINDER)]
    assert stage_provided_alignments({}, [TARGET], tmp_path / "none", writer) == {}
    assert not (tmp_path / "none").exists()


def test_the_directory_holds_a_file_per_sequence_that_has_an_alignment(tmp_path: Path) -> None:
    provided = tmp_path / "provided.pqt"
    provided.write_text("provided")
    fetched = tmp_path / "fetched.pqt"
    fetched.write_text("fetched")
    directory = assemble_msa_directory(
        [TARGET, BINDER, BINDER], {0: provided, 1: fetched, 2: fetched}, tmp_path / "msas", basename
    )
    assert directory == tmp_path / "msas"
    assert sorted(p.name for p in directory.iterdir()) == sorted([basename(TARGET), basename(BINDER)])
    assert (directory / basename(TARGET)).read_text() == "provided"


def test_chains_without_an_alignment_get_no_file_and_no_files_means_no_directory(tmp_path: Path) -> None:
    fetched = tmp_path / "fetched.pqt"
    fetched.write_text("fetched")
    directory = assemble_msa_directory([TARGET, BINDER], {0: fetched, 1: EMPTY}, tmp_path / "msas", basename)
    assert directory is not None and [p.name for p in directory.iterdir()] == [basename(TARGET)]
    assert assemble_msa_directory([TARGET], {0: EMPTY}, tmp_path / "none", basename) is None
    assert not (tmp_path / "none").exists()


def test_cache_keys_depend_on_the_sequence_and_the_server() -> None:
    base = cache_key(TARGET, "https://api.colabfold.com")
    assert base == cache_key(TARGET, "https://api.colabfold.com")
    assert base != cache_key(BINDER, "https://api.colabfold.com")
    assert base != cache_key(TARGET, "https://other.example")
