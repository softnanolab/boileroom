"""Unit tests for the Boltz-2 side of MSA handling (no Boltz or GPU needed)."""

from __future__ import annotations

from pathlib import Path

from boileroom.models.boltz.msa import AUTO, EMPTY, cache_key, chains_to_fasta, split_chains, stage_provided_alignments
from boileroom.msa_cache import MSACache

TARGET = "MKVWPQGHSTNRYLAEFCID"
BINDER_1 = "ACDEFGHIKLMNPQRSTVWY"
BINDER_2 = "YWVTSRQPNMLKIHGFEDCA"


def test_cache_keys_depend_on_the_sequence_and_the_server_request() -> None:
    base = cache_key(TARGET, "https://api.colabfold.com", 8192)
    assert base == cache_key(TARGET, "https://api.colabfold.com", 8192)
    assert base != cache_key(BINDER_1, "https://api.colabfold.com", 8192)
    assert base != cache_key(TARGET, "https://other.example", 8192)
    assert base != cache_key(TARGET, "https://api.colabfold.com", 4096)
    assert base != MSACache.hash_key(TARGET)  # entries of earlier versions (key = the sequence) are never reused


def test_only_alignments_of_sequences_that_occur_are_written(tmp_path: Path) -> None:
    alignments = {TARGET: f">q\n{TARGET}\n", BINDER_1: f">q\n{BINDER_1}\n"}
    staged = stage_provided_alignments(alignments, [TARGET, BINDER_2], tmp_path / "provided")
    assert list(staged) == [TARGET]  # the binder was changed: its old alignment is not used
    assert staged[TARGET].read_text(encoding="utf-8") == alignments[TARGET] and staged[TARGET].suffix == ".a3m"
    assert stage_provided_alignments({}, [TARGET], tmp_path / "none") == {}
    assert not (tmp_path / "none").exists()


def test_fasta_headers(tmp_path: Path) -> None:
    provided = tmp_path / "provided.a3m"
    fasta = chains_to_fasta([TARGET, BINDER_1, BINDER_2], {0: provided, 1: EMPTY, 2: AUTO}).splitlines()
    assert fasta == [
        f">A|protein|{provided.resolve()}",
        TARGET,
        ">B|protein|empty",
        BINDER_1,
        ">C|protein|",
        BINDER_2,
    ]


def test_split_chains() -> None:
    assert split_chains(["AC:DE", "FG", " HI : "]) == ["AC", "DE", "FG", "HI"]
