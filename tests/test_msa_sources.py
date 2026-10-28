"""Unit tests for where each chain's MSA comes from (no model, GPU or server needed)."""

from __future__ import annotations

from pathlib import Path

import pytest

from boileroom.msa_sources import AUTO, EMPTY, provided_alignments, resolve_chain_msas

TARGET = "MKVWPQGHSTNRYLAEFCID"
TARGET_2 = "MKVWPQGHSTNRYLAEFCIE"  # the target with one residue changed
BINDER_1 = "ACDEFGHIKLMNPQRSTVWY"
BINDER_2 = "YWVTSRQPNMLKIHGFEDCA"


def a3m(sequence: str, rows: int = 2) -> str:
    return ">query\n" + sequence + "\n" + "".join(f">hit{i}\n{sequence}\n" for i in range(rows - 1))


class Recorder:
    """Stands in for the cache and the server: records every call and returns files in ``tmp_path``."""

    def __init__(self, tmp_path: Path, cached: dict[str, Path] | None = None) -> None:
        self.tmp_path = tmp_path
        self.cached = dict(cached or {})
        self.lookups: list[list[str]] = []
        self.fetches: list[str] = []
        self.stores: list[str] = []
        self.sources: dict[int, str] = {}

    def lookup(self, chains):
        self.lookups.append(list(chains))
        return {s: self.cached[s] for s in chains if s in self.cached}

    def fetch(self, sequence):
        self.fetches.append(sequence)
        path = self.tmp_path / f"fetched_{len(self.fetches)}.csv"
        path.write_text("key,sequence\n-1," + sequence + "\n")
        return path

    def store(self, sequence, path):
        self.stores.append(sequence)
        self.cached[sequence] = path

    def resolve(self, chains, *, provided=None, use_msa_server=True, cache_enabled=True, always_per_chain=False):
        self.sources = {}
        return resolve_chain_msas(
            chains,
            provided=provided or {},
            use_msa_server=use_msa_server,
            cache_enabled=cache_enabled,
            lookup=self.lookup,
            fetch=self.fetch,
            store=self.store,
            sources=self.sources,
            always_per_chain=always_per_chain,
        )


# ---------------------------------------------------------------------------------------------------------------------
# the behaviour that was asked for
# ---------------------------------------------------------------------------------------------------------------------
def test_no_msa_mode_uses_no_msa_and_never_touches_the_cache(tmp_path: Path) -> None:
    """Even with the target's MSA sitting in the cache, no-MSA mode folds every chain alone."""
    recorder = Recorder(tmp_path, cached={TARGET: tmp_path / "cached_target.csv"})
    fields = recorder.resolve([TARGET, BINDER_1], use_msa_server=False, cache_enabled=True)
    assert fields == {0: EMPTY, 1: EMPTY}
    assert recorder.lookups == [] and recorder.fetches == [] and recorder.stores == []


def test_a_provided_alignment_is_used_without_a_server_call(tmp_path: Path) -> None:
    provided_file = tmp_path / "target.a3m"
    recorder = Recorder(tmp_path, cached={TARGET: tmp_path / "old.csv"})  # even a cached entry for the same sequence
    fields = recorder.resolve([TARGET, BINDER_1], provided={TARGET: provided_file})
    assert fields[0] == provided_file
    assert recorder.fetches == [BINDER_1]  # only the chain that has no alignment is fetched
    assert recorder.sources == {0: "provided", 1: "fetched"}


def test_server_mode_fetches_once_and_caches_then_only_new_sequences_are_fetched(tmp_path: Path) -> None:
    recorder = Recorder(tmp_path)
    first = recorder.resolve([TARGET, BINDER_1])
    assert recorder.fetches == [TARGET, BINDER_1] and recorder.stores == [TARGET, BINDER_1]
    second = recorder.resolve([TARGET, BINDER_1])  # nothing changed: nothing is asked again
    assert recorder.fetches == [TARGET, BINDER_1] and second == first
    third = recorder.resolve([TARGET, BINDER_2])  # a new binder: only the new sequence is asked
    assert recorder.fetches == [TARGET, BINDER_1, BINDER_2] and third[0] == first[0]
    assert AUTO not in third.values()


def test_a_changed_chain_with_a_provided_alignment_gets_a_new_msa_from_the_server(tmp_path: Path) -> None:
    """The provided alignment is of the original sequence; a changed chain is treated like any new sequence."""
    provided_file = tmp_path / "target.a3m"
    recorder = Recorder(tmp_path)

    start = recorder.resolve([TARGET, BINDER_1], provided={TARGET: provided_file})
    assert start[0] == provided_file and recorder.fetches == [BINDER_1]

    changed = recorder.resolve([TARGET_2, BINDER_1], provided={TARGET: provided_file})  # the target changed
    assert recorder.sources == {0: "fetched", 1: "cache"}
    assert recorder.fetches == [BINDER_1, TARGET_2]  # the new target sequence is asked for once ...
    assert isinstance(changed[0], Path) and changed[0] != provided_file

    again = recorder.resolve([TARGET_2, BINDER_1], provided={TARGET: provided_file})  # ... and then cached
    assert recorder.fetches == [BINDER_1, TARGET_2] and recorder.sources == {0: "cache", 1: "cache"}
    assert again == changed

    back = recorder.resolve([TARGET, BINDER_1], provided={TARGET: provided_file})  # the original sequence again
    assert back[0] == provided_file and recorder.sources[0] == "provided"
    assert recorder.fetches == [BINDER_1, TARGET_2]


def test_a_changed_chain_without_the_server_gets_no_msa(tmp_path: Path) -> None:
    provided_file = tmp_path / "target.a3m"
    recorder = Recorder(tmp_path)
    fields = recorder.resolve([TARGET_2, BINDER_1], provided={TARGET: provided_file}, use_msa_server=False)
    assert fields == {0: EMPTY, 1: EMPTY}
    assert recorder.fetches == [] and recorder.stores == []  # the server is not called to change it


def test_an_alignment_provided_for_any_chain_applies_to_its_sequence(tmp_path: Path) -> None:
    """The provided alignment is of a sequence: it is used for that sequence wherever it occurs, e.g. a homodimer."""
    provided_file = tmp_path / "target.a3m"
    fields = Recorder(tmp_path).resolve([TARGET, TARGET], provided={TARGET: provided_file}, use_msa_server=False)
    assert fields == {0: provided_file, 1: provided_file}


# ---------------------------------------------------------------------------------------------------------------------
# the other cases
# ---------------------------------------------------------------------------------------------------------------------
def test_cached_and_new_chains_never_mix_custom_and_generated_msas(tmp_path: Path) -> None:
    recorder = Recorder(tmp_path, cached={TARGET: tmp_path / "cached_target.csv"})
    fields = recorder.resolve([TARGET, BINDER_1])
    assert fields[0] == tmp_path / "cached_target.csv" and isinstance(fields[1], Path)
    assert AUTO not in fields.values()


def test_cache_off_without_an_alignment_leaves_the_msa_to_the_model(tmp_path: Path) -> None:
    recorder = Recorder(tmp_path)
    fields = recorder.resolve([TARGET, BINDER_1], cache_enabled=False)
    assert fields == {0: AUTO, 1: AUTO}
    assert recorder.lookups == [] and recorder.fetches == [] and recorder.stores == []
    forced = recorder.resolve([TARGET, BINDER_1], cache_enabled=False, always_per_chain=True)
    assert all(isinstance(field, Path) for field in forced.values()) and recorder.stores == []


def test_cache_off_with_an_alignment_fetches_the_other_chains_without_caching(tmp_path: Path) -> None:
    provided_file = tmp_path / "target.a3m"
    recorder = Recorder(tmp_path)
    fields = recorder.resolve([TARGET, BINDER_1], provided={TARGET: provided_file}, cache_enabled=False)
    assert fields[0] == provided_file and isinstance(fields[1], Path)  # no auto next to a custom MSA
    assert recorder.fetches == [BINDER_1] and recorder.stores == [] and recorder.lookups == []


def test_an_unused_provided_alignment_does_not_force_per_chain_fetching(tmp_path: Path) -> None:
    """The alignment is of a sequence that no chain has: with the cache off the model may still generate everything."""
    recorder = Recorder(tmp_path)
    fields = recorder.resolve([TARGET_2, BINDER_1], provided={TARGET: tmp_path / "x.a3m"}, cache_enabled=False)
    assert fields == {0: AUTO, 1: AUTO}


def test_identical_chains_are_fetched_once(tmp_path: Path) -> None:
    recorder = Recorder(tmp_path)
    fields = recorder.resolve([TARGET, TARGET])
    assert recorder.fetches == [TARGET] and fields[0] == fields[1]


def test_lookup_only_asks_for_sequences_without_a_provided_alignment(tmp_path: Path) -> None:
    recorder = Recorder(tmp_path)
    recorder.resolve([TARGET, BINDER_1], provided={TARGET: tmp_path / "t.a3m"})
    assert recorder.lookups == [[BINDER_1]]


# ---------------------------------------------------------------------------------------------------------------------
# provided_alignments
# ---------------------------------------------------------------------------------------------------------------------
def test_provided_alignments_are_keyed_by_their_query_sequence() -> None:
    alignments = provided_alignments([a3m(TARGET), None], 2)
    assert alignments == {TARGET: a3m(TARGET)}
    assert provided_alignments(None, 2) == {}
    assert provided_alignments([None, ""], 2) == {}


def test_an_alignment_of_another_sequence_is_not_an_error() -> None:
    """The chain at its position may have changed since the alignment was made: the alignment simply does not apply."""
    assert provided_alignments([a3m(TARGET), None], 2) == {TARGET: a3m(TARGET)}  # whatever chain 0 is now


def test_provided_alignments_are_validated() -> None:
    with pytest.raises(ValueError, match="one A3M string or None per chain"):
        provided_alignments([a3m(TARGET)], 2)
    with pytest.raises(ValueError, match="A3M text"):
        provided_alignments(["not an alignment", None], 2)
    with pytest.raises(ValueError, match="aligned length"):
        provided_alignments([f">q\n{TARGET}\n>h\n{TARGET[:-1]}\n", None], 2)
    with pytest.raises(ValueError, match="Two different alignments"):
        provided_alignments([a3m(TARGET), a3m(TARGET, rows=3)], 2)
    assert provided_alignments([a3m(TARGET), a3m(TARGET)], 2) == {TARGET: a3m(TARGET)}  # the same one twice is fine
