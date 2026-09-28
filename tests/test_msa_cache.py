"""Unit tests for the shared content-addressed MSA cache."""

from __future__ import annotations

import json
from pathlib import Path

from boileroom.msa_cache import MSACache, sequence_hash


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_hash_key_matches_sequence_hash():
    assert MSACache.hash_key("ACDEFG") == sequence_hash("ACDEFG")
    # Stable, sequence-dependent digest.
    assert MSACache.hash_key("ACDEFG") != MSACache.hash_key("GFEDCA")


def test_put_then_get_round_trip(tmp_path: Path):
    cache = MSACache(tmp_path, suffix=".a3m")
    src = _write(tmp_path / "src.a3m", ">q\nACDE\n")
    key = MSACache.hash_key("ACDE")

    assert cache.get(key) is None  # miss before put

    cache.put(key, src)
    hit = cache.get(key)
    assert hit is not None
    assert hit.read_text(encoding="utf-8") == ">q\nACDE\n"


def test_hash_fanned_layout_and_index_schema(tmp_path: Path):
    cache = MSACache(tmp_path, suffix=".csv")
    key = MSACache.hash_key("SEQ")
    cache.put(key, _write(tmp_path / "src.csv", "a,b\n"))

    expected = cache.cache_dir / key[:2] / key[2:4] / f"{key}.csv"
    assert expected.exists()

    index = json.loads((cache.cache_dir / "msa_index.json").read_text())
    entry = index[key]
    assert entry["msa_path"] == f"{key[:2]}/{key[2:4]}/{key}.csv"
    assert set(entry) == {"msa_path", "created_at", "last_accessed", "file_size"}
    assert entry["file_size"] == expected.stat().st_size


def test_get_evicts_entry_when_file_missing(tmp_path: Path):
    cache = MSACache(tmp_path, suffix=".a3m")
    key = MSACache.hash_key("ACDE")
    cache.put(key, _write(tmp_path / "src.a3m", ">q\nACDE\n"))

    # Delete the backing file but leave the index entry behind.
    cached = cache.get(key)
    assert cached is not None
    cached.unlink()
    assert cache.get(key) is None
    index = json.loads((cache.cache_dir / "msa_index.json").read_text())
    assert key not in index


def test_put_is_idempotent_and_does_not_overwrite(tmp_path: Path):
    cache = MSACache(tmp_path, suffix=".a3m")
    key = MSACache.hash_key("ACDE")
    cache.put(key, _write(tmp_path / "first.a3m", "FIRST"))
    cache.put(key, _write(tmp_path / "second.a3m", "SECOND"))
    cached = cache.get(key)
    assert cached is not None
    assert cached.read_text(encoding="utf-8") == "FIRST"


def test_missing_source_is_ignored(tmp_path: Path):
    cache = MSACache(tmp_path, suffix=".a3m")
    key = MSACache.hash_key("ACDE")
    cache.put(key, tmp_path / "does_not_exist.a3m")
    assert cache.get(key) is None


def test_locked_index_round_trips_writes(tmp_path: Path):
    cache = MSACache(tmp_path, suffix=".a3m")
    with cache.locked_index() as index:
        assert index == {}
        index["abc"] = {"msa_path": "ab/cd/abc.a3m"}
        cache._save_index(index)
    with cache.locked_index() as index:
        assert index["abc"]["msa_path"] == "ab/cd/abc.a3m"


def test_corrupt_index_is_backed_up_and_reset(tmp_path: Path):
    cache = MSACache(tmp_path, suffix=".a3m")
    cache.index_path.write_text("{ not valid json", encoding="utf-8")
    # A read through the public API recovers gracefully.
    assert cache.get("whatever") is None
    assert cache.index_path.with_suffix(".json.bak").exists()


def test_failed_copy_leaves_no_partial_cache_file(tmp_path: Path, monkeypatch):
    import shutil

    cache = MSACache(tmp_path, suffix=".a3m")
    key = MSACache.hash_key("ACDE")
    src = _write(tmp_path / "src.a3m", ">q\nACDE\n")

    def failing_copy(source, dest, *args, **kwargs):
        Path(dest).write_text(">q\nAC", encoding="utf-8")
        raise OSError("disk full")

    monkeypatch.setattr(shutil, "copy2", failing_copy)
    cache.put(key, src)  # put logs and swallows cache write failures
    monkeypatch.undo()

    dest = cache.cache_dir / key[:2] / key[2:4] / f"{key}.a3m"
    assert not dest.exists()
    assert not list(dest.parent.glob(f".{dest.name}.*"))
    cache.put(key, src)
    hit = cache.get(key)
    assert hit is not None and hit.read_text(encoding="utf-8") == ">q\nACDE\n"
