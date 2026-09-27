"""Content-addressed cache for MSA files, shared across folding adapters.

Keys are hex digests (typically the SHA256 of a sequence, or of a sequence
combined with the settings that affect its alignment). Cached artifacts are
stored under a hash-fanned directory and tracked in a JSON index guarded by an
exclusive file lock, so concurrent folds never clobber one another's entries.

The cache is format-agnostic: Boltz stores ``.csv`` alignments and the
ColabFold-backed AlphaFold2-Multimer adapter stores ``.a3m`` alignments, both
through the same :class:`MSACache`.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import shutil
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .utils import safe_mkdir

logger = logging.getLogger(__name__)


def sequence_hash(sequence: str) -> str:
    """Return the SHA256 hex digest of ``sequence``."""
    return hashlib.sha256(sequence.encode()).hexdigest()


class MSACache:
    """Content-addressed, lock-guarded cache for MSA files.

    Parameters
    ----------
    base_dir : Path
        Directory under which the cache lives; the cache root is
        ``base_dir / subdir``.
    subdir : str
        Name of the cache subdirectory (default ``"msa_cache"``).
    suffix : str
        File extension for cached artifacts, including the leading dot
        (e.g. ``".csv"`` for Boltz, ``".a3m"`` for ColabFold).
    """

    def __init__(self, base_dir: Path, *, subdir: str = "msa_cache", suffix: str = ".a3m") -> None:
        self.cache_dir = Path(base_dir) / subdir
        self.suffix = suffix
        safe_mkdir(self.cache_dir, parents=True)

    @staticmethod
    def hash_key(value: str) -> str:
        """Return the cache key for ``value`` (its SHA256 hex digest)."""
        return sequence_hash(value)

    @property
    def index_path(self) -> Path:
        """Path to the JSON index that maps keys to cached files."""
        return self.cache_dir / "msa_index.json"

    def _relative_path(self, key: str) -> str:
        return f"{key[:2]}/{key[2:4]}/{key}{self.suffix}"

    def _load_index(self) -> dict[str, dict[str, Any]]:
        index_path = self.index_path
        if not index_path.exists():
            return {}
        try:
            with index_path.open("r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load MSA cache index from {index_path}: {e}. Recreating index.")
            backup_path = index_path.with_suffix(".json.bak")
            with contextlib.suppress(OSError):
                shutil.move(str(index_path), str(backup_path))
                logger.info(f"Moved corrupted index to {backup_path}")
            return {}

    def _save_index(self, index: dict[str, dict[str, Any]]) -> None:
        index_path = self.index_path
        safe_mkdir(index_path.parent, parents=True)
        # Write to a temporary file first, then rename, so readers never observe
        # a partially written index.
        temp_path = index_path.with_name(f"{index_path.name}.{os.getpid()}.tmp")
        try:
            with temp_path.open("w") as f:
                json.dump(index, f, indent=2)
            temp_path.replace(index_path)
        except OSError as e:
            logger.warning(f"Failed to save MSA cache index to {index_path}: {e}")
            if temp_path.exists():
                with contextlib.suppress(OSError):
                    temp_path.unlink()

    @contextlib.contextmanager
    def locked_index(self) -> Iterator[dict[str, dict[str, Any]]]:
        """Yield the index while holding an exclusive cross-process file lock."""
        import fcntl

        index_path = self.index_path
        safe_mkdir(index_path.parent, parents=True)
        lock_path = index_path.with_suffix(".json.lock")
        with lock_path.open("a+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield self._load_index()
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def get(self, key: str) -> Path | None:
        """Return the cached file for ``key``, or ``None`` on a miss.

        Refreshes the entry's ``last_accessed`` timestamp on a hit and evicts
        index entries whose backing file has disappeared.
        """
        try:
            with self.locked_index() as index:
                entry = index.get(key)
                if entry is None:
                    return None
                path = self.cache_dir / (entry.get("msa_path") or self._relative_path(key))
                if path.exists() and path.is_file():
                    now = datetime.now(UTC).isoformat()
                    if entry.get("last_accessed") != now:
                        entry["last_accessed"] = now
                        self._save_index(index)
                    logger.info("Using cached MSA for key %s at %s", key, path)
                    return path
                # Stale entry: the file is gone, so drop it from the index.
                del index[key]
                self._save_index(index)
                return None
        except Exception as e:
            logger.warning(f"Error checking MSA cache: {e}. Continuing without cache.")
            return None

    def put(self, key: str, source_path: Path) -> None:
        """Copy ``source_path`` into the cache under ``key`` if not already present."""
        try:
            source_path = Path(source_path)
            if not source_path.exists():
                logger.warning(f"Source MSA file not found: {source_path}")
                return
            dest = self.cache_dir / self._relative_path(key)
            if dest.exists():
                logger.debug(f"MSA already cached for key {key}")
                return
            safe_mkdir(dest.parent, parents=True)
            shutil.copy2(source_path, dest)
            file_size = dest.stat().st_size
            with self.locked_index() as index:
                now = datetime.now(UTC).isoformat()
                index[key] = {
                    "msa_path": self._relative_path(key),
                    "created_at": now,
                    "last_accessed": now,
                    "file_size": file_size,
                }
                self._save_index(index)
            logger.debug(f"Cached MSA for key {key}")
        except Exception as e:
            logger.warning(f"Error saving MSA to cache: {e}. Continuing without caching.")
