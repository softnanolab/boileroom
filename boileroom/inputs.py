"""Shared lightweight input dataclasses for BoilerRoom model adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MSAInput:
    """Portable multiple-sequence-alignment input.

    Parameters
    ----------
    sequences
        In-memory MSA rows for models that accept sequence-list MSAs directly.
    path
        File-backed MSA location for models that consume MSA files. Model adapters
        may support only one representation; unsupported representations should
        fail with a clear model-specific error.
    remove_insertions
        Whether supported adapters should strip insertion columns from sequence
        rows before constructing the model-native MSA object.
    """

    sequences: list[str] | None = None
    path: str | Path | None = None
    remove_insertions: bool = False

    def __post_init__(self) -> None:
        """Validate that the MSA points to exactly one source."""
        has_sequences = self.sequences is not None
        has_path = self.path is not None
        if has_sequences == has_path:
            raise ValueError("MSAInput requires exactly one of sequences or path.")
        if self.sequences is not None and not all(isinstance(sequence, str) for sequence in self.sequences):
            raise TypeError("MSAInput sequences must be a list of strings.")
        if self.path is not None and not isinstance(self.path, str | Path):
            raise TypeError("MSAInput path must be a string or pathlib.Path.")
        if not isinstance(self.remove_insertions, bool):
            raise TypeError("MSAInput remove_insertions must be a bool.")
