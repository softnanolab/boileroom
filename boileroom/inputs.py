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


def parse_a3m(text: str) -> list[tuple[str, str]]:
    """Parse A3M/FASTA text into ``(header, sequence)`` records.

    This is the one A3M reader shared by every model family; family-specific rules
    (aligned length, query row, allowed characters) are layered on its records.
    Lines are stripped and blank lines skipped. A ``>`` line starts a record and
    every other line is appended verbatim to the current record's sequence, so a
    wrapped row is joined, while internal whitespace and ``#`` comment lines stay in
    the sequence for the caller's checks to reject.

    Parameters
    ----------
    text
        A3M or FASTA text.

    Returns
    -------
    list[tuple[str, str]]
        The records in file order; repeated headers are kept.

    Raises
    ------
    ValueError
        If sequence data precedes the first ``>`` header or the text holds no records.
    """
    records: list[tuple[str, str]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            records.append((line[1:], ""))
        elif records:
            header, sequence = records[-1]
            records[-1] = (header, sequence + line)
        else:
            raise ValueError("A3M text has sequence data before the first '>' header")
    if not records:
        raise ValueError("A3M text contains no sequences")
    return records


def aligned_columns(row: str) -> str:
    """Return the aligned (match-state) columns of an A3M row: lowercase insertions and ``.`` removed."""
    return "".join(char for char in row if not char.islower() and char != ".")


def a3m_rows(text: object, sequence: str) -> list[str]:
    """Parse A3M text into aligned rows (insertions dropped), checking them against the query chain.

    Rows are kept in a list, so alignments that repeat a header (as many search tools write) stay intact.

    Raises
    ------
    ValueError
        If ``text`` is not A3M text, its first row is not ``sequence``, or a row's aligned length differs.
    """
    if not isinstance(text, str) or not text.lstrip().startswith(">"):
        raise ValueError("MSA entries must be A3M text or None")
    rows = [aligned_columns(row) for _, row in parse_a3m(text)]
    if rows[0] != sequence:
        raise ValueError("The first A3M row must match its input protein chain")
    if any(len(row) != len(sequence) for row in rows):
        raise ValueError("Every A3M row must have the input chain's aligned length")
    return rows
