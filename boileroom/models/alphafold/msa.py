"""Lightweight MSA handling for AlphaFold2-Multimer (no heavy model dependencies).

``options["msa"]`` accepts two portable forms:

* :class:`~boileroom.inputs.MSAInput` -- ``sequences`` rows (``:``-joined, one
  segment per chain, for complexes) or a ``path`` to an a3m file (plain a3m for a
  single chain, ColabFold complex a3m with a ``#<lengths>\\t<cardinalities>``
  header for complexes).
* A list with one entry per chain (counting repeated chains), each either A3M
  text for that chain's *unpaired* MSA or ``None``. This matches the other
  boileroom cores.

Everything is checked against the requested chain sequences and rendered into the
ColabFold a3m the resident runner consumes. The encoded form
(:func:`encode_msa_option`) is JSON-safe, with file contents read on the caller's
side, so it also crosses the Apptainer HTTP boundary and Modal containers that
cannot see local paths.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ...inputs import MSAInput

_MSA_KIND = "msa_input"


def split_chains(sequence_entry: str) -> list[str]:
    """Split a ``:``-joined sequence entry into chains, rejecting empty chains."""
    chains = [part.strip() for part in sequence_entry.split(":")]
    if not any(chains):
        raise ValueError("AlphaFold2-Multimer input must contain at least one chain")
    if not all(chains):
        raise ValueError("AlphaFold2-Multimer input must not contain empty chains (check for stray ':')")
    return chains


def aligned(row: str) -> str:
    """Return the aligned columns of an a3m row (lowercase insertions and ``.`` removed)."""
    return "".join(char for char in row if not char.islower() and char != ".")


def strip_insertions(a3m_text: str) -> str:
    """Drop insertion columns from every sequence line, keeping headers."""
    lines = []
    for line in a3m_text.splitlines():
        if line.startswith((">", "#")) or not line:
            lines.append(line)
        else:
            lines.append(aligned(line))
    return "\n".join(lines) + "\n"


def parse_a3m(text: str) -> list[tuple[str, str]]:
    """Parse a3m/FASTA text into ``(header, sequence)`` records, skipping ``#`` lines."""
    records: list[tuple[str, str]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(">"):
            records.append((line[1:], ""))
        elif records:
            header, sequence = records[-1]
            records[-1] = (header, sequence + "".join(line.split()))
        else:
            raise ValueError("Provided MSA has sequence data before the first '>' header")
    if not records:
        raise ValueError("Provided MSA contains no sequences")
    return records


def _unique_chains(chains: Sequence[str]) -> tuple[list[str], list[int], list[int]]:
    """Return ``(unique chains, first index of each, cardinality of each)`` in first-seen order."""
    unique = list(dict.fromkeys(chains))
    return unique, [chains.index(chain) for chain in unique], [chains.count(chain) for chain in unique]


def _check_records(records: Sequence[tuple[str, str]], expected: str, label: str) -> None:
    """Require every row to span ``len(expected)`` columns and the first row to equal ``expected``."""
    for index, (_, row) in enumerate(records):
        if len(aligned(row)) != len(expected):
            raise ValueError(
                f"{label}: MSA row {index} has {len(aligned(row))} aligned columns; expected {len(expected)}"
            )
    if aligned(records[0][1]) != expected:
        raise ValueError(f"{label}: the first MSA row must be the requested sequence")


def _validate_a3m_text(text: str, chains: Sequence[str]) -> None:
    """Validate a provided a3m file/text against the requested chains."""
    unique, _, cardinalities = _unique_chains(chains)
    records = parse_a3m(text)
    first_line = text.lstrip().splitlines()[0]
    if first_line.startswith("#"):
        fields = first_line[1:].split("\t")
        try:
            lengths = [int(part) for part in fields[0].split(",")]
            counts = [int(part) for part in fields[1].split(",")]
        except (IndexError, ValueError) as exc:
            raise ValueError(
                "Complex a3m header must be '#<lengths>\\t<cardinalities>' (comma-separated ints)"
            ) from exc
        if lengths != [len(chain) for chain in unique] or counts != cardinalities:
            raise ValueError(
                f"Complex a3m header (lengths {lengths}, cardinalities {counts}) does not match the requested "
                f"chains (lengths {[len(c) for c in unique]}, cardinalities {cardinalities})"
            )
        _check_records(records, "".join(unique), "Provided MSA")
    elif len(chains) > 1:
        raise ValueError(
            "A multi-chain MSA file must be ColabFold complex a3m (first line '#<lengths>\\t<cardinalities>'); "
            "a plain a3m would be folded as a single chain. Alternatively pass MSAInput(sequences=...) with "
            "':'-joined rows or a list of per-chain A3M text."
        )
    else:
        _check_records(records, chains[0], "Provided MSA")


def _rows_to_a3m(rows: Sequence[str], chains: Sequence[str]) -> str:
    """Validate ``:``-joined MSAInput rows and render ColabFold a3m text."""
    if not rows:
        raise ValueError("Provided MSA contains no sequences")
    for index, row in enumerate(rows):
        segments = row.split(":")
        if len(segments) != len(chains):
            raise ValueError(
                f"MSA row {index} has {len(segments)} ':'-separated segments; expected one per chain ({len(chains)})"
            )
        for chain_index, (segment, chain) in enumerate(zip(segments, chains, strict=True)):
            if len(aligned(segment)) != len(chain):
                raise ValueError(
                    f"MSA row {index}, chain {chain_index} has {len(aligned(segment))} aligned columns; "
                    f"expected {len(chain)}"
                )
    for chain_index, (segment, chain) in enumerate(zip(rows[0].split(":"), chains, strict=True)):
        if aligned(segment) != chain:
            raise ValueError(f"The first MSA row must be the requested sequence (chain {chain_index} differs)")
    if len(chains) > 1:
        return complex_a3m(rows, chains)
    return "\n".join(f">seq_{index}\n{row}" for index, row in enumerate(rows)) + "\n"


def complex_a3m(rows: Sequence[str], chains: Sequence[str]) -> str:
    """Serialize ``:``-joined complex MSA rows into ColabFold's complex a3m format.

    ColabFold expects a ``#<lengths>\\t<cardinalities>`` header over the unique chains,
    followed by the concatenated unique query and then one concatenated row per hit.
    Each hit contributes the segment of the first copy of every unique chain. Rows
    that align to several chains are used as the paired MSA; rows covering a single
    chain (the rest gaps) are used as unpaired. A heteromer also gets every chain's
    own segments as gap-padded unpaired rows, as ColabFold's own complex a3m does:
    with only paired rows a chain's unpaired MSA is empty and ColabFold refuses it.
    """
    unique, first_index, cardinalities = _unique_chains(chains)
    header = f"#{','.join(str(len(c)) for c in unique)}\t{','.join(str(n) for n in cardinalities)}"
    labels = "\t".join(str(101 + index) for index in range(len(unique)))
    lines = [header, f">{labels}", "".join(unique)]
    for index, row in enumerate(rows):
        segments = row.split(":")
        if len(segments) != len(chains):
            raise ValueError(
                f"MSA row {index} has {len(segments)} ':'-separated segments; expected one per chain ({len(chains)})"
            )
        lines.extend([f">seq_{index}", "".join(segments[i] for i in first_index)])
    if len(unique) > 1:
        widths = [len(chain) for chain in unique]
        for index, chain in enumerate(unique):
            before, after = "-" * sum(widths[:index]), "-" * sum(widths[index + 1 :])
            seen = {chain}
            lines.extend([f">{101 + index}", f"{before}{chain}{after}"])
            for row in rows:
                segment = row.split(":")[first_index[index]]
                if segment not in seen and set(aligned(segment)) != {"-"}:
                    seen.add(segment)
                    lines.extend([f">{101 + index}", f"{before}{segment}{after}"])
    return "\n".join(lines) + "\n"


def _per_chain_to_a3m(msas: Sequence[Any], chains: Sequence[str]) -> str:
    """Render one-A3M-text-per-chain (unpaired) MSAs as ColabFold a3m.

    A single chain passes straight through. For complexes, each unique chain's rows
    are gap-padded into the full complex width as unpaired rows; the paired MSA
    contains only the concatenated query, so AlphaFold gets no cross-chain pairing
    evidence from these inputs. ``None`` leaves that chain single-sequence (the
    server is not queried). Repeated chains must carry the same MSA or ``None``.
    """
    if len(msas) != len(chains):
        raise ValueError(f"options['msa'] needs one entry per chain ({len(chains)}); got {len(msas)}")
    unique, first_index, cardinalities = _unique_chains(chains)
    per_unique: list[list[tuple[str, str]] | None] = []
    for chain in unique:
        candidates: list[list[tuple[str, str]]] = []
        for position, (entry, entry_chain) in enumerate(zip(msas, chains, strict=True)):
            if entry_chain != chain or entry is None:
                continue
            if not isinstance(entry, str) or not entry.lstrip().startswith(">"):
                raise ValueError(f"options['msa'] entry {position} must be A3M text (starting with '>') or None")
            records = parse_a3m(entry)
            _check_records(records, chain, f"options['msa'] entry {position}")
            candidates.append(records)
        if any(candidate != candidates[0] for candidate in candidates[1:]):
            raise ValueError(
                f"options['msa'] entries for repeated chain {chain[:12]!r}... differ; "
                "identical chains must share one MSA"
            )
        per_unique.append(candidates[0] if candidates else None)
    if all(item is None for item in per_unique):
        raise ValueError(
            "options['msa'] has no A3M text for any chain; omit it to use the MSA server, or set use_msa_server=False"
        )

    if len(chains) == 1:
        single = per_unique[0]
        assert single is not None
        return "".join(f">{header}\n{row}\n" for header, row in single)

    widths = [len(chain) for chain in unique]
    header = f"#{','.join(str(width) for width in widths)}\t{','.join(str(n) for n in cardinalities)}"
    labels = "\t".join(str(101 + index) for index in range(len(unique)))
    lines = [header, f">{labels}", "".join(unique)]
    for index, chain in enumerate(unique):
        before, after = "-" * sum(widths[:index]), "-" * sum(widths[index + 1 :])
        unpaired = per_unique[index] or [("query", chain)]
        lines.append(f">{101 + index}")
        lines.append(f"{before}{chain}{after}")
        for row_header, row in unpaired[1:]:
            lines.extend([f">{row_header}", f"{before}{row}{after}"])
    return "\n".join(lines) + "\n"


def encode_msa_option(value: Any) -> Any:
    """Convert ``options["msa"]`` into a JSON-safe form, reading path-backed MSAs locally.

    Raises
    ------
    ValueError
        If the value is not an ``MSAInput``, a list of ``str | None``, or an
        already-encoded payload.
    """
    if value is None:
        return None
    if isinstance(value, MSAInput):
        payload: dict[str, Any] = {"kind": _MSA_KIND, "remove_insertions": value.remove_insertions}
        if value.sequences is not None:
            payload["sequences"] = list(value.sequences)
        else:
            payload["a3m"] = Path(str(value.path)).read_text(encoding="utf-8")
        return payload
    if isinstance(value, list | tuple) and all(item is None or isinstance(item, str) for item in value):
        return list(value)
    if isinstance(value, Mapping) and value.get("kind") == _MSA_KIND:
        return dict(value)
    raise ValueError("options['msa'] must be an MSAInput or a list of A3M text (str | None), one entry per chain")


def materialize_msa(msa: Any, chains: Sequence[str]) -> str:
    """Validate ``options["msa"]`` against ``chains`` and render ColabFold a3m text.

    Accepts an ``MSAInput``, an encoded payload from :func:`encode_msa_option`, or
    a list of per-chain A3M text. Raises ``ValueError`` on malformed input or when
    the alignment does not match the requested sequence(s).
    """
    payload = encode_msa_option(msa)
    if payload is None:
        raise ValueError("options['msa'] must not be None here")
    if isinstance(payload, list):
        return _per_chain_to_a3m(payload, chains)

    remove_insertions = payload.get("remove_insertions", False)
    if not isinstance(remove_insertions, bool):
        raise ValueError("Encoded MSA remove_insertions must be a boolean")
    sequences, text = payload.get("sequences"), payload.get("a3m")
    if (sequences is None) == (text is None):
        raise ValueError("Encoded MSA needs exactly one of 'sequences' or 'a3m'")
    if sequences is not None:
        if not isinstance(sequences, list) or not all(isinstance(row, str) for row in sequences):
            raise ValueError("Encoded MSA sequences must be a list of strings")
        text = _rows_to_a3m(sequences, chains)
    elif not isinstance(text, str) or not text.lstrip().startswith((">", "#")):
        raise ValueError("Provided MSA must be in a3m/FASTA format (first line starts with '>' or a '#' header)")
    else:
        _validate_a3m_text(text, chains)
    if remove_insertions:
        text = strip_insertions(text)
    return text
