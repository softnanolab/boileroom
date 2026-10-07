"""Lightweight MSA handling for AlphaFold2-Multimer (no heavy model dependencies).

``options["msa"]`` accepts two portable forms:

* :class:`~boileroom.inputs.MSAInput` with either

  - ``sequences``: one row per alignment entry, ``:``-joined with one segment per
    chain (counting repeated chains) for complexes. Rows that cover several chains
    become ColabFold's *paired* MSA; for a heteromer every chain's own segments are
    also added as gap-padded *unpaired* rows. ColabFold's complex a3m holds one
    segment per unique chain, so all copies of a repeated chain must carry the
    same segment in every row.
  - ``path``: an a3m file, read on the caller's side. A single chain takes plain
    a3m; a complex needs ColabFold's complex a3m, whose first line is the
    ``#<lengths>\\t<cardinalities>`` header over the unique chains (a headerless
    file would be folded as one chain, so it is refused).

* A list with one entry per chain (counting repeated chains), each either A3M text
  for that chain's *unpaired* MSA or ``None`` (query only for that chain). This
  form is unpaired-only: the paired MSA holds just the concatenated query, so
  AlphaFold sees no cross-chain pairing evidence. Repeated chains must carry the
  same text or ``None``, and at least one entry must be text.

A3M text (the ``path`` and per-chain list forms) is parsed with the shared
:func:`boileroom.inputs.parse_a3m`; ``:``-joined ``sequences`` rows are split into
segments directly. Every form is then held to the rules of the ColabFold consumer:

* the first row must equal the requested chain(s) exactly: ColabFold folds the
  raw first row and slices it by the header lengths, so it must not carry
  insertions;
* every row has one aligned column per residue of the chain(s) it covers;
* rows hold only residue letters, ``-`` gaps and lowercase insertions. ``.`` is
  refused: ColabFold counts it as an aligned column when it splits complex rows
  into chains (shifting chain boundaries) and AlphaFold cannot encode it.
  ``MSAInput(remove_insertions=True)`` drops lowercase insertions and ``.``
  before these checks; the per-chain list form has no such option, so ``.`` must
  be deleted from its text.

The validated rows are re-rendered as the ColabFold a3m the resident runner
consumes. The encoded form (:func:`encode_msa_option`) is JSON-safe, with file
contents read on the caller's side, so it also crosses the Apptainer HTTP
boundary and Modal containers that cannot see local paths.
"""

from __future__ import annotations

import string
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ...inputs import MSAInput, aligned_columns, parse_a3m

_MSA_KIND = "msa_input"
_ROW_ALPHABET = frozenset(string.ascii_letters + "-")
_MSA_INPUT_DOT_HINT = "delete it or pass MSAInput(..., remove_insertions=True)"
_PER_CHAIN_DOT_HINT = "delete the '.' characters (the per-chain list form has no remove_insertions option)"


def split_chains(sequence_entry: str) -> list[str]:
    """Split a ``:``-joined sequence entry into chains, rejecting empty chains."""
    chains = [part.strip() for part in sequence_entry.split(":")]
    if not any(chains):
        raise ValueError("AlphaFold2-Multimer input must contain at least one chain")
    if not all(chains):
        raise ValueError("AlphaFold2-Multimer input must not contain empty chains (check for stray ':')")
    return chains


def _unique_chains(chains: Sequence[str]) -> tuple[list[str], list[int], list[int]]:
    """Return ``(unique chains, first index of each, cardinality of each)`` in first-seen order."""
    unique = list(dict.fromkeys(chains))
    return unique, [chains.index(chain) for chain in unique], [chains.count(chain) for chain in unique]


# -- validation ----------------------------------------------------------------


def _check_row(row: str, width: int, where: str, dot_hint: str = _MSA_INPUT_DOT_HINT) -> None:
    """Require ColabFold-safe characters and ``width`` aligned columns in one row or segment.

    ``dot_hint`` is the remedy named when the row holds ``.``; it differs by input form.
    """
    if "." in row:
        raise ValueError(
            f"{where} contains '.', an insert-state gap that ColabFold counts as an aligned column (shifting "
            f"chain boundaries) and AlphaFold cannot encode; {dot_hint}"
        )
    invalid = sorted(set(row) - _ROW_ALPHABET)
    if invalid:
        raise ValueError(
            f"{where} contains invalid characters {', '.join(map(repr, invalid))}; rows may hold only residue "
            "letters, '-' gaps and lowercase insertions (no whitespace, comment lines or terminators)"
        )
    if len(aligned_columns(row)) != width:
        raise ValueError(f"{where} has {len(aligned_columns(row))} aligned columns; expected {width}")


def _check_query(row: str, expected: str, message: str) -> None:
    """Require the first row (or one of its segments) to equal the requested sequence exactly."""
    if row == expected:
        return
    if aligned_columns(row) == expected:
        message += "; ColabFold folds the first row as written, so it must not contain insertions"
    raise ValueError(message)


def _validated_records(
    records: Sequence[tuple[str, str]],
    expected: str,
    label: str,
    remove_insertions: bool,
    dot_hint: str = _MSA_INPUT_DOT_HINT,
) -> list[tuple[str, str]]:
    """Check parsed records against ``expected`` and return them, insertions dropped if requested."""
    if remove_insertions:
        records = [(header, aligned_columns(row)) for header, row in records]
    for index, (_, row) in enumerate(records):
        _check_row(row, len(expected), f"{label}: MSA row {index}", dot_hint)
    _check_query(records[0][1], expected, f"{label}: the first MSA row must be the requested sequence")
    return list(records)


def _validated_segments(rows: Sequence[str], chains: Sequence[str], remove_insertions: bool) -> list[list[str]]:
    """Split ``:``-joined rows into per-chain segments and check them against ``chains``."""
    if not rows:
        raise ValueError("Provided MSA contains no sequences")
    first_index = [chains.index(chain) for chain in chains]
    split_rows: list[list[str]] = []
    for index, row in enumerate(rows):
        segments = row.split(":")
        if len(segments) != len(chains):
            raise ValueError(
                f"MSA row {index} has {len(segments)} ':'-separated segments; expected one per chain ({len(chains)})"
            )
        if remove_insertions:
            segments = [aligned_columns(segment) for segment in segments]
        for chain_index, (segment, chain) in enumerate(zip(segments, chains, strict=True)):
            _check_row(segment, len(chain), f"MSA row {index}, chain {chain_index}")
            if index == 0:
                _check_query(
                    segment,
                    chain,
                    f"The first MSA row must be the requested sequence (chain {chain_index} differs)",
                )
        for chain_index, first in enumerate(first_index):
            if segments[chain_index] != segments[first]:
                raise ValueError(
                    f"MSA row {index}: chains {first} and {chain_index} are copies of one sequence but carry "
                    "different segments; ColabFold's complex a3m holds one segment per unique chain, so give "
                    "every copy the same segment"
                )
        split_rows.append(segments)
    return split_rows


def _parse_complex_header(line: str) -> tuple[list[int], list[int]]:
    """Parse ColabFold's ``#<lengths>\\t<cardinalities>`` complex a3m header."""
    try:
        # Exactly two tab-separated fields, as ColabFold's unserialize_msa requires.
        lengths, cardinalities = ([int(part) for part in field.split(",")] for field in line[1:].split("\t"))
        return lengths, cardinalities
    except ValueError as exc:
        raise ValueError("Complex a3m header must be '#<lengths>\\t<cardinalities>' (comma-separated ints)") from exc


# -- rendering -----------------------------------------------------------------


def _render(records: Iterable[tuple[str, str]]) -> str:
    """Render ``(header, row)`` records as a3m text."""
    return "".join(f">{header}\n{row}\n" for header, row in records)


def _complex_header(unique: Sequence[str], cardinalities: Sequence[int]) -> str:
    """Return ColabFold's ``#<lengths>\\t<cardinalities>`` line for the unique chains."""
    return f"#{','.join(str(len(chain)) for chain in unique)}\t{','.join(str(n) for n in cardinalities)}"


def _complex_preamble(unique: Sequence[str], cardinalities: Sequence[int]) -> list[str]:
    """Return the complex a3m header, the query's label line and the concatenated query."""
    labels = "\t".join(str(101 + index) for index in range(len(unique)))
    return [_complex_header(unique, cardinalities), f">{labels}", "".join(unique)]


def _unpaired_block(index: int, widths: Sequence[int], records: Iterable[tuple[str, str]]) -> list[str]:
    """Gap-pad one unique chain's ``(header, row)`` records into the full complex width."""
    before, after = "-" * sum(widths[:index]), "-" * sum(widths[index + 1 :])
    lines: list[str] = []
    for header, row in records:
        lines.extend([f">{header}", f"{before}{row}{after}"])
    return lines


def _rows_to_a3m(rows: Sequence[str], chains: Sequence[str], remove_insertions: bool) -> str:
    """Validate ``:``-joined MSAInput rows and render ColabFold a3m text.

    A single chain becomes plain a3m. A complex becomes ColabFold's complex a3m:
    the header over the unique chains, the concatenated query, then one row per hit
    built from each unique chain's segment (all copies carry the same segment).
    Rows that align to several chains are used as the paired MSA; rows covering a
    single chain (the rest gaps) are used as unpaired. A heteromer also gets every
    chain's own segments as gap-padded unpaired rows, as ColabFold's own complex
    a3m does: with only paired rows a chain's unpaired MSA is empty and ColabFold
    refuses it.
    """
    split_rows = _validated_segments(rows, chains, remove_insertions)
    if len(chains) == 1:
        return _render((f"seq_{index}", segments[0]) for index, segments in enumerate(split_rows))

    unique, first_index, cardinalities = _unique_chains(chains)
    lines = _complex_preamble(unique, cardinalities)
    for index, segments in enumerate(split_rows):
        lines.extend([f">seq_{index}", "".join(segments[i] for i in first_index)])
    if len(unique) > 1:
        widths = [len(chain) for chain in unique]
        for index, chain in enumerate(unique):
            label = str(101 + index)
            unpaired = [(label, chain)]
            seen = {chain}
            for segments in split_rows:
                segment = segments[first_index[index]]
                if segment not in seen and set(aligned_columns(segment)) != {"-"}:
                    seen.add(segment)
                    unpaired.append((label, segment))
            lines.extend(_unpaired_block(index, widths, unpaired))
    return "\n".join(lines) + "\n"


def _text_to_a3m(text: str, chains: Sequence[str], remove_insertions: bool) -> str:
    """Validate a provided a3m file's text against ``chains`` and re-render it."""
    unique, _, cardinalities = _unique_chains(chains)
    lines = text.lstrip().splitlines()
    if lines[0].startswith("#"):
        lengths, counts = _parse_complex_header(lines[0].rstrip())
        if lengths != [len(chain) for chain in unique] or counts != cardinalities:
            raise ValueError(
                f"Complex a3m header (lengths {lengths}, cardinalities {counts}) does not match the requested "
                f"chains (lengths {[len(c) for c in unique]}, cardinalities {cardinalities})"
            )
        records = _validated_records(
            parse_a3m("\n".join(lines[1:])), "".join(unique), "Provided MSA", remove_insertions
        )
        return f"{_complex_header(unique, cardinalities)}\n{_render(records)}"
    if len(chains) > 1:
        raise ValueError(
            "A multi-chain MSA file must be ColabFold complex a3m (first line '#<lengths>\\t<cardinalities>'); "
            "a plain a3m would be folded as a single chain. Alternatively pass MSAInput(sequences=...) with "
            "':'-joined rows or a list of per-chain A3M text."
        )
    return _render(_validated_records(parse_a3m(text), chains[0], "Provided MSA", remove_insertions))


def _per_chain_to_a3m(msas: Sequence[Any], chains: Sequence[str]) -> str:
    """Render one-A3M-text-per-chain (unpaired) MSAs as ColabFold a3m.

    A single chain becomes plain a3m. For complexes, each unique chain's rows are
    gap-padded into the full complex width as unpaired rows; the paired MSA
    contains only the concatenated query, so AlphaFold gets no cross-chain pairing
    evidence from these inputs. ``None`` leaves that chain single-sequence (the
    server is not queried). Repeated chains must carry the same MSA or ``None``.
    """
    if len(msas) != len(chains):
        raise ValueError(f"options['msa'] needs one entry per chain ({len(chains)}); got {len(msas)}")
    unique, _, cardinalities = _unique_chains(chains)
    per_unique: list[list[tuple[str, str]] | None] = []
    for chain in unique:
        candidates: list[list[tuple[str, str]]] = []
        for position, (entry, entry_chain) in enumerate(zip(msas, chains, strict=True)):
            if entry_chain != chain or entry is None:
                continue
            if not isinstance(entry, str) or not entry.lstrip().startswith(">"):
                raise ValueError(f"options['msa'] entry {position} must be A3M text (starting with '>') or None")
            candidates.append(
                _validated_records(
                    parse_a3m(entry),
                    chain,
                    f"options['msa'] entry {position}",
                    remove_insertions=False,
                    dot_hint=_PER_CHAIN_DOT_HINT,
                )
            )
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
        return _render(single)

    widths = [len(chain) for chain in unique]
    lines = _complex_preamble(unique, cardinalities)
    for index, chain in enumerate(unique):
        hits = (per_unique[index] or [])[1:]
        lines.extend(_unpaired_block(index, widths, [(str(101 + index), chain), *hits]))
    return "\n".join(lines) + "\n"


# -- public entry points -------------------------------------------------------


def encode_msa_option(value: Any) -> Any:
    """Convert ``options["msa"]`` into a JSON-safe form, reading path-backed MSAs locally.

    Parameters
    ----------
    value
        ``None``, an :class:`~boileroom.inputs.MSAInput`, a list or tuple of
        ``str | None`` (one per chain), or an already-encoded payload.

    Returns
    -------
    Any
        ``None``, a list, or a ``{"kind": "msa_input", ...}`` dict with the rows
        (``sequences``) or the file's text (``a3m``).

    Raises
    ------
    ValueError
        If the value is none of the accepted forms.
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

    Parameters
    ----------
    msa
        An ``MSAInput``, an encoded payload from :func:`encode_msa_option`, or a
        list of per-chain A3M text (unpaired only); see the module docstring for
        the rules each form must meet.
    chains
        The requested chain sequences, repeated chains included.

    Returns
    -------
    str
        ColabFold a3m text: plain a3m for one chain, complex a3m otherwise.

    Raises
    ------
    ValueError
        On malformed input, when the alignment does not match the requested
        sequence(s), or when it holds content ColabFold would misread (``.``,
        insertions in the first row, differing segments for copies of a chain).
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
        return _rows_to_a3m(sequences, chains, remove_insertions)
    if not isinstance(text, str) or not text.lstrip().startswith((">", "#")):
        raise ValueError("Provided MSA must be in a3m/FASTA format (first line starts with '>' or a '#' header)")
    return _text_to_a3m(text, chains, remove_insertions)
