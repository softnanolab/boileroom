"""The Boltz-2 side of MSA handling: staging provided alignments, cache keys and the FASTA headers.

Which source each chain's MSA comes from is decided by :func:`boileroom.msa_sources.resolve_chain_msas`. Boltz refuses an
input that mixes chains with custom MSAs (a file path, or ``empty``, in the FASTA header) and chains whose MSA it should
generate itself ("Cannot mix custom and auto-generated MSAs in the same input!"). The MSA cache was exactly such a mix
whenever some chains were cached and others new (a fixed target and a new binder at every step of a design loop), so Boltz
is only ever handed custom files or ``empty``, except when nothing is cached or provided and it generates all MSAs itself.

This module has no heavy imports, so it can be tested without Boltz.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from ...msa_cache import MSACache
from ...msa_sources import AUTO, EMPTY, MSAField, provided_alignments, resolve_chain_msas

__all__ = [
    "AUTO",
    "EMPTY",
    "cache_key",
    "chains_to_fasta",
    "provided_alignments",
    "resolve_chain_msas",
    "split_chains",
    "stage_provided_alignments",
]

#: Cache entries made by earlier versions are keyed by the sequence alone and hold MSAs that were paired with whichever
#: chains the first fold happened to contain; this namespace keeps them from ever being reused.
CACHE_KEY_VERSION = "boltz2-unpaired-v2"


def split_chains(sequences: Sequence[str]) -> list[str]:
    """Individual chain sequences of ``sequences`` (entries may be ``:``-joined multimers), in order."""
    chains: list[str] = []
    for entry in sequences:
        chains.extend(part.strip() for part in entry.split(":") if part.strip())
    return chains


def cache_key(sequence: str, server_url: str, max_msa_seqs: int) -> str:
    """Cache key of the MSA of ``sequence``: it depends on the sequence and on what was asked of the server."""
    return MSACache.hash_key(f"{CACHE_KEY_VERSION}|{server_url}|{max_msa_seqs}|{sequence}")


def stage_provided_alignments(alignments: Mapping[str, str], chains: Sequence[str], directory: Path) -> dict[str, Path]:
    """Write the provided alignments of the sequences that occur in ``chains`` to ``.a3m`` files (Boltz reads those).

    Returns the file of each such sequence. An alignment of a sequence that no chain has (the chain was changed) is not
    used and not written.
    """
    staged: dict[str, Path] = {}
    for sequence in dict.fromkeys(chains):
        if sequence not in alignments:
            continue
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"provided_{len(staged)}.a3m"
        path.write_text(alignments[sequence], encoding="utf-8")
        staged[sequence] = path
    return staged


def chains_to_fasta(chains: Sequence[str], fields: Mapping[int, MSAField]) -> str:
    """Boltz-2 FASTA text with chain ids A, B, C, ... and each chain's MSA field in its header."""
    lines: list[str] = []
    for index, sequence in enumerate(chains):
        field = fields.get(index, AUTO)
        value = str(Path(field).resolve()) if isinstance(field, Path) else field
        lines.append(f">{chr(65 + index)}|protein|{value}")
        lines.append(sequence)
    return "\n".join(lines)
