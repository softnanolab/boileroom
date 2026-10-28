"""Where the MSA of each chain comes from, for models that can fetch MSAs from a server (Boltz-2, Chai-1).

Every chain is given one source, in this order of precedence:

1. **a provided alignment** (``msa``) *of that sequence*: used as given. A provided alignment is a starting point for the
   sequence it is the alignment of (its query row), so it is used whichever chain has that sequence, and it is never
   searched for, fetched again or overwritten;
2. **no MSA server** (``use_msa_server=False``): the chain is folded from its sequence alone. The cache is not consulted,
   so a cached MSA is never used when no MSA is wanted;
3. **the cache**, when the server is used and the cache is on: the MSA of this sequence from an earlier fold;
4. **the MSA server**: the MSA of the chain on its own (unpaired) is fetched, and cached by sequence when the cache is
   on, so it is not asked for again unless the sequence changes.

So a provided alignment and an MSA fetched from the server behave the same afterwards. The only difference is that the
provided one is there from the start, so the server is not called for that sequence. If the sequence of a chain changes,
the provided alignment no longer applies to it and the new sequence gets an MSA by rules 2 to 4.

A model whose input format cannot mix chains that have a custom MSA with chains whose MSA it generates itself (Boltz) must
never be handed such a mix; that is why chains are fetched one by one whenever a custom MSA is in play.

No heavy imports here, so the logic can be tested without any of the models.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from .inputs import a3m_rows, aligned_columns, parse_a3m

#: Field value for a chain that is folded from its sequence alone.
EMPTY = "empty"
#: Field value for a chain whose MSA the model generates itself, together with the other chains.
AUTO = ""

#: A chain's MSA field: the file of its MSA, :data:`EMPTY` or :data:`AUTO`.
MSAField = Path | str


def provided_alignments(msa: Sequence[str | None] | None, n_chains: int) -> dict[str, str]:
    """The caller's alignments by the sequence each one is the alignment of (its query row).

    Parameters
    ----------
    msa
        One A3M text (or ``None``) per chain, or ``None`` for no alignments. The position of an entry does not matter
        beyond that: an alignment applies to the sequence it is the alignment of, wherever that sequence occurs.
    n_chains
        The number of chains of the input.

    Raises
    ------
    ValueError
        If there is not one entry per chain, an entry is not A3M text (rows of the same aligned length, the first
        being the query), or two different alignments are given for one sequence.
    """
    if msa is None:
        return {}
    if len(msa) != n_chains:
        raise ValueError(f"msa must have one A3M string or None per chain ({n_chains}); got {len(msa)}")
    alignments: dict[str, str] = {}
    for position, alignment in enumerate(msa):
        if not alignment:
            continue
        if not isinstance(alignment, str) or not alignment.lstrip().startswith(">"):
            raise ValueError(f"msa entry {position} must be A3M text (starting with '>') or None")
        query = aligned_columns(parse_a3m(alignment)[0][1])
        a3m_rows(alignment, query)  # every row has the aligned length of the query
        if alignments.setdefault(query, alignment) != alignment:
            raise ValueError("Two different alignments were given for the same sequence")
    return alignments


def resolve_chain_msas(
    chains: Sequence[str],
    *,
    provided: Mapping[str, Path],
    use_msa_server: bool,
    cache_enabled: bool,
    lookup: Callable[[Sequence[str]], Mapping[str, Path]],
    fetch: Callable[[str], Path],
    store: Callable[[str, Path], None],
    sources: dict[int, str] | None = None,
    always_per_chain: bool = False,
) -> dict[int, MSAField]:
    """Decide the MSA field of every chain (see the module docstring for the rules).

    Parameters
    ----------
    chains
        The individual chain sequences.
    provided
        Files of the caller's alignments, by the sequence they are the alignment of.
    use_msa_server
        Whether chains without a provided alignment may get one from the MSA server.
    cache_enabled
        Whether MSAs fetched from the server are cached by sequence.
    lookup
        Cached MSA files for the sequences that have one. Only called when the cache is used.
    fetch
        Fetch the MSA of one sequence on its own from the server and return its file.
    store
        Put a fetched MSA file into the cache for a sequence.
    sources
        If given, filled with where each chain's MSA comes from, by chain index: ``"provided"``, ``"cache"``,
        ``"fetched"``, ``"empty"`` or ``"auto"``.
    always_per_chain
        Fetch chains one by one even with the cache off and nothing provided (instead of leaving all MSAs to the model).

    Returns
    -------
    dict[int, Path | str]
        By chain index: a file, :data:`EMPTY` or :data:`AUTO`.
    """
    fields: dict[int, MSAField] = {}
    sources = {} if sources is None else sources
    seeded = any(sequence in provided for sequence in chains)
    # chains are fetched one by one when a provided alignment is in play (a model cannot generate the others next to it)
    # or the cache is on (the MSA of a sequence must not depend on the other chains for it to be cacheable)
    per_chain = use_msa_server and (always_per_chain or cache_enabled or seeded)
    open_sequences = [sequence for sequence in chains if sequence not in provided]
    cached: Mapping[str, Path] = lookup(open_sequences) if (use_msa_server and cache_enabled and open_sequences) else {}
    fetched: dict[str, Path] = {}
    for index, sequence in enumerate(chains):
        if sequence in provided:
            fields[index], sources[index] = provided[sequence], "provided"
        elif not use_msa_server:
            fields[index], sources[index] = EMPTY, "empty"
        elif sequence in cached:
            fields[index], sources[index] = cached[sequence], "cache"
        elif per_chain:
            if sequence not in fetched:
                fetched[sequence] = fetch(sequence)
                if cache_enabled:
                    store(sequence, fetched[sequence])
            fields[index], sources[index] = fetched[sequence], "fetched"
        else:
            fields[index], sources[index] = AUTO, "auto"
    return fields
