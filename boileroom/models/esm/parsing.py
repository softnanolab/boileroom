"""Token-aware parsing utilities for ESM-family sequence inputs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

MASK_TOKEN = "<mask>"
ESM2_RESIDUE_TOKENS = frozenset("ACDEFGHIKLMNPQRSTVWYXBZOU")


@dataclass(frozen=True)
class ESMChainTokens:
    """Parsed residue tokens for a single chain."""

    tokens: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.tokens)

    def to_string(self) -> str:
        return "".join(self.tokens)


@dataclass(frozen=True)
class ESMSequenceTokens:
    """Parsed residue tokens for one ESM input sequence."""

    original: str
    chains: tuple[ESMChainTokens, ...]

    @property
    def is_multimer(self) -> bool:
        return len(self.chains) > 1

    @property
    def residue_count(self) -> int:
        return sum(len(chain) for chain in self.chains)

    def to_string(self, glycine_linker: str = "") -> str:
        return glycine_linker.join(chain.to_string() for chain in self.chains)


def _invalid_fragment(sequence: str, start: int) -> str:
    end = start + 1
    while end < len(sequence):
        if sequence.startswith(MASK_TOKEN, end) or sequence[end] == ":" or sequence[end] in ESM2_RESIDUE_TOKENS:
            break
        end += 1
    return sequence[start:end]


def parse_esm2_sequence(sequence: str) -> ESMSequenceTokens:
    """Parse an ESM2 sequence into residue-aware tokens.

    Parameters
    ----------
    sequence : str
        Sequence containing one-letter residues, optional inline ``<mask>`` tokens,
        and optional ``:`` chain separators.

    Returns
    -------
    ESMSequenceTokens
        Parsed representation with per-chain residue tokens.

    Raises
    ------
    ValueError
        If the sequence contains unsupported fragments or empty chains.
    """

    if sequence == "":
        raise ValueError("ESM2 sequence must contain at least one residue token.")

    chains: list[ESMChainTokens] = []
    current_chain: list[str] = []
    index = 0

    while index < len(sequence):
        if sequence.startswith(MASK_TOKEN, index):
            current_chain.append(MASK_TOKEN)
            index += len(MASK_TOKEN)
            continue

        token = sequence[index]
        if token == ":":
            if not current_chain:
                raise ValueError(f"Invalid ESM2 sequence {sequence!r}: empty chain near ':'.")
            chains.append(ESMChainTokens(tuple(current_chain)))
            current_chain = []
            index += 1
            continue

        if token in ESM2_RESIDUE_TOKENS:
            current_chain.append(token)
            index += 1
            continue

        fragment = _invalid_fragment(sequence, index)
        raise ValueError(f"Invalid ESM2 token fragment {fragment!r} in sequence {sequence!r}.")

    if not current_chain:
        raise ValueError(f"Invalid ESM2 sequence {sequence!r}: empty chain near ':'.")

    chains.append(ESMChainTokens(tuple(current_chain)))
    return ESMSequenceTokens(original=sequence, chains=tuple(chains))


def parse_esm2_sequences(sequences: str | Sequence[str]) -> list[ESMSequenceTokens]:
    """Parse one or more ESM2 input sequences."""

    sequence_list = [sequences] if isinstance(sequences, str) else list(sequences)
    return [parse_esm2_sequence(sequence) for sequence in sequence_list]
