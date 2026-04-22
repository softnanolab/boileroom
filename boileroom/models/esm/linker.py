import numpy as np
import torch
import torch.nn.functional as F

from .parsing import ESMSequenceTokens


ESMSequenceLike = str | ESMSequenceTokens


def _tokenize_raw_sequence(sequence: str) -> list[tuple[str, ...]]:
    chains: list[tuple[str, ...]] = []
    current_chain: list[str] = []
    index = 0

    while index < len(sequence):
        if sequence.startswith("<mask>", index):
            current_chain.append("<mask>")
            index += len("<mask>")
            continue

        token = sequence[index]
        if token == ":":
            if not current_chain:
                raise ValueError(f"Invalid multimer sequence {sequence!r}: empty chain near ':'.")
            chains.append(tuple(current_chain))
            current_chain = []
            index += 1
            continue

        current_chain.append(token)
        index += 1

    if not current_chain:
        raise ValueError(f"Invalid multimer sequence {sequence!r}: empty chain near ':'.")

    chains.append(tuple(current_chain))
    return chains


def _split_chains(sequence: ESMSequenceLike) -> list[tuple[str, ...]]:
    if isinstance(sequence, ESMSequenceTokens):
        return [chain.tokens for chain in sequence.chains]
    return _tokenize_raw_sequence(sequence)


# --- Glycine linker and positional skip utilities ---
def compute_position_ids(
    sequences: list[ESMSequenceLike],
    glycine_linker: str,
    position_ids_skip: int,
    *,
    add_special_tokens: bool = False,
) -> torch.Tensor:
    """
    Compute the position ids for the sequences.
    Parameters
    ----------
    sequences: List of sequences, each containing chains separated by ":".
    glycine_linker: The glycine linker string used between chains represented as a string (e.g. "GGGG").
    position_ids_skip: The number of positions to skip between chains.
    add_special_tokens: Whether to pad the position ids with leading and trailing
        positions for tokenizer-added special tokens.
    Returns
    -------
    torch.Tensor: The position ids for the sequences
    """
    position_ids = []
    linker_length = len(glycine_linker)
    for multimer_seq in sequences:
        multimer_position_ids = []
        previous_chain_end = 0
        chains = _split_chains(multimer_seq)
        for chain_id, chain_seq in enumerate(chains):
            intrachain_position_ids = np.arange(len(chain_seq), dtype=np.int64)
            if chain_id != 0 and intrachain_position_ids.size > 0:
                intrachain_position_ids = (intrachain_position_ids + (previous_chain_end + 1)) + position_ids_skip
            # add linker if not last chain
            if chain_id != len(chains) - 1 and linker_length > 0:
                linker_start = previous_chain_end + 1
                if intrachain_position_ids.size > 0:
                    linker_start = int(intrachain_position_ids[-1]) + 1
                linker_position_ids = np.arange(linker_length, dtype=np.int64) + linker_start
                intrachain_position_ids = np.concatenate([intrachain_position_ids, linker_position_ids])
            if intrachain_position_ids.size > 0:
                previous_chain_end = int(intrachain_position_ids[-1])
            multimer_position_ids += intrachain_position_ids.tolist()
        position_tensor = torch.tensor(multimer_position_ids, dtype=torch.long)
        if add_special_tokens:
            position_tensor = torch.cat(
                [torch.zeros(1, dtype=torch.long), position_tensor, torch.zeros(1, dtype=torch.long)]
            )
        position_ids.append(position_tensor)
    # add padding to the position ids
    max_length = max(len(ids) for ids in position_ids)
    for i, pos_ids in enumerate(position_ids):
        position_ids[i] = torch.cat([pos_ids, torch.zeros(max_length - len(pos_ids), dtype=torch.long)])
    return torch.stack(position_ids)


def store_multimer_properties(_sequences: list[ESMSequenceLike], glycine_linker: str):
    """Store properties needed for multimer processing.

    Parameters
    ----------
    _sequences : List[str]
        List of sequences, each containing chains separated by ":".
    glycine_linker : str
        The glycine linker string used between chains.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        A tuple containing:
        - linker_map: tensor of shape (batch_size, sequence_length) where 0 indicates
          linker positions and 1 indicates chain positions.
        - residue_index: tensor of shape (batch_size, sequence_length) containing
          residue indices that restart at 1 for each chain.
        - chain_index: tensor of shape (batch_size, sequence_length) containing
          chain indices (0, 1, 2, etc.).
    """
    linker_map = []
    residue_index = []
    chain_index = []
    assert len(_sequences) > 0, "Sequences must not be empty"
    linker_length = len(glycine_linker)
    for seq in _sequences:
        chains = _split_chains(seq)
        full_seq_len = sum(len(chain) for chain in chains) + max(len(chains) - 1, 0) * linker_length
        seq_mask = torch.ones(full_seq_len, dtype=torch.long)
        res_index = torch.zeros(full_seq_len, dtype=torch.long)
        ch_index = torch.zeros(full_seq_len, dtype=torch.long)
        current_pos = 0
        for i, chain in enumerate(chains):
            ch_index[current_pos : current_pos + len(chain)] = i
            res_index[current_pos : current_pos + len(chain)] = torch.arange(0, len(chain))
            current_pos += len(chain)
            if i < len(chains) - 1:
                seq_mask[current_pos : current_pos + linker_length] = 0
                ch_index[current_pos : current_pos + linker_length] = i
                res_index[current_pos : current_pos + linker_length] = torch.arange(
                    len(chain) + 1, len(chain) + linker_length + 1
                )
                current_pos += linker_length
        linker_map.append(seq_mask)
        residue_index.append(res_index)
        chain_index.append(ch_index)
    linker_max_size = max(tensor.size(0) for tensor in linker_map)
    residue_index_max_size = max(tensor.size(0) for tensor in residue_index)
    chain_index_max_size = max(tensor.size(0) for tensor in chain_index)
    max_size = max(linker_max_size, residue_index_max_size, chain_index_max_size)
    padded_linker_map = [F.pad(tensor, (0, max_size - tensor.size(0)), value=-1) for tensor in linker_map]
    padded_residue_index = [F.pad(tensor, (0, max_size - tensor.size(0)), value=-1) for tensor in residue_index]
    padded_chain_index = [F.pad(tensor, (0, max_size - tensor.size(0)), value=-1) for tensor in chain_index]
    return (
        torch.stack(padded_linker_map),
        torch.stack(padded_residue_index),
        torch.stack(padded_chain_index),
    )


def replace_glycine_linkers(sequences: list[ESMSequenceLike], glycine_linker: str) -> list[str]:
    return [glycine_linker.join("".join(chain) for chain in _split_chains(multimer_seq)) for multimer_seq in sequences]
