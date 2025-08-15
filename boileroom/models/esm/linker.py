import numpy as np
from typing import List

from ...images import esm_image

with esm_image.imports():
    import torch
    import torch.nn.functional as F


# --- Glycine linker and positional skip utilities ---
def compute_position_ids(sequences: List[str], glycine_linker: str, position_ids_skip: int) -> torch.Tensor:
    """
    Compute the position ids for the sequences.
    Parameters
    ----------
    sequences: List of sequences, each containing chains separated by ":".
    glycine_linker: The glycine linker string used between chains represented as a string (e.g. "GGGG").
    position_ids_skip: The number of positions to skip between chains.
    Returns
    -------
    torch.Tensor: The position ids for the sequences
    """
    position_ids = []
    for multimer_seq in sequences:
        multimer_position_ids = []
        previous_chain_end = 0
        for chain_id, chain_seq in enumerate(multimer_seq.split(":")):
            intrachain_position_ids = np.arange(len(chain_seq))
            if chain_id != 0:
                intrachain_position_ids = (intrachain_position_ids + (previous_chain_end + 1)) + position_ids_skip
            # add linker if not last chain
            if chain_id != len(multimer_seq.split(":")) - 1:
                linker_position_ids = np.arange(len(glycine_linker)) + intrachain_position_ids[-1] + 1
                intrachain_position_ids = np.concatenate([intrachain_position_ids, linker_position_ids])
            previous_chain_end = intrachain_position_ids[-1]
            multimer_position_ids += intrachain_position_ids.tolist()
        position_ids.append(torch.tensor(multimer_position_ids))
    # add padding to the position ids
    max_length = max(len(ids) for ids in position_ids)
    for i, pos_ids in enumerate(position_ids):
        position_ids[i] = torch.cat([pos_ids, torch.zeros(max_length - len(pos_ids), dtype=torch.long)])
    return torch.stack(position_ids)


def store_multimer_properties(_sequences: List[str], glycine_linker: str):
    """Store properties needed for multimer processing.
    Args:
        _sequences: List of sequences, each containing chains separated by ":"
        glycine_linker: The glycine linker string used between chains
    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - linker_map: tensor of shape (batch_size, sequence_length) where 0 indicates
            linker positions and 1 indicates chain positions
            - residue_index: tensor of shape (batch_size, sequence_length) containing
            residue indices that restart at 1 for each chain
            - chain_index: tensor of shape (batch_size, sequence_length) containing
            chain indices (0, 1, 2, etc.)
    """
    linker_map = []
    residue_index = []
    chain_index = []
    assert len(_sequences) > 0, "Sequences must not be empty"
    for seq in _sequences:
        full_seq_len = len(seq.replace(":", glycine_linker))
        seq_mask = torch.ones(full_seq_len, dtype=torch.long)
        res_index = torch.zeros(full_seq_len, dtype=torch.long)
        ch_index = torch.zeros(full_seq_len, dtype=torch.long)
        current_pos = 0
        chains = seq.split(":")
        for i, chain in enumerate(chains):
            ch_index[current_pos : current_pos + len(chain)] = i
            res_index[current_pos : current_pos + len(chain)] = torch.arange(0, len(chain))
            current_pos += len(chain)
            if i < len(chains) - 1:
                seq_mask[current_pos : current_pos + len(glycine_linker)] = 0
                ch_index[current_pos : current_pos + len(glycine_linker)] = i
                res_index[current_pos : current_pos + len(glycine_linker)] = torch.arange(
                    len(chain) + 1, len(chain) + len(glycine_linker) + 1
                )
                current_pos += len(glycine_linker)
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


def replace_glycine_linkers(sequences: List[str], glycine_linker: str) -> List[str]:
    return [multimer_seq.replace(":", glycine_linker) for multimer_seq in sequences]
