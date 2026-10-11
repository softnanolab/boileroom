"""Type definitions for ESM outputs without heavy dependencies."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ...base import EmbeddingPrediction, PredictionMetadata, StructurePrediction

if TYPE_CHECKING:
    from biotite.structure import AtomArray

CA_ATOM37_INDEX = 1


@dataclass
class ESM2Output(EmbeddingPrediction):
    """Output from ESM2 prediction including all model outputs."""

    embeddings: np.ndarray  # (batch_size, seq_len, embedding_dim)
    metadata: PredictionMetadata
    chain_index: np.ndarray  # (batch_size, seq_len)
    residue_index: np.ndarray  # (batch_size, seq_len)
    hidden_states: np.ndarray | None = None  # (batch_size, hidden_state_iter, seq_len, embedding_dim)
    lm_logits: np.ndarray | None = None  # (batch_size, seq_len, vocab_size)


@dataclass
class ESMFoldOutput(StructurePrediction):
    """Output from ESMFold prediction including all model outputs."""

    # Required by StructurePrediction protocol
    metadata: PredictionMetadata
    atom_array: list["AtomArray"] | None = None  # Always generated, one AtomArray per sample

    # Additional ESMFold-specific outputs (all optional, filtered by include_fields)
    frames: np.ndarray | None = None  # (model_layer, batch_size, residue, qxyz=7)
    sidechain_frames: np.ndarray | None = None  # (model_layer, batch_size, residue, 8, 4, 4) [rot matrix per sidechain]
    unnormalized_angles: np.ndarray | None = None  # (model_layer, batch_size, residue, 7, 2) [torsion angles]
    angles: np.ndarray | None = None  # (model_layer, batch_size, residue, 7, 2) [torsion angles]
    states: np.ndarray | None = None  # (model_layer, batch_size, residue, ???)
    s_s: np.ndarray | None = None  # (batch_size, residue, 1024)
    s_z: np.ndarray | None = None  # (batch_size, residue, residue, 128)
    distogram_logits: np.ndarray | None = None  # (batch_size, residue, residue, 64) ???
    lm_logits: np.ndarray | None = None  # (batch_size, residue, 23) ???
    aatype: np.ndarray | None = None  # (batch_size, residue) amino acid identity
    atom14_atom_exists: np.ndarray | None = None  # (batch_size, residue, atom=14)
    residx_atom14_to_atom37: np.ndarray | None = None  # (batch_size, residue, atom=14)
    residx_atom37_to_atom14: np.ndarray | None = None  # (batch_size, residue, atom=37)
    atom37_atom_exists: np.ndarray | None = None  # (batch_size, residue, atom=37)
    residue_index: np.ndarray | None = None  # (batch_size, residue)
    lddt_head: np.ndarray | None = None  # (model_layer, batch_size, residue, atom=37, 50) ??
    plddt: list[np.ndarray | None] | None = None  # one unit-scale (residue,) array per sample
    ptm_logits: np.ndarray | None = None  # (batch_size, residue, residue, 64) ???
    ptm: list[np.ndarray | None] | None = None  # one shape-(1,) array per sample
    aligned_confidence_probs: np.ndarray | None = None  # (batch_size, residue, residue, 64)
    pae: np.ndarray | None = None  # (batch_size, residue, residue)
    max_pae: np.ndarray | None = None  # float # TODO: make it into a float when sending to the client
    chain_index: np.ndarray | None = None  # (batch_size, residue)
    pdb: list[str] | None = None  # 0-indexed
    cif: list[str] | None = None  # 0-indexed

    # TODO: can add a save method here (to a pickle and a pdb file) that can be run locally
    # TODO: add verification of the outputs, and primarily the shape of all the arrays
    # (see test_esmfold_batch_multimer_linkers for the exact batched shapes)

    def __post_init__(self) -> None:
        """Normalize confidence metrics to the public output contract."""
        self.plddt = _normalize_esmfold_plddt(self.plddt, self.metadata.sequence_lengths)
        if self.ptm is not None:
            ptm = np.asarray(self.ptm, dtype=np.float32).reshape(-1)
            self.ptm = [np.array([score], dtype=np.float32) for score in ptm]


def _normalize_esmfold_plddt(
    values: np.ndarray | list[np.ndarray | None] | None,
    sequence_lengths: list[int] | None,
) -> list[np.ndarray | None] | None:
    if values is None:
        return None

    if isinstance(values, list):
        samples = [_normalize_esmfold_plddt_sample(sample) if sample is not None else None for sample in values]
        return samples if any(sample is not None for sample in samples) else None

    plddt = np.asarray(values, dtype=np.float32)
    if plddt.ndim == 3:
        plddt = plddt[:, :, CA_ATOM37_INDEX]
    elif plddt.ndim == 1:
        plddt = plddt[None, :]
    if plddt.ndim != 2:
        raise ValueError(f"ESMFold pLDDT expected shape (batch, residue); got {plddt.shape}")

    if plddt.size and np.nanmax(plddt) > 1.0:
        plddt = plddt / 100.0

    lengths = sequence_lengths or [plddt.shape[1]] * plddt.shape[0]
    if len(lengths) != plddt.shape[0]:
        raise ValueError(f"ESMFold pLDDT batch size {plddt.shape[0]} does not match sequence lengths {len(lengths)}")
    return [plddt[index, :length].astype(np.float32, copy=False) for index, length in enumerate(lengths)]


def _normalize_esmfold_plddt_sample(sample: np.ndarray) -> np.ndarray:
    plddt = np.asarray(sample, dtype=np.float32)
    if plddt.ndim == 2 and plddt.shape[-1] == 37:
        plddt = plddt[:, CA_ATOM37_INDEX]
    while plddt.ndim > 1 and plddt.shape[0] == 1:
        plddt = plddt[0]
    if plddt.ndim != 1:
        raise ValueError(f"ESMFold pLDDT expected a 1D array; got {plddt.shape}")
    if plddt.size and np.nanmax(plddt) > 1.0:
        plddt = plddt / 100.0
    return plddt.astype(np.float32, copy=False)
