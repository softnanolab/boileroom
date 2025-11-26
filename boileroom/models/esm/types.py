"""Type definitions for ESM outputs without heavy dependencies."""

from dataclasses import dataclass
from typing import Optional, List, TYPE_CHECKING

import numpy as np

from ...base import EmbeddingPrediction, PredictionMetadata, StructurePrediction

if TYPE_CHECKING:
    from biotite.structure import AtomArray


@dataclass
class ESM2Output(EmbeddingPrediction):
    """Output from ESM2 prediction including all model outputs."""

    embeddings: np.ndarray  # (batch_size, seq_len, embedding_dim)
    metadata: PredictionMetadata
    chain_index: np.ndarray  # (batch_size, seq_len)
    residue_index: np.ndarray  # (batch_size, seq_len)
    hidden_states: Optional[np.ndarray] = None  # (batch_size, hidden_state_iter, seq_len, embedding_dim)


@dataclass
class ESMFoldOutput(StructurePrediction):
    """Output from ESMFold prediction including all model outputs."""

    # Required by StructurePrediction protocol
    metadata: PredictionMetadata
    atom_array: Optional[List["AtomArray"]] = None  # Always generated, one AtomArray per sample

    # Additional ESMFold-specific outputs (all optional, filtered by include_fields)
    frames: Optional[np.ndarray] = None  # (model_layer, batch_size, residue, qxyz=7)
    sidechain_frames: Optional[np.ndarray] = (
        None  # (model_layer, batch_size, residue, 8, 4, 4) [rot matrix per sidechain]
    )
    unnormalized_angles: Optional[np.ndarray] = None  # (model_layer, batch_size, residue, 7, 2) [torsion angles]
    angles: Optional[np.ndarray] = None  # (model_layer, batch_size, residue, 7, 2) [torsion angles]
    states: Optional[np.ndarray] = None  # (model_layer, batch_size, residue, ???)
    s_s: Optional[np.ndarray] = None  # (batch_size, residue, 1024)
    s_z: Optional[np.ndarray] = None  # (batch_size, residue, residue, 128)
    distogram_logits: Optional[np.ndarray] = None  # (batch_size, residue, residue, 64) ???
    lm_logits: Optional[np.ndarray] = None  # (batch_size, residue, 23) ???
    aatype: Optional[np.ndarray] = None  # (batch_size, residue) amino acid identity
    atom14_atom_exists: Optional[np.ndarray] = None  # (batch_size, residue, atom=14)
    residx_atom14_to_atom37: Optional[np.ndarray] = None  # (batch_size, residue, atom=14)
    residx_atom37_to_atom14: Optional[np.ndarray] = None  # (batch_size, residue, atom=37)
    atom37_atom_exists: Optional[np.ndarray] = None  # (batch_size, residue, atom=37)
    residue_index: Optional[np.ndarray] = None  # (batch_size, residue)
    lddt_head: Optional[np.ndarray] = None  # (model_layer, batch_size, residue, atom=37, 50) ??
    plddt: Optional[np.ndarray] = None  # (batch_size, residue, atom=37)
    ptm_logits: Optional[np.ndarray] = None  # (batch_size, residue, residue, 64) ???
    ptm: Optional[np.ndarray] = None  # float # TODO: make it into a float when sending to the client
    aligned_confidence_probs: Optional[np.ndarray] = None  # (batch_size, residue, residue, 64)
    predicted_aligned_error: Optional[np.ndarray] = None  # (batch_size, residue, residue)
    max_predicted_aligned_error: Optional[np.ndarray] = (
        None  # float # TODO: make it into a float when sending to the client
    )
    chain_index: Optional[np.ndarray] = None  # (batch_size, residue)
    pdb: Optional[list[str]] = None  # 0-indexed
    cif: Optional[list[str]] = None  # 0-indexed

    # TODO: can add a save method here (to a pickle and a pdb file) that can be run locally
    # TODO: add verification of the outputs, and primarily the shape of all the arrays
    # (see test_esmfold_batch_multimer_linkers for the exact batched shapes)
