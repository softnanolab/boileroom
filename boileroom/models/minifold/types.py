"""MiniFold output types."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ...base import PredictionMetadata, StructurePrediction

if TYPE_CHECKING:
    from biotite.structure import AtomArray


@dataclass
class MiniFoldOutput(StructurePrediction):
    """MiniFold monomer prediction, with CPU arrays only.

    plddt is per-residue on the 0–1 scale; atom-array/PDB/mmCIF B-factors
    retain upstream 0–100 confidence. residue_index is zero-based with -1
    batch padding; structural res_id is one-based. PAE is not produced.
    """

    # Required by StructurePrediction protocol
    metadata: PredictionMetadata
    atom_array: list["AtomArray"] | None = None  # Always generated, one AtomArray per sample

    # Confidence-related outputs (one list entry per sample)
    plddt: list[np.ndarray] | None = None  # [num_residues] per sequence
    pae: list[np.ndarray] | None = None  # [num_residues, num_residues] per sequence

    # Multimer-related outputs
    residue_index: np.ndarray | None = None  # (batch_size, residue)
    chain_index: np.ndarray | None = None  # (batch_size, residue)

    # Optional serialized structures (one string per sample)
    pdb: list[str] | None = None
    cif: list[str] | None = None
