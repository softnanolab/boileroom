"""Type definitions for MiniFold outputs without heavy dependencies."""

from dataclasses import dataclass
from typing import Optional, List, TYPE_CHECKING

import numpy as np

from ...base import StructurePrediction, PredictionMetadata

if TYPE_CHECKING:
    from biotite.structure import AtomArray


@dataclass
class MiniFoldOutput(StructurePrediction):
    """Output from MiniFold prediction including all model outputs."""

    # Required by StructurePrediction protocol
    metadata: PredictionMetadata
    atom_array: Optional[List["AtomArray"]] = None  # Always generated, one AtomArray per sample

    # Confidence-related outputs (one list entry per sample)
    plddt: Optional[List[np.ndarray]] = None  # [num_residues] per sequence
    pae: Optional[List[np.ndarray]] = None  # [num_residues, num_residues] per sequence

    # Multimer-related outputs
    residue_index: Optional[np.ndarray] = None  # (batch_size, residue)
    chain_index: Optional[np.ndarray] = None  # (batch_size, residue)

    # Optional serialized structures (one string per sample)
    pdb: Optional[list[str]] = None
    cif: Optional[list[str]] = None
