"""Type definitions for Boltz2 outputs without heavy dependencies."""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import numpy as np

from ...base import StructurePrediction, PredictionMetadata


@dataclass
class Boltz2Output(StructurePrediction):
    """Output from Boltz-2 prediction including all model outputs."""

    # Required by StructurePrediction protocol
    metadata: PredictionMetadata
    atom_array: Optional[List[Any]] = None  # Always generated, one AtomArray per sample

    # Confidence-related outputs (one entry per sample)
    confidence: Optional[List[Dict[str, Any]]] = None
    plddt: Optional[List[np.ndarray]] = None
    pae: Optional[List[np.ndarray]] = None
    pde: Optional[List[np.ndarray]] = None
    # Optional serialized structures (one string per sample)
    pdb: Optional[List[str]] = None
    cif: Optional[List[str]] = None
