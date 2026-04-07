"""Type definitions for Boltz2 outputs without heavy dependencies."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from ...base import PredictionMetadata, StructurePrediction


@dataclass
class Boltz2Output(StructurePrediction):
    """Output from Boltz-2 prediction including all model outputs."""

    # Required by StructurePrediction protocol
    metadata: PredictionMetadata
    atom_array: list[Any] | None = None  # Always generated, one AtomArray per sample

    # Confidence-related outputs (one entry per sample)
    confidence: list[dict[str, Any] | None] | None = None
    plddt: list[np.ndarray | None] | None = None
    pae: list[np.ndarray | None] | None = None
    pde: list[np.ndarray | None] | None = None
    # Optional serialized structures (one string per sample)
    pdb: list[str] | None = None
    cif: list[str] | None = None
