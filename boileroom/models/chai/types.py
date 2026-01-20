"""Type definitions for Chai1 outputs without heavy dependencies."""

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import numpy as np

from ...base import StructurePrediction, PredictionMetadata

if TYPE_CHECKING:
    from biotite.structure import AtomArray


@dataclass
class Chai1Output(StructurePrediction):
    """Output from Chai-1 prediction including all model outputs."""

    metadata: PredictionMetadata
    atom_array: Optional["list[AtomArray]"] = None  # Always generated, one AtomArray per sample

    # Additional Chai-1-specific outputs (all optional, filtered by include_fields)
    pae: Optional[list[np.ndarray]] = None
    pde: Optional[list[np.ndarray]] = None
    plddt: Optional[list[np.ndarray]] = None
    ptm: Optional[list[np.ndarray]] = None
    iptm: Optional[list[np.ndarray]] = None
    per_chain_iptm: Optional[list[np.ndarray]] = None
    cif: Optional[list[str]] = None
