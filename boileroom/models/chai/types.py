"""Type definitions for Chai1 outputs without heavy dependencies."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ...base import PredictionMetadata, StructurePrediction

if TYPE_CHECKING:
    from biotite.structure import AtomArray


@dataclass
class Chai1Output(StructurePrediction):
    """Output from Chai-1 prediction including all model outputs."""

    metadata: PredictionMetadata
    atom_array: "list[AtomArray] | None" = None  # Always generated, one AtomArray per sample

    # Additional Chai-1-specific outputs (all optional, filtered by include_fields)
    pae: list[np.ndarray] | None = None
    pde: list[np.ndarray] | None = None
    plddt: list[np.ndarray] | None = None
    ptm: list[np.ndarray] | None = None
    iptm: list[np.ndarray] | None = None
    per_chain_iptm: list[np.ndarray] | None = None
    cif: list[str] | None = None
