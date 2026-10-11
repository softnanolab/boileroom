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
    plddt: list[np.ndarray | None] | None = None
    ptm: list[np.ndarray | None] | None = None
    iptm: list[np.ndarray | None] | None = None
    per_chain_iptm: list[np.ndarray] | None = None
    cif: list[str] | None = None

    def __post_init__(self) -> None:
        """Normalize confidence metrics to the public output contract."""
        if self.plddt is not None:
            plddt_samples: list[np.ndarray | None] = []
            for sample in self.plddt:
                if sample is None:
                    plddt_samples.append(None)
                    continue
                plddt = np.asarray(sample, dtype=np.float32)
                while plddt.ndim > 1 and plddt.shape[0] == 1:
                    plddt = plddt[0]
                if plddt.ndim != 1:
                    raise ValueError(f"Chai-1 pLDDT expected a 1D array; got {plddt.shape}")
                if plddt.size and np.nanmax(plddt) > 1.0:
                    plddt = plddt / 100.0
                plddt_samples.append(plddt.astype(np.float32, copy=False))
            self.plddt = plddt_samples if any(sample is not None for sample in plddt_samples) else None

        self.ptm = _normalize_scalar_scores(self.ptm, "Chai-1 pTM")
        self.iptm = _normalize_scalar_scores(self.iptm, "Chai-1 iPTM")


def _normalize_scalar_scores(values: list[np.ndarray | None] | None, label: str) -> list[np.ndarray | None] | None:
    if values is None:
        return None

    scores: list[np.ndarray | None] = []
    for value in values:
        if value is None:
            scores.append(None)
            continue
        score = np.asarray(value, dtype=np.float32).squeeze()
        if score.ndim != 0:
            raise ValueError(f"{label} expected a scalar score; got {score.shape}")
        scores.append(score.reshape(1))
    return scores if any(score is not None for score in scores) else None
