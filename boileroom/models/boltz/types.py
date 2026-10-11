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
    ptm: list[np.ndarray | None] | None = None
    iptm: list[np.ndarray | None] | None = None
    pae: list[np.ndarray | None] | None = None
    pde: list[np.ndarray | None] | None = None
    # Optional serialized structures (one string per sample)
    pdb: list[str] | None = None
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
                    raise ValueError(f"Boltz-2 pLDDT expected a 1D array; got {plddt.shape}")
                if plddt.size and np.nanmax(plddt) > 1.0:
                    plddt = plddt / 100.0
                plddt_samples.append(plddt.astype(np.float32, copy=False))
            self.plddt = plddt_samples if any(sample is not None for sample in plddt_samples) else None

        confidence_ptm, self.confidence = _extract_score(self.confidence, "ptm")
        confidence_iptm, self.confidence = _extract_score(self.confidence, "iptm")
        self.ptm = _normalize_scalar_scores(self.ptm, "Boltz-2 pTM") or confidence_ptm
        self.iptm = _normalize_scalar_scores(self.iptm, "Boltz-2 iPTM") or confidence_iptm


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


def _extract_score(
    confidence: list[dict[str, Any] | None] | None,
    key: str,
) -> tuple[list[np.ndarray | None] | None, list[dict[str, Any] | None] | None]:
    if confidence is None:
        return None, None

    scores: list[np.ndarray | None] = []
    stripped: list[dict[str, Any] | None] = []
    for item in confidence:
        if item is None:
            scores.append(None)
            stripped.append(None)
            continue
        next_item = dict(item)
        raw_score = next_item.pop(key, None)
        score = None if raw_score is None else np.asarray(raw_score, dtype=np.float32).squeeze()
        if score is not None:
            if score.ndim != 0:
                raise ValueError(f"Boltz-2 {key} expected a scalar score; got {score.shape}")
            score = score.reshape(1)
        scores.append(score)
        stripped.append(next_item or None)

    return (
        scores if any(score is not None for score in scores) else None,
        stripped if any(item is not None for item in stripped) else None,
    )
