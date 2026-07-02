"""Type definitions for Protenix outputs without heavy runtime dependencies."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from ...base import PredictionMetadata, StructurePrediction


@dataclass
class ProtenixOutput(StructurePrediction):
    """Output from Protenix structure prediction."""

    metadata: PredictionMetadata
    atom_array: list[Any] | None = None

    confidence: list[dict[str, Any] | None] | None = None
    plddt: list[np.ndarray | None] | None = None
    ptm: list[np.ndarray | None] | None = None
    iptm: list[np.ndarray | None] | None = None
    pdb: list[str] | None = None
    cif: list[str] | None = None

    def __post_init__(self) -> None:
        """Normalize common confidence scalars to the public output contract."""
        self.ptm = _normalize_scalar_scores(self.ptm, "Protenix pTM") or _extract_scalar(self.confidence, "ptm")
        self.iptm = _normalize_scalar_scores(self.iptm, "Protenix iPTM") or _extract_scalar(self.confidence, "iptm")

        if self.plddt is not None:
            normalized: list[np.ndarray | None] = []
            for value in self.plddt:
                if value is None:
                    normalized.append(None)
                    continue
                arr = np.asarray(value, dtype=np.float32).squeeze()
                if arr.size and np.isfinite(arr).any() and np.nanmax(arr) > 1.0:
                    arr = arr / 100.0
                normalized.append(arr.reshape(1) if arr.ndim == 0 else arr)
            self.plddt = normalized if any(value is not None for value in normalized) else None


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


def _extract_scalar(
    confidence: list[dict[str, Any] | None] | None,
    key: str,
) -> list[np.ndarray | None] | None:
    if confidence is None:
        return None

    scores: list[np.ndarray | None] = []
    for item in confidence:
        if item is None or key not in item:
            scores.append(None)
            continue
        score = np.asarray(item[key], dtype=np.float32).squeeze()
        if score.ndim != 0:
            raise ValueError(f"Protenix {key} expected a scalar score; got {score.shape}")
        scores.append(score.reshape(1))
    return scores if any(score is not None for score in scores) else None
