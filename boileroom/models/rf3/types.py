"""Type definitions for RoseTTAFold 3 outputs without heavy runtime dependencies."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from ...base import PredictionMetadata, StructurePrediction


@dataclass
class RF3Output(StructurePrediction):
    """Output from RoseTTAFold 3 structure prediction.

    Samples are listed best first: ``sample_ranks`` is the position by ``ranking_score``
    (``0.8 * ipTM + 0.2 * pTM - 100 * has_clash``; RF3 reports ipTM 0 for a single chain, so a
    monomer ranks by ``0.2 * pTM``), and ``sample_indices`` is the index RF3 gave the sample within
    its diffusion batch.
    """

    metadata: PredictionMetadata
    atom_array: list[Any] | None = None

    confidence: list[dict[str, Any] | None] | None = None
    plddt: list[np.ndarray | None] | None = None
    ptm: list[np.ndarray | None] | None = None
    iptm: list[np.ndarray | None] | None = None
    pae: list[np.ndarray] | None = None
    token_chain_ids: list[np.ndarray] | None = None
    token_res_ids: list[np.ndarray] | None = None
    atom_plddt: list[np.ndarray] | None = None
    seeds: list[int] | None = None
    sample_ranks: list[int] | None = None
    sample_indices: list[int] | None = None
    pdb: list[str] | None = None
    cif: list[str] | None = None

    def __post_init__(self) -> None:
        """Normalize common confidence scalars and pLDDT to the public output contract."""
        self.ptm = _normalize_scalar_scores(self.ptm, "RF3 pTM") or _extract_scalar(self.confidence, "ptm")
        self.iptm = _normalize_scalar_scores(self.iptm, "RF3 ipTM") or _extract_scalar(self.confidence, "iptm")
        self.plddt = _normalize_plddt(self.plddt)
        self.atom_plddt = cast(list[np.ndarray] | None, _normalize_plddt(self.atom_plddt))


def _normalize_plddt(values: Sequence[np.ndarray | None] | None) -> list[np.ndarray | None] | None:
    """Return pLDDT arrays on a 0-1 scale (RF3 reports 0-1; a 0-100 input is rescaled)."""
    if values is None:
        return None
    normalized: list[np.ndarray | None] = []
    for value in values:
        if value is None:
            normalized.append(None)
            continue
        arr = np.asarray(value, dtype=np.float32).squeeze()
        if arr.size and np.isfinite(arr).any() and np.nanmax(arr) > 1.0:
            arr = arr / 100.0
        normalized.append(arr.reshape(1) if arr.ndim == 0 else arr)
    return normalized if any(value is not None for value in normalized) else None


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
        if item is None or item.get(key) is None:
            scores.append(None)
            continue
        score = np.asarray(item[key], dtype=np.float32).squeeze()
        if score.ndim != 0:
            raise ValueError(f"RF3 {key} expected a scalar score; got {score.shape}")
        scores.append(score.reshape(1))
    return scores if any(score is not None for score in scores) else None
