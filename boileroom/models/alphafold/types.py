"""Type definitions for AlphaFold2-Multimer outputs."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from ...base import PredictionMetadata, StructurePrediction


@dataclass
class AlphaFold2MultimerOutput(StructurePrediction):
    """Output from AlphaFold2-Multimer prediction."""

    metadata: PredictionMetadata
    atom_array: list[Any] | None = None

    ranking: dict[str, Any] | None = None
    plddt: list[np.ndarray | None] | None = None
    ptm: list[np.ndarray | None] | None = None
    iptm: list[np.ndarray | None] | None = None
    pae: list[np.ndarray | None] | None = None
    pdb: list[str] | None = None
    cif: list[str] | None = None

    def __post_init__(self) -> None:
        """Normalize confidence arrays to stable shapes."""
        self.ptm = _normalize_scalar_scores(self.ptm, "AlphaFold2-Multimer pTM")
        self.iptm = _normalize_scalar_scores(self.iptm, "AlphaFold2-Multimer iPTM")
        if self.plddt is not None:
            self.plddt = [_normalize_vector(value, "AlphaFold2-Multimer pLDDT") for value in self.plddt]
        if self.pae is not None:
            self.pae = [_normalize_matrix(value, "AlphaFold2-Multimer PAE") for value in self.pae]


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


def _normalize_vector(value: np.ndarray | None, label: str) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32).squeeze()
    if arr.ndim != 1:
        raise ValueError(f"{label} expected a 1D array; got {arr.shape}")
    if arr.size and np.nanmax(arr) > 1.0:
        arr = arr / 100.0
    return arr


def _normalize_matrix(value: np.ndarray | None, label: str) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32).squeeze()
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{label} expected a square matrix; got {arr.shape}")
    return arr
