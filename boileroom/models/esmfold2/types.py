"""Type definitions for ESMFold2 inputs and outputs without heavy dependencies."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ...base import PredictionMetadata, StructurePrediction

if TYPE_CHECKING:
    from biotite.structure import AtomArray


@dataclass(frozen=True)
class MSAInput:
    """Lightweight MSA specification for ESMFold2 protein inputs."""

    sequences: list[str]
    remove_insertions: bool = False


@dataclass(frozen=True)
class Modification:
    """A residue modification specified by zero-indexed position and CCD code."""

    position: int
    ccd: str
    smiles: str | None = None


@dataclass(frozen=True)
class ProteinInput:
    """Protein chain input for ESMFold2."""

    id: str | list[str]
    sequence: str
    modifications: list[Modification] | None = None
    msa: MSAInput | Any | None = None


@dataclass(frozen=True)
class RNAInput:
    """RNA chain input for ESMFold2."""

    id: str | list[str]
    sequence: str
    modifications: list[Modification] | None = None


@dataclass(frozen=True)
class DNAInput:
    """DNA chain input for ESMFold2."""

    id: str | list[str]
    sequence: str
    modifications: list[Modification] | None = None


@dataclass(frozen=True)
class LigandInput:
    """Ligand input for ESMFold2, specified by SMILES or CCD code."""

    id: str | list[str]
    smiles: str | None = None
    ccd: list[str] | None = None


@dataclass(frozen=True)
class DistogramConditioning:
    """Optional distogram conditioning for a chain."""

    chain_id: str
    distogram: np.ndarray


@dataclass(frozen=True)
class PocketConditioning:
    """Optional pocket conditioning with target contacts for a binder chain."""

    binder_chain_id: str
    contacts: list[tuple[str, int]]


@dataclass(frozen=True)
class CovalentBond:
    """Optional covalent bond between two chain atoms."""

    chain_id1: str
    res_idx1: int
    atom_idx1: int
    chain_id2: str
    res_idx2: int
    atom_idx2: int


@dataclass(frozen=True)
class StructurePredictionInput:
    """All-atom ESMFold2 input containing protein, nucleic-acid, and ligand chains."""

    sequences: Sequence[ProteinInput | RNAInput | DNAInput | LigandInput]
    pocket: PocketConditioning | None = None
    distogram_conditioning: list[DistogramConditioning] | None = None
    covalent_bonds: list[CovalentBond] | None = None


@dataclass
class ESMFold2Output(StructurePrediction):
    """Output from ESMFold2 prediction."""

    metadata: PredictionMetadata
    atom_array: list[AtomArray] | None = None

    plddt: list[np.ndarray | None] | None = None
    ptm: list[np.ndarray | None] | None = None
    iptm: list[np.ndarray | None] | None = None
    pae: list[np.ndarray | None] | None = None
    distogram: list[np.ndarray | None] | None = None
    pair_chains_iptm: list[np.ndarray | None] | None = None
    residue_index: list[np.ndarray | None] | None = None
    entity_id: list[np.ndarray | None] | None = None
    pdb: list[str] | None = None
    cif: list[str] | None = None

    def __post_init__(self) -> None:
        """Normalize confidence metrics to unit-scale per-sample arrays."""
        self.plddt = _normalize_array_list(self.plddt, unit_scale=True)
        self.ptm = _normalize_scalar_list(self.ptm)
        self.iptm = _normalize_scalar_list(self.iptm)
        self.pae = _normalize_array_list(self.pae)
        self.distogram = _normalize_array_list(self.distogram)
        self.pair_chains_iptm = _normalize_array_list(self.pair_chains_iptm)
        self.residue_index = _normalize_array_list(self.residue_index)
        self.entity_id = _normalize_array_list(self.entity_id)


def _normalize_array_list(values: Any, unit_scale: bool = False) -> list[np.ndarray | None] | None:
    """Normalize scalar, vector, matrix, or list values into per-sample arrays."""
    if values is None:
        return None

    if isinstance(values, list):
        normalized = [_normalize_array(value, unit_scale=unit_scale) if value is not None else None for value in values]
        return normalized if any(value is not None for value in normalized) else None

    array = np.asarray(values)
    if array.ndim <= 1:
        return [_normalize_array(array, unit_scale=unit_scale)]
    return [_normalize_array(sample, unit_scale=unit_scale) for sample in array]


def _normalize_array(value: Any, unit_scale: bool = False) -> np.ndarray:
    """Convert one confidence/geometry value to an array, optionally scaling percentages."""
    array = np.asarray(value, dtype=np.float32 if unit_scale else None)
    if unit_scale and array.size and np.nanmax(array) > 1.0:
        array = array / 100.0
    return array.astype(np.float32, copy=False) if unit_scale else array


def _normalize_scalar_list(values: Any) -> list[np.ndarray | None] | None:
    """Normalize scalar metrics into one length-one array per sample."""
    if values is None:
        return None

    if isinstance(values, list):
        normalized = [_normalize_scalar(value) if value is not None else None for value in values]
        return normalized if any(value is not None for value in normalized) else None

    array = np.asarray(values, dtype=np.float32).reshape(-1)
    return [np.array([value], dtype=np.float32) for value in array]


def _normalize_scalar(value: Any) -> np.ndarray:
    """Return a single scalar metric as a float32 length-one array."""
    return np.asarray([value], dtype=np.float32).reshape(1)
