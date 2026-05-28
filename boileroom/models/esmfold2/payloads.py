"""JSON-compatible ESMFold2 input payloads for non-Modal backends."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np

from .types import (
    CovalentBond,
    DistogramConditioning,
    DNAInput,
    LigandInput,
    Modification,
    MSAInput,
    ProteinInput,
    RNAInput,
    StructurePredictionInput,
)

SequenceInput = ProteinInput | RNAInput | DNAInput | LigandInput
EncodedFoldInput = str | list[str] | dict[str, Any] | list[dict[str, Any]]

_STRUCTURE_KIND = "structure_prediction_input"
_PROTEIN_KIND = "protein"
_RNA_KIND = "rna"
_DNA_KIND = "dna"
_LIGAND_KIND = "ligand"


def encode_fold_input(value: Any) -> EncodedFoldInput:
    """Convert public ESMFold2 inputs into JSON-compatible values."""
    if isinstance(value, str):
        return value
    if isinstance(value, StructurePredictionInput):
        return encode_structure_input(value)
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, Sequence):
        raise TypeError("ESMFold2.fold expects a string, structure input, or sequence of those values.")

    items = list(value)
    if not items:
        raise ValueError("ESMFold2.fold received an empty input sequence.")
    if all(isinstance(item, str) for item in items):
        return [cast(str, item) for item in items]
    if all(isinstance(item, StructurePredictionInput) for item in items):
        return [encode_structure_input(cast(StructurePredictionInput, item)) for item in items]
    if all(isinstance(item, ProteinInput | RNAInput | DNAInput | LigandInput) for item in items):
        return encode_structure_input(StructurePredictionInput(sequences=cast(list[SequenceInput], items)))
    if all(isinstance(item, Mapping) for item in items):
        return [dict(cast(Mapping[str, Any], item)) for item in items]
    raise TypeError(
        "ESMFold2.fold input lists must contain only strings, only StructurePredictionInput objects, "
        "or only molecule input dataclasses."
    )


def encode_structure_input(value: StructurePredictionInput) -> dict[str, Any]:
    """Encode a structure input as plain JSON-compatible data."""
    return {
        "kind": _STRUCTURE_KIND,
        "sequences": [encode_sequence_input(item) for item in value.sequences],
        "distogram_conditioning": (
            [encode_distogram_conditioning(item) for item in value.distogram_conditioning]
            if value.distogram_conditioning is not None
            else None
        ),
        "covalent_bonds": (
            [encode_covalent_bond(item) for item in value.covalent_bonds] if value.covalent_bonds is not None else None
        ),
    }


def decode_structure_input(value: Mapping[str, Any]) -> StructurePredictionInput:
    """Decode an encoded structure payload into lightweight dataclasses."""
    if value.get("kind") != _STRUCTURE_KIND:
        raise ValueError("ESMFold2 structure payload must have kind='structure_prediction_input'.")
    return StructurePredictionInput(
        sequences=[decode_sequence_input(item) for item in _mapping_sequence(value.get("sequences"), "sequences")],
        distogram_conditioning=(
            [
                decode_distogram_conditioning(item)
                for item in _mapping_sequence(value.get("distogram_conditioning"), "distogram_conditioning")
            ]
            if value.get("distogram_conditioning") is not None
            else None
        ),
        covalent_bonds=(
            [decode_covalent_bond(item) for item in _mapping_sequence(value.get("covalent_bonds"), "covalent_bonds")]
            if value.get("covalent_bonds") is not None
            else None
        ),
    )


def encode_sequence_input(value: SequenceInput) -> dict[str, Any]:
    """Encode one protein, nucleic-acid, or ligand input."""
    if isinstance(value, ProteinInput):
        return {
            "kind": _PROTEIN_KIND,
            "id": value.id,
            "sequence": value.sequence,
            "modifications": encode_modifications(value.modifications),
            "msa": encode_msa(value.msa),
        }
    if isinstance(value, RNAInput):
        return {
            "kind": _RNA_KIND,
            "id": value.id,
            "sequence": value.sequence,
            "modifications": encode_modifications(value.modifications),
        }
    if isinstance(value, DNAInput):
        return {
            "kind": _DNA_KIND,
            "id": value.id,
            "sequence": value.sequence,
            "modifications": encode_modifications(value.modifications),
        }
    return {"kind": _LIGAND_KIND, "id": value.id, "smiles": value.smiles, "ccd": value.ccd}


def decode_sequence_input(value: Mapping[str, Any]) -> SequenceInput:
    """Decode one encoded chain or ligand input."""
    kind = value.get("kind")
    if kind == _PROTEIN_KIND:
        return ProteinInput(
            id=_decode_id(value.get("id")),
            sequence=str(value.get("sequence", "")),
            modifications=decode_modifications(value.get("modifications")),
            msa=decode_msa(value.get("msa")),
        )
    if kind == _RNA_KIND:
        return RNAInput(
            id=_decode_id(value.get("id")),
            sequence=str(value.get("sequence", "")),
            modifications=decode_modifications(value.get("modifications")),
        )
    if kind == _DNA_KIND:
        return DNAInput(
            id=_decode_id(value.get("id")),
            sequence=str(value.get("sequence", "")),
            modifications=decode_modifications(value.get("modifications")),
        )
    if kind == _LIGAND_KIND:
        ccd = value.get("ccd")
        return LigandInput(
            id=_decode_id(value.get("id")),
            smiles=cast(str | None, value.get("smiles")),
            ccd=[str(item) for item in ccd] if isinstance(ccd, list) else None,
        )
    raise ValueError(f"Unsupported ESMFold2 sequence payload kind: {kind!r}")


def encode_modifications(values: list[Modification] | None) -> list[dict[str, Any]] | None:
    """Encode residue modifications."""
    if values is None:
        return None
    return [
        {"position": modification.position, "ccd": modification.ccd, "smiles": modification.smiles}
        for modification in values
    ]


def decode_modifications(value: Any) -> list[Modification] | None:
    """Decode residue modifications."""
    if value is None:
        return None
    return [
        Modification(position=int(item["position"]), ccd=str(item["ccd"]), smiles=cast(str | None, item.get("smiles")))
        for item in _mapping_sequence(value, "modifications")
    ]


def encode_msa(value: MSAInput | Any | None) -> dict[str, Any] | list[str] | None:
    """Encode lightweight MSA inputs.

    Biohub-native MSA objects are Modal-only because the Apptainer bridge uses
    JSON. Use ``MSAInput`` for portable Modal/Apptainer calls.
    """
    if value is None:
        return None
    if isinstance(value, MSAInput):
        return {"sequences": value.sequences, "remove_insertions": value.remove_insertions}
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return [cast(str, item) for item in value]
    raise TypeError("Apptainer ESMFold2 inputs must use MSAInput or list[str] for protein MSA values.")


def decode_msa(value: Any) -> MSAInput | list[str] | None:
    """Decode a lightweight MSA payload."""
    if value is None:
        return None
    if isinstance(value, Mapping):
        sequences = value.get("sequences")
        if not isinstance(sequences, list) or not all(isinstance(item, str) for item in sequences):
            raise TypeError("Encoded ESMFold2 MSA payload requires a string sequence list.")
        return MSAInput(sequences=sequences, remove_insertions=bool(value.get("remove_insertions", False)))
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return [cast(str, item) for item in value]
    raise TypeError("Encoded ESMFold2 MSA payload must be a mapping, list[str], or None.")


def encode_distogram_conditioning(value: DistogramConditioning) -> dict[str, Any]:
    """Encode distogram conditioning."""
    return {"chain_id": value.chain_id, "distogram": np.asarray(value.distogram).tolist()}


def decode_distogram_conditioning(value: Mapping[str, Any]) -> DistogramConditioning:
    """Decode distogram conditioning."""
    return DistogramConditioning(chain_id=str(value["chain_id"]), distogram=np.asarray(value["distogram"]))


def encode_covalent_bond(value: CovalentBond) -> dict[str, Any]:
    """Encode one covalent bond."""
    return {
        "chain_id1": value.chain_id1,
        "res_idx1": value.res_idx1,
        "atom_idx1": value.atom_idx1,
        "chain_id2": value.chain_id2,
        "res_idx2": value.res_idx2,
        "atom_idx2": value.atom_idx2,
    }


def decode_covalent_bond(value: Mapping[str, Any]) -> CovalentBond:
    """Decode one covalent bond."""
    return CovalentBond(
        chain_id1=str(value["chain_id1"]),
        res_idx1=int(value["res_idx1"]),
        atom_idx1=int(value["atom_idx1"]),
        chain_id2=str(value["chain_id2"]),
        res_idx2=int(value["res_idx2"]),
        atom_idx2=int(value["atom_idx2"]),
    )


def _decode_id(value: Any) -> str | list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    return str(value)


def _mapping_sequence(value: Any, field_name: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(item, Mapping) for item in value):
        raise TypeError(f"Encoded ESMFold2 {field_name} must be a list of mappings.")
    return [cast(Mapping[str, Any], item) for item in value]
