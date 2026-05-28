"""JSON-compatible ESMFold2 input payloads for non-Modal backends."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

from .types import (
    CovalentBond,
    DistogramConditioning,
    DNAInput,
    LigandInput,
    Modification,
    MSAInput,
    PocketConditioning,
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
        "pocket": encode_pocket_conditioning(value.pocket),
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
        pocket=decode_pocket_conditioning(value.get("pocket")),
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
            sequence=_decode_sequence(value.get("sequence"), "protein.sequence"),
            modifications=decode_modifications(value.get("modifications")),
            msa=decode_msa(value.get("msa")),
        )
    if kind == _RNA_KIND:
        return RNAInput(
            id=_decode_id(value.get("id")),
            sequence=_decode_sequence(value.get("sequence"), "rna.sequence"),
            modifications=decode_modifications(value.get("modifications")),
        )
    if kind == _DNA_KIND:
        return DNAInput(
            id=_decode_id(value.get("id")),
            sequence=_decode_sequence(value.get("sequence"), "dna.sequence"),
            modifications=decode_modifications(value.get("modifications")),
        )
    if kind == _LIGAND_KIND:
        ccd = value.get("ccd")
        return LigandInput(
            id=_decode_id(value.get("id")),
            smiles=_decode_optional_str(value.get("smiles"), "ligand.smiles"),
            ccd=_decode_optional_str_list(ccd, "ligand.ccd"),
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
    JSON. Use ``MSAInput`` for portable Modal/Apptainer calls. ESMFold2 currently
    consumes in-memory MSA sequences; path-backed MSA values are preserved in the
    payload so model adapters can reject unsupported sources clearly.
    """
    if value is None:
        return None
    if isinstance(value, MSAInput):
        payload: dict[str, Any] = {"remove_insertions": value.remove_insertions}
        if value.sequences is not None:
            payload["sequences"] = value.sequences
        if value.path is not None:
            payload["path"] = str(value.path)
        return payload
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return [cast(str, item) for item in value]
    raise TypeError("Apptainer ESMFold2 inputs must use MSAInput or list[str] for protein MSA values.")


def decode_msa(value: Any) -> MSAInput | list[str] | None:
    """Decode a lightweight MSA payload."""
    if value is None:
        return None
    if isinstance(value, Mapping):
        sequences = value.get("sequences")
        path = value.get("path")
        remove_insertions = value.get("remove_insertions", False)
        if sequences is not None and (
            not isinstance(sequences, list) or not all(isinstance(item, str) for item in sequences)
        ):
            raise TypeError("Encoded ESMFold2 MSA payload sequences must be a string sequence list.")
        if path is not None and not isinstance(path, str | Path):
            raise TypeError("Encoded ESMFold2 MSA payload path must be a string path.")
        if not isinstance(remove_insertions, bool):
            raise TypeError("Encoded ESMFold2 MSA payload remove_insertions must be a boolean.")
        return MSAInput(
            sequences=[cast(str, item) for item in sequences] if sequences is not None else None,
            path=cast(str | Path | None, path),
            remove_insertions=remove_insertions,
        )
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return [cast(str, item) for item in value]
    raise TypeError("Encoded ESMFold2 MSA payload must be a mapping, list[str], or None.")


def encode_pocket_conditioning(value: PocketConditioning | None) -> dict[str, Any] | None:
    """Encode pocket conditioning."""
    if value is None:
        return None
    return {"binder_chain_id": value.binder_chain_id, "contacts": [list(contact) for contact in value.contacts]}


def decode_pocket_conditioning(value: Any) -> PocketConditioning | None:
    """Decode pocket conditioning."""
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("Encoded ESMFold2 pocket payload must be a mapping or None.")
    contacts = value.get("contacts")
    if not isinstance(contacts, list):
        raise TypeError("Encoded ESMFold2 pocket contacts must be a list.")
    return PocketConditioning(
        binder_chain_id=_decode_sequence(value.get("binder_chain_id"), "pocket.binder_chain_id"),
        contacts=[_decode_pocket_contact(contact) for contact in contacts],
    )


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
    """Decode a chain identifier that may name one or many copies."""
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return [cast(str, item) for item in value]
    raise TypeError("Encoded ESMFold2 id must be str or list[str].")


def _decode_sequence(value: Any, field_name: str) -> str:
    """Decode a required string field."""
    if not isinstance(value, str):
        raise TypeError(f"Encoded ESMFold2 {field_name} must be a string.")
    return value


def _decode_optional_str(value: Any, field_name: str) -> str | None:
    """Decode an optional string field."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"Encoded ESMFold2 {field_name} must be a string or None.")
    return value


def _decode_optional_str_list(value: Any, field_name: str) -> list[str] | None:
    """Decode an optional list of strings."""
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TypeError(f"Encoded ESMFold2 {field_name} must be a list of strings or None.")
    return [cast(str, item) for item in value]


def _decode_pocket_contact(value: Any) -> tuple[str, int]:
    """Decode one pocket contact pair as ``(chain_id, residue_index)``."""
    if not isinstance(value, Sequence) or isinstance(value, str) or len(value) != 2:
        raise TypeError("Encoded ESMFold2 pocket contacts must be [chain_id, residue_index] pairs.")
    chain_id = _decode_sequence(value[0], "pocket.contacts.chain_id")
    residue_index = value[1]
    if not isinstance(residue_index, int) or isinstance(residue_index, bool):
        raise TypeError("Encoded ESMFold2 pocket contact residue index must be an integer.")
    return chain_id, residue_index


def _mapping_sequence(value: Any, field_name: str) -> list[Mapping[str, Any]]:
    """Validate and cast an encoded list-of-mappings field."""
    if not isinstance(value, list) or not all(isinstance(item, Mapping) for item in value):
        raise TypeError(f"Encoded ESMFold2 {field_name} must be a list of mappings.")
    return [cast(Mapping[str, Any], item) for item in value]
