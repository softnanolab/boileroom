"""Stateless ESMFold2 request validation, upstream input conversion and output formatting."""

from __future__ import annotations

import dataclasses
import importlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from io import StringIO
from typing import Any, cast

import numpy as np

from ...base import FoldingAlgorithm, PredictionMetadata
from ...inputs import a3m_rows
from ...utils import validate_sequence
from .payloads import decode_structure_input
from .types import (
    CovalentBond,
    DistogramConditioning,
    DNAInput,
    ESMFold2Output,
    LigandInput,
    Modification,
    MSAInput,
    PocketConditioning,
    ProteinInput,
    RNAInput,
    StructurePredictionInput,
)

SequenceInput = ProteinInput | RNAInput | DNAInput | LigandInput
ESMFold2FoldInput = (
    str
    | Sequence[str]
    | StructurePredictionInput
    | Sequence[StructurePredictionInput]
    | Sequence[SequenceInput]
    | Mapping[str, Any]
    | Sequence[Mapping[str, Any]]
)


@dataclass(frozen=True)
class _FoldRequest:
    """One normalized fold request plus its sequence-length metadata."""

    input: StructurePredictionInput
    sequence_length: int


#: Heavy atoms of an unmodified residue, as both esm builders (3.3.0 and 3.4.1) tokenize it: ``PROTEIN_HEAVY_ATOMS``,
#: ``DNA_HEAVY_ATOMS`` and ``RNA_HEAVY_ATOMS`` in ``esm.models.esmfold2.constants``, by one-letter code. A covalent bond's
#: atom index is bounded by these before either builder sees it (esm 3.3.0 drops an out-of-range bond silently). A
#: protein letter outside the table becomes ``UNK``, whose four backbone atoms esm builds; an unknown nucleotide letter
#: has no static count.
# fmt: off
PROTEIN_RESIDUE_ATOM_COUNTS: dict[str, int] = {
    "A": 5, "R": 11, "N": 8, "D": 8, "C": 6, "Q": 9, "E": 9, "G": 4, "H": 10, "I": 8,
    "L": 8, "K": 9, "M": 8, "F": 11, "P": 7, "S": 6, "T": 7, "W": 14, "Y": 12, "V": 7,
}
# fmt: on
UNKNOWN_PROTEIN_RESIDUE_ATOMS = 4
DNA_RESIDUE_ATOM_COUNTS: dict[str, int] = {"A": 21, "G": 22, "C": 19, "T": 20}
RNA_RESIDUE_ATOM_COUNTS: dict[str, int] = {"A": 22, "G": 23, "C": 20, "U": 20}


def _atom_past_residue(index: int, atom_idx: int, res_idx: int, chain_id: str, residue: str, count: int) -> ValueError:
    """The refusal of a covalent bond whose atom index is past the atoms esm builds for its residue."""
    return ValueError(
        f"ESMFold2 covalent bond {index}: atom index {atom_idx} is past residue {res_idx} of chain {chain_id!r} "
        f"({residue}: {count} atoms, 0-based 0-{count - 1})"
    )


def _polymer_residue_atoms(item: SequenceInput, res_idx: int) -> tuple[str, int] | None:
    """The name and heavy-atom count of an unmodified protein, DNA or RNA residue, or ``None`` without a static count."""
    if isinstance(item, LigandInput):
        return None
    if any(modification.position == res_idx for modification in item.modifications or []):
        return None
    letter = item.sequence[res_idx]
    if isinstance(item, ProteinInput):
        return f"protein {letter}", PROTEIN_RESIDUE_ATOM_COUNTS.get(letter, UNKNOWN_PROTEIN_RESIDUE_ATOMS)
    table = DNA_RESIDUE_ATOM_COUNTS if isinstance(item, DNAInput) else RNA_RESIDUE_ATOM_COUNTS
    if letter not in table:
        return None
    return f"{'DNA' if isinstance(item, DNAInput) else 'RNA'} {letter}", table[letter]


def _has_user_msa(msa: object) -> bool:
    """Whether ``options['msa']`` carries an MSA: ``None`` and an all-``None`` list do not."""
    if msa is None:
        return False
    if isinstance(msa, list | tuple):
        return any(entry is not None for entry in msa)
    return True


def _attach_user_msa(request: _FoldRequest, msa: object) -> _FoldRequest:
    """Attach ``options['msa']`` (one A3M text or None per input entry) to the protein entries."""
    items = request.input.sequences
    if not isinstance(msa, list | tuple) or len(msa) != len(items):
        raise ValueError("ESMFold2 option 'msa' must have one A3M string or None per input sequence entry.")
    attached: list[SequenceInput] = []
    for index, (item, text) in enumerate(zip(items, msa, strict=True)):
        if text is None:
            attached.append(item)
            continue
        if not isinstance(item, ProteinInput):
            raise ValueError(f"ESMFold2 option 'msa' entry {index} targets a non-protein input; use None.")
        if item.msa is not None:
            raise ValueError(f"ESMFold2 input entry {index} already carries an MSA; do not also pass options['msa'].")
        if ":" in item.sequence or "|" in item.sequence:
            raise ValueError(f"ESMFold2 option 'msa' entry {index} needs a single-chain input entry.")
        rows = a3m_rows(text, item.sequence)
        attached.append(dataclasses.replace(item, msa=MSAInput(sequences=rows)))
    return dataclasses.replace(request, input=dataclasses.replace(request.input, sequences=attached))


def _check_covalent_bonds(prediction_input: StructurePredictionInput) -> None:
    """Validate covalent bonds before either esm builder sees them, and before any load.

    esm 3.4.1 raises on a bond naming a missing chain, residue or atom; the kit's esm 3.3.0 silently drops it (and
    reads a negative atom index from the end of the residue), folding the ligand unbonded. Residue and atom
    indices are 0-based, as in esm. Bonds cannot be combined with ``:`` / ``|`` chainbreaks (esm 3.4.1 refuses
    that too). The atom index of an unmodified protein, DNA or RNA residue is bounded here by its heavy-atom
    count (:data:`PROTEIN_RESIDUE_ATOM_COUNTS` and the nucleotide tables); a CCD ligand's is bounded after load
    by :meth:`_check_ligand_bond_atoms`. A modified residue, an unknown nucleotide and a SMILES ligand have no
    count without esm's own tokenizer and stay with esm.
    """
    bonds = prediction_input.covalent_bonds
    if not bonds:
        return
    chains: dict[str, tuple[SequenceInput, int]] = {}
    for item in prediction_input.sequences:
        if isinstance(item, ProteinInput | RNAInput | DNAInput):
            if ":" in item.sequence or "|" in item.sequence:
                raise ValueError(
                    f"ESMFold2 covalent bonds cannot be combined with chainbreaks (':' or '|') in entry "
                    f"{item.id!r}; give each chain its own input entry."
                )
            count = len(item.sequence)
        else:
            # A CCD list is one residue per code; a SMILES ligand is a single residue.
            count = len(item.ccd) if isinstance(item.ccd, list) and item.ccd else 1
        for chain_id in item.id if isinstance(item.id, list) else [item.id]:
            chains[chain_id] = (item, count)
    for index, bond in enumerate(bonds):
        for chain_id, res_idx, atom_idx in (
            (bond.chain_id1, bond.res_idx1, bond.atom_idx1),
            (bond.chain_id2, bond.res_idx2, bond.atom_idx2),
        ):
            if chain_id not in chains:
                raise ValueError(
                    f"ESMFold2 covalent bond {index}: chain_id {chain_id!r} does not exist; "
                    f"available chain ids: {sorted(chains)}"
                )
            item, count = chains[chain_id]
            if not isinstance(res_idx, int) or isinstance(res_idx, bool) or not 0 <= res_idx < count:
                raise ValueError(
                    f"ESMFold2 covalent bond {index}: residue index {res_idx!r} is not in chain {chain_id!r} "
                    f"(0-based, 0-{count - 1})"
                )
            if not isinstance(atom_idx, int) or isinstance(atom_idx, bool) or atom_idx < 0:
                raise ValueError(
                    f"ESMFold2 covalent bond {index}: atom index {atom_idx!r} must be a non-negative integer "
                    "(0-based within the residue)"
                )
            atoms = _polymer_residue_atoms(item, res_idx)
            if atoms is not None and atom_idx >= atoms[1]:
                raise _atom_past_residue(index, atom_idx, res_idx, chain_id, *atoms)


def _check_ligand_bond_atoms(prediction_input: StructurePredictionInput) -> None:
    """Bound the atom index of a covalent bond on a CCD ligand by the component's atoms in the loaded CCD.

    Runs after load, when the input builder has loaded the CCD. Both esm builders read the same
    ``esm.models.esmfold2.conformers`` (identical in esm 3.3.0 and 3.4.1) and tokenize a covalently bonded CCD
    ligand as one token per atom without its leaving atoms, so the bound is that count. A code missing from the
    CCD is left to esm, which refuses it.
    """
    bonds = prediction_input.covalent_bonds
    if not bonds:
        return
    ligands: dict[str, list[str]] = {}
    for item in prediction_input.sequences:
        if isinstance(item, LigandInput) and isinstance(item.ccd, list) and item.ccd:
            for chain_id in item.id if isinstance(item.id, list) else [item.id]:
                ligands[chain_id] = item.ccd
    if not ligands:
        return
    conformers = importlib.import_module("esm.models.esmfold2.conformers")
    for index, bond in enumerate(bonds):
        for chain_id, res_idx, atom_idx in (
            (bond.chain_id1, bond.res_idx1, bond.atom_idx1),
            (bond.chain_id2, bond.res_idx2, bond.atom_idx2),
        ):
            if chain_id not in ligands:
                continue
            code = ligands[chain_id][res_idx]
            ccd_atoms = conformers.get_ligand_ccd_atoms_with_charges(code)
            if ccd_atoms is None:
                continue
            leaving = conformers.get_ccd_leaving_atoms(code)
            count = sum(1 for atom in ccd_atoms if atom[0] not in leaving)
            if atom_idx >= count:
                raise _atom_past_residue(index, atom_idx, res_idx, chain_id, f"CCD {code}", count)


def _coerce_requests(sequences: ESMFold2FoldInput) -> list[_FoldRequest]:
    """Coerce public BoilerRoom inputs into one or more ESMFold2 structure inputs."""
    if isinstance(sequences, str):
        return [_request_from_protein_string(sequences)]

    if isinstance(sequences, StructurePredictionInput):
        return [_request_from_structure_input(sequences)]

    if isinstance(sequences, Mapping):
        return [_request_from_structure_input(decode_structure_input(cast(Mapping[str, Any], sequences)))]

    if not isinstance(sequences, Sequence):
        raise TypeError("ESMFold2.fold expects a sequence string, structure input, or a sequence of those inputs.")

    items = list(sequences)
    if not items:
        raise ValueError("ESMFold2.fold received an empty input sequence.")

    if all(isinstance(item, str) for item in items):
        return [_request_from_protein_string(cast(str, item)) for item in items]

    if all(isinstance(item, StructurePredictionInput) for item in items):
        return [_request_from_structure_input(cast(StructurePredictionInput, item)) for item in items]

    if all(isinstance(item, Mapping) for item in items):
        return [_request_from_structure_input(decode_structure_input(cast(Mapping[str, Any], item))) for item in items]

    if all(_is_sequence_input(item) for item in items):
        structure_input = StructurePredictionInput(sequences=cast(list[SequenceInput], items))
        return [_request_from_structure_input(structure_input)]

    raise TypeError(
        "ESMFold2.fold input lists must contain only strings, only StructurePredictionInput objects, "
        "or only molecule input dataclasses."
    )


def _is_sequence_input(value: object) -> bool:
    """Return whether a value is one lightweight molecule input dataclass."""
    return isinstance(value, ProteinInput | RNAInput | DNAInput | LigandInput)


def _request_from_protein_string(sequence: str) -> _FoldRequest:
    """Convert a colon-delimited protein string into one structure request."""
    chains = sequence.replace("|", ":").split(":")
    if any(chain == "" for chain in chains):
        raise ValueError(f"Invalid ESMFold2 protein sequence {sequence!r}: empty chain near ':'.")
    for chain in chains:
        validate_sequence(chain)
    structure_input = StructurePredictionInput(
        sequences=[ProteinInput(id=_chain_id(index), sequence=chain) for index, chain in enumerate(chains)]
    )
    return _FoldRequest(input=structure_input, sequence_length=sum(len(chain) for chain in chains))


def _request_from_structure_input(prediction_input: StructurePredictionInput) -> _FoldRequest:
    """Validate a rich structure input and attach sequence-length metadata."""
    if not prediction_input.sequences:
        raise ValueError("StructurePredictionInput.sequences must not be empty.")
    return _FoldRequest(input=prediction_input, sequence_length=_structure_input_length(prediction_input))


def _chain_id(index: int) -> str:
    """Return spreadsheet-style chain IDs: A, B, ..., Z, AA, AB, ..."""
    index += 1
    parts = []
    while index:
        index, remainder = divmod(index - 1, 26)
        parts.append(chr(65 + remainder))
    return "".join(reversed(parts))


def _structure_input_length(prediction_input: StructurePredictionInput) -> int:
    """Return the metadata length for all sequence-like entities in an input."""
    return sum(_sequence_input_length(item) for item in prediction_input.sequences)


def _sequence_input_length(item: SequenceInput) -> int:
    """Return the metadata length contribution for one chain or ligand input."""
    multiplier = _id_count(item.id)
    if isinstance(item, ProteinInput | RNAInput | DNAInput):
        sequence = item.sequence.replace("|", ":")
        chains = sequence.split(":")
        if any(chain == "" for chain in chains):
            raise ValueError(f"Invalid ESMFold2 sequence {sequence!r}: empty chain near ':'.")
        if isinstance(item, ProteinInput):
            for chain in chains:
                validate_sequence(chain)
        return sum(len(chain) for chain in chains) * multiplier
    ccd_count = len(item.ccd or [])
    ligand_count = ccd_count or (1 if item.smiles else 0)
    return max(multiplier, ligand_count)


def _id_count(value: str | list[str]) -> int:
    """Return how many entity copies an ESMFold2 id represents."""
    return len(value) if isinstance(value, list) else 1


def _to_esm_structure_prediction_input(prediction_input: StructurePredictionInput) -> Any:
    """Convert a lightweight structure input to Biohub's native dataclass."""
    from esm.models.esmfold2 import StructurePredictionInput as ESMStructurePredictionInput

    return ESMStructurePredictionInput(
        sequences=[_to_esm_sequence_input(item) for item in prediction_input.sequences],
        pocket=_to_esm_pocket_conditioning(prediction_input.pocket),
        distogram_conditioning=_to_esm_distogram_conditioning(prediction_input.distogram_conditioning),
        covalent_bonds=_to_esm_covalent_bonds(prediction_input.covalent_bonds),
    )


def _to_esm_sequence_input(item: SequenceInput) -> Any:
    """Convert one lightweight chain or ligand input to Biohub's native type."""
    from esm.models.esmfold2 import DNAInput as ESMDNAInput
    from esm.models.esmfold2 import LigandInput as ESMLigandInput
    from esm.models.esmfold2 import ProteinInput as ESMProteinInput
    from esm.models.esmfold2 import RNAInput as ESMRNAInput

    if isinstance(item, ProteinInput):
        return ESMProteinInput(
            id=item.id,
            sequence=item.sequence,
            modifications=_to_esm_modifications(item.modifications),
            msa=_to_esm_msa(item.msa),
        )
    if isinstance(item, RNAInput):
        return ESMRNAInput(
            id=item.id,
            sequence=item.sequence,
            modifications=_to_esm_modifications(item.modifications),
        )
    if isinstance(item, DNAInput):
        return ESMDNAInput(
            id=item.id,
            sequence=item.sequence,
            modifications=_to_esm_modifications(item.modifications),
        )
    return ESMLigandInput(id=item.id, smiles=item.smiles, ccd=item.ccd)


def _to_esm_modifications(modifications: list[Modification] | None) -> list[Any] | None:
    """Convert optional lightweight residue modifications to Biohub objects."""
    if modifications is None:
        return None
    from esm.models.esmfold2 import Modification as ESMModification

    return [
        ESMModification(position=modification.position, ccd=modification.ccd, smiles=modification.smiles)
        for modification in modifications
    ]


def _to_esm_msa(msa: MSAInput | Any | None) -> Any | None:
    """Convert lightweight or native MSA inputs to Biohub's MSA object."""
    if msa is None:
        return None
    if isinstance(msa, MSAInput):
        if msa.sequences is None:
            raise ValueError(
                "ESMFold2 currently requires in-memory MSA sequences; "
                "file-backed MSA paths are reserved for other model adapters."
            )
        from esm.models.esmfold2 import MSA

        return MSA.from_sequences(msa.sequences, remove_insertions=msa.remove_insertions)
    from esm.models.esmfold2 import MSA

    if isinstance(msa, MSA):
        return msa
    if isinstance(msa, list):
        return MSA.from_sequences(msa)
    return msa


def _to_esm_pocket_conditioning(pocket: PocketConditioning | None) -> Any | None:
    """Convert optional pocket conditioning to Biohub's native dataclass."""
    if pocket is None:
        return None
    from esm.utils.structure.input_builder import PocketConditioning as ESMPocketConditioning

    return ESMPocketConditioning(binder_chain_id=pocket.binder_chain_id, contacts=pocket.contacts)


def _to_esm_distogram_conditioning(
    distograms: list[DistogramConditioning] | None,
) -> list[Any] | None:
    """Convert optional distogram conditioning entries to Biohub objects."""
    if distograms is None:
        return None
    from esm.models.esmfold2 import DistogramConditioning as ESMDistogramConditioning

    return [
        ESMDistogramConditioning(chain_id=distogram.chain_id, distogram=distogram.distogram) for distogram in distograms
    ]


def _to_esm_covalent_bonds(bonds: list[CovalentBond] | None) -> list[Any] | None:
    """Convert optional covalent-bond constraints to Biohub objects."""
    if bonds is None:
        return None
    from esm.models.esmfold2 import CovalentBond as ESMCovalentBond

    return [
        ESMCovalentBond(
            chain_id1=bond.chain_id1,
            res_idx1=bond.res_idx1,
            atom_idx1=bond.atom_idx1,
            chain_id2=bond.chain_id2,
            res_idx2=bond.res_idx2,
            atom_idx2=bond.atom_idx2,
        )
        for bond in bonds
    ]


def _convert_results(
    results: list[Any],
    metadata: PredictionMetadata,
    config: dict[str, Any],
) -> ESMFold2Output:
    """Convert Biohub decoded outputs to BoilerRoom's normalized output type."""
    include_fields = cast(list[str] | None, config.get("include_fields"))
    atom_arrays: list[Any] = []
    cif_values: list[str] | None = [] if _wants_field(include_fields, "cif") else None
    pdb_values: list[str] | None = [] if _wants_field(include_fields, "pdb") else None

    plddt: list[np.ndarray | None] | None = [] if _wants_field(include_fields, "plddt") else None
    ptm: list[Any] | None = [] if _wants_field(include_fields, "ptm") else None
    iptm: list[Any] | None = [] if _wants_field(include_fields, "iptm") else None
    pae: list[np.ndarray | None] | None = [] if _wants_field(include_fields, "pae") else None
    distogram: list[np.ndarray | None] | None = [] if _wants_field(include_fields, "distogram") else None
    pair_chains_iptm: list[np.ndarray | None] | None = [] if _wants_field(include_fields, "pair_chains_iptm") else None
    residue_index: list[np.ndarray | None] | None = [] if _wants_field(include_fields, "residue_index") else None
    entity_id: list[np.ndarray | None] | None = [] if _wants_field(include_fields, "entity_id") else None

    for result in results:
        cif_string = result.complex.to_mmcif()
        atom_array = _cif_to_atom_array(cif_string)
        atom_arrays.append(atom_array)

        if cif_values is not None:
            cif_values.append(cif_string)
        if pdb_values is not None:
            pdb_values.append(_atom_array_to_pdb(atom_array))
        if plddt is not None:
            plddt.append(_tensor_to_numpy(result.plddt))
        if ptm is not None:
            ptm.append(result.ptm)
        if iptm is not None:
            iptm.append(result.iptm)
        if pae is not None:
            pae.append(_tensor_to_numpy(result.pae))
        if distogram is not None:
            distogram.append(_tensor_to_numpy(result.distogram))
        if pair_chains_iptm is not None:
            pair_chains_iptm.append(_tensor_to_numpy(result.pair_chains_iptm))
        if residue_index is not None:
            residue_index.append(_tensor_to_numpy(result.residue_index))
        if entity_id is not None:
            entity_id.append(_tensor_to_numpy(result.entity_id))

    full_output = ESMFold2Output(
        metadata=metadata,
        atom_array=atom_arrays,
        plddt=plddt,
        ptm=ptm,
        iptm=iptm,
        pae=pae,
        distogram=distogram,
        pair_chains_iptm=pair_chains_iptm,
        residue_index=residue_index,
        entity_id=entity_id,
        pdb=pdb_values,
        cif=cif_values,
    )
    filtered = FoldingAlgorithm._filter_include_fields(full_output, include_fields)
    return cast(ESMFold2Output, filtered)


def _wants_field(include_fields: list[str] | None, field: str) -> bool:
    """Return whether a named optional output field was requested."""
    return include_fields is not None and ("*" in include_fields or field in include_fields)


def _tensor_to_numpy(value: Any) -> np.ndarray | None:
    """Detach a tensor-like value and return it as a NumPy array."""
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return np.asarray(value)


def _cif_to_atom_array(cif_string: str) -> Any:
    """Parse an mmCIF string into a Biotite atom array."""
    from biotite.structure.io.pdbx import CIFFile, get_structure

    return get_structure(CIFFile.read(StringIO(cif_string)), model=1)


def _atom_array_to_pdb(atom_array: Any) -> str:
    """Serialize a Biotite atom array to PDB text."""
    from biotite.structure.io.pdb import PDBFile, set_structure

    pdb_file = PDBFile()
    set_structure(pdb_file, atom_array)
    buffer = StringIO()
    pdb_file.write(buffer)
    return buffer.getvalue()
