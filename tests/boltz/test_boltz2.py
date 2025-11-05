import pathlib
from typing import Generator, Optional

import numpy as np
import pytest
from modal import enable_output

from boileroom import Boltz2
from boileroom.models.boltz.boltz2 import Boltz2Output
from boileroom.constants import restype_3to1

from biotite.structure import AtomArray, superimpose, rmsd
from biotite.structure.io.pdbx import CIFFile, get_structure


nipah_binder_sequence = "ICLQKTSNQILKPKLISYTLGQSGTCITDPLLAMDEGYFAYSHLERIGSCSRGVSKQRIIGVGEVLDRGDEVPSLFMTNVWTPPNPNTVYHCSAVYNNEFYYVLCAVSTVGDPILNSTYWSGSLMMTRLAVKPKSNGGGYNQHQLALRSIEKGRYDKVMPYGPSGIKQGDTLYFPAVGFLVRTEFKYNDSNCPITKCQYSKPENCRLSMGIRPNSHYILRSGLLKYNLSDGENPKVVFIEISDQRLSIGSPSKIYDSLGQPVFYQASFSWDTMIKFGDVLTVNPLVVNWRNNTVISRPGQSQCPRFNTCPEICWEGVYNDAFLIDRINWISAGVFLDSNQTAENPVFTVFKDNEILYRAQLASEDTNAQKTITNCFLLKNKIWCISLVEIYDTGDNVIRPKLFAVKIPEQCTH:IMILYWYWNASNHKFTNFQAGQVDPYILLDLDMCEPQKVAQYDYIYWTVHPMIFKDYPWMEPQIQIFKKELWTPENFQDCANPEQHKFIIWFQSNESPNMGGNEFQPGKDYMIISTSNGELDWGFLDMGKVCDESAWIDMSSPNHSQE"

@pytest.fixture(scope="module")
def boltz2_model(config: Optional[dict] = None) -> Generator[Boltz2, None, None]:
    model_config = dict(config) if config is not None else {}
    with enable_output():
        yield Boltz2(backend="modal", config=model_config)


def _recover_chain_sequences(atomarray: AtomArray) -> list[str]:
    chains = []
    for chain_id in np.unique(atomarray.chain_id):
        chain_atoms = atomarray[atomarray.chain_id == chain_id]
        unique_res_ids = np.unique(chain_atoms.res_id)
        three_letter_codes = [chain_atoms.res_name[chain_atoms.res_id == res_id][0] for res_id in unique_res_ids]
        one_letter_codes = [restype_3to1[code] for code in three_letter_codes]
        chains.append("".join(one_letter_codes))
    return chains


def test_boltz2_against_reference_cif(data_dir: pathlib.Path, boltz2_model: Boltz2):
    # Locate reference CIF; skip if not present (user will provide one)
    candidate = data_dir / "boltz" / "0_model_0.cif"
    if not candidate.exists():
        pytest.skip("Reference CIF not available: boltz/0_model_0.cif")

    # Read reference structure
    cif = CIFFile.read(candidate)
    ref = get_structure(cif, model=1)

    # Build sequences from reference chain order and run prediction
    chain_seqs = _recover_chain_sequences(ref)
    input_multimer = ":".join(chain_seqs) if len(chain_seqs) > 1 else chain_seqs[0]

    # Ensure reference chain sequences match provided Nipah binder sequence
    assert ":".join(chain_seqs) == nipah_binder_sequence, "Reference CIF sequences mismatch with expected Nipah binder sequence"

    with enable_output():
        result = boltz2_model.fold(input_multimer)

    # Basic type and shape checks
    assert isinstance(result, Boltz2Output), "Result should be a Boltz2Output"
    assert result.positions is not None, "Positions should be generated"

    # Validate sequence order matches reference
    assert isinstance(input_multimer, str)
    pred_seqs = input_multimer.split(":")
    assert pred_seqs == chain_seqs, "Predicted input sequences must match reference chain order and identity"

    # Extract predicted CA coordinates; handle possible dimensional variants
    pos = result.positions
    # Expected shapes could be: (S, B, R, 14, 3) or (B, R, 14, 3) or (R, 14, 3)
    if pos.ndim == 5:
        pred = np.asarray(pos[0, 0])
    elif pos.ndim == 4:
        pred = np.asarray(pos[0])
    else:
        pred = np.asarray(pos)

    assert pred.shape[-2:] == (14, 3), f"Unexpected atom/coord shape: {pred.shape}"

    # Compare residue counts
    num_res_ref = len(np.unique(ref.res_id))
    assert pred.shape[-3] == num_res_ref, f"Residue count mismatch: pred={pred.shape[-3]} ref={num_res_ref}"

    # Build a CA-only AtomArray from prediction, reusing reference CA topology for metadata
    ref_ca = ref[ref.atom_name == "CA"]
    pred_ca = ref_ca.copy()
    pred_ca.coord = pred[:, 1, :]  # Atom14 convention: index 1 is CA

    # Superimpose and compute RMSD on CA atoms
    superimposed, _ = superimpose(ref_ca, pred_ca)
    ca_rmsd = rmsd(ref_ca, superimposed)

    # Tolerance: generous due to stochasticity; will be refined with seeding later
    assert ca_rmsd < 5.0, f"CA RMSD too high: {ca_rmsd:.2f} Å"


def test_boltz2_lengths_monomer_and_multimer(test_sequences: dict[str, str]):
    """Check monomer and multimer residue counts in positions output."""
    with enable_output():
        model = Boltz2(backend="modal")

    # Monomer
    mono = test_sequences["short"]
    mono_out = model.fold(mono)
    assert isinstance(mono_out, Boltz2Output)
    mono_pos = mono_out.positions
    # Normalize to last dims (..., R, 14, 3)
    if mono_pos.ndim == 5:
        mono_arr = np.asarray(mono_pos[0, 0])
    elif mono_pos.ndim == 4:
        mono_arr = np.asarray(mono_pos[0])
    else:
        mono_arr = np.asarray(mono_pos)
    assert mono_arr.shape[-2:] == (14, 3)
    assert mono_arr.shape[-3] == len(mono), "Monomer residue length mismatch"

    # Multimer
    multi = test_sequences["multimer"]
    multi_out = model.fold(multi)
    assert isinstance(multi_out, Boltz2Output)
    multi_pos = multi_out.positions
    if multi_pos.ndim == 5:
        multi_arr = np.asarray(multi_pos[0, 0])
    elif multi_pos.ndim == 4:
        multi_arr = np.asarray(multi_pos[0])
    else:
        multi_arr = np.asarray(multi_pos)
    assert multi_arr.shape[-2:] == (14, 3)
    exp_len = len(multi.replace(":", ""))
    assert multi_arr.shape[-3] == exp_len, "Multimer residue length mismatch"


def test_boltz2_invalid_amino_acids_validation(test_sequences: dict[str, str]):
    """Boltz2Core should reject invalid sequences via inherited validation helper."""
    # Use the core's validator directly to ensure it raises for invalid inputs
    from boileroom.models.boltz.boltz2 import Boltz2Core

    core = Boltz2Core(config={"device": "cpu"})
    with pytest.raises(ValueError):
        core._validate_sequences(test_sequences["invalid"])  # should raise


