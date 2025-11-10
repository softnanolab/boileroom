import json
import pathlib
from typing import Generator, Optional

import numpy as np
import pytest
from modal import enable_output

from boileroom import Boltz2
from boileroom.models.boltz.boltz2 import Boltz2Output
from boileroom.constants import restype_3to1

from biotite.structure import AtomArray
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


def test_boltz2_nipah_matches_reference():
    """Run Nipah system once and compare confidence + arrays to tests/data/boltz; also check basic shapes."""
    base_dir = pathlib.Path(__file__).resolve().parents[1] / "data" / "boltz"
    conf_path = base_dir / "confidence_0_model_0.json"
    plddt_npz = base_dir / "plddt_0_model_0.npz"
    pae_npz = base_dir / "pae_0_model_0.npz"
    pde_npz = base_dir / "pde_0_model_0.npz"
    assert conf_path.exists(), "tests/data/boltz/confidence_0_model_0.json must exist"
    assert plddt_npz.exists(), "tests/data/boltz/plddt_0_model_0.npz must exist"
    assert pae_npz.exists(), "tests/data/boltz/pae_0_model_0.npz must exist"
    assert pde_npz.exists(), "tests/data/boltz/pde_0_model_0.npz must exist"

    with enable_output():
        model = Boltz2(backend="modal", config={
            "write_full_pae": True, 
            "write_full_pde": True,
            "output_attributes": ["*"]  # Request all attributes for comprehensive testing
        })
        out = model.fold(nipah_binder_sequence)

    assert isinstance(out, Boltz2Output)
    
    # Verify minimal defaults: atom_array should always be present
    assert out.atom_array is not None, "atom_array should always be generated"
    assert len(out.atom_array) > 0, "atom_array should contain at least one structure"

    # Basic positions check (if positions are requested)
    pos = out.positions
    if pos.ndim == 5:
        pred = np.asarray(pos[0, 0])
    elif pos.ndim == 4:
        pred = np.asarray(pos[0])
    else:
        pred = np.asarray(pos)
    assert pred.shape[-2:] == (14, 3)
    assert pred.shape[-3] == len(nipah_binder_sequence.replace(":", ""))

    # Confidence comparison
    reference_data = json.loads(conf_path.read_text())
    assert out.confidence is not None and len(out.confidence) > 0
    pred_conf = out.confidence[0]
    tolerance = 1e-3
    for key in [
        "confidence_score",
        "ptm",
        "iptm",
        "ligand_iptm",
        "protein_iptm",
        "complex_plddt",
        "complex_iplddt",
        "complex_pde",
        "complex_ipde",
    ]:
        if key in reference_data:
            assert key in pred_conf
            assert abs(reference_data[key] - pred_conf[key]) < tolerance
    if "chains_ptm" in reference_data:
        for k, v in reference_data["chains_ptm"].items():
            assert k in pred_conf["chains_ptm"]
            assert abs(v - pred_conf["chains_ptm"][k]) < tolerance
    if "pair_chains_iptm" in reference_data:
        for i, inner in reference_data["pair_chains_iptm"].items():
            for j, v in inner.items():
                assert abs(v - pred_conf["pair_chains_iptm"][i][j]) < tolerance

    # Array comparisons
    assert out.plddt is not None and out.pae is not None and out.pde is not None
    ref_plddt = np.load(plddt_npz)["arr_0"]
    ref_pae = np.load(pae_npz)["arr_0"]
    ref_pde = np.load(pde_npz)["arr_0"]
    pred_plddt = out.plddt[0]
    pred_pae = out.pae[0]
    pred_pde = out.pde[0]
    assert ref_plddt.shape == pred_plddt.shape
    assert ref_pae.shape == pred_pae.shape
    assert ref_pde.shape == pred_pde.shape
    assert np.max(np.abs(ref_plddt - pred_plddt)) < tolerance
    assert np.max(np.abs(ref_pae - pred_pae)) < tolerance
    assert np.max(np.abs(ref_pde - pred_pde)) < tolerance


    # Note: Avoid extra inference-based tests; one Nipah run is sufficient for shapes and value checks.


def test_boltz2_minimal_output(test_sequences: dict[str, str]):
    """Test that Boltz2 returns minimal output by default (metadata + atom_array)."""
    with enable_output():
        model = Boltz2(backend="modal", config={})  # No output_attributes = minimal output
        out = model.fold(test_sequences["short"])
    
    assert isinstance(out, Boltz2Output)
    assert out.metadata is not None, "metadata should always be present"
    assert out.atom_array is not None, "atom_array should always be generated"
    # With minimal output, other fields should be None
    assert out.positions is None, "positions should be None in minimal output"
    assert out.confidence is None, "confidence should be None in minimal output"
    assert out.plddt is None, "plddt should be None in minimal output"
    assert out.pae is None, "pae should be None in minimal output"
    assert out.pde is None, "pde should be None in minimal output"


def test_boltz2_invalid_amino_acids_validation(test_sequences: dict[str, str]):
    """Boltz2Core should reject invalid sequences via inherited validation helper."""
    # Use the core's validator directly to ensure it raises for invalid inputs
    from boileroom.models.boltz.boltz2 import Boltz2Core

    core = Boltz2Core(config={"device": "cpu"})
    with pytest.raises(ValueError):
        core._validate_sequences(test_sequences["invalid"])  # should raise


def test_boltz2_static_config_enforcement(test_sequences: dict[str, str]):
    """Test that static config keys cannot be overridden in options."""
    from boileroom.models.boltz.boltz2 import Boltz2Core

    core = Boltz2Core(config={"device": "cpu"})
    # device, cache_dir, and no_kernels are static config keys
    with pytest.raises(ValueError, match="device"):
        core.fold(test_sequences["short"], options={"device": "cuda:0"})
    with pytest.raises(ValueError, match="cache_dir"):
        core.fold(test_sequences["short"], options={"cache_dir": "/tmp/test"})
    with pytest.raises(ValueError, match="no_kernels"):
        core.fold(test_sequences["short"], options={"no_kernels": True})
