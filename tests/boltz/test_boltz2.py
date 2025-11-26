import pathlib
from typing import Generator, Optional

import numpy as np
import pytest
from modal import enable_output

from boileroom import Boltz2
from boileroom.models.boltz.types import Boltz2Output
from boileroom.constants import restype_3to1

from biotite.structure import AtomArray, rmsd, superimpose
from biotite.structure.io.pdbx import CIFFile, get_structure


nipah_virus_sequence = "ICLQKTSNQILKPKLISYTLGQSGTCITDPLLAMDEGYFAYSHLERIGSCSRGVSKQRIIGVGEVLDRGDEVPSLFMTNVWTPPNPNTVYHCSAVYNNEFYYVLCAVSTVGDPILNSTYWSGSLMMTRLAVKPKSNGGGYNQHQLALRSIEKGRYDKVMPYGPSGIKQGDTLYFPAVGFLVRTEFKYNDSNCPITKCQYSKPENCRLSMGIRPNSHYILRSGLLKYNLSDGENPKVVFIEISDQRLSIGSPSKIYDSLGQPVFYQASFSWDTMIKFGDVLTVNPLVVNWRNNTVISRPGQSQCPRFNTCPEICWEGVYNDAFLIDRINWISAGVFLDSNQTAENPVFTVFKDNEILYRAQLASEDTNAQKTITNCFLLKNKIWCISLVEIYDTGDNVIRPKLFAVKIPEQCTH"


@pytest.fixture(scope="module")
def boltz2_model(config: Optional[dict] = None, gpu_device: Optional[str] = None) -> Generator[Boltz2, None, None]:
    """Provide a Boltz2 model instance configured for the Modal backend.

    Parameters
    ----------
    config : Optional[dict]
        Optional model configuration overrides to apply when constructing the Boltz2 instance.
    gpu_device : Optional[str]
        Optional device identifier to run the model on (for example, "cuda:0" or similar).

    Yields
    ------
    Boltz2
        A Boltz2 instance configured with backend="modal", the specified device, and the provided configuration.
    """
    model_config = dict(config) if config is not None else {}
    with enable_output():
        yield Boltz2(backend="modal", device=gpu_device, config=model_config)


def _recover_chain_sequences(atomarray: AtomArray) -> list[str]:
    """Extract one-letter amino-acid sequences for each chain in an AtomArray.

    Parameters
    ----------
    atomarray : AtomArray
        Biotite AtomArray containing residues with `chain_id`, `res_id`, and three-letter `res_name` fields.

    Returns
    -------
    list[str]
        A list of one-letter sequences, one string per unique chain in the order of numpy.unique on `chain_id`.
    """
    chains = []
    for chain_id in np.unique(atomarray.chain_id):
        chain_atoms = atomarray[atomarray.chain_id == chain_id]
        unique_res_ids = np.unique(chain_atoms.res_id)
        three_letter_codes = [chain_atoms.res_name[chain_atoms.res_id == res_id][0] for res_id in unique_res_ids]
        one_letter_codes = [restype_3to1[code] for code in three_letter_codes]
        chains.append("".join(one_letter_codes))
    return chains


def test_boltz2_nipah_matches_reference(gpu_device: Optional[str]):
    """Run Boltz2 on the Nipah virus sequence and validate the predicted structure against reference data.

    Checks that required reference files exist, runs the model requesting all output fields, verifies an atom array was produced, compares C-alpha (CA) atoms between the predicted and reference structures, and asserts the superimposed CA RMSD is less than 0.5 Å.

    Parameters
    ----------
    gpu_device : Optional[str]
        GPU device identifier to use for the model, or `None` to run on CPU.
    """
    base_dir = pathlib.Path(__file__).resolve().parents[1] / "data" / "boltz"
    conf_path = base_dir / "confidence_0_model_0.json"
    cif_path = base_dir / "0_model_0.cif"
    plddt_npz = base_dir / "plddt_0_model_0.npz"
    pae_npz = base_dir / "pae_0_model_0.npz"
    pde_npz = base_dir / "pde_0_model_0.npz"
    assert conf_path.exists(), "tests/data/boltz/confidence_0_model_0.json must exist"
    assert cif_path.exists(), "tests/data/boltz/0_model_0.cif must exist"
    assert plddt_npz.exists(), "tests/data/boltz/plddt_0_model_0.npz must exist"
    assert pae_npz.exists(), "tests/data/boltz/pae_0_model_0.npz must exist"
    assert pde_npz.exists(), "tests/data/boltz/pde_0_model_0.npz must exist"

    with enable_output():
        model = Boltz2(
            backend="modal",
            device=gpu_device,
            config={
                "include_fields": ["*"],  # Request all fields for comprehensive testing
            },
        )
        # Note: we cannot guarantee fully deterministic output across different hardware
        # Current Boltz-2 implementation also does not set CUDA-based RNG to deterministic mode
        out = model.fold(nipah_virus_sequence, options={"seed": 42})

    assert isinstance(out, Boltz2Output)

    # Verify minimal defaults: atom_array should always be present
    assert out.atom_array is not None, "atom_array should always be generated"
    assert len(out.atom_array) > 0, "atom_array should contain at least one structure"

    # load the reference cif
    reference_atom_array = get_structure(CIFFile.read(cif_path), model=1)
    predicted_atom_array = out.atom_array[0]

    # Use CA atoms for backbone comparison (standard practice for protein structure comparison)
    predicted_ca = predicted_atom_array[predicted_atom_array.atom_name == "CA"]
    reference_ca = reference_atom_array[reference_atom_array.atom_name == "CA"]

    # Ensure both structures have the same number of CA atoms
    assert len(predicted_ca) == len(reference_ca), (
        f"Number of CA atoms must match: predicted has {len(predicted_ca)}, " f"reference has {len(reference_ca)}"
    )

    # Calculate RMSD between reference and superimposed predicted structure
    predicted_superimposed, _ = superimpose(reference_ca, predicted_ca)
    rmsd_value = rmsd(reference_ca, predicted_superimposed)
    assert rmsd_value < 0.5, f"RMSD {rmsd_value:.4f} Å exceeds threshold of 0.5 Å"

    # TODO: check confidence metrics within tolerance


def test_boltz2_minimal_output(test_sequences: dict[str, str], gpu_device: Optional[str]):
    """Test that Boltz2 returns minimal output by default (metadata + atom_array)."""
    with enable_output():
        model = Boltz2(backend="modal", device=gpu_device, config={})  # No include_fields = minimal output
        out = model.fold(test_sequences["short"])

    assert isinstance(out, Boltz2Output)
    assert out.metadata is not None, "metadata should always be present"
    assert out.atom_array is not None, "atom_array should always be generated"
    # With minimal output, other fields should be None
    assert out.confidence is None, "confidence should be None in minimal output"
    assert out.plddt is None, "plddt should be None in minimal output"
    assert out.pae is None, "pae should be None in minimal output"
    assert out.pde is None, "pde should be None in minimal output"


def test_boltz2_invalid_amino_acids_validation(test_sequences: dict[str, str]):
    """Verify that Boltz2Core's sequence validator raises a ValueError for invalid amino-acid sequences.

    This test calls Boltz2Core._validate_sequences with a sequence labelled "invalid" in the provided fixtures and expects a ValueError to be raised.

    Parameters
    ----------
    test_sequences : dict[str, str]
        Mapping of named test sequences; must include an "invalid" entry containing a sequence with invalid/unsupported amino-acid codes.
    """
    # Use the core's validator directly to ensure it raises for invalid inputs
    from boileroom.models.boltz.core import Boltz2Core

    core = Boltz2Core(config={"device": "cpu"})
    with pytest.raises(ValueError):
        core._validate_sequences(test_sequences["invalid"])  # should raise


def test_boltz2_static_config_enforcement(test_sequences: dict[str, str]):
    """Test that static config keys cannot be overridden in options."""
    from boileroom.models.boltz.core import Boltz2Core

    core = Boltz2Core(config={"device": "cpu"})
    # device, cache_dir, and no_kernels are static config keys
    with pytest.raises(ValueError, match="device"):
        core.fold(test_sequences["short"], options={"device": "cuda:0"})
    with pytest.raises(ValueError, match="cache_dir"):
        core.fold(test_sequences["short"], options={"cache_dir": "/tmp/test"})
    with pytest.raises(ValueError, match="no_kernels"):
        core.fold(test_sequences["short"], options={"no_kernels": True})
