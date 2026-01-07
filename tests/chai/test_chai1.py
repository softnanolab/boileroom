import json
from io import StringIO
from pathlib import Path
from typing import Any, Generator, Optional

import numpy as np
import pytest
from biotite.structure import AtomArray, rmsd, superimpose
from biotite.structure.io.pdbx import CIFFile, get_structure
from modal import enable_output

from boileroom import Chai1
from boileroom.models.chai.types import Chai1Output

nipah_virus_sequence = "ICLQKTSNQILKPKLISYTLGQSGTCITDPLLAMDEGYFAYSHLERIGSCSRGVSKQRIIGVGEVLDRGDEVPSLFMTNVWTPPNPNTVYHCSAVYNNEFYYVLCAVSTVGDPILNSTYWSGSLMMTRLAVKPKSNGGGYNQHQLALRSIEKGRYDKVMPYGPSGIKQGDTLYFPAVGFLVRTEFKYNDSNCPITKCQYSKPENCRLSMGIRPNSHYILRSGLLKYNLSDGENPKVVFIEISDQRLSIGSPSKIYDSLGQPVFYQASFSWDTMIKFGDVLTVNPLVVNWRNNTVISRPGQSQCPRFNTCPEICWEGVYNDAFLIDRINWISAGVFLDSNQTAENPVFTVFKDNEILYRAQLASEDTNAQKTITNCFLLKNKIWCISLVEIYDTGDNVIRPKLFAVKIPEQCTH"

REFERENCE_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "chai"
REFERENCE_RANK = 0
RMSD_TOLERANCE_ANGSTROM = 0.5
PAE_ABS_TOLERANCE = 1.0
PAE_REL_TOLERANCE = 5e-2
SCORE_TOLERANCE = 5e-2


@pytest.fixture(scope="session")
def chai_reference_package() -> dict[str, Any]:
    """Load reference Chai-1 outputs shipped with the repository."""
    cif_path = REFERENCE_DATA_DIR / f"pred.rank_{REFERENCE_RANK}.cif"
    pae_path = REFERENCE_DATA_DIR / f"pae.rank_{REFERENCE_RANK}.npy"
    scores_path = REFERENCE_DATA_DIR / f"scores.rank_{REFERENCE_RANK}.json"

    assert cif_path.exists(), f"{cif_path} must exist"
    assert pae_path.exists(), f"{pae_path} must exist"
    assert scores_path.exists(), f"{scores_path} must exist"

    cif_file = CIFFile.read(str(cif_path))
    reference_structure = get_structure(cif_file, model=1)
    reference_pae = np.load(str(pae_path))
    with scores_path.open(encoding="utf-8") as handle:
        reference_scores = json.load(handle)

    return {
        "atom_array": reference_structure,
        "pae": reference_pae,
        "scores": reference_scores,
    }


def _assert_ca_rmsd_within_tolerance(predicted: AtomArray, reference: AtomArray) -> None:
    """Ensure the C-alpha RMSD between predicted and reference structures stays within tolerance."""
    predicted_ca = predicted[predicted.atom_name == "CA"]
    reference_ca = reference[reference.atom_name == "CA"]
    assert len(predicted_ca) == len(reference_ca), (
        "Predicted and reference structures must contain the same number of CA atoms; "
        f"got {len(predicted_ca)} vs {len(reference_ca)}"
    )
    predicted_superimposed, _ = superimpose(reference_ca, predicted_ca)
    rmsd_value = rmsd(reference_ca, predicted_superimposed)
    assert (
        rmsd_value < RMSD_TOLERANCE_ANGSTROM
    ), f"CA RMSD {rmsd_value:.3f} Å exceeds tolerance {RMSD_TOLERANCE_ANGSTROM} Å"


def _assert_score_close(actual: float, expected: float, label: str) -> None:
    """Compare scalar confidence metrics within SCORE_TOLERANCE."""
    delta = abs(actual - expected)
    assert delta <= SCORE_TOLERANCE, f"{label} delta {delta:.4f} exceeds tolerance {SCORE_TOLERANCE}"


# Each test instantiates its own model; keeping function scope avoids long-lived Modal handles.
@pytest.fixture
def chai1_model(config: Optional[dict] = None, gpu_device: Optional[str] = None) -> Generator[Chai1, None, None]:
    """Provide a Chai1 model instance configured to run with the Modal backend.

    Parameters
    ----------
    config : Optional[dict]
        Optional model configuration mapping used when creating the model.
    gpu_device : Optional[str]
        Optional device identifier passed to the model (e.g., "cpu" or a CUDA device string).

    Yields
    ------
    Chai1
        A Chai1 model instance yielded by the generator for use in tests.
    """
    model_config = dict(config) if config is not None else {}
    with enable_output():
        yield Chai1(backend="modal", device=gpu_device, config=model_config)


def test_chai1_minimal_output(test_sequences: dict[str, str], chai1_model: Chai1):
    """Test that Chai1 returns minimal output by default (metadata + atom_array)."""
    quick_options = {
        "num_diffn_samples": 1,
        "num_trunk_samples": 1,
        "use_esm_embeddings": True,
        "num_trunk_recycles": 1,
        "num_diffn_timesteps": 10,
    }
    result = chai1_model.fold(test_sequences["short"], options=quick_options)

    assert isinstance(result, Chai1Output), "Result should be a Chai1Output"
    assert result.metadata is not None, "metadata should always be present"
    assert result.atom_array is not None, "atom_array should always be generated"
    assert len(result.atom_array) > 0, "atom_array should contain at least one structure"

    # With minimal output, other fields should be None
    assert result.plddt is None, "plddt should be None in minimal output"
    assert result.pae is None, "pae should be None in minimal output"
    assert result.pde is None, "pde should be None in minimal output"
    assert result.cif is None, "cif should be None in minimal output"


def test_chai1_full_output(
    test_sequences: dict[str, str],
    chai1_model: Chai1,
    chai_reference_package: dict[str, Any],
):
    """
    Verify that Chai1.fold returns all requested output fields and that reported pLDDT scores are valid.

    Checks that an atom array is produced, that the pLDDT field is present and non-empty, and that all pLDDT values are between 0 and 100 inclusive.
    """
    quick_options = {
        "num_diffn_samples": 1,
        "num_trunk_samples": 1,
        "use_esm_embeddings": True,
        "num_trunk_recycles": 1,
        "num_diffn_timesteps": 10,
        "include_fields": ["*"],  # Request all fields
    }
    result = chai1_model.fold(nipah_virus_sequence, options=quick_options)

    assert isinstance(result, Chai1Output), "Result should be a Chai1Output"
    assert result.atom_array is not None, "atom_array should always be generated"
    assert len(result.atom_array) > 0, "atom_array should contain at least one structure"
    assert result.plddt is not None, "plddt should be present when requested"
    assert len(result.plddt) > 0, "plddt should contain values"
    assert np.all(np.array(result.plddt[0]) >= 0), "pLDDT scores should be non-negative"
    assert np.all(np.array(result.plddt[0]) <= 100), "pLDDT scores should be less than or equal to 100"

    reference_structure = chai_reference_package["atom_array"]
    _assert_ca_rmsd_within_tolerance(result.atom_array[0], reference_structure)

    assert result.cif is not None and len(result.cif) > 0, "CIF string should be returned when requested"
    predicted_cif_structure = get_structure(CIFFile.read(StringIO(result.cif[0])), model=1)
    _assert_ca_rmsd_within_tolerance(predicted_cif_structure, reference_structure)

    assert result.pae is not None and len(result.pae) > 0, "PAE should be included when requested"
    np.testing.assert_allclose(
        result.pae[0],
        chai_reference_package["pae"],
        rtol=PAE_REL_TOLERANCE,
        atol=PAE_ABS_TOLERANCE,
        err_msg="PAE mismatch relative to reference data",
    )

    reference_scores = chai_reference_package["scores"]
    assert result.ptm is not None and len(result.ptm) > 0, "ptm values must be present"
    assert result.iptm is not None and len(result.iptm) > 0, "iptm values must be present"
    assert result.per_chain_iptm is not None and len(result.per_chain_iptm) > 0, "per_chain_iptm must be present"

    _assert_score_close(float(result.ptm[0]), float(reference_scores["ptm"]), "ptm")
    _assert_score_close(float(result.iptm[0]), float(reference_scores["iptm"]), "iptm")

    predicted_chain_scores = np.atleast_1d(np.array(result.per_chain_iptm[0], dtype=float))
    reference_chain_scores = np.atleast_1d(np.array(reference_scores.get("per_chain_pair_iptm"), dtype=float))
    np.testing.assert_allclose(
        predicted_chain_scores,
        reference_chain_scores,
        atol=SCORE_TOLERANCE,
        err_msg="per_chain_iptm mismatch relative to reference data",
    )


def test_chai1_static_config_enforcement(test_sequences: dict[str, str], chai1_model: Chai1):
    """Test that static config keys cannot be overridden in options."""
    # device is a static config key
    with pytest.raises(ValueError, match="device"):
        chai1_model.fold(test_sequences["short"], options={"device": "cpu"})
