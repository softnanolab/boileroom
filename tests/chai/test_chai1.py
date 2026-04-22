from collections.abc import Generator
from io import StringIO

import numpy as np
import pytest
from biotite.structure import AtomArray, rmsd, superimpose
from biotite.structure.io.pdbx import CIFFile, get_structure
from modal import enable_output

from boileroom import Chai1
from boileroom.models.chai.types import Chai1Output

pytestmark = [pytest.mark.integration, pytest.mark.slow, pytest.mark.gpu, pytest.mark.xdist_group("chai1")]

nipah_virus_sequence = "ICLQKTSNQILKPKLISYTLGQSGTCITDPLLAMDEGYFAYSHLERIGSCSRGVSKQRIIGVGEVLDRGDEVPSLFMTNVWTPPNPNTVYHCSAVYNNEFYYVLCAVSTVGDPILNSTYWSGSLMMTRLAVKPKSNGGGYNQHQLALRSIEKGRYDKVMPYGPSGIKQGDTLYFPAVGFLVRTEFKYNDSNCPITKCQYSKPENCRLSMGIRPNSHYILRSGLLKYNLSDGENPKVVFIEISDQRLSIGSPSKIYDSLGQPVFYQASFSWDTMIKFGDVLTVNPLVVNWRNNTVISRPGQSQCPRFNTCPEICWEGVYNDAFLIDRINWISAGVFLDSNQTAENPVFTVFKDNEILYRAQLASEDTNAQKTITNCFLLKNKIWCISLVEIYDTGDNVIRPKLFAVKIPEQCTH"

ROUNDTRIP_RMSD_TOLERANCE_ANGSTROM = 1e-3
DEFAULT_QUICK_OPTIONS = {
    "num_diffn_samples": 1,
    "num_trunk_samples": 1,
    "use_esm_embeddings": True,
    "num_trunk_recycles": 1,
    "num_diffn_timesteps": 10,
    "seed": 42,
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
    assert rmsd_value < ROUNDTRIP_RMSD_TOLERANCE_ANGSTROM, (
        f"CA RMSD {rmsd_value:.6f} Å exceeds tolerance {ROUNDTRIP_RMSD_TOLERANCE_ANGSTROM} Å"
    )


def _scalar(value: np.ndarray | list[float] | float) -> float:
    """Convert a Chai scalar-like value into a Python float."""
    return float(np.asarray(value, dtype=float).reshape(-1)[0])


def _quick_options(include_fields: list[str] | None = None) -> dict[str, object]:
    """Build the reduced-cost Chai options used in integration tests."""
    options: dict[str, object] = dict(DEFAULT_QUICK_OPTIONS)
    if include_fields is not None:
        options["include_fields"] = include_fields
    return options


# Module scope keeps a single Modal app lifecycle for the Chai integration shard.
@pytest.fixture(scope="module")
def chai1_model(config: dict | None = None, gpu_device: str | None = None) -> Generator[Chai1, None, None]:
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
    with enable_output(), Chai1(backend="modal", device=gpu_device, config=model_config) as model:
        yield model


def test_chai1_minimal_output(test_sequences: dict[str, str], chai1_model: Chai1):
    """Test that Chai1 returns minimal output by default (metadata + atom_array)."""
    result = chai1_model.fold(test_sequences["short"], options=_quick_options())

    assert isinstance(result, Chai1Output), "Result should be a Chai1Output"
    assert result.metadata is not None, "metadata should always be present"
    assert result.atom_array is not None, "atom_array should always be generated"
    assert len(result.atom_array) > 0, "atom_array should contain at least one structure"
    assert len(result.atom_array[0]) > 0, "predicted structure should contain atoms"

    # With minimal output, other fields should be None
    assert result.plddt is None, "plddt should be None in minimal output"
    assert result.pae is None, "pae should be None in minimal output"
    assert result.pde is None, "pde should be None in minimal output"
    assert result.cif is None, "cif should be None in minimal output"


def test_chai1_full_output(chai1_model: Chai1):
    """
    Verify that Chai1.fold returns all requested output fields and stable structural invariants.

    Checks that an atom array is produced, confidence fields are populated with finite values in expected
    ranges, PAE has the expected square shape for the sequence, and CIF serialization round-trips back to
    the predicted coordinates. Chai still shows modest run-to-run GPU variation even with a fixed seed, so
    this test intentionally avoids exact comparison against a stored artifact.
    """
    result = chai1_model.fold(nipah_virus_sequence, options=_quick_options(include_fields=["*"]))

    assert isinstance(result, Chai1Output), "Result should be a Chai1Output"
    assert result.atom_array is not None, "atom_array should always be generated"
    assert len(result.atom_array) > 0, "atom_array should contain at least one structure"
    assert len(result.atom_array[0]) > 0, "predicted structure should contain atoms"
    assert result.metadata.model_name == "Chai-1"
    assert result.metadata.sequence_lengths == [len(nipah_virus_sequence)]
    assert result.metadata.inference_time is not None and result.metadata.inference_time > 0
    assert result.plddt is not None, "plddt should be present when requested"
    assert len(result.plddt) > 0, "plddt should contain values"
    plddt = np.asarray(result.plddt[0], dtype=float)
    assert plddt.shape == (len(nipah_virus_sequence),)
    assert np.all(np.isfinite(plddt)), "pLDDT values should be finite"
    assert np.all(plddt >= 0), "pLDDT scores should be non-negative"
    assert np.all(plddt <= 100), "pLDDT scores should be less than or equal to 100"

    assert result.cif is not None and len(result.cif) > 0, "CIF string should be returned when requested"
    predicted_cif_structure = get_structure(CIFFile.read(StringIO(result.cif[0])), model=1)
    _assert_ca_rmsd_within_tolerance(predicted_cif_structure, result.atom_array[0])

    assert result.pae is not None and len(result.pae) > 0, "PAE should be included when requested"
    pae = np.asarray(result.pae[0], dtype=float)
    sequence_length = len(nipah_virus_sequence)
    assert pae.shape == (sequence_length, sequence_length)
    assert np.all(np.isfinite(pae)), "PAE values should be finite"
    assert np.all(pae >= 0), "PAE values should be non-negative"

    assert result.ptm is not None and len(result.ptm) > 0, "ptm values must be present"
    assert result.iptm is not None and len(result.iptm) > 0, "iptm values must be present"
    assert result.per_chain_iptm is not None and len(result.per_chain_iptm) > 0, "per_chain_iptm must be present"
    ptm = _scalar(result.ptm[0])
    iptm = _scalar(result.iptm[0])
    per_chain_iptm = np.atleast_1d(np.asarray(result.per_chain_iptm[0], dtype=float))

    assert 0.0 <= ptm <= 1.0, "ptm should be a probability-like score"
    assert 0.0 <= iptm <= 1.0, "iptm should be a probability-like score"
    assert np.all(np.isfinite(per_chain_iptm)), "per_chain_iptm values should be finite"
    assert np.all(per_chain_iptm >= 0.0), "per_chain_iptm should be non-negative"
    assert np.all(per_chain_iptm <= 1.0), "per_chain_iptm should be less than or equal to 1"


def test_chai1_static_config_enforcement(test_sequences: dict[str, str], chai1_model: Chai1):
    """Test that static config keys cannot be overridden in options."""
    # device is a static config key
    with pytest.raises(ValueError, match="device"):
        chai1_model.fold(test_sequences["short"], options={"device": "cpu"})
