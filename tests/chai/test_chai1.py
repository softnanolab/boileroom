import pytest
import numpy as np

from boileroom import Chai1
from boileroom.models.chai.chai1 import Chai1Output

from typing import Generator, Optional
from modal import enable_output


# Each test instantiates its own model; keeping function scope avoids long-lived Modal handles.
@pytest.fixture
def chai1_model(config: Optional[dict] = None, gpu_device: Optional[str] = None) -> Generator[Chai1, None, None]:
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


def test_chai1_full_output(test_sequences: dict[str, str], chai1_model: Chai1):
    """Test Chai1 with full output requested."""
    quick_options = {
        "num_diffn_samples": 1,
        "num_trunk_samples": 1,
        "use_esm_embeddings": True,
        "num_trunk_recycles": 1,
        "num_diffn_timesteps": 10,
        "output_attributes": ["*"],  # Request all attributes
    }
    result = chai1_model.fold(test_sequences["short"], options=quick_options)

    assert isinstance(result, Chai1Output), "Result should be a Chai1Output"
    assert result.atom_array is not None, "atom_array should always be generated"
    assert result.plddt is not None, "plddt should be present when requested"
    assert len(result.plddt) > 0, "plddt should contain values"
    assert np.all(np.array(result.plddt[0]) >= 0), "pLDDT scores should be non-negative"
    assert np.all(np.array(result.plddt[0]) <= 100), "pLDDT scores should be less than or equal to 100"


# def test_chai1_multimer(test_sequences: dict[str, str], chai1_model: Chai1):
#     """Test Chai-1 multimer functionality."""
#     result = chai1_model.fold(test_sequences["multimer"])

#     assert isinstance(result, Chai1Output), "Result should be a Chai1Output"


def test_chai1_static_config_enforcement(test_sequences: dict[str, str], chai1_model: Chai1):
    """Test that static config keys cannot be overridden in options."""
    # device is a static config key
    with pytest.raises(ValueError, match="device"):
        chai1_model.fold(test_sequences["short"], options={"device": "cpu"})
