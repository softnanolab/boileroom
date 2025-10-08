import pytest
import numpy as np

from boileroom import Chai1
from boileroom.models.chai.chai1 import Chai1Output

from typing import Generator, Optional
from modal import enable_output


# Each test instantiates its own model; keeping function scope avoids long-lived Modal handles.
@pytest.fixture
def chai1_model(config: Optional[dict] = None) -> Generator[Chai1, None, None]:
    model_config = dict(config) if config is not None else {}
    with enable_output():
        yield Chai1(backend="modal", config=model_config)
    

def test_chai1_basic(test_sequences: dict[str, str], chai1_model: Chai1):
    """Test basic Chai-1 functionality."""
    result = chai1_model.fold(test_sequences["short"])

    assert isinstance(result, Chai1Output), "Result should be a Chai1Output"

    assert result.positions.shape[2] == 3, "Number of dimensions should be 3"
    assert result.positions.shape[0] == len(test_sequences["short"]), "Number of residues mismatch"
    assert result.positions.shape[1] == 3, "Number of dimensions should be 3"

    assert result.plddt.shape == (len(test_sequences["short"]),), "pLDDT scores should be a single dimension"
    assert np.all(result.plddt >= 0), "pLDDT scores should be non-negative"
    assert np.all(result.plddt <= 100), "pLDDT scores should be less than or equal to 100"