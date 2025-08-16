import pytest
import numpy as np
from typing import Generator
from modal import enable_output

from boileroom import app
from boileroom.models.chai.chai1 import Chai1, Chai1Output
from boileroom.convert import pdb_string_to_atomarray


@pytest.fixture
def chai1_model(config={}) -> Generator[Chai1, None, None]:
	with enable_output(), app.run():
		yield Chai1(config=config)


def test_chai1_basic(test_sequences: dict[str, str], chai1_model: Chai1, run_backend):
	sequence = test_sequences["short"]
	result = run_backend(chai1_model.fold)(sequence)
	assert isinstance(result, Chai1Output)
	assert result.pdb is not None and isinstance(result.pdb, list) and len(result.pdb) == 1
	assert result.pdb[0].startswith("HEADER") or "ATOM" in result.pdb[0]
	# Parse AtomArray if requested
	if result.atom_array is not None:
		atomarray = result.atom_array[0]
		# Sanity checks: there should be atoms and residues
		assert atomarray.coord.shape[0] > 0
		assert len(np.unique(atomarray.res_id)) == len(sequence)