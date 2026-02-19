import pytest
import numpy as np
from typing import Generator, Optional
from modal import enable_output

from boileroom import MiniFold
from boileroom.models.minifold.core import MiniFoldCore
from boileroom.models.minifold.types import MiniFoldOutput
from biotite.structure import AtomArray


# Module scope keeps a single Modal container alive for the duration of the suite.
@pytest.fixture(scope="module")
def minifold_model(gpu_device: Optional[str] = None) -> Generator[MiniFold, None, None]:
    """Provide a configured MiniFold model instance for tests."""
    with enable_output():
        yield MiniFold(backend="modal", device=gpu_device, config={})


def test_minifold_basic(test_sequences: dict[str, str], minifold_model: MiniFold):
    """Test basic MiniFold functionality with minimal output."""
    result = minifold_model.fold(test_sequences["short"])

    assert isinstance(result, MiniFoldOutput), "Result should be a MiniFoldOutput"
    assert result.metadata is not None, "metadata should always be present"
    assert result.atom_array is not None, "atom_array should always be generated"
    assert len(result.atom_array) > 0, "atom_array should contain at least one structure"
    assert isinstance(result.atom_array[0], AtomArray), "atom_array elements should be AtomArray"

    # With minimal output, pdb/cif should be None
    assert result.pdb is None, "pdb should be None in minimal output"
    assert result.cif is None, "cif should be None in minimal output"


def test_minifold_full_output(test_sequences: dict[str, str], gpu_device: Optional[str]):
    """Test MiniFold with full output requested."""
    with enable_output():
        model = MiniFold(backend="modal", device=gpu_device, config={"include_fields": ["*"]})
        result = model.fold(test_sequences["short"])

    assert isinstance(result, MiniFoldOutput), "Result should be a MiniFoldOutput"

    assert result.plddt is not None, "plddt should be present in full output"
    assert len(result.plddt) == 1, "plddt should have one entry for single sequence"
    assert np.all(result.plddt[0] >= 0), "pLDDT scores should be non-negative"
    assert np.all(result.plddt[0] <= 100), "pLDDT scores should be less than or equal to 100"

    assert result.pdb is not None, "pdb should be present in full output"
    assert result.cif is not None, "cif should be present in full output"


def test_minifold_batch(test_sequences: dict[str, str], gpu_device: Optional[str]):
    """Test MiniFold batch prediction."""
    with enable_output():
        model = MiniFold(backend="modal", device=gpu_device, config={"include_fields": ["*"]})
        sequences = [test_sequences["short"], test_sequences["medium"]]
        result = model.fold(sequences)

    assert result.atom_array is not None, "atom_array should be present"
    assert len(result.atom_array) == 2, "Should have two atom arrays for two sequences"

    assert result.plddt is not None, "plddt should be present"
    assert len(result.plddt) == 2, "Should have two plddt arrays for two sequences"

    # Check sequence lengths match
    assert len(result.plddt[0]) == len(test_sequences["short"]), "plddt length should match short sequence"
    assert len(result.plddt[1]) == len(test_sequences["medium"]), "plddt length should match medium sequence"


def test_minifold_multimer(test_sequences: dict[str, str], gpu_device: Optional[str]):
    """Test MiniFold multimer functionality."""
    with enable_output():
        model = MiniFold(backend="modal", device=gpu_device, config={"include_fields": ["*"]})
        result = model.fold(test_sequences["multimer"])

    assert result.pdb is not None, "PDB output should be generated"
    assert result.residue_index is not None, "residue_index should be generated"
    assert result.chain_index is not None, "chain_index should be generated"

    # The multimer test sequence is two identical 54-residue chains
    n_residues = len(test_sequences["multimer"].replace(":", ""))
    chain_len = 54  # Each chain is 54 residues

    assert np.all(result.residue_index[0][:chain_len] == np.arange(0, chain_len)), "First chain residue index mismatch"
    assert np.all(result.residue_index[0][chain_len:n_residues] == np.arange(0, chain_len)), "Second chain residue index mismatch"
    assert np.all(result.chain_index[0][:chain_len] == 0), "First chain index mismatch"
    assert np.all(result.chain_index[0][chain_len:n_residues] == 1), "Second chain index mismatch"

    # Check chain assignments in atom array
    structure = result.atom_array[0]
    unique_chains = np.unique(structure.chain_id)
    assert len(unique_chains) == 2, f"Expected 2 chains, got {len(unique_chains)}"


def test_minifold_output_pdb_cif(test_sequences: dict[str, str], gpu_device: Optional[str]):
    """Validate that MiniFold produces consistent PDB, CIF, and AtomArray outputs."""
    from io import StringIO
    from biotite.structure.io.pdb import PDBFile

    with enable_output():
        model = MiniFold(
            backend="modal", device=gpu_device, config={"include_fields": ["pdb", "atom_array"]}
        )
        sequences = [test_sequences["short"], test_sequences["medium"]]
        result = model.fold(sequences)

    assert result.pdb is not None, "PDB output should be generated"
    assert result.cif is None, "CIF output should be None (not requested)"
    assert result.atom_array is not None, "Atom array output should always be generated"

    assert len(result.pdb) == len(result.atom_array) == len(sequences) == 2, "Batch output match"

    short_pdb = PDBFile.read(StringIO(result.pdb[0])).get_structure(model=1)
    short_atomarray = result.atom_array[0]

    # Check residues are 0-indexed
    num_residues = len(sequences[0])
    assert np.all(
        np.unique(short_atomarray.res_id) == np.arange(0, num_residues)
    ), "AtomArray residues should be 0-indexed"

    # Compare coordinates with tolerance
    assert np.allclose(
        short_pdb.coord, short_atomarray.coord, atol=0.1
    ), "Atom coordinates should be equal within 0.1A tolerance"

    # Compare other attributes exactly
    assert np.array_equal(short_pdb.chain_id, short_atomarray.chain_id), "Chain IDs should match exactly"
    assert np.array_equal(short_pdb.res_id, short_atomarray.res_id), "Residue IDs should match exactly"
    assert np.array_equal(short_pdb.res_name, short_atomarray.res_name), "Residue names should match exactly"
    assert np.array_equal(short_pdb.atom_name, short_atomarray.atom_name), "Atom names should match exactly"


def test_minifold_static_config_enforcement(test_sequences: dict[str, str]):
    """Test that static config keys cannot be overridden in options."""
    core = MiniFoldCore(config={"device": "cpu"})
    # device is a static config key
    with pytest.raises(ValueError, match="device"):
        core.fold(test_sequences["short"], options={"device": "cuda:0"})


def test_sequence_validation(test_sequences: dict[str, str]):
    """Test sequence validation in MiniFoldCore."""
    core = MiniFoldCore(config={"device": "cpu"})

    # Test single sequence
    single_seq = test_sequences["short"]
    validated = core._validate_sequences(single_seq)
    assert isinstance(validated, list), "Single sequence should be converted to list"
    assert len(validated) == 1, "Should contain one sequence"
    assert validated[0] == single_seq, "Sequence should be unchanged"

    # Test sequence list
    seq_list = [test_sequences["short"], test_sequences["medium"]]
    validated = core._validate_sequences(seq_list)
    assert isinstance(validated, list), "Should return a list"
    assert len(validated) == 2, "Should contain two sequences"
    assert validated == seq_list, "Sequences should be unchanged"

    # Test invalid sequence
    with pytest.raises(ValueError) as exc_info:
        core._validate_sequences(test_sequences["invalid"])
    assert "Invalid amino acid" in str(exc_info.value), f"Expected 'Invalid amino acid', got {str(exc_info.value)}"


def test_minifold_model_sizes(test_sequences: dict[str, str], gpu_device: Optional[str]):
    """Verify that both 48L and 12L model sizes can be used."""
    for model_size in ["48L", "12L"]:
        with enable_output():
            model = MiniFold(backend="modal", device=gpu_device, config={"model_size": model_size})
            result = model.fold(test_sequences["short"])

        assert isinstance(result, MiniFoldOutput), f"Result should be MiniFoldOutput for {model_size}"
        assert result.atom_array is not None, f"atom_array should be generated for {model_size}"
        assert len(result.atom_array) > 0, f"atom_array should contain structures for {model_size}"
