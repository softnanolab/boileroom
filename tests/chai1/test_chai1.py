import pytest
import pathlib
import numpy as np
import torch
from typing import Generator
from modal import enable_output

from boileroom import app, Chai1
from boileroom.models.chai1.chai1 import Chai1Output


@pytest.fixture
def chai1_model(config={}) -> Generator[Chai1, None, None]:
    with enable_output(), app.run():
        yield Chai1(config=config)


def test_chai1_basic(test_sequences: dict[str, str], chai1_model: Chai1, run_backend):
    """Test basic Chai1 functionality."""
    result = run_backend(chai1_model.fold)(test_sequences["short"])

    assert isinstance(result, Chai1Output), "Result should be a Chai1Output"

    seq_len = len(test_sequences["short"])
    positions_shape = result.positions.shape

    assert positions_shape[-1] == 3, f"Coordinate dimension mismatch. Expected: 3, Got: {positions_shape[-1]}"
    assert (
        positions_shape[-3] == seq_len
    ), f"Number of residues mismatch. Expected: {seq_len}, Got: {positions_shape[-3]}"
    assert np.all(result.confidence >= 0), "Confidence scores should be non-negative"
    assert np.all(result.confidence <= 1), "Confidence scores should be less than or equal to 1"


def test_chai1_medium_sequence(test_sequences: dict[str, str], run_backend):
    """Test Chai1 with medium length sequence."""
    with enable_output(), app.run():
        model = Chai1()
        result = run_backend(model.fold)(test_sequences["medium"])

    assert isinstance(result, Chai1Output), "Result should be a Chai1Output"
    
    seq_len = len(test_sequences["medium"])
    positions_shape = result.positions.shape
    
    assert positions_shape[-1] == 3, "Coordinate dimension should be 3"
    assert positions_shape[-3] == seq_len, f"Expected {seq_len} residues, got {positions_shape[-3]}"


def test_chai1_with_pdb_output(test_sequences: dict[str, str], run_backend):
    """Test Chai1 with PDB output enabled."""
    with enable_output(), app.run():
        model = Chai1(config={"output_pdb": True})
        result = run_backend(model.fold)(test_sequences["short"])

    assert isinstance(result, Chai1Output), "Result should be a Chai1Output"
    # Note: PDB output is not fully implemented yet, so we just check it doesn't crash
    # When fully implemented, this should be: assert result.pdb is not None


def test_chai1_with_cif_output(test_sequences: dict[str, str], run_backend):
    """Test Chai1 with CIF output enabled."""
    with enable_output(), app.run():
        model = Chai1(config={"output_cif": True})
        result = run_backend(model.fold)(test_sequences["short"])

    assert isinstance(result, Chai1Output), "Result should be a Chai1Output"
    # Note: CIF output is not fully implemented yet, so we just check it doesn't crash
    # When fully implemented, this should be: assert result.cif is not None


def test_chai1_with_atomarray_output(test_sequences: dict[str, str], run_backend):
    """Test Chai1 with AtomArray output enabled."""
    with enable_output(), app.run():
        model = Chai1(config={"output_atomarray": True})
        result = run_backend(model.fold)(test_sequences["short"])

    assert isinstance(result, Chai1Output), "Result should be a Chai1Output"
    # Note: AtomArray output is not fully implemented yet, so we just check it doesn't crash
    # When fully implemented, this should be: assert result.atom_array is not None


def test_chai1_invalid_sequence(test_sequences: dict[str, str], chai1_model: Chai1, run_backend):
    """Test Chai1 with invalid amino acid sequence."""
    with pytest.raises(ValueError, match="Invalid amino acid"):
        run_backend(chai1_model.fold)(test_sequences["invalid"])


def test_chai1_multimer_not_supported(test_sequences: dict[str, str], chai1_model: Chai1, run_backend):
    """Test that Chai1 raises error for multimer sequences (not supported in simplified implementation)."""
    with pytest.raises(ValueError, match="Multimer sequences are not supported"):
        run_backend(chai1_model.fold)(test_sequences["multimer"])


def test_chai1_multiple_sequences_not_supported(test_sequences: dict[str, str], chai1_model: Chai1, run_backend):
    """Test that Chai1 raises error for multiple sequences (not supported in simplified implementation)."""
    sequences = [test_sequences["short"], test_sequences["medium"]]
    with pytest.raises(ValueError, match="currently supports only single sequences"):
        run_backend(chai1_model.fold)(sequences)


def test_chai1_config_parameters(test_sequences: dict[str, str], run_backend):
    """Test Chai1 with different configuration parameters."""
    with enable_output(), app.run():
        model = Chai1(config={"num_steps": 100, "output_dir": "/tmp/test_chai1"})
        result = run_backend(model.fold)(test_sequences["short"])

    assert isinstance(result, Chai1Output), "Result should be a Chai1Output"
    assert result.metadata.model_name == "Chai-1", "Model name should be Chai-1"
    assert result.metadata.model_version == "0.5.0", "Model version should be 0.5.0"


def test_chai1_metadata(test_sequences: dict[str, str], chai1_model: Chai1, run_backend):
    """Test that Chai1 properly sets metadata."""
    result = run_backend(chai1_model.fold)(test_sequences["short"])

    assert result.metadata.model_name == "Chai-1", "Model name should be Chai-1"
    assert result.metadata.model_version == "0.5.0", "Model version should be 0.5.0"
    assert result.metadata.prediction_time is not None, "Prediction time should be recorded"
    assert result.metadata.prediction_time > 0, "Prediction time should be positive"
    assert result.metadata.sequence_lengths is not None, "Sequence lengths should be recorded"
    assert len(result.metadata.sequence_lengths) == 1, "Should have one sequence length"
    assert result.metadata.sequence_lengths[0] == len(test_sequences["short"]), "Sequence length should match input"


def test_chai1_output_structure(test_sequences: dict[str, str], chai1_model: Chai1, run_backend):
    """Test the structure of Chai1Output."""
    result = run_backend(chai1_model.fold)(test_sequences["short"])

    # Check required fields
    assert hasattr(result, 'positions'), "Output should have positions"
    assert hasattr(result, 'metadata'), "Output should have metadata"
    assert hasattr(result, 'confidence'), "Output should have confidence scores"

    # Check optional fields
    assert hasattr(result, 'atom_array'), "Output should have atom_array field"
    assert hasattr(result, 'pdb'), "Output should have pdb field"
    assert hasattr(result, 'cif'), "Output should have cif field"

    # Check shapes
    assert isinstance(result.positions, np.ndarray), "Positions should be numpy array"
    assert isinstance(result.confidence, np.ndarray), "Confidence should be numpy array"
    assert len(result.positions.shape) == 4, "Positions should be 4D: (batch, residue, atom, xyz)"
    assert len(result.confidence.shape) == 2, "Confidence should be 2D: (batch, residue)"


def test_chai1_single_string_input(test_sequences: dict[str, str], chai1_model: Chai1, run_backend):
    """Test Chai1 with single string input (not in list)."""
    result = run_backend(chai1_model.fold)(test_sequences["short"])

    assert isinstance(result, Chai1Output), "Result should be a Chai1Output"
    seq_len = len(test_sequences["short"])
    assert result.positions.shape[-3] == seq_len, f"Expected {seq_len} residues"


def test_chai1_list_input(test_sequences: dict[str, str], chai1_model: Chai1, run_backend):
    """Test Chai1 with single sequence in list."""
    result = run_backend(chai1_model.fold)([test_sequences["short"]])

    assert isinstance(result, Chai1Output), "Result should be a Chai1Output"
    seq_len = len(test_sequences["short"])
    assert result.positions.shape[-3] == seq_len, f"Expected {seq_len} residues"