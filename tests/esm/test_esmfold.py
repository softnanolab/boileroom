import pytest
import pathlib
import numpy as np
import torch
from typing import Generator
from modal import enable_output

from boileroom import ESMFold
from boileroom.models.esm.esmfold import ESMFoldCore, ESMFoldOutput
from boileroom.models.esm.linker import store_multimer_properties
from boileroom.convert import pdb_string_to_atomarray
from boileroom.constants import restype_3to1
from biotite.structure import AtomArray, rmsd
from io import StringIO
from biotite.structure.io.pdb import PDBFile


# Module scope keeps a single Modal container alive for the duration of the suite.
@pytest.fixture(scope="module")
def esmfold_model(config={}) -> Generator[ESMFold, None, None]:
    with enable_output():
        yield ESMFold(backend="modal", config=config)


def test_esmfold_basic(test_sequences: dict[str, str], esmfold_model: ESMFold):
    """Test basic ESMFold functionality."""
    result = esmfold_model.fold(test_sequences["short"])

    assert isinstance(result, ESMFoldOutput), "Result should be an ESMFoldOutput"

    seq_len = len(test_sequences["short"])
    positions_shape = result.positions.shape

    assert positions_shape[-1] == 3, "Coordinate dimension mismatch. Expected: 3, Got: {positions_shape[-1]}"
    assert (
        positions_shape[-3] == seq_len
    ), "Number of residues mismatch. Expected: {seq_len}, Got: {positions_shape[-3]}"
    assert np.all(result.plddt >= 0), "pLDDT scores should be non-negative"
    assert np.all(result.plddt <= 100), "pLDDT scores should be less than or equal to 100"


def test_esmfold_multimer(test_sequences):
    """Test ESMFold multimer functionality."""
    with enable_output():  # TODO: make this better with a fixture, re-using the logic
        model = ESMFold(config={"output_pdb": True})
        result = model.fold(test_sequences["multimer"])

    assert result.pdb is not None, "PDB output should be generated"
    assert result.positions.shape[2] == len(test_sequences["multimer"].replace(":", "")), "Number of residues mismatch"
    assert np.all(result.residue_index[0][:54] == np.arange(0, 54)), "First chain residue index mismatch"
    assert np.all(result.residue_index[0][54:] == np.arange(0, 54)), "Second chain residue index mismatch"
    assert np.all(result.chain_index[0][:54] == 0), "First chain index mismatch"
    assert np.all(result.chain_index[0][54:] == 1), "Second chain index mismatch"

    structure = pdb_string_to_atomarray(result.pdb[0])

    n_residues = len(set((chain, res) for chain, res in zip(structure.chain_id, structure.res_id, strict=True)))

    assert n_residues == len(test_sequences["multimer"].replace(":", "")), "Number of residues mismatch"
    assert len(result.chain_index[0]) == n_residues, "Chain index length mismatch"
    assert len(result.residue_index[0]) == n_residues, "Residue index length mismatch"

    # Check chain assignments
    unique_chains = np.unique(structure.chain_id)
    assert len(unique_chains) == 2, f"Expected 2 chains, got {len(unique_chains)}"

    # Check residues per chain
    chain_a_residues = len(np.unique(structure.res_id[structure.chain_id == "A"]))
    chain_b_residues = len(np.unique(structure.res_id[structure.chain_id == "B"]))
    assert chain_a_residues == 54, f"Chain A should have 54 residues, got {chain_a_residues}"
    assert chain_b_residues == 54, f"Chain B should have 54 residues, got {chain_b_residues}"

    # Assert correct folding outputs metrics (need to do it as we slice the linker out)
    assert result.predicted_aligned_error.shape == (1, n_residues, n_residues), "PAE matrix shape mismatch"
    assert result.plddt.shape == (1, n_residues, 37), "pLDDT matrix shape mismatch"
    assert result.ptm_logits.shape == (1, n_residues, n_residues, 64), "pTM matrix shape mismatch"
    assert result.aligned_confidence_probs.shape == (1, n_residues, n_residues, 64), "aligned confidence shape mismatch"
    assert result.s_z.shape == (1, n_residues, n_residues, 128), "s_z matrix shape mismatch"
    assert result.s_s.shape == (1, n_residues, 1024), "s_s matrix shape mismatch"
    assert result.distogram_logits.shape == (1, n_residues, n_residues, 64), "distogram logits matrix shape mismatch"
    assert result.lm_logits.shape == (1, n_residues, 23), "lm logits matrix shape mismatch"
    assert result.lddt_head.shape == (8, 1, n_residues, 37, 50), "lddt head matrix shape mismatch"
    assert result.plddt.shape == (1, n_residues, 37), "pLDDT matrix shape mismatch"


def test_esmfold_linker_map():
    """
    Test ESMFold linker map.
    The linker map has 1 for residues to keep (i.e. those not part of the linker),
    and 0 for residues to remove (i.e. those part of the linker).
    """
    sequences = ["AAAAAA:BBBBBBBBB", "CCCCC:DDDDDDD:EEEEEEE", "HHHH"]
    GLYCINE_LINKER = "G" * 50
    N = len(GLYCINE_LINKER)
    linker_map, _, _ = store_multimer_properties([sequences[0]], GLYCINE_LINKER)
    gt_map = torch.tensor([1] * 6 + [0] * N + [1] * 9)
    assert torch.all(linker_map == gt_map), "Linker map mismatch"

    linker_map, _, _ = store_multimer_properties([sequences[1]], GLYCINE_LINKER)
    gt_map = torch.tensor([1] * 5 + [0] * N + [1] * 7 + [0] * N + [1] * 7)
    assert torch.all(linker_map == gt_map), "Linker map mismatch"

    linker_map, _, _ = store_multimer_properties([sequences[2]], GLYCINE_LINKER)
    gt_map = torch.tensor([1] * 4)
    assert torch.all(linker_map == gt_map), "Linker map mismatch"


def test_esmfold_no_glycine_linker(test_sequences):
    """Test ESMFold no glycine linker."""
    model = ESMFold(
        config={
            "glycine_linker": "",
        }
    )

    with enable_output():
        result = model.fold(test_sequences["multimer"])

    assert result.positions is not None, "Positions should be generated"
    assert result.positions.shape[2] == len(test_sequences["multimer"].replace(":", "")), "Number of residues mismatch"

    assert result.residue_index is not None, "Residue index should be generated"
    assert result.plddt is not None, "pLDDT should be generated"
    assert result.ptm is not None, "pTM should be generated"

    # assert correct chain_indices
    assert np.all(result.chain_index[0] == np.array([0] * 54 + [1] * 54)), "Chain indices mismatch"
    assert np.all(
        result.residue_index[0] == np.concatenate([np.arange(0, 54), np.arange(0, 54)])
    ), "Residue index mismatch"


def test_esmfold_chain_indices():
    """
    Test ESMFold chain indices. Note that this is before we slice the linker out, that
    is why we need to check the presence of the linker indices here as well. And by construction,
    it is assigned to the first chain, i.e. 0.
    """
    sequences = ["AAAAAA:CCCCCCCCC", "CCCCC:DDDDDDD:EEEEEEE", "HHHH"]
    GLYCINE_LINKER = "G" * 50
    N = len(GLYCINE_LINKER)

    _, _, chain_indices = store_multimer_properties([sequences[0]], GLYCINE_LINKER)

    expected_chain_indices = np.concatenate(
        [
            np.zeros(6),  # First chain (6 residues)
            np.zeros(N),  # Linker region (N residues) - belongs to first chain
            np.ones(9),  # Second chain (9 residues)
        ]
    )
    assert np.array_equal(chain_indices[0], expected_chain_indices), "Chain indices mismatch"


def test_esmfold_batch(esmfold_model: ESMFold, test_sequences: dict[str, str]):
    """Test ESMFold batch prediction."""

    # Define input sequences
    sequences = [test_sequences["short"], test_sequences["medium"]]

    # Make prediction
    result = esmfold_model.fold(sequences)

    max_seq_length = max(len(seq) for seq in sequences)
    assert (
        result.positions.shape == (8, len(sequences), max_seq_length, 14, 3)
    ), f"Position shape mismatch. Expected: (8, {len(sequences)}, {max_seq_length}, 14, 3), Got: {result.positions.shape}"

    # Check that batch outputs have correct sequence lengths
    assert result.aatype.shape[0] == len(sequences), "Batch size mismatch in aatype"
    assert result.plddt.shape[0] == len(sequences), "Batch size mismatch in plddt"
    assert result.ptm_logits.shape[0] == len(sequences), "Batch size mismatch in ptm_logits"
    assert result.predicted_aligned_error.shape[0] == len(sequences), "Batch size mismatch in predicted_aligned_error"


def test_tokenize_sequences_with_mocker(mocker):
    """Test tokenization of multimer sequences using pytest-mock."""
    from boileroom.models.esm.esmfold import ESMFoldCore

    # Test data
    sequences = ["AAAAAA:CCCCCCCCC", "CCCCC:DDDDDDD:EEEEEEE", "HHHH"]
    GLYCINE_LINKER = ""
    POSITION_IDS_SKIP = 512

    # Create a model instance
    model = ESMFoldCore(config={"glycine_linker": GLYCINE_LINKER, "position_ids_skip": POSITION_IDS_SKIP})
    model.device = "cpu"

    # Mock the tokenizer
    mock_tokenizer = mocker.patch.object(model, 'tokenizer')
    mock_tokenizer.return_value = {
        "input_ids": torch.tensor([
            [1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, -1, -1, -1, -1],
            [3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5],
            [8, 8, 8, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
            ]),
        "attention_mask": torch.ones(3, 19),
        }

    # Call the method to test
    tokenized_input, multimer_properties = model._tokenize_sequences(sequences)

    # Assert the tokenizer was called with the expected arguments
    expected_sequences = [seq.replace(":", GLYCINE_LINKER) for seq in sequences]
    mock_tokenizer.assert_called_once_with(
        expected_sequences,
        padding=True,
        truncation=True,
        return_tensors="pt",
        add_special_tokens=False
    )

    # Verify the output contains the expected keys
    assert set(tokenized_input.keys()) >= {"input_ids", "attention_mask", "position_ids"}
    assert multimer_properties is not None


def test_sequence_validation(test_sequences: dict[str, str]):
    """Test sequence validation in FoldingAlgorithm."""

    esmfold_core = ESMFoldCore(config={})

    # Test single sequence
    single_seq = test_sequences["short"]
    validated = esmfold_core._validate_sequences(single_seq)
    assert isinstance(validated, list), "Single sequence should be converted to list"
    assert len(validated) == 1, "Should contain one sequence"
    assert validated[0] == single_seq, "Sequence should be unchanged"

    # Test sequence list
    seq_list = [test_sequences["short"], test_sequences["medium"]]
    validated = esmfold_core._validate_sequences(seq_list)
    assert isinstance(validated, list), "Should return a list"
    assert len(validated) == 2, "Should contain two sequences"
    assert validated == seq_list, "Sequences should be unchanged"

    # Test invalid sequence
    with pytest.raises(ValueError) as exc_info:
        esmfold_core._validate_sequences(test_sequences["invalid"])
    assert "Invalid amino acid" in str(exc_info.value), f"Expected 'Invalid amino acid', got {str(exc_info.value)}"

    # Test that fold method uses validation
    with pytest.raises(ValueError) as exc_info:
        esmfold_core.fold(test_sequences["invalid"])
    assert "Invalid amino acid" in str(exc_info.value), f"Expected 'Invalid amino acid', got {str(exc_info.value)}"


def test_esmfold_output_pdb_cif(data_dir: pathlib.Path, test_sequences: dict[str, str]):
    """Test ESMFold output PDB and CIF."""

    def recover_sequence(atomarray: AtomArray) -> str:
        unique_res_ids = np.unique(atomarray.res_id)
        three_letter_codes = [atomarray.res_name[atomarray.res_id == res_id][0] for res_id in unique_res_ids]
        one_letter_codes = [restype_3to1[three_letter_code] for three_letter_code in three_letter_codes]
        return "".join(one_letter_codes)

    with enable_output():
        model = ESMFold(config={"output_pdb": True, "output_cif": False, "output_atomarray": True})
        # Define input sequences
        sequences = [test_sequences["short"], test_sequences["medium"]]
        result = model.fold(sequences)

    assert result.pdb is not None, "PDB output should be generated"
    assert result.cif is None, "CIF output should be None"
    assert len(result.pdb) == len(result.atom_array) == len(sequences) == 2, "Batching output match!"
    assert isinstance(result.pdb, list), "PDB output should be a list"
    assert len(result.pdb) == len(sequences), "PDB output should have same length as input sequences"
    assert isinstance(result.atom_array, list), "Atom array should be a list"
    assert isinstance(result.atom_array[0], AtomArray), "Atom array should be an AtomArray"

    short_pdb = PDBFile.read(StringIO(result.pdb[0])).get_structure(model=1)
    medium_pdb = PDBFile.read(StringIO(result.pdb[1])).get_structure(model=1)
    short_atomarray = result.atom_array[0]
    medium_atomarray = result.atom_array[1]

    # Short protein checks
    num_residues = len(sequences[0])
    assert np.all(
        np.unique(short_atomarray.res_id) == np.arange(0, num_residues)
    ), "AtomArray residues should be 0-indexed"
    recovered_seq = recover_sequence(short_atomarray)
    assert recovered_seq == sequences[0], "Recovered sequence should be equal to the input sequence"
    assert np.all(np.unique(short_pdb.res_id) == np.arange(0, num_residues)), "Residues should be 0-indexed"
    # Compare coordinates with tolerance
    assert np.allclose(
        short_pdb.coord, short_atomarray.coord, atol=0.1
    ), "Atom coordinates should be equal within 0.1Å tolerance"
    # Compare other attributes exactly
    assert np.array_equal(short_pdb.chain_id, short_atomarray.chain_id), "Chain IDs should match exactly"
    assert np.array_equal(short_pdb.res_id, short_atomarray.res_id), "Residue IDs should match exactly"
    assert np.array_equal(short_pdb.res_name, short_atomarray.res_name), "Residue names should match exactly"
    assert np.array_equal(short_pdb.atom_name, short_atomarray.atom_name), "Atom names should match exactly"

    # Medium protein checks
    num_residues = len(sequences[1])
    assert np.all(
        np.unique(medium_atomarray.res_id) == np.arange(0, num_residues)
    ), "AtomArray residues should be 0-indexed"
    recovered_seq = recover_sequence(medium_atomarray)
    assert recovered_seq == sequences[1], "Recovered sequence should be equal to the input sequence"
    assert np.all(np.unique(medium_pdb.res_id) == np.arange(0, num_residues)), "Residues should be 0-indexed"

    # Compare coordinates with tolerance
    assert np.allclose(
        medium_pdb.coord, medium_atomarray.coord, atol=0.1
    ), "Atom coordinates should be equal within 0.1Å tolerance"
    # Compare other attributes exactly
    assert np.array_equal(medium_pdb.chain_id, medium_atomarray.chain_id), "Chain IDs should match exactly"
    assert np.array_equal(medium_pdb.res_id, medium_atomarray.res_id), "Residue IDs should match exactly"
    assert np.array_equal(medium_pdb.res_name, medium_atomarray.res_name), "Residue names should match exactly"
    assert np.array_equal(medium_pdb.atom_name, medium_atomarray.atom_name), "Atom names should match exactly"

    short_pdbfile = PDBFile().read(data_dir / "esmfold_server_short.pdb")
    saved_short_pdb = short_pdbfile.get_structure(model=1)
    saved_short_bfactor = short_pdbfile.get_b_factor()
    rmsd_value = rmsd(short_pdb, saved_short_pdb)
    assert (
        rmsd_value < 1.5
    ), "PDB file should be almost equal to the saved ESMFold Server PDB file. Difference comes from HF vs. Meta implementation differences."

    medium_pdbfile = PDBFile().read(data_dir / "esmfold_server_medium.pdb")
    saved_medium_pdb = medium_pdbfile.get_structure(model=1)
    saved_medium_bfactor = medium_pdbfile.get_b_factor()
    rmsd_value = rmsd(medium_pdb, saved_medium_pdb)
    assert (
        rmsd_value < 1.5
    ), "PDB file should be almost equal to the saved ESMFold Server PDB file. Difference comes from HF vs. Meta implementation differences."

    # compare b-factor
    short_bfactor = short_atomarray.get_annotation("b_factor")
    medium_bfactor = medium_atomarray.get_annotation("b_factor")
    assert np.allclose(
        short_bfactor, saved_short_bfactor, atol=0.05
    ), "B-factor should match within a tolerance (HF vs. Meta)"
    assert np.allclose(
        medium_bfactor, saved_medium_bfactor, atol=0.05
    ), "B-factor should match within a tolerance (HF vs. Meta)"
