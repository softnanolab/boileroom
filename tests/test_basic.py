"""Test suite for BoilerRoom package."""

from typing import Generator
import pathlib
import numpy as np
import pytest
import torch
from io import StringIO
from biotite.structure.io.pdb import PDBFile
from biotite.structure.io.pdbx import CIFFile
from biotite.structure import AtomArray, rmsd
from modal import enable_output

from boileroom import app
from boileroom.esmfold import ESMFold, ESMFoldOutput
from boileroom.utils import validate_sequence, format_time
from boileroom.esm2 import get_esm2
from boileroom.linker import compute_position_ids, store_multimer_properties
from boileroom.convert import pdb_string_to_atomarray
from boileroom.constants import restype_3to1

# Test sequences
TEST_SEQUENCES = {
    "short": "MLKNVHVLVLGAGDVGSVVVRLLEK",  # 25 residues
    "medium": "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKT",  # Insulin
    "invalid": "MALWMRLLPX123LLALWGPD",  # Contains invalid characters
    "multimer": "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKT:MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKT",  # 2x Insulin
}


@pytest.fixture
def esmfold_model(config={}) -> Generator[ESMFold, None, None]:
    """Fixture for ESMFold model.

    Args:
        config: Optional configuration dictionary for ESMFold
    """
    with enable_output():
        with app.run():
            model = ESMFold(config=config)
            yield model


def test_validate_sequence():
    """Test sequence validation."""
    # Valid sequences
    assert validate_sequence(TEST_SEQUENCES["short"]) is True
    assert validate_sequence(TEST_SEQUENCES["medium"]) is True

    # Invalid sequences
    with pytest.raises(ValueError):
        validate_sequence(TEST_SEQUENCES["invalid"])
    with pytest.raises(ValueError):
        validate_sequence("NOT A SEQUENCE")


def test_format_time():
    """Test time formatting."""
    assert format_time(30) == "30s", f"Expected '30s', got {format_time(30)}"
    assert format_time(90) == "1m 30s", f"Expected '1m 30s', got {format_time(90)}"
    assert format_time(3600) == "1h", f"Expected '1h', got {format_time(3600)}"
    assert format_time(3661) == "1h 1m 1s", f"Expected '1h 1m 1s', got {format_time(3661)}"


def test_esmfold_basic():
    """Test basic ESMFold functionality."""
    with enable_output():
        with app.run():
            model = ESMFold()
            result = model.fold.remote(TEST_SEQUENCES["short"])

            assert isinstance(result, ESMFoldOutput), "Result should be an ESMFoldOutput"

            seq_len = len(TEST_SEQUENCES["short"])
            positions_shape = result.positions.shape

            assert positions_shape[-1] == 3, "Coordinate dimension mismatch. Expected: 3, Got: {positions_shape[-1]}"
            assert (
                positions_shape[-3] == seq_len
            ), "Number of residues mismatch. Expected: {seq_len}, Got: {positions_shape[-3]}"
            assert np.all(result.plddt >= 0), "pLDDT scores should be non-negative"
            assert np.all(result.plddt <= 100), "pLDDT scores should be less than or equal to 100"


def test_esmfold_multimer():
    """Test ESMFold multimer functionality."""
    with enable_output():
        with app.run():
            model = ESMFold(config={"output_pdb": True})
            result = model.fold.remote(TEST_SEQUENCES["multimer"])

    assert result.pdb is not None, "PDB output should be generated"
    assert result.positions.shape[2] == len(TEST_SEQUENCES["multimer"].replace(":", "")), "Number of residues mismatch"
    assert np.all(result.residue_index[0][:54] == np.arange(0, 54)), "First chain residue index mismatch"
    assert np.all(result.residue_index[0][54:] == np.arange(0, 54)), "Second chain residue index mismatch"
    assert np.all(result.chain_index[0][:54] == 0), "First chain index mismatch"
    assert np.all(result.chain_index[0][54:] == 1), "Second chain index mismatch"

    structure = pdb_string_to_atomarray(result.pdb[0])

    n_residues = len(set((chain, res) for chain, res in zip(structure.chain_id, structure.res_id)))

    assert n_residues == len(TEST_SEQUENCES["multimer"].replace(":", "")), "Number of residues mismatch"
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


def test_esmfold_no_glycine_linker():
    """Test ESMFold no glycine linker."""
    model = ESMFold(
        config={
            "glycine_linker": "",
        }
    )

    with enable_output():
        with app.run():
            result = model.fold.remote(TEST_SEQUENCES["multimer"])

    assert result.positions is not None, "Positions should be generated"
    assert result.positions.shape[2] == len(TEST_SEQUENCES["multimer"].replace(":", "")), "Number of residues mismatch"

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


def test_compute_position_ids_batch():
    """Test position IDs computation with batch processing."""
    sequences = ["AAAAAA:CCCCCCCCC", "CCCCC:DDDDDDD:EEEEEEE", "HHHH"]

    GLYCINE_LINKER = "G" * 50
    POSITION_IDS_SKIP = 512

    position_ids = compute_position_ids([sequences[0]], GLYCINE_LINKER, POSITION_IDS_SKIP)

    # Check first sequence (AAAAAA:CCCCCCCCC)
    seq1_ids = position_ids[0]
    assert position_ids.shape[0] == 1, "Batch size should be 1"
    assert torch.all(seq1_ids[: 6 + 50] == torch.arange(6 + 50)), "First chain + linker positions incorrect"
    assert torch.all(
        seq1_ids[6 + 50 :] == torch.arange(1 + 512 + 55, 1 + 512 + 55 + 9)
    ), "Second chain positions incorrect"

    # Check second sequence (CCCCC:DDDDDDD:EEEEEEE)
    position_ids = compute_position_ids([sequences[1]], GLYCINE_LINKER, POSITION_IDS_SKIP)
    seq2_ids = position_ids[0]

    assert torch.all(seq2_ids[:55] == torch.arange(55)), "First chain + linker positions incorrect"
    assert torch.all(seq2_ids[55:112] == torch.arange(567, 624)), "Second chain + linker positions incorrect"
    assert torch.all(seq2_ids[112:] == torch.arange(1136, 1143)), "Third chain positions incorrect"

    # Check third sequence (HHHH), do entire batch, but only check the third sequence
    position_ids = compute_position_ids(sequences, GLYCINE_LINKER, POSITION_IDS_SKIP)
    assert position_ids.shape[0] == 3, "Batch size should be 3"

    seq3_ids = position_ids[2]

    assert torch.all(seq3_ids[:4] == torch.arange(4)), "First chain positions incorrect"
    # Check its padding is all zeros
    assert torch.all(seq3_ids[4:] == torch.zeros(len(seq3_ids) - 4)), "Padding should be all zeros"

    # length of it should be 19 + 100, i.e. the longest sequence + 2 glycine linkers, which is the second sequence
    assert len(seq3_ids) == 19 + 100, "Length of sequence 3 should be 19 + 100"

    # Test with empty glycine linker
    EMPTY_LINKER = ""
    position_ids = compute_position_ids(sequences, EMPTY_LINKER, POSITION_IDS_SKIP)
    assert position_ids.shape[0] == 3, "Batch size should be 3"

    # Check first sequence (AAAAAA:CCCCCCCCC)
    seq1_ids = position_ids[0]
    assert torch.all(seq1_ids[:6] == torch.arange(6)), "First chain positions incorrect"
    assert torch.all(seq1_ids[6:15] == torch.arange(512 + 6, 512 + 6 + 9)), "Second chain positions incorrect"

    # Check second sequence (CCCCC:DDDDDDD:EEEEEEE)
    seq2_ids = position_ids[1]
    assert torch.all(seq2_ids[:5] == torch.arange(5)), "First chain positions incorrect"
    assert torch.all(seq2_ids[5:12] == torch.arange(512 + 5, 512 + 5 + 7)), "Second chain positions incorrect"
    assert torch.all(seq2_ids[12:] == torch.arange(2 * 512 + 12, 2 * 512 + 12 + 7)), "Third chain positions incorrect"

    # Check third sequence (HHHH)
    seq3_ids = position_ids[2]
    assert torch.all(seq3_ids[:4] == torch.arange(4)), "First chain positions incorrect"
    # Check its padding is all zeros
    assert torch.all(seq3_ids[4] == torch.zeros(len(seq3_ids) - 4)), "Padding should be all zeros"

    # length should be 19, which is the length of the longest sequence (second sequence)
    assert len(seq3_ids) == 19, "Length of sequence 3 should be 19"


# TODO: This is not obvious to do, given the way we wrap things around in Modal
# This shows well how fragile relying on Modal is going to be moving forward, and we should think
# of ways to make it more managable through local execution as well

# def test_tokenize_sequences_with_mocker(mocker):
#     """Test tokenization of multimer sequences using pytest-mock."""
#     from boileroom.esmfold import ESMFold

#     # Test data
#     sequences = ["AAAAAA:CCCCCCCCC", "CCCCC:DDDDDDD:EEEEEEE", "HHHH"]
#     GLYCINE_LINKER = ""
#     POSITION_IDS_SKIP = 512

#     # Create a model instance
#     model = ESMFold(config={"glycine_linker": GLYCINE_LINKER, "position_ids_skip": POSITION_IDS_SKIP})

#     # Mock the tokenizer
#     mock_tokenizer = mocker.patch.object(model, 'tokenizer')
#     mock_tokenizer.return_value = {
#         "input_ids": torch.tensor([
#             [1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, -1, -1, -1, -1],
#             [3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5],
#             [8, 8, 8, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
#             ]),
#         "attention_mask": torch.ones(3, 19),
#         }

#     # Call the method to test
#     tokenized_input = model._tokenize_sequences(sequences)

#     # Assert the tokenizer was called with the expected arguments
#     expected_sequences = [seq.replace(":", GLYCINE_LINKER) for seq in sequences]
#     mock_tokenizer.assert_called_once_with(
#         expected_sequences,
#         padding=True,
#         truncation=True,
#         return_tensors="pt",
#         add_special_tokens=False
#     )

#     # Verify the output contains the expected keys
#     assert set(tokenized_input.keys()) >= {"input_ids", "attention_mask", "position_ids"}


def test_esmfold_batch(esmfold_model: ESMFold):
    """Test ESMFold batch prediction."""

    # Define input sequences
    sequences = [TEST_SEQUENCES["short"], TEST_SEQUENCES["medium"]]

    # Make prediction
    result = esmfold_model.fold.remote(sequences)

    max_seq_length = max(len(seq) for seq in sequences)
    assert (
        result.positions.shape == (8, len(sequences), max_seq_length, 14, 3)
    ), f"Position shape mismatch. Expected: (8, {len(sequences)}, {max_seq_length}, 14, 3), Got: {result.positions.shape}"

    # Check that batch outputs have correct sequence lengths
    assert result.aatype.shape[0] == len(sequences), "Batch size mismatch in aatype"
    assert result.plddt.shape[0] == len(sequences), "Batch size mismatch in plddt"
    assert result.ptm_logits.shape[0] == len(sequences), "Batch size mismatch in ptm_logits"
    assert result.predicted_aligned_error.shape[0] == len(sequences), "Batch size mismatch in predicted_aligned_error"


def test_sequence_validation(esmfold_model: ESMFold):
    """Test sequence validation in FoldingAlgorithm."""

    # Test single sequence
    single_seq = TEST_SEQUENCES["short"]
    validated = esmfold_model._validate_sequences(single_seq)
    assert isinstance(validated, list), "Single sequence should be converted to list"
    assert len(validated) == 1, "Should contain one sequence"
    assert validated[0] == single_seq, "Sequence should be unchanged"

    # Test sequence list
    seq_list = [TEST_SEQUENCES["short"], TEST_SEQUENCES["medium"]]
    validated = esmfold_model._validate_sequences(seq_list)
    assert isinstance(validated, list), "Should return a list"
    assert len(validated) == 2, "Should contain two sequences"
    assert validated == seq_list, "Sequences should be unchanged"

    # Test invalid sequence
    with pytest.raises(ValueError) as exc_info:
        esmfold_model._validate_sequences(TEST_SEQUENCES["invalid"])
    assert "Invalid amino acid" in str(exc_info.value), f"Expected 'Invalid amino acid', got {str(exc_info.value)}"

    # Test that fold method uses validation
    with pytest.raises(ValueError) as exc_info:
        esmfold_model.fold.remote(TEST_SEQUENCES["invalid"])
    assert "Invalid amino acid" in str(exc_info.value), f"Expected 'Invalid amino acid', got {str(exc_info.value)}"


def test_esmfold_output_pdb_cif():
    """Test ESMFold output PDB and CIF."""

    def recover_sequence(atomarray: AtomArray) -> str:
        unique_res_ids = np.unique(atomarray.res_id)
        three_letter_codes = [atomarray.res_name[atomarray.res_id == res_id][0] for res_id in unique_res_ids]
        one_letter_codes = [restype_3to1[three_letter_code] for three_letter_code in three_letter_codes]
        return "".join(one_letter_codes)

    with enable_output():
        with app.run():
            model = ESMFold(config={"output_pdb": True, "output_cif": False, "output_atomarray": True})
            # Define input sequences
            sequences = [TEST_SEQUENCES["short"], TEST_SEQUENCES["medium"]]
            result = model.fold.remote(sequences)

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
    ), "AtomArray residues should be 1-indexed"
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
    ), "AtomArray residues should be 1-indexed"
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

    # also load the PDB file and compare
    current_file = pathlib.Path(__file__).parent

    short_pdbfile = PDBFile().read(current_file / "data/esmfold_server_short.pdb")
    saved_short_pdb = short_pdbfile.get_structure(model=1)
    saved_short_bfactor = short_pdbfile.get_b_factor()
    rmsd_value = rmsd(short_pdb, saved_short_pdb)
    assert (
        rmsd_value < 1.5
    ), "PDB file should be almost equal to the saved ESMFold Server PDB file. Difference comes from HF vs. Meta implementation differences."

    medium_pdbfile = PDBFile().read(current_file / "data/esmfold_server_medium.pdb")
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


@pytest.fixture(params=[10, 25, 50])
def glycine_linker(request) -> str:
    return "G" * request.param


def test_esmfold_batch_multimer_linkers(esmfold_model, glycine_linker):
    """Test ESMFold batch prediction with different glycine linker lengths.
    Tests various edge cases:
    - Different sized multimers in the same batch
    - Mixing monomers and multimers
    - Very short and very long sequences
    - Sequences with different numbers of chains
    """
    model = esmfold_model
    model.config["glycine_linker"] = glycine_linker

    # Test sequences with different properties
    sequences = [
        "AAA:CCC",  # Very short 2-chain multimer
        TEST_SEQUENCES["short"],  # Monomer (25 residues)
        "A" * 50 + ":" + "C" * 100 + ":" + "D" * 75,  # Long 3-chain multimer with different chain lengths
        "M" * 10 + ":" + "K" * 10,  # Small symmetric 2-chain multimer
        "M" * 1 + ":" + "Y" * 1,  # Edge case: minimal 2-chain multimer (1 residue each)
    ]

    result = model.fold.remote(sequences)

    # Basic output checks
    assert result.positions is not None, "Positions should be generated"
    assert result.residue_index is not None, "Residue index should be generated"
    assert result.plddt is not None, "pLDDT should be generated"
    assert result.ptm is not None, "pTM should be generated"
    assert result.chain_index is not None, "Chain index should be generated"

    # Check batch size
    max_seq_length = max(len(seq.replace(":", "")) for seq in sequences)
    assert result.positions.shape == (8, len(sequences), max_seq_length, 14, 3), "Positions shape mismatch"
    assert result.frames.shape == (8, len(sequences), max_seq_length, 7), "Frames shape mismatch"
    assert result.sidechain_frames.shape == (
        8,
        len(sequences),
        max_seq_length,
        8,
        4,
        4,
    ), "Sidechain frames shape mismatch"
    assert result.unnormalized_angles.shape == (
        8,
        len(sequences),
        max_seq_length,
        7,
        2,
    ), "Unnormalized angles shape mismatch"
    assert result.angles.shape == (8, len(sequences), max_seq_length, 7, 2), "Angles shape mismatch"
    assert result.states.shape == (8, len(sequences), max_seq_length, 384), "States shape mismatch"
    assert result.lddt_head.shape == (8, len(sequences), max_seq_length, 37, 50), "LDDT head shape mismatch"

    assert result.s_s.shape == (len(sequences), max_seq_length, 1024), "s_s shape mismatch"
    assert result.lm_logits.shape == (len(sequences), max_seq_length, 23), "lm_logits shape mismatch"
    assert result.aatype.shape == (len(sequences), max_seq_length), "aatype shape mismatch"
    assert result.residx_atom14_to_atom37.shape == (
        len(sequences),
        max_seq_length,
        14,
    ), "residx_atom14_to_atom37 shape mismatch"
    assert result.residx_atom37_to_atom14.shape == (
        len(sequences),
        max_seq_length,
        37,
    ), "residx_atom37_to_atom14 shape mismatch"
    assert result.atom14_atom_exists.shape == (len(sequences), max_seq_length, 14), "atom14_atom_exists shape mismatch"
    assert result.plddt.shape == (len(sequences), max_seq_length, 37), "plddt shape mismatch"

    assert result.s_z.shape == (len(sequences), max_seq_length, max_seq_length, 128), "s_z shape mismatch"
    assert result.distogram_logits.shape == (
        len(sequences),
        max_seq_length,
        max_seq_length,
        64,
    ), "distogram_logits shape mismatch"
    assert result.ptm_logits.shape == (len(sequences), max_seq_length, max_seq_length, 64), "ptm_logits shape mismatch"
    assert result.aligned_confidence_probs.shape == (
        len(sequences),
        max_seq_length,
        max_seq_length,
        64,
    ), "aligned_confidence_probs shape mismatch"
    assert result.predicted_aligned_error.shape == (
        len(sequences),
        max_seq_length,
        max_seq_length,
    ), "predicted_aligned_error shape mismatch"

    assert result.chain_index.shape == (len(sequences), max_seq_length), "chain_index shape mismatch"
    assert result.residue_index.shape == (len(sequences), max_seq_length), "residue_index shape mismatch"

    # Test sequences[0] = "AAA:CCC"
    assert np.all(result.chain_index[0][:3] == 0), "Chain index should be 0 for the first chain"
    assert np.all(result.chain_index[0][3:6] == 1), "Chain index should be 1 for the second chain"
    assert np.all(result.chain_index[0][6:] == -1), "Padding should be -1"
    assert np.all(result.residue_index[0][:3] == np.arange(0, 3)), "Residue index should be 0-2 for the first chain"
    assert np.all(result.residue_index[0][3:6] == np.arange(0, 3)), "Residue index should be 0-2 for the second chain"
    assert np.all(result.residue_index[0][6:] == -1), "Padding should be -1"
    assert np.all(result.positions[:, 0, 6:, :, :] == -1), "Padding should be -1"
    assert np.all(result.positions[:, 0, :6, :, :] != -1), "No positions should look like padding (-1)"
    assert np.all(result.aatype[0, :6] != -1), "No aatype should look like padding (-1)"
    assert np.all(result.aatype[0, 6:] == -1), "Padding should be -1"

    # Test sequences[1] = TEST_SEQUENCES["short"]
    assert np.all(result.chain_index[1][:25] == 0), "Chain index should be 0 for the first chain"
    assert np.all(result.chain_index[1][25:] == -1), "Padding should be -1"
    assert np.all(result.residue_index[1][:25] == np.arange(0, 25)), "Residue index should be 0-24 for the first chain"
    assert np.all(result.residue_index[1][25:] == -1), "Padding should be -1"
    assert np.all(result.positions[:, 1, 25:, :, :] == -1), "Padding should be -1"
    assert np.all(result.positions[:, 1, :25, :, :] != -1), "No positions should look like padding (-1)"
    assert np.all(result.aatype[1, :25] != -1), "No aatype should look like padding (-1)"
    assert np.all(result.aatype[1, 25:] == -1), "Padding should be -1"

    # Test sequences[2] = "A" * 50 + ":" + "C" * 100 + ":" + "D" * 75
    assert np.all(result.chain_index[2][:50] == 0), "Chain index should be 0 for the first chain"
    assert np.all(result.chain_index[2][50:150] == 1), "Chain index should be 1 for the second chain"
    assert np.all(result.chain_index[2][150:] == 2), "Chain index should be 2 for the third chain"
    assert np.all(result.residue_index[2][:50] == np.arange(0, 50)), "Residue index should be 0-49 for the first chain"
    assert np.all(
        result.residue_index[2][50:150] == np.arange(0, 100)
    ), "Residue index should be 0-99 for the second chain"
    assert np.all(result.residue_index[2][150:] == np.arange(0, 75)), "Residue index should be 0-74 for the third chain"
    assert np.all(result.positions[:, 2, :50, :, :] != -1), "No positions should look like padding (-1)"
    assert np.all(result.aatype[2, :50] != -1), "No aatype should look like padding (-1)"

    # Test sequences[3] = "M" * 10 + ":" + "K" * 10
    assert np.all(result.chain_index[3][:10] == 0), "Chain index should be 0 for the first chain"
    assert np.all(result.chain_index[3][10:20] == 1), "Chain index should be 1 for the second chain"
    assert np.all(result.chain_index[3][20:] == -1), "Padding should be -1"
    assert np.all(result.residue_index[3][:10] == np.arange(0, 10)), "Residue index should be 0-9 for the first chain"
    assert np.all(
        result.residue_index[3][10:20] == np.arange(0, 10)
    ), "Residue index should be 0-9 for the second chain"
    assert np.all(result.residue_index[3][20:] == -1), "Padding should be -1"
    assert np.all(result.positions[:, 3, 20:, :, :] == -1), "Padding should be -1"
    assert np.all(result.positions[:, 3, :20, :, :] != -1), "No positions should look like padding (-1)"
    assert np.all(result.aatype[3, 20:] == -1), "Padding should be -1"
    assert np.all(result.aatype[3, :20] != -1), "No aatype should look like padding (-1)"

    # Test sequences[4] = "M" * 1 + ":" + "Y" * 1
    assert np.all(result.chain_index[4][0] == 0), "Chain index should be 0 for the first chain"
    assert np.all(result.chain_index[4][1] == 1), "Chain index should be 1 for the second chain"
    assert np.all(result.chain_index[4][2:] == -1), "Padding should be -1"
    assert np.all(result.residue_index[4][0] == 0), "Residue index should be 0 for the first chain"
    assert np.all(result.residue_index[4][1] == 0), "Residue index should be 0 for the second chain"
    assert np.all(result.residue_index[4][2:] == -1), "Padding should be -1"
    assert np.all(result.positions[:, 4, 2:, :, :] == -1), "Padding should be -1"
    assert np.all(result.positions[:, 4, :2, :, :] != -1), "No positions should look like padding (-1)"
    assert np.all(result.aatype[4, 2:] == -1), "Padding should be -1"
    assert np.all(result.aatype[4, :2] != -1), "No aatype should look like padding (-1)"


def test_esmfold_multimer_reference():
    """Test ESMFold multimer reference."""
    from boileroom.esmfold import get_esmfold
    from biotite.structure.io.pdbx import get_structure

    with app.run():
        config = {
            "output_pdb": True,
            "output_cif": True,
            "output_atomarray": True,
            "position_ids_skip": 512,
            "glycine_linker": 50 * "G",
        }
        model = get_esmfold(gpu_type="A100-40GB", config=config)
        sequence = open(pathlib.Path(__file__).parent / "data/multimer-check.txt").read().strip()
        result = model.fold.remote([sequence])

        reference_pdb = PDBFile().read(pathlib.Path(__file__).parent / "data/multimer-check-A100.pdb")
        reference_structure = reference_pdb.get_structure(model=1)

        # Check that all outputs exist
        assert result.pdb is not None, "PDB output should be generated"
        assert result.cif is not None, "CIF output should be generated"
        assert result.atom_array is not None, "Atom array output should be generated"

        # Get the first entry from each output
        pdb_structure = PDBFile.read(StringIO(result.pdb[0])).get_structure(model=1)

        cif_structure = get_structure(CIFFile.read(StringIO(result.cif[0])))
        atom_array = result.atom_array[0]

        # Compare coordinates with tolerance
        assert np.allclose(
            pdb_structure.coord, cif_structure.coord, atol=0.1
        ), "PDB and CIF coordinates should match within 0.1Å"
        assert np.allclose(
            pdb_structure.coord, atom_array.coord, atol=0.1
        ), "PDB and atom array coordinates should match within 0.1Å"
        assert np.allclose(
            pdb_structure.coord, reference_structure.coord, atol=0.1
        ), "PDB and reference coordinates should match within 0.1Å"

        # Compare other attributes exactly
        assert np.array_equal(pdb_structure.chain_id, cif_structure.chain_id), "PDB and CIF chain IDs should match"
        assert np.array_equal(pdb_structure.chain_id, atom_array.chain_id), "PDB and atom array chain IDs should match"
        assert np.array_equal(
            pdb_structure.chain_id, reference_structure.chain_id
        ), "PDB and reference chain IDs should match"

        assert np.array_equal(pdb_structure.res_id, cif_structure.res_id), "PDB and CIF residue IDs should match"
        assert np.array_equal(pdb_structure.res_id, atom_array.res_id), "PDB and atom array residue IDs should match"
        assert np.array_equal(
            1 +pdb_structure.res_id, reference_structure.res_id
        ), "PDB and reference residue IDs should match"

        assert np.array_equal(pdb_structure.res_name, cif_structure.res_name), "PDB and CIF residue names should match"
        assert np.array_equal(
            pdb_structure.res_name, atom_array.res_name
        ), "PDB and atom array residue names should match"
        assert np.array_equal(
            pdb_structure.res_name, reference_structure.res_name
        ), "PDB and reference residue names should match"

        assert np.array_equal(pdb_structure.atom_name, cif_structure.atom_name), "PDB and CIF atom names should match"
        assert np.array_equal(
            pdb_structure.atom_name, atom_array.atom_name
        ), "PDB and atom array atom names should match"
        assert np.array_equal(
            pdb_structure.atom_name, reference_structure.atom_name
        ), "PDB and reference atom names should match"

        # DO the same but with an T4
        model = get_esmfold(gpu_type="T4", config=config)
        result = model.fold.remote([sequence])
        reference_pdb = PDBFile().read(pathlib.Path(__file__).parent / "data/multimer-check-T4.pdb")
        reference_structure = reference_pdb.get_structure(model=1)
        pdb_structure = PDBFile.read(StringIO(result.pdb[0])).get_structure(model=1)
        cif_structure = get_structure(CIFFile.read(StringIO(result.cif[0])))
        atom_array = result.atom_array[0]
        assert np.allclose(
            pdb_structure.coord, reference_structure.coord, atol=0.1
        ), "PDB and reference coordinates should match within 0.1Å"


########################
# ESM2
########################


@pytest.fixture
def esm2_model_factory():
    def _make_model(**kwargs):
        config = {**kwargs}

        if "15B" in config["model_name"]:
            model = get_esm2(gpu_type="A100-80GB", config=config)
        elif "3B" in config["model_name"]:
            model = get_esm2(gpu_type="A100-40GB", config=config)
        else:
            model = get_esm2(gpu_type="T4", config=config)

        return model

    return _make_model


@pytest.mark.parametrize(
    "model_config",
    [
        {"model_name": "esm2_t6_8M_UR50D", "latent_dim": 320, "num_layers": 6},
        {"model_name": "esm2_t12_35M_UR50D", "latent_dim": 480, "num_layers": 12},
        {"model_name": "esm2_t30_150M_UR50D", "latent_dim": 640, "num_layers": 30},
        {"model_name": "esm2_t33_650M_UR50D", "latent_dim": 1280, "num_layers": 33},
        {"model_name": "esm2_t36_3B_UR50D", "latent_dim": 2560, "num_layers": 36},
        # {"model_name": "esm2_t48_15B_UR50D", "latent_dim": 5120, "num_layers": 48},
    ],
)
def test_esm2_embed_basic(esm2_model_factory, model_config):
    """Test ESM2 embedding."""
    sequence = "MALWMRLLPLLALLALWGPDPAAA"

    with app.run():
        model = esm2_model_factory(model_name=model_config["model_name"])
        result = model.embed.remote([sequence])
        # +2 for the two extra tokens (start of sequence and end of sequence)
        assert result.embeddings.shape == (1, len(sequence), model_config["latent_dim"])
        assert result.hidden_states is not None
        # +1 for the extra layer of the transformer ??? UNCLEAR WHY THIS IS THE CASE
        assert result.hidden_states.shape == (
            1,
            model_config["num_layers"] + 1,
            len(sequence),
            model_config["latent_dim"],
        )
        del model


def test_esm2_embed_hidden_states(esm2_model_factory):
    """Test ESM2 embedding hidden states."""
    with app.run():
        sequence = "MALWMRLLPLLALLALWGPDPAAA"
        model = esm2_model_factory(model_name="esm2_t33_650M_UR50D", output_hidden_states=False)
        result = model.embed.remote([sequence])
        assert result.hidden_states is None
        del model


def test_esm2_embed_multimer(esm2_model_factory):
    """Test ESM2 embedding multimer functionality.

    Tests various aspects of multimer handling:
    - Basic multimer embedding
    - Chain indices and residue indices
    - Padding mask
    - Hidden states (when enabled)
    - Different glycine linker lengths
    """
    with app.run():
        # Test with different glycine linker lengths
        for linker_length in [0, 10, 50]:
            model = esm2_model_factory(
                model_name="esm2_t33_650M_UR50D",
                output_hidden_states=True,
                glycine_linker="G" * linker_length,
                position_ids_skip=512,
            )

            # Test with a simple multimer sequence
            sequence = TEST_SEQUENCES["multimer"]
            result = model.embed.remote([sequence])

            # Check basic shape
            expected_length = len(sequence.replace(":", ""))
            assert result.embeddings.shape == (1, expected_length, 1280), "Embedding shape mismatch"

            # Check chain indices
            assert result.chain_index is not None, "Chain index should be present"
            assert result.chain_index.shape == (1, expected_length), "Chain index shape mismatch"

            # First chain should be 0, second chain should be 1
            first_chain_length = len(sequence.split(":")[0])
            assert np.all(result.chain_index[0, :first_chain_length] == 0), "First chain indices should be 0"
            assert np.all(result.chain_index[0, first_chain_length:] == 1), "Second chain indices should be 1"

            # Check residue indices
            assert result.residue_index is not None, "Residue index should be present"
            assert result.residue_index.shape == (1, expected_length), "Residue index shape mismatch"

            # Check hidden states
            assert result.hidden_states is not None, "Hidden states should be present"
            assert result.hidden_states.shape == (1, 34, expected_length, 1280), "Hidden states shape mismatch"

            # Test with a more complex multimer sequence
            complex_sequence = "MALWMRLLPLLALLALLAADASDASLLALWGPDPAAA:MADLLALWGPDPAAA:MALWMRLLPLLAADLLALWGPDPWGPDPAAA"
            result = model.embed.remote([complex_sequence])

            # Check basic shape for complex sequence
            expected_length = len(complex_sequence.replace(":", ""))
            assert result.embeddings.shape == (1, expected_length, 1280), "Complex sequence embedding shape mismatch"

            # Check chain indices for complex sequence
            assert result.chain_index.shape == (1, expected_length), "Complex sequence chain index shape mismatch"

            # First chain should be 0, second chain should be 1, third chain should be 2
            first_chain_length = len(complex_sequence.split(":")[0])
            second_chain_length = len(complex_sequence.split(":")[1])
            third_chain_length = len(complex_sequence.split(":")[2])
            assert np.all(result.chain_index[0, :first_chain_length] == 0), "First chain indices should be 0"
            assert np.all(
                result.chain_index[0, first_chain_length : first_chain_length + second_chain_length] == 1
            ), "Second chain indices should be 1"
            assert np.all(
                result.chain_index[0, first_chain_length + second_chain_length :] == 2
            ), "Third chain indices should be 2"
            assert np.all(
                result.chain_index[0, first_chain_length + second_chain_length + third_chain_length :] == 3
            ), "Fourth chain indices should be 3"

            # Last test for a batched multimer, each sequence has different number of chains and length
            sequences = [
                "AAA:CCC",  # Very short 2-chain multimer
                TEST_SEQUENCES["short"],  # Monomer (25 residues)
                "A" * 50 + ":" + "C" * 100 + ":" + "D" * 75,  # Long 3-chain multimer with different chain lengths
                "M" * 10 + ":" + "K" * 10,  # Small symmetric 2-chain multimer
                "M" * 1 + ":" + "Y" * 1,  # Edge case: minimal 2-chain multimer (1 residue each)
            ]
            result = model.embed.remote(sequences)
            assert result.embeddings.shape == (
                len(sequences),
                max(len(seq.replace(":", "")) for seq in sequences),
                1280,
            ), "Embedding shape mismatch"
            assert result.chain_index.shape == (
                len(sequences),
                max(len(seq.replace(":", "")) for seq in sequences),
            ), "Chain index shape mismatch"
            assert result.residue_index.shape == (
                len(sequences),
                max(len(seq.replace(":", "")) for seq in sequences),
            ), "Residue index shape mismatch"
            assert result.hidden_states.shape == (
                len(sequences),
                34,
                max(len(seq.replace(":", "")) for seq in sequences),
                1280,
            ), "Hidden states shape mismatch"

            for i, seq in enumerate(sequences):
                expected_length = len(seq.replace(":", ""))
                assert np.all(result.embeddings[i, :expected_length] != 0), "No padding should be 0"
                assert np.all(result.embeddings[i, expected_length:] == 0), "Padding should be 0"
                assert np.all(result.chain_index[i, :expected_length] != -1), "No padding should be -1"
                assert np.all(result.chain_index[i, expected_length:] == -1), "Padding should be -1"
                assert np.all(result.residue_index[i, :expected_length] != -1), "No padding should be -1"
                assert np.all(result.residue_index[i, expected_length:] == -1), "Padding should be -1"
                # Count the number of zeros in the non-padding region; allow up to 16 zeros due to possible sparsity
                num_zeros = np.sum(result.hidden_states[i, :, :expected_length] == 0)
                assert num_zeros < 16, f"Too many zeros ({num_zeros}) in non-padding hidden states"
                assert np.all(result.hidden_states[i, :, expected_length:] == 0), "Padding should be 0"
            del model
