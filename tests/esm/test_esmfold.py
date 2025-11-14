import pytest
import pathlib
import numpy as np
import torch
from typing import Generator, Optional
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
def esmfold_model(config: Optional[dict] = None, gpu_device: Optional[str] = None) -> Generator[ESMFold, None, None]:
    """
    Provide a configured ESMFold model instance for tests.
    
    Parameters:
        config (Optional[dict]): Optional configuration overrides passed to the ESMFold constructor.
        gpu_device (Optional[str]): Optional device identifier (e.g., "cuda:0" or "cpu") to initialize the model on.
    
    Returns:
        ESMFold: A configured ESMFold instance ready for use in tests.
    """
    model_config = dict(config) if config is not None else {}
    with enable_output():
        yield ESMFold(backend="modal", device=gpu_device, config=model_config)


def test_esmfold_basic(test_sequences: dict[str, str], esmfold_model: ESMFold):
    """Test basic ESMFold functionality with minimal output."""
    result = esmfold_model.fold(test_sequences["short"])

    assert isinstance(result, ESMFoldOutput), "Result should be an ESMFoldOutput"
    assert result.metadata is not None, "metadata should always be present"
    assert result.atom_array is not None, "atom_array should always be generated"
    assert len(result.atom_array) > 0, "atom_array should contain at least one structure"

    # With minimal output, plddt should be None
    assert result.plddt is None, "plddt should be None in minimal output"


def test_esmfold_full_output(test_sequences: dict[str, str], gpu_device: Optional[str]):
    """Test ESMFold with full output requested."""
    with enable_output():
        model = ESMFold(backend="modal", device=gpu_device, config={"include_fields": ["*"]})
        result = model.fold(test_sequences["short"])

    assert isinstance(result, ESMFoldOutput), "Result should be an ESMFoldOutput"

    assert result.plddt is not None, "plddt should be present in full output"
    assert np.all(result.plddt >= 0), "pLDDT scores should be non-negative"
    assert np.all(result.plddt <= 100), "pLDDT scores should be less than or equal to 100"


def test_esmfold_multimer(test_sequences, gpu_device: Optional[str]):
    """Test ESMFold multimer functionality."""
    with enable_output():  # TODO: make this better with a fixture, re-using the logic
        model = ESMFold(backend="modal", device=gpu_device, config={"include_fields": ["*"]})  # Request all fields
        result = model.fold(test_sequences["multimer"])

    assert result.pdb is not None, "PDB output should be generated"
    assert result.residue_index is not None, "residue_index should be generated"
    assert result.chain_index is not None, "chain_index should be generated"
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
    assert result.predicted_aligned_error is not None, "predicted_aligned_error should be generated"
    assert result.plddt is not None, "plddt should be generated"
    assert result.ptm_logits is not None, "ptm_logits should be generated"
    assert result.aligned_confidence_probs is not None, "aligned_confidence_probs should be generated"
    assert result.s_z is not None, "s_z should be generated"
    assert result.s_s is not None, "s_s should be generated"
    assert result.distogram_logits is not None, "distogram_logits should be generated"
    assert result.lm_logits is not None, "lm_logits should be generated"
    assert result.lddt_head is not None, "lddt_head should be generated"
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


def test_esmfold_no_glycine_linker(test_sequences, gpu_device: Optional[str]):
    """Test ESMFold no glycine linker."""
    model = ESMFold(
        backend="modal",
        device=gpu_device,
        config={
            "glycine_linker": "",
            "include_fields": ["*"],  # Request all fields
        },
    )

    with enable_output():
        result = model.fold(test_sequences["multimer"])

    assert result.residue_index is not None, "Residue index should be generated"
    assert result.plddt is not None, "pLDDT should be generated"
    assert result.ptm is not None, "pTM should be generated"

    # assert correct chain_indices
    assert result.chain_index is not None, "chain_index should be generated"
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


def test_esmfold_batch(test_sequences: dict[str, str], gpu_device: Optional[str]):
    """Test ESMFold batch prediction."""
    with enable_output():
        model = ESMFold(backend="modal", device=gpu_device, config={"include_fields": ["*"]})  # Request all fields
        # Define input sequences
        sequences = [test_sequences["short"], test_sequences["medium"]]
        # Make prediction
        result = model.fold(sequences)

    # Check that batch outputs have correct sequence lengths
    assert result.aatype is not None, "aatype should be present in full output"
    assert result.plddt is not None, "plddt should be present in full output"
    assert result.ptm_logits is not None, "ptm_logits should be present in full output"
    assert result.predicted_aligned_error is not None, "predicted_aligned_error should be present in full output"
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
    mock_tokenizer = mocker.patch.object(model, "tokenizer")
    mock_tokenizer.return_value = {
        "input_ids": torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, -1, -1, -1, -1],
                [3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5],
                [8, 8, 8, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            ]
        ),
        "attention_mask": torch.ones(3, 19),
    }

    # Call the method to test
    effective_config = model._merge_options(None)
    tokenized_input, multimer_properties = model._tokenize_sequences(sequences, effective_config)

    # Assert the tokenizer was called with the expected arguments
    expected_sequences = [seq.replace(":", GLYCINE_LINKER) for seq in sequences]
    mock_tokenizer.assert_called_once_with(
        expected_sequences, padding=True, truncation=True, return_tensors="pt", add_special_tokens=False
    )

    # Verify the output contains the expected keys
    assert set(tokenized_input.keys()) >= {"input_ids", "attention_mask", "position_ids"}
    assert multimer_properties is not None


def test_sequence_validation(test_sequences: dict[str, str]):
    """Test sequence validation in FoldingAlgorithm."""

    esmfold_core = ESMFoldCore(config={"device": "cpu"})

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


def test_esmfold_static_config_enforcement(test_sequences: dict[str, str]):
    """Test that static config keys cannot be overridden in options."""
    esmfold_core = ESMFoldCore(config={"device": "cpu"})
    # device is a static config key
    with pytest.raises(ValueError, match="device"):
        esmfold_core.fold(test_sequences["short"], options={"device": "cuda:0"})


def test_esmfold_output_pdb_cif(data_dir: pathlib.Path, test_sequences: dict[str, str], gpu_device: Optional[str]):
    """
    Validate that ESMFold produces consistent PDB and AtomArray outputs and matches saved reference PDB files.
    
    This test:
    - Requests only `pdb` and `atom_array` outputs from ESMFold and asserts that `pdb` is produced, `cif` is not, and `atom_array` is produced.
    - For both short and medium sequences, verifies residues are 0-indexed, the sequence recovered from the AtomArray equals the input sequence, coordinates match between the PDB and AtomArray within 0.1 Å, and `chain_id`, `res_id`, `res_name`, and `atom_name` match exactly.
    - Compares produced PDB structures to saved reference PDB files by RMSD and requires RMSD < 1.5 Å.
    - Compares predicted B-factors from the AtomArray to the saved reference B-factors and requires agreement within an absolute tolerance of 0.05.
    
    No return value.
    """

    def recover_sequence(atomarray: AtomArray) -> str:
        """
        Reconstructs the amino-acid sequence from an AtomArray by mapping unique residue IDs to one-letter codes in ascending residue order.
        
        Parameters:
            atomarray (AtomArray): AtomArray containing `res_id` (residue identifiers) and `res_name` (three-letter residue codes).
        
        Returns:
            str: Concatenated one-letter amino-acid sequence corresponding to the residues ordered by ascending `res_id`.
        """
        unique_res_ids = np.unique(atomarray.res_id)
        three_letter_codes = [atomarray.res_name[atomarray.res_id == res_id][0] for res_id in unique_res_ids]
        one_letter_codes = [restype_3to1[three_letter_code] for three_letter_code in three_letter_codes]
        return "".join(one_letter_codes)

    with enable_output():
        model = ESMFold(
            backend="modal", device=gpu_device, config={"include_fields": ["pdb", "atom_array"]}
        )  # Request PDB and atom_array
        # Define input sequences
        sequences = [test_sequences["short"], test_sequences["medium"]]
        result = model.fold(sequences)

    assert result.pdb is not None, "PDB output should be generated"
    assert result.cif is None, "CIF output should be None (not requested)"
    assert result.atom_array is not None, "Atom array output should always be generated"

    pdb_outputs = result.pdb
    atom_array_outputs = result.atom_array

    assert len(pdb_outputs) == len(atom_array_outputs) == len(sequences) == 2, "Batching output match!"
    assert isinstance(pdb_outputs, list), "PDB output should be a list"
    assert len(pdb_outputs) == len(sequences), "PDB output should have same length as input sequences"
    assert isinstance(atom_array_outputs, list), "Atom array should be a list"
    assert isinstance(atom_array_outputs[0], AtomArray), "Atom array should be an AtomArray"

    short_pdb = PDBFile.read(StringIO(pdb_outputs[0])).get_structure(model=1)
    medium_pdb = PDBFile.read(StringIO(pdb_outputs[1])).get_structure(model=1)
    short_atomarray = atom_array_outputs[0]
    medium_atomarray = atom_array_outputs[1]

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