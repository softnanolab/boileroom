import pytest
import numpy as np

from boileroom import app
from boileroom.models.esm.esm2 import get_esm2
from conftest import TEST_SEQUENCES


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
