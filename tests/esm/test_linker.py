import torch

from boileroom.models.esm.linker import compute_position_ids


def test_compute_position_ids_padding_and_offsets():
    """Compute position ids for multimers without redundant splits."""
    sequences = ["AA:BB", "CC"]

    result = compute_position_ids(sequences, glycine_linker="G", position_ids_skip=3)

    expected = torch.tensor([[0, 1, 2, 6, 7], [0, 1, 0, 0, 0]], dtype=torch.long)
    assert torch.equal(result, expected)
