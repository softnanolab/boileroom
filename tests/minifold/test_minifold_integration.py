"""Explicit real-runtime MiniFold checks; never selected by offline CI."""

import numpy as np
import pytest

from boileroom import MiniFold

pytestmark = [pytest.mark.integration, pytest.mark.gpu, pytest.mark.slow, pytest.mark.xdist_group("minifold")]


@pytest.mark.parametrize("model_size", ["12L", "48L"])
def test_minifold_variants_repeated_and_mixed_length(backend_option, device_option, output_ctx, model_size):
    """Check both checkpoints, CPU transport, and unpadded list/single consistency."""
    sequences = ["MLKNVHVLVLGAGDVGSVVVRLLEK", "ACDEFGHIKLMNPQRSTVWY"]
    with (
        output_ctx(),
        MiniFold(backend=backend_option, device=device_option, config={"model_size": model_size}) as model,
    ):
        single = model.fold(sequences[0], {"include_fields": ["*"]})
        batch = model.fold(sequences, {"include_fields": ["*"]})
        again = model.fold(sequences[1])
    assert single.metadata.sequence_lengths == [len(sequences[0])]
    assert batch.metadata.sequence_lengths == list(map(len, sequences))
    assert single.metadata is not batch.metadata
    for result in (single, batch, again):
        assert result.atom_array and all(np.isfinite(a.coord).all() for a in result.atom_array)
        assert result.metadata.runtime["minifold.checkpoint_revision"]
    np.testing.assert_allclose(single.atom_array[0].coord, batch.atom_array[0].coord, atol=1e-4, rtol=1e-4)
    for scores in batch.plddt:
        assert np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all()
    assert batch.pae is None and batch.pdb and batch.cif
    assert again.plddt is None
