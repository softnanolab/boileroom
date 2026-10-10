"""Live production-image regressions for ESM-C and ESM3 embeddings."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.gpu, pytest.mark.xdist_group("esm3")]

SEQUENCES = ("MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG", "ACDEFGHIKL")


@pytest.mark.parametrize(("model_name", "dimension"), [("ESMC", 960), ("ESM3", 1536)])
def test_live_embeddings_preserve_residues(model_name, dimension, backend_option, device_option, output_ctx):
    """The public wrapper loads a real checkpoint and returns finite residue-aligned embeddings."""
    import boileroom

    with output_ctx(), getattr(boileroom, model_name)(backend=backend_option, device=device_option) as model:
        output = model.embed(SEQUENCES, options={"include_fields": ["lm_logits"]})

    assert output.embeddings.shape == (2, len(SEQUENCES[0]), dimension)
    assert np.isfinite(output.embeddings).all() and np.any(output.embeddings)
    assert output.lm_logits is not None and output.lm_logits.shape[:2] == output.embeddings.shape[:2]
    assert np.isfinite(output.lm_logits).all()
    for row, sequence in enumerate(SEQUENCES):
        length = len(sequence)
        np.testing.assert_array_equal(output.residue_index[row, :length], np.arange(length))
        np.testing.assert_array_equal(output.chain_index[row, :length], np.zeros(length))
        assert not output.embeddings[row, length:].any()
        assert (output.residue_index[row, length:] == -1).all()
    assert output.metadata.model_version and output.metadata.model_version != "fake-version"
    print(f"{model_name}: embeddings={output.embeddings.shape}, sdk_version={output.metadata.model_version}")
