"""ESMFold2 integration tests against a real backend."""

import numpy as np
import pytest

from boileroom import ESMFold2

pytestmark = [pytest.mark.integration, pytest.mark.slow, pytest.mark.gpu, pytest.mark.xdist_group("esmfold2")]


def test_esmfold2_modal_fold_basic(backend_option: str, device_option: str | None, output_ctx) -> None:
    """Fold a short protein with ESMFold2 and validate real structure outputs."""
    sequence = "MLKNVHVLVLGAGDVGSVVVRLLEK"
    config = {"model_name": "biohub/ESMFold2-Fast"}
    options = {
        "include_fields": ["plddt", "ptm", "cif"],
        "num_loops": 1,
        "num_sampling_steps": 5,
        "num_diffusion_samples": 1,
        "seed": 0,
    }

    with output_ctx(), ESMFold2(backend=backend_option, device=device_option, config=config) as model:
        result = model.fold(sequence, options=options)

    assert result.metadata.sequence_lengths == [len(sequence)]
    assert result.cif is not None
    assert len(result.cif) == 1
    assert result.cif[0].startswith("data_")
    assert result.atom_array is not None
    assert len(result.atom_array) == 1
    assert len(result.atom_array[0]) > 0
    assert result.plddt is not None
    assert result.plddt[0] is not None
    assert result.plddt[0].shape == (len(sequence),)
    assert result.ptm is not None
    assert result.ptm[0] is not None
    assert np.isfinite(result.ptm[0]).all()
