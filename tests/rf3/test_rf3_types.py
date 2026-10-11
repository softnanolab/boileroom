"""RF3Output normalization without importing RF3 or any model dependency."""

import numpy as np
import pytest

from boileroom.base import PredictionMetadata
from boileroom.models.rf3.types import RF3Output


def _metadata() -> PredictionMetadata:
    return PredictionMetadata(model_name="RF3", model_version="test", sequence_lengths=[4])


def test_scalars_come_from_the_summary_confidence_when_not_given() -> None:
    """ptm and iptm are lifted out of each sample's summary and stay aligned with the samples."""
    output = RF3Output(
        metadata=_metadata(),
        confidence=[{"ptm": 0.9, "iptm": 0.8}, {"ptm": 0.5, "iptm": None}, None],
    )

    assert output.ptm is not None and output.iptm is not None
    first_ptm, second_ptm, third_ptm = output.ptm
    first_iptm, second_iptm, third_iptm = output.iptm
    assert first_ptm is not None and second_ptm is not None and third_ptm is None
    assert first_ptm.item() == pytest.approx(0.9) and second_ptm.item() == pytest.approx(0.5)
    assert first_iptm is not None and first_iptm.item() == pytest.approx(0.8)
    assert second_iptm is None and third_iptm is None
    assert first_ptm.shape == (1,) and second_ptm.shape == (1,)


def test_explicit_scalars_win_over_the_summary() -> None:
    """Scalars passed directly are kept and reshaped, not recomputed from ``confidence``."""
    output = RF3Output(metadata=_metadata(), ptm=[np.asarray(0.25, dtype=np.float32)], confidence=[{"ptm": 0.9}])

    assert output.ptm is not None and output.ptm[0] is not None
    assert output.ptm[0].shape == (1,) and output.ptm[0].item() == pytest.approx(0.25)


def test_non_scalar_scores_are_rejected() -> None:
    with pytest.raises(ValueError, match="RF3 pTM expected a scalar"):
        RF3Output(metadata=_metadata(), ptm=[np.zeros(3)])
    with pytest.raises(ValueError, match="RF3 iptm expected a scalar"):
        RF3Output(metadata=_metadata(), confidence=[{"iptm": [0.1, 0.2]}])


def test_plddt_is_reported_on_a_zero_to_one_scale() -> None:
    """RF3 writes 0-1 pLDDT; a 0-100 input is rescaled so both end up comparable."""
    output = RF3Output(
        metadata=_metadata(),
        plddt=[np.array([0.5, 0.9]), np.array([50.0, 90.0]), None],
        atom_plddt=[np.array([0.25]), np.array([25.0])],
    )

    assert output.plddt is not None and output.atom_plddt is not None
    first, second, missing = output.plddt
    assert first is not None and second is not None and missing is None
    np.testing.assert_allclose(first, [0.5, 0.9])
    np.testing.assert_allclose(second, [0.5, 0.9])
    np.testing.assert_allclose(np.concatenate(list(output.atom_plddt)), [0.25, 0.25])
    assert first.dtype == np.float32


def test_missing_fields_stay_none() -> None:
    output = RF3Output(metadata=_metadata())

    assert output.ptm is None and output.iptm is None and output.plddt is None and output.atom_plddt is None
