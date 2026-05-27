"""Contract tests for normalized confidence metric outputs."""

from dataclasses import replace

import numpy as np

from boileroom.base import PredictionMetadata
from boileroom.models.boltz.types import Boltz2Output
from boileroom.models.chai.types import Chai1Output
from boileroom.models.esm.types import ESMFoldOutput
from boileroom.models.esmfold2.types import ESMFold2Output


def _metadata(model_name: str = "test", sequence_lengths: list[int] | None = None) -> PredictionMetadata:
    return PredictionMetadata(model_name=model_name, model_version="test", sequence_lengths=sequence_lengths or [3])


def test_esmfold_confidence_metrics_are_per_sample_unit_scale_arrays() -> None:
    """ESMFold should expose per-residue pLDDT and scalar pTM in consumer-friendly shapes."""
    raw_plddt = np.zeros((2, 3, 37), dtype=np.float32)
    raw_plddt[0, :, 1] = np.array([10.0, 50.0, 100.0], dtype=np.float32)
    raw_plddt[1, :, 1] = np.array([25.0, 75.0, 90.0], dtype=np.float32)

    output = ESMFoldOutput(
        metadata=_metadata("ESMFold", [3, 1]),
        atom_array=[object(), object()],
        plddt=raw_plddt,
        ptm=np.array([0.7, 0.8], dtype=np.float32),
    )

    assert output.plddt is not None
    assert len(output.plddt) == 2
    np.testing.assert_allclose(output.plddt[0], np.array([0.1, 0.5, 1.0], dtype=np.float32))
    np.testing.assert_allclose(output.plddt[1], np.array([0.25], dtype=np.float32))

    replaced = replace(output)
    assert replaced.plddt is not None
    np.testing.assert_allclose(replaced.plddt[1], np.array([0.25], dtype=np.float32))

    assert output.ptm is not None
    assert [score.shape for score in output.ptm if score is not None] == [(1,), (1,)]
    np.testing.assert_allclose(output.ptm[0], np.array([0.7], dtype=np.float32))
    np.testing.assert_allclose(output.ptm[1], np.array([0.8], dtype=np.float32))


def test_chai_confidence_metrics_are_per_sample_unit_scale_arrays() -> None:
    """Chai-1 should normalize percent pLDDT and scalar pTM/iPTM arrays."""
    output = Chai1Output(
        metadata=_metadata("Chai-1"),
        atom_array=[object()],
        plddt=[np.array([50.0], dtype=np.float32)],
        ptm=[np.asarray(0.6, dtype=np.float32)],
        iptm=[np.asarray(0.4, dtype=np.float32)],
    )

    assert output.plddt is not None
    np.testing.assert_allclose(output.plddt[0], np.array([0.5], dtype=np.float32))
    assert output.ptm is not None and output.ptm[0] is not None
    assert output.iptm is not None and output.iptm[0] is not None
    assert output.ptm[0].shape == (1,)
    assert output.iptm[0].shape == (1,)
    np.testing.assert_allclose(output.ptm[0], np.array([0.6], dtype=np.float32))
    np.testing.assert_allclose(output.iptm[0], np.array([0.4], dtype=np.float32))


def test_esmfold2_confidence_metrics_are_per_sample_unit_scale_arrays() -> None:
    """ESMFold2 should normalize pLDDT and scalar pTM/iPTM outputs."""
    output = ESMFold2Output(
        metadata=_metadata("ESMFold2"),
        atom_array=[object(), object()],
        plddt=[np.array([10.0, 50.0], dtype=np.float32), np.array([0.25], dtype=np.float32)],
        ptm=[0.7, None],
        iptm=[np.asarray(0.6, dtype=np.float32), np.asarray(0.4, dtype=np.float32)],
    )

    assert output.plddt is not None
    np.testing.assert_allclose(output.plddt[0], np.array([0.1, 0.5], dtype=np.float32))
    np.testing.assert_allclose(output.plddt[1], np.array([0.25], dtype=np.float32))

    assert output.ptm is not None and output.ptm[0] is not None
    assert output.ptm[0].shape == (1,)
    assert output.ptm[1] is None
    np.testing.assert_allclose(output.ptm[0], np.array([0.7], dtype=np.float32))

    assert output.iptm is not None and output.iptm[0] is not None and output.iptm[1] is not None
    np.testing.assert_allclose(output.iptm[0], np.array([0.6], dtype=np.float32))
    np.testing.assert_allclose(output.iptm[1], np.array([0.4], dtype=np.float32))


def test_boltz_confidence_metrics_are_top_level_and_unit_scale() -> None:
    """Boltz-2 should lift pTM/iPTM out of nested confidence dictionaries."""
    output = Boltz2Output(
        metadata=_metadata("Boltz-2"),
        atom_array=[object(), object()],
        confidence=[
            {"confidence_score": 0.9, "ptm": 0.7, "iptm": np.asarray(0.5, dtype=np.float32)},
            None,
        ],
        plddt=[np.array([0.2], dtype=np.float32), None],
    )

    assert output.plddt is not None
    np.testing.assert_allclose(output.plddt[0], np.array([0.2], dtype=np.float32))
    assert output.plddt[1] is None

    assert output.ptm is not None and output.ptm[0] is not None
    assert output.iptm is not None and output.iptm[0] is not None
    assert output.ptm[0].shape == (1,)
    assert output.iptm[0].shape == (1,)
    assert output.ptm[1] is None
    assert output.iptm[1] is None
    np.testing.assert_allclose(output.ptm[0], np.array([0.7], dtype=np.float32))
    np.testing.assert_allclose(output.iptm[0], np.array([0.5], dtype=np.float32))

    assert output.confidence == [{"confidence_score": 0.9}, None]
