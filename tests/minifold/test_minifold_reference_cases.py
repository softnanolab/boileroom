"""Offline audit of externally sourced fixtures and scientific metric implementation."""

import numpy as np
import pytest
from _minifold_cases import CASES, experimental_ca, quality_metrics

pytestmark = pytest.mark.contract


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_experimental_sequences_and_residue_mapping(case):
    ids, coordinates = experimental_ca(case)
    assert len(ids) == len(case["sequence"])
    assert coordinates.shape == (len(ids), 3)


def test_metrics_are_rigid_transform_invariant_and_detect_distortion():
    _, coordinates = experimental_ca(CASES[0])
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    same = quality_metrics(coordinates @ rotation + 12.0, coordinates)
    assert same["ca_rmsd_angstrom"] < 1e-10
    assert same["ca_local_distance_agreement"] == 1.0
    distorted = quality_metrics(coordinates * 2, coordinates)
    assert distorted["ca_rmsd_angstrom"] > 1.0
    assert distorted["ca_local_distance_agreement"] < 0.5


def test_runtime_validation_refuses_missing_gpu_without_launch(monkeypatch):
    import json

    import torch
    from runtime_validation import run_validation

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    result = run_validation("12L")
    assert result["passed"] is False
    assert result["phases"]["independent_stock_reference"]["error_type"] == "AssertionError"
    assert "CUDA" in result["phases"]["independent_stock_reference"]["message"]
    json.dumps(result)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_memory_inputs_vary_identity_at_fixed_length_and_composition(case):
    from runtime_validation import _memory_sequence

    source = case["sequence"]
    sequences = [_memory_sequence(source, cycle) for cycle in range(4)]
    assert len(set(sequences + [source])) == 5
    assert all(len(seq) == len(source) and sorted(seq) == sorted(source) for seq in sequences)
    assert all(seq[0] == source[0] and seq[-1] == source[-1] for seq in sequences)
