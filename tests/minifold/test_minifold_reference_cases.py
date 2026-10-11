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
    from minifold_runtime_validation import run_validation

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    result = run_validation("12L")
    assert result["passed"] is False
    assert result["phases"]["independent_stock_reference"]["error_type"] == "AssertionError"
    assert "CUDA" in result["phases"]["independent_stock_reference"]["message"]
    json.dumps(result)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_memory_inputs_vary_identity_at_fixed_length_and_composition(case):
    from minifold_runtime_validation import _memory_sequence

    source = case["sequence"]
    sequences = [_memory_sequence(source, cycle) for cycle in range(4)]
    assert len(set(sequences + [source])) == 5
    assert all(len(seq) == len(source) and sorted(seq) == sorted(source) for seq in sequences)
    assert all(seq[0] == source[0] and seq[-1] == source[-1] for seq in sequences)


@pytest.fixture
def reference_manifest():
    from _minifold_reference import MINIFORMER_SHA256, PREDICT_SHA256

    from boileroom.models.minifold.loading import CHECKPOINT_REVISION, CHECKPOINT_SHA256, ESM_ARTIFACTS, UPSTREAM_COMMIT

    files = {f"{case['id']}-r3.npz": "0" * 64 for case in CASES} | {f"{CASES[0]['id']}-r0.npz": "0" * 64}
    return {
        "model_size": "12L",
        "upstream_commit": UPSTREAM_COMMIT,
        "predict_sha256": PREDICT_SHA256,
        "miniformer_sha256": MINIFORMER_SHA256,
        "triangular_implementation": "original-pinned-upstream",
        "checkpoint_revision": CHECKPOINT_REVISION,
        "checkpoint_sha256": CHECKPOINT_SHA256["12L"],
        "esm_artifacts": ESM_ARTIFACTS.copy(),
        "runtime": {"image_ref": "source:test-only"},
        "compile": False,
        "kernels": False,
        "batch_mode": "one-unpadded-sequence",
        "matmul_precision": "highest",
        "gpu": "test-only L4",
        "files": files,
        "cases": [
            {"id": case["id"], "sequence": case["sequence"], "experimental_sha256": case["sha256"]} for case in CASES
        ],
    }


def test_known_manifest_contract(reference_manifest):
    from _minifold_cases import _validate_reference_manifest

    _validate_reference_manifest(reference_manifest, "12L")


@pytest.mark.parametrize(
    "field,value",
    [
        ("compile", True),
        ("kernels", True),
        ("checkpoint_sha256", "wrong"),
        ("upstream_commit", "wrong"),
        ("miniformer_sha256", "wrong"),
        ("triangular_implementation", "adapter-patched"),
        ("esm_artifacts", {}),
        ("cases", []),
        ("files", {}),
        ("runtime", {"image_ref": "unknown"}),
        ("matmul_precision", "medium"),
        ("gpu", "unknown"),
        ("model_size", "48L"),
    ],
)
def test_unproven_reference_is_rejected_before_public_backend(reference_manifest, field, value):
    from unittest.mock import Mock

    from minifold_public_validation import run_public_modal_validation

    reference_manifest[field] = value
    factory = Mock(side_effect=AssertionError("must not construct a backend"))
    result = run_public_modal_validation(
        factory, "12L", {"reference_manifest": reference_manifest, "reference_artifacts_base64": {}}
    )
    assert result["passed"] is False
    factory.assert_not_called()
