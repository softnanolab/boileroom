"""Offline tests of validation gates; these do not count as real-model validation."""

from types import SimpleNamespace

import pytest
from profam_runtime_validation import FAMILIES, _assert_same, _live_http, _plateau

from boileroom.base import PredictionMetadata
from boileroom.models.profam.types import ConditioningPrompt, ProFamOutput


def test_reference_fixtures_and_parity_gates() -> None:
    assert [len(value[0][0]) for value in FAMILIES.values()] == [76, 46]
    conditioning = [ConditioningPrompt(["ACD"], ["accession"])]
    actual = ProFamOutput(PredictionMetadata("ProFam", "test", [3]), ["ACD"], [-0.5], conditioning)
    reference = SimpleNamespace(sequences=["ACD"], scores=[-0.5], conditioning_prompts=conditioning)
    _assert_same(actual, reference)
    reference.scores = [-0.4]
    with pytest.raises(AssertionError, match="score parity"):
        _assert_same(actual, reference)
    reference.scores = [-0.5]
    reference.conditioning_prompts = [ConditioningPrompt(["ACE"], ["accession"])]
    with pytest.raises(AssertionError, match="conditioning parity"):
        _assert_same(actual, reference)


def test_memory_gate_refuses_sustained_growth() -> None:
    def trace(values):
        return [[dict.fromkeys(("allocated", "reserved", "device_used"), value)] for value in values]

    _plateau(trace([100, 120, 125, 120]))
    with pytest.raises(AssertionError, match="grew"):
        _plateau(trace([100, 120, 300, 400]))


def test_live_http_helper_restores_state_after_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("uvicorn")
    from boileroom.backend import server
    from boileroom.backend.transport import TRANSPORT_HMAC_KEY_ENV

    original = object()
    monkeypatch.setattr(server, "_model_instance", original)
    monkeypatch.setenv(TRANSPORT_HMAC_KEY_ENV, "previous")
    output = ProFamOutput(PredictionMetadata("ProFam", "test", [3]), ["ACD"], [-0.1])
    core = SimpleNamespace(generate=lambda *args, **kwargs: output)
    with pytest.raises(RuntimeError, match="injected harness failure"), _live_http(core) as (proxy, _):
        received = proxy.generate("ACD")
        assert received.sequences == output.sequences
        raise RuntimeError("injected harness failure")
    assert server._model_instance is original
    import os

    assert os.environ[TRANSPORT_HMAC_KEY_ENV] == "previous"


def test_public_backend_harness_has_explicit_success_and_failure_paths() -> None:
    from profam_public_validation import run_public_validation

    class Wrapper:
        fail = False

        def __init__(self, **kwargs):
            self._backend = SimpleNamespace(model=object(), _process=None)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self._backend.model = None

        def generate(self, sequences, options):
            from boileroom.models.profam.core import ProFamCore

            ProFamCore().validate_request(sequences, options)
            if self.fail:
                raise RuntimeError("injected runtime failure")
            from profam_public_validation import PROMPT

            runtime = {
                "profam.checkpoint_sha256": "8ec54d80323f188d3454fb4457f63f12158fe55552b919554806b12730c30e08",
                "profam.commit": "a19061606dc6469424035db3fdcaae6e63688017",
            }
            return ProFamOutput(
                PredictionMetadata("ProFam", "test", [3, 3], runtime=runtime),
                ["ACD", "ACE"],
                [-0.1, -0.2],
                [ConditioningPrompt(PROMPT.copy(), ["1UBQ_A", "1UBQ_G76A_synthetic"])],
            )

    report = run_public_validation(Wrapper, backend="apptainer", device="cpu")
    assert report["passed"] and report["client_backend_stopped"]
    assert report["invalid_inputs_refused"] == 4
    Wrapper.fail = True
    report = run_public_validation(Wrapper, backend="apptainer", device="cpu")
    assert not report["passed"] and report["client_backend_stopped"]
    assert "injected runtime failure" in report["error"]


def test_cleanup_diagnostics_report_metadata_without_retaining_tensors(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    import profam_runtime_validation as validation

    class Tensor:
        def __init__(self, cuda):
            self.is_cuda = cuda
            self.shape = (2, 3)
            self.dtype = "float16"

        def numel(self):
            return 6

        def element_size(self):
            return 2

    blocks = [
        {
            "blocks": [
                {"size": 8519680, "requested_size": 8519680, "state": "active_allocated"},
                {"size": 42, "state": "inactive"},
            ]
        }
    ]
    fake = SimpleNamespace(Tensor=Tensor, cuda=SimpleNamespace(memory_snapshot=lambda: blocks))
    monkeypatch.setitem(sys.modules, "torch", fake)
    monkeypatch.setattr(validation.gc, "get_objects", lambda: [Tensor(True), Tensor(False), object()])
    result = validation._allocation_diagnostics()
    assert result["active_blocks"] == blocks[0]["blocks"][:1]
    assert result["live_cuda_tensor_count"] == 1
    assert result["live_cuda_tensor_examples"] == [{"shape": [2, 3], "dtype": "float16", "bytes": 12}]
