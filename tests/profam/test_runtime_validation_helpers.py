"""Offline tests of validation gates; these do not count as real-model validation."""

from types import SimpleNamespace

import pytest
from runtime_validation import FAMILIES, _assert_same, _live_http, _plateau

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
