"""Generation uses the same authenticated transport and guarded lifecycle as other tasks."""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from boileroom.backend.apptainer import _ApptainerModelProxy
from boileroom.backend.transport import deserialize_transport_payload, serialize_transport_payload
from boileroom.base import PredictionMetadata
from boileroom.models.profam.types import ProFamOutput


def _output() -> ProFamOutput:
    return ProFamOutput(PredictionMetadata("ProFam", "test", [3]), ["ACD"], [-0.1])


def test_proxy_generation_roundtrip_and_signature_verification() -> None:
    expected = _output()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/generate"
        assert json.loads(request.content) == {"sequences": ["ACD", "ACE"], "options": {"seed": 42}}
        return httpx.Response(200, json=serialize_transport_payload(expected, "secret"))

    with httpx.Client(transport=httpx.MockTransport(handler), base_url="http://server") as client:
        assert _ApptainerModelProxy(client, "secret").generate(["ACD", "ACE"], {"seed": 42}) == expected
        with pytest.raises(ValueError, match="signature"):
            _ApptainerModelProxy(client, "wrong").generate(["ACD", "ACE"], {"seed": 42})


def test_generation_route_and_error_statuses(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("fastapi")
    from fastapi import HTTPException

    from boileroom.backend import server
    from boileroom.backend.transport import TRANSPORT_HMAC_KEY_ENV

    expected = _output()
    recorded: list[ProFamOutput] = []
    monkeypatch.setenv(TRANSPORT_HMAC_KEY_ENV, "secret")
    monkeypatch.setattr(server, "record_gpu_memory", recorded.append)
    request = server.GenerateRequest(sequences=["ACD"], options={"seed": 42})

    def generate(sequences, options=None):
        assert sequences == ["ACD"] and options == {"seed": 42}
        return expected

    monkeypatch.setattr(server, "_model_instance", SimpleNamespace(generate=generate))
    response = asyncio.run(server.generate(request))
    assert deserialize_transport_payload(json.loads(response.body), "secret") == expected
    assert recorded == [expected]

    for core, status in [(None, 503), (SimpleNamespace(), 501)]:
        monkeypatch.setattr(server, "_model_instance", core)
        with pytest.raises(HTTPException) as info:
            asyncio.run(server.generate(request))
        assert info.value.status_code == status

    def invalid(*args, **kwargs):
        raise ValueError("invalid prompt")

    monkeypatch.setattr(server, "_model_instance", SimpleNamespace(generate=invalid))
    with pytest.raises(HTTPException) as info:
        asyncio.run(server.generate(request))
    assert info.value.status_code == 422

    def failure(*args, **kwargs):
        raise RuntimeError("load failed")

    monkeypatch.setattr(server, "_model_instance", SimpleNamespace(generate=failure))
    response = asyncio.run(server.generate(request))
    assert response.status_code == 500
    assert json.loads(response.body)["error_type"] == "RuntimeError"
