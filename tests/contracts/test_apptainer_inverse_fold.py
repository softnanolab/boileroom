"""Transport tests for the ESM3 ``inverse_fold`` Apptainer proxy and server route."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import numpy as np
import pytest

from boileroom.backend.apptainer import _ApptainerModelProxy
from boileroom.backend.transport import serialize_transport_payload

SECRET = "test-secret"


def _proxy(handler: httpx.MockTransport, log_file_path: Path | None = None) -> _ApptainerModelProxy:
    """Build a proxy whose HTTP client is served by ``handler``."""
    client = httpx.Client(transport=handler, base_url="http://server")
    return _ApptainerModelProxy(client, SECRET, log_file_path=log_file_path)


def test_proxy_posts_json_safe_payload_with_nan_as_null_and_deserializes_response() -> None:
    """NaN coordinates are sent as null, ints are coerced, and the signed response is verified and decoded."""
    seen: dict[str, Any] = {}
    expected = SimpleNamespace(logits=np.ones((1, 20), dtype=np.float32))

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)  # strict JSON parse of what was sent
        return httpx.Response(200, json=serialize_transport_payload(expected, SECRET))

    coords = np.arange(2 * 3 * 3, dtype=np.float32).reshape(2, 3, 3)
    coords[1, 2] = np.nan
    result = _proxy(httpx.MockTransport(handler)).inverse_fold("AC", coords, [np.int64(1)])

    assert seen["path"] == "/inverse_fold"
    assert seen["body"]["sequence"] == "AC"
    assert seen["body"]["positions"] == [1]
    assert seen["body"]["backbone_coordinates"][1][2] == [None, None, None]
    assert seen["body"]["backbone_coordinates"][0][1] == [3.0, 4.0, 5.0]
    assert np.array_equal(result.logits, expected.logits)


def test_proxy_500_raises_runtime_error_with_log_path(tmp_path: Path) -> None:
    """Server errors surface as RuntimeError mentioning the server log file."""
    transport = httpx.MockTransport(lambda request: httpx.Response(500, json={"detail": "boom"}))
    proxy = _proxy(transport, log_file_path=tmp_path / "server.log")
    with pytest.raises(RuntimeError, match="server.log"):
        proxy.inverse_fold("AC", np.zeros((2, 3, 3)), [0])


def test_proxy_rejects_tampered_response() -> None:
    """A response signed with a different secret fails verification."""
    payload = serialize_transport_payload(SimpleNamespace(logits=np.zeros((1, 20))), "other-secret")
    proxy = _proxy(httpx.MockTransport(lambda request: httpx.Response(200, json=payload)))
    with pytest.raises(ValueError, match="signature"):
        proxy.inverse_fold("AC", np.zeros((2, 3, 3)), [0])


def test_server_route_converts_null_to_nan_and_forwards_to_core(monkeypatch: pytest.MonkeyPatch) -> None:
    """The route rebuilds a float32 array (null -> NaN), calls the core, and returns a signed payload."""
    pytest.importorskip("fastapi")
    from fastapi import HTTPException

    from boileroom.backend import server
    from boileroom.backend.transport import TRANSPORT_HMAC_KEY_ENV, deserialize_transport_payload

    calls: list[tuple[str, np.ndarray, list[int]]] = []

    class _Core:
        def inverse_fold(self, sequence: str, coordinates: np.ndarray, positions: list[int]) -> SimpleNamespace:
            """Record the call and return a small output."""
            calls.append((sequence, coordinates, positions))
            return SimpleNamespace(logits=np.full((1, 20), 2.0, dtype=np.float32))

    monkeypatch.setenv(TRANSPORT_HMAC_KEY_ENV, SECRET)
    monkeypatch.setattr(server, "_model_instance", _Core())
    coords = np.zeros((2, 3, 3)).astype(object)
    coords[1, 2] = None
    request = server.InverseFoldRequest(sequence="AC", backbone_coordinates=coords.tolist(), positions=[1])

    response = asyncio.run(server.inverse_fold(request))

    sequence, coordinates, positions = calls[0]
    assert (sequence, positions) == ("AC", [1])
    assert coordinates.dtype == np.float32 and coordinates.shape == (2, 3, 3)
    assert np.isnan(coordinates[1, 2]).all() and not np.isnan(coordinates[0]).any()
    assert deserialize_transport_payload(json.loads(response.body), SECRET).logits.shape == (1, 20)

    monkeypatch.setattr(server, "_model_instance", SimpleNamespace())  # e.g. ESM-C: no inverse_fold
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(server.inverse_fold(request))
    assert excinfo.value.status_code == 501
    monkeypatch.setattr(server, "_model_instance", None)
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(server.inverse_fold(request))
    assert excinfo.value.status_code == 503


def test_server_route_maps_value_error_to_422_and_other_errors_to_500(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid caller input (ValueError from the core) is a 422; unexpected failures stay a 500."""
    pytest.importorskip("fastapi")
    from fastapi import HTTPException

    from boileroom.backend import server
    from boileroom.backend.transport import TRANSPORT_HMAC_KEY_ENV

    class _Core:
        error: Exception = ValueError("positions must not contain duplicates.")

        def inverse_fold(self, sequence: str, coordinates: np.ndarray, positions: list[int]) -> SimpleNamespace:
            """Always fail with the configured error."""
            raise self.error

    core = _Core()
    monkeypatch.setenv(TRANSPORT_HMAC_KEY_ENV, SECRET)
    monkeypatch.setattr(server, "_model_instance", core)
    request = server.InverseFoldRequest(
        sequence="AC", backbone_coordinates=np.zeros((2, 3, 3)).tolist(), positions=[1, 1]
    )

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(server.inverse_fold(request))
    assert excinfo.value.status_code == 422
    assert "duplicates" in str(excinfo.value.detail)

    core.error = RuntimeError("CUDA out of memory")
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(server.inverse_fold(request))
    assert excinfo.value.status_code == 500
