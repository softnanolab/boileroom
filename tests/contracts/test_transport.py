"""Contract tests for the local Apptainer transport helpers."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from boileroom.backend.transport import deserialize_transport_payload, serialize_transport_payload


@dataclass
class SamplePayload:
    """Simple payload used to validate signed transport round-trips."""

    values: list[int]


def test_transport_payload_round_trip() -> None:
    """Signed transport payloads should deserialize back to the original object."""
    payload = SamplePayload(values=[1, 2, 3])
    serialized = serialize_transport_payload(payload, "secret")
    restored = deserialize_transport_payload(serialized, "secret")
    assert restored == payload


def test_transport_payload_rejects_tampering() -> None:
    """Tampered transport payloads should fail signature verification."""
    payload = serialize_transport_payload(SamplePayload(values=[1, 2, 3]), "secret")
    payload["pickled"] = payload["pickled"][:-4] + "AAAA"
    with pytest.raises(ValueError, match="signature verification failed"):
        deserialize_transport_payload(payload, "secret")
