"""Shared transport helpers for the Apptainer HTTP bridge."""

from __future__ import annotations

import base64
import hashlib
import hmac
import pickle
from collections.abc import Mapping
from typing import Any, Final

TRANSPORT_HMAC_KEY_ENV: Final = "BOILEROOM_TRANSPORT_HMAC_KEY"


def serialize_transport_payload(output: Any, secret: str) -> dict[str, str]:
    """Serialize a Python object for local transport and sign it."""
    pickled_data = pickle.dumps(output)
    signature = hmac.new(secret.encode("utf-8"), pickled_data, hashlib.sha256).hexdigest()
    return {
        "pickled": base64.b64encode(pickled_data).decode("utf-8"),
        "hmac": signature,
    }


def deserialize_transport_payload(data: Mapping[str, Any], secret: str) -> Any:
    """Validate and deserialize a signed transport payload."""
    if "pickled" not in data:
        raise ValueError("Response does not contain pickled data")
    if "hmac" not in data:
        raise ValueError("Response does not contain payload signature")

    pickled_data = base64.b64decode(str(data["pickled"]).encode("utf-8"))
    expected_signature = hmac.new(secret.encode("utf-8"), pickled_data, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(str(data["hmac"]), expected_signature):
        raise ValueError("Response payload signature verification failed")
    return pickle.loads(pickled_data)
