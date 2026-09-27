"""Shared environment and output helpers for resident model workers."""

from __future__ import annotations

import os
from typing import Any


def include_field(include_fields: list[str] | None, field: str) -> bool:
    """Return whether ``field`` should be collected given an inclusion list."""
    return include_fields is not None and ("*" in include_fields or field in include_fields)


def command_env(config: dict[str, Any], extra_env: dict[str, str] | None = None) -> dict[str, str]:
    """Build the environment for an isolated model worker.

    Parameters
    ----------
    config : dict[str, Any]
        Effective core configuration; an optional ``device`` value of the form
        ``cuda:<n>`` or ``cpu`` sets ``CUDA_VISIBLE_DEVICES`` accordingly.
    extra_env : dict[str, str] | None
        Defaults applied with ``setdefault`` so the caller never overrides a
        value the surrounding environment already provides.

    Returns
    -------
    dict[str, str]
        A copy of ``os.environ`` with the requested overrides applied.
    """
    env = os.environ.copy()
    for key, value in (extra_env or {}).items():
        env.setdefault(key, value)
    device = config.get("device")
    if device is None:
        return env
    device_name = str(device)
    if device_name.startswith("cuda:"):
        env["CUDA_VISIBLE_DEVICES"] = device_name.split(":", 1)[1]
    elif device_name == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env
