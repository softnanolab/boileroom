"""Shared helpers for CLI-backed folding cores (Protenix, AlphaFold2-Multimer).

These cores drive an upstream command-line tool via ``subprocess``. They share
the same boolean-flag rendering, output-field selection, CUDA device plumbing,
and failure handling, which live here so each core stays focused on its own
input/output mapping.
"""

from __future__ import annotations

import os
import subprocess
from typing import Any


def bool_arg(value: Any) -> str:
    """Render a boolean as the lowercase ``true``/``false`` the CLIs expect."""
    return "true" if bool(value) else "false"


def include_field(include_fields: list[str] | None, field: str) -> bool:
    """Return whether ``field`` should be collected given an inclusion list."""
    return include_fields is not None and ("*" in include_fields or field in include_fields)


def command_env(config: dict[str, Any], extra_env: dict[str, str] | None = None) -> dict[str, str]:
    """Build the subprocess environment for a CLI-backed core.

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


def run_command(command: list[str], label: str, env: dict[str, str], timeout: float | None) -> None:
    """Run a CLI command, raising with a trimmed log tail on failure.

    Parameters
    ----------
    command : list[str]
        Argument vector passed to ``subprocess.run``.
    label : str
        Human-readable model name used in the error message.
    env : dict[str, str]
        Environment for the subprocess (see :func:`command_env`).
    timeout : float | None
        Seconds before the command is killed; ``None`` disables the timeout.
    """
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
    )
    if result.returncode != 0:
        tail = "\n".join((result.stdout + "\n" + result.stderr).splitlines()[-80:])
        raise RuntimeError(f"{label} command failed with exit code {result.returncode}:\n{tail}")
