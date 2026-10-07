"""Runtime provenance for prediction metadata: which code, image, stack and GPU produced an output.

Standard library only, so any core (and the lightweight wrappers) can call it without pulling in model dependencies.
"""

from __future__ import annotations

import importlib.metadata
import os
import platform
import sys
import tomllib
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .optimization import GpuInfo

#: Environment variable a runtime image sets to its own reference (e.g. ``docker.io/org/boileroom-x:cuda12.6-0.3.1``).
IMAGE_REF_ENV = "BOILEROOM_IMAGE_REF"
_UNKNOWN = "unknown"
_NOT_LOADED = "not-loaded"
_ABSENT = "absent"
_NONE = "none"
#: ``metadata.runtime`` keys every kit family records in a kit mode (see :func:`kit_provenance`).
KIT_RUNTIME_KEYS: tuple[str, ...] = ("kit.commit", "kit.levers_applied", "kit.levers_fallback", "kit.partial")


def runtime_provenance(
    packages: Iterable[str] = (),
    *,
    gpu: GpuInfo | None = None,
    extra: Mapping[str, object] | None = None,
) -> dict[str, str]:
    """Describe the runtime that produced a prediction, as a flat string mapping.

    Never imports torch or a model library: ``torch`` and ``cuda`` are read from modules that are already loaded
    (``"not-loaded"`` if torch is not; ``cuda`` is ``"none"`` for a loaded CPU-only torch build).

    Parameters
    ----------
    packages : Iterable[str]
        Distribution names whose installed versions to record, one key each (``"absent"`` if not installed).
    gpu : GpuInfo | None
        The device the model ran on (see :func:`boileroom.optimization.describe_gpu`), or ``None`` for none.
    extra : Mapping[str, object] | None
        Further entries (e.g. what a worker's ``describe()`` reported), stringified. They win over the keys above.

    Returns
    -------
    dict[str, str]
        ``boileroom``, ``python``, ``image_ref``, ``torch``, ``cuda``, ``gpu``, ``gpu_capability``, then one key per
        package in the order given, then the ``extra`` keys in their order.
    """
    torch = sys.modules.get("torch")
    if torch is None:
        torch_version = cuda_version = _NOT_LOADED
    else:
        torch_version = str(getattr(torch, "__version__", None) or _UNKNOWN)
        # A loaded torch without CUDA (a CPU-only build) is recorded as such, not as missing data.
        cuda_version = str(getattr(getattr(torch, "version", None), "cuda", None) or _NONE)
    record: dict[str, str] = {
        "boileroom": _boileroom_version(),
        "python": platform.python_version(),
        "image_ref": os.environ.get(IMAGE_REF_ENV) or _UNKNOWN,
        "torch": torch_version,
        "cuda": cuda_version,
        "gpu": gpu.name if gpu is not None else _NONE,
        "gpu_capability": gpu.sm if gpu is not None else _NONE,
    }
    for package in packages:
        record[package] = _distribution_version(package)
    for key, value in (extra or {}).items():
        record[str(key)] = str(value)
    return record


def kit_provenance(
    *,
    commit: str | None,
    levers_applied: Iterable[object],
    levers_fallback: Iterable[object],
    partial: bool,
) -> dict[str, str]:
    """The ``metadata.runtime`` entries every kit family records in a kit mode, under one set of names.

    ESMFold2, Protenix and OpenDDE all write these keys (:data:`KIT_RUNTIME_KEYS`) in the same format, so a caller can
    check which kit commit and which levers served a run without per-family lookups. Family-specific detail keeps
    its own keys.

    Parameters
    ----------
    commit : str | None
        The kit's source commit, or ``None`` when it is not known.
    levers_applied : Iterable[object]
        The levers the kit reported as serving.
    levers_fallback : Iterable[object]
        The levers the kit reported as fallen back to the stock path.
    partial : bool
        Whether the kit reported a partial lever set.

    Returns
    -------
    dict[str, str]
        ``kit.commit`` (``"unknown"`` without one), ``kit.levers_applied`` and ``kit.levers_fallback`` (comma-joined,
        ``"none"`` when empty) and ``kit.partial`` (``"true"`` or ``"false"``).
    """
    return {
        "kit.commit": commit or _UNKNOWN,
        "kit.levers_applied": _joined(levers_applied),
        "kit.levers_fallback": _joined(levers_fallback),
        "kit.partial": "true" if partial else "false",
    }


def _joined(levers: Iterable[object], separator: str = ",") -> str:
    """Join lever names (or notes) with ``separator``, or ``"none"`` for none."""
    return separator.join(str(lever) for lever in levers) or _NONE


def _distribution_version(name: str) -> str:
    """Return the installed version of distribution ``name``, or ``"absent"``."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return _ABSENT


def _boileroom_version() -> str:
    """Return the boileroom version: the source tree's ``pyproject.toml`` if present, else the installed one."""
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    try:
        return str(tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["version"]).strip()
    except (OSError, KeyError, tomllib.TOMLDecodeError):
        pass
    version = _distribution_version("boileroom")
    return _UNKNOWN if version == _ABSENT else version
