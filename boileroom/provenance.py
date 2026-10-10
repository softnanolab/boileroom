"""Runtime provenance for prediction metadata: which code, image, stack and GPU produced an output.

Standard library only, so any core (and the lightweight wrappers) can call it without pulling in model dependencies.
"""

from __future__ import annotations

import dataclasses
import importlib.metadata
import logging
import os
import platform
import subprocess
import sys
import tomllib
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .optimization import GpuInfo

logger = logging.getLogger(__name__)

#: Environment variable a runtime image sets to its own reference (e.g. ``docker.io/org/boileroom-x:cuda12.6-0.3.1``).
IMAGE_REF_ENV = "BOILEROOM_IMAGE_REF"
_UNKNOWN = "unknown"
_NOT_LOADED = "not-loaded"
_ABSENT = "absent"
_NONE = "none"
#: ``metadata.runtime`` keys every kit family records in a kit mode (see :func:`kit_provenance`).
KIT_RUNTIME_KEYS: tuple[str, ...] = ("kit.commit", "kit.levers_applied", "kit.levers_fallback", "kit.partial")
#: ``metadata.runtime`` keys of the GPU memory a backend records after every call (see :func:`gpu_memory_facts`).
GPU_MEMORY_KEYS: tuple[str, ...] = (
    "gpu.mem.used_mib",
    "gpu.mem.total_mib",
    "gpu.mem.allocated_mib",
    "gpu.mem.reserved_mib",
)
_MIB = 2**20


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


def gpu_memory_facts() -> dict[str, str]:
    """Return the GPU memory in use now, as ``metadata.runtime`` entries in whole MiB.

    Best effort: never raises and never imports torch.

    Returns
    -------
    dict[str, str]
        ``gpu.mem.used_mib`` and ``gpu.mem.total_mib`` for the whole device, every process on it included (a model
        worker child's too): from torch's current device when this process has initialised CUDA, else from
        ``nvidia-smi`` for the first device in ``CUDA_VISIBLE_DEVICES`` (device 0 without it). When this process has
        initialised CUDA, also ``gpu.mem.allocated_mib`` and ``gpu.mem.reserved_mib``: the memory of its live tensors and
        all the memory its torch caching allocator holds. Empty when no GPU memory can be read.
    """
    cuda = getattr(sys.modules.get("torch"), "cuda", None)
    try:
        if cuda is not None and cuda.is_initialized():
            free, total = cuda.mem_get_info()
            return {
                "gpu.mem.used_mib": str((total - free) // _MIB),
                "gpu.mem.total_mib": str(total // _MIB),
                "gpu.mem.allocated_mib": str(cuda.memory_allocated() // _MIB),
                "gpu.mem.reserved_mib": str(cuda.memory_reserved() // _MIB),
            }
    except Exception:  # a CUDA context that cannot answer: nvidia-smi still can
        pass
    return _nvidia_smi_memory()


def _nvidia_smi_memory() -> dict[str, str]:
    """Read ``gpu.mem.used_mib`` and ``gpu.mem.total_mib`` of the first visible device through ``nvidia-smi``."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    selector = "0" if visible is None else visible.split(",")[0].strip()
    # "" and "-1" hide every GPU.
    if not selector or selector.startswith("-"):
        return {}
    try:
        line = subprocess.run(
            ["nvidia-smi", f"--id={selector}", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        used, total = (int(value) for value in line.splitlines()[0].split(","))
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return {}
    return {"gpu.mem.used_mib": str(used), "gpu.mem.total_mib": str(total)}


def record_gpu_memory(output: object) -> None:
    """Add :func:`gpu_memory_facts` to ``output.metadata.runtime``.

    The metadata is replaced, not changed in place, since a core may hand the same metadata object to several outputs.
    Nothing changes when no GPU memory can be read or ``output`` carries no metadata with a ``runtime`` field.

    Parameters
    ----------
    output : object
        A core's output, normally with a ``metadata`` :class:`~boileroom.base.PredictionMetadata`.
    """
    metadata = getattr(output, "metadata", None)
    if not dataclasses.is_dataclass(metadata) or isinstance(metadata, type) or not hasattr(metadata, "runtime"):
        return
    facts = gpu_memory_facts()
    if not facts:
        return
    try:
        output.metadata = dataclasses.replace(metadata, runtime={**(metadata.runtime or {}), **facts})  # type: ignore[attr-defined]
    except (AttributeError, TypeError, ValueError) as error:  # a frozen output: its prediction still stands
        logger.warning(f"GPU memory not recorded on {type(output).__name__}: {error}")


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
