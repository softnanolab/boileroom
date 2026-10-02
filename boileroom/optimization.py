"""Static ``optimization`` option: vanilla | exact | fast, resolved per GPU.

``exact`` and ``fast`` drive the kits of anthropics/uplifting-biomolecular-modeling (Apache-2.0).
A kit mode is all of its levers on a GPU class, so a card the kit cannot fully serve is refused
by name here, before any weights load, instead of the kit's own ``os._exit(3)``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

OPTIMIZATION_MODES = ("vanilla", "exact", "fast")
DEFAULT_OPTIMIZATION = "vanilla"

# Compute capabilities on which each kit family was benchmarked here; the kit's configs exist per card.
_KIT_CONFIG_BY_CAPABILITY: dict[str, dict[tuple[int, int], str]] = {
    "esmfold2": {(8, 0): "a100", (9, 0): "h100"},
    "protenix": {(8, 0): "a100", (9, 0): "h100"},
    "opendde": {(8, 0): "a100", (9, 0): "h100"},
}
# Card names the kit config was validated on; H200 is the same Hopper (GH100) silicon Modal serves for H100 requests.
_KIT_CARD_NAMES: dict[str, tuple[str, ...]] = {"a100": ("a100",), "h100": ("h100", "h200")}
_UNSERVED_REASON: dict[str, str] = {
    "esmfold2": (
        "the kit refuses a partial lever set on sm89 (e.g. fast: t10 needs 168 KB of shared memory and t3 has no "
        "tile table; exact: t3 and t6 (>99 KB shared memory))"
    ),
    "protenix": "the kit has no BLK2 launch cells for sm89 (blk2_block_path, blk2_chunked_*), so it refuses the mode",
    "opendde": "the kit ships configs only for a100, h100 and h200 (no sm89 launch cells)",
}


class OptimizationUnavailableError(RuntimeError):
    """The requested optimization mode cannot run on this GPU or stack."""


def initialize_core(core: Any) -> RuntimeError | None:
    """Run ``core._initialize()``; in a kit mode, return a refusal instead of raising it.

    A Modal wrapper calls this from ``@modal.enter()``. An exception raised there crashes the container and Modal
    restarts it until the call times out, so the caller never sees why a kit mode was refused (wrong GPU, no kit in
    the image, kit inactive). The wrapper raises the returned error from ``fold()`` instead. Vanilla failures raise
    as before.
    """
    try:
        core._initialize()
    except RuntimeError as error:
        if core.config.get("optimization", DEFAULT_OPTIMIZATION) == DEFAULT_OPTIMIZATION:
            raise
        return error
    return None


@dataclass(frozen=True)
class GpuInfo:
    """The device an optimization mode is resolved against."""

    name: str
    capability: tuple[int, int]


@dataclass(frozen=True)
class OptimizationResolution:
    """Outcome of resolving an ``optimization`` option against a GPU."""

    requested: str
    active: str
    kit_config: str | None
    gpu_name: str | None
    capability: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record for output metadata."""
        return asdict(self)


def validate_optimization(mode: object) -> str:
    """Return ``mode`` if it is a known optimization mode, else raise ``ValueError``."""
    if mode not in OPTIMIZATION_MODES:
        raise ValueError(f"optimization must be one of {list(OPTIMIZATION_MODES)}, got {mode!r}")
    return str(mode)


def detect_gpu(device: str | None = None) -> GpuInfo:
    """Read name and compute capability of the CUDA device the model will run on."""
    index = int(device.split(":")[1]) if device and device.startswith("cuda:") else 0
    try:
        import torch
    except ImportError:
        return _detect_gpu_nvidia_smi(index)
    if not torch.cuda.is_available():
        raise OptimizationUnavailableError("optimization exact/fast needs a CUDA GPU; none is visible")
    return GpuInfo(name=torch.cuda.get_device_name(index), capability=torch.cuda.get_device_capability(index))


def _detect_gpu_nvidia_smi(index: int) -> GpuInfo:
    import subprocess

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
        name, cap = (part.strip() for part in out[index].rsplit(",", 1))
        major, minor = cap.split(".")
    except (OSError, subprocess.CalledProcessError, IndexError, ValueError) as error:
        raise OptimizationUnavailableError(
            "optimization exact/fast could not read the GPU (nvidia-smi failed)"
        ) from error
    return GpuInfo(name=name, capability=(int(major), int(minor)))


def resolve_optimization(family: str, mode: str, gpu: GpuInfo | None = None) -> OptimizationResolution:
    """Resolve ``mode`` for a model family on ``gpu``; raise by name when it cannot be served."""
    validate_optimization(mode)
    if mode == "vanilla":
        return OptimizationResolution(mode, "vanilla", None, gpu.name if gpu else None, None)
    if family not in _KIT_CONFIG_BY_CAPABILITY:
        raise OptimizationUnavailableError(f"optimization={mode!r} has no kit for model family {family!r}")
    if gpu is None:
        raise OptimizationUnavailableError(f"optimization={mode!r} for {family} needs a GPU to resolve against")
    sm = f"sm{gpu.capability[0]}{gpu.capability[1]}"
    config = _KIT_CONFIG_BY_CAPABILITY[family].get(gpu.capability)
    if config is not None and not any(card in gpu.name.lower() for card in _KIT_CARD_NAMES[config]):
        config = None  # same compute capability, different card (e.g. A30 is sm80): the kit was not validated on it
    if config is None:
        served = ", ".join(f"sm{a}{b}" for a, b in _KIT_CONFIG_BY_CAPABILITY[family])
        why = _UNSERVED_REASON[family] if gpu.capability == (8, 9) else f"the kit is only validated on {served}"
        raise OptimizationUnavailableError(
            f"optimization={mode!r} cannot run {family} on {gpu.name} ({sm}): {why}. "
            f"Use optimization='vanilla' here, or a GPU with {served}."
        )
    return OptimizationResolution(mode, mode, config, gpu.name, sm)
