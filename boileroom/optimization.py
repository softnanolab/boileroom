"""Static ``optimization`` option: vanilla | exact | fast, resolved per GPU.

``exact`` and ``fast`` drive the kits of anthropics/uplifting-biomolecular-modeling (Apache-2.0).
A kit mode is all of its levers on a GPU class, so a card the kit cannot fully serve is refused
by name here, before any weights load, instead of the kit's own ``os._exit(3)``.
"""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, Generic, TypeVar

OPTIMIZATION_MODES = ("vanilla", "exact", "fast")
DEFAULT_OPTIMIZATION = "vanilla"

#: Model families that have an optimization kit; every other family runs ``vanilla`` only.
KIT_FAMILIES: frozenset[str] = frozenset({"esmfold2", "protenix", "opendde"})
# Compute capabilities the kits were benchmarked on here, mapped to the kit's per-card config.
_KIT_CONFIG_BY_CAPABILITY: dict[tuple[int, int], str] = {(8, 0): "a100", (9, 0): "h100"}
# Card names each kit config was validated on, matched as whole tokens of the device name; H200 is the same Hopper
# (GH100) silicon Modal serves for H100 requests. GH200 (Grace Hopper, aarch64) is a different card and is not served.
_KIT_CARD_NAMES: dict[str, tuple[str, ...]] = {"a100": ("a100",), "h100": ("h100", "h200")}
_SERVED_CARDS = "NVIDIA A100 (sm80), H100 or H200 (sm90)"
_UNSERVED_REASON: dict[str, str] = {
    "esmfold2": (
        "the kit refuses a partial lever set on sm89 (e.g. fast: t10 needs 168 KB of shared memory and t3 has no "
        "tile table; exact: t3 and t6 (>99 KB shared memory))"
    ),
    "protenix": "the kit has no BLK2 launch cells for sm89 (blk2_block_path, blk2_chunked_*), so it refuses the mode",
    "opendde": "the kit ships configs only for a100, h100 and h200 (no sm89 launch cells)",
}
#: Exit status of a *process* that refused the optimization mode: the boileroom worker/server contract. A kit that
#: refuses at interpreter start ends the model worker's child with ``os._exit(3)``, and the Apptainer model server
#: exits with this code when its core refuses (any other load failure exits 1). It is the only refusal status a
#: process reports, since the kits' in-process refusal ``SystemExit`` codes (:data:`KIT_REFUSAL_EXIT_CODES`) are mapped
#: before they can end one.
REFUSAL_PROCESS_EXIT_CODE = 3
#: ``SystemExit`` codes a kit call raises *in-process* for a refusal: 3 (not active) and 5 (OpenDDE's kernel census,
#: ``opendde_opt.lncensus.KernelsRefused``, a ``SystemExit`` subclass). Any other code (1 = failed, 2 = usage) is a
#: failure. The worker files, which cannot import boileroom, keep literal copies (``tests/contracts/
#: test_kit_refusal_sets.py`` holds them equal).
KIT_REFUSAL_EXIT_CODES: frozenset[int] = frozenset({3, 5})
#: Exception classes the kits raise for a refusal, matched by name anywhere in the exception's MRO (the kits' own
#: subclasses, such as OpenDDE's ``BigRefusal`` of ``OpenModeError``, included). Only ``Exception`` subclasses belong
#: here: ``KernelsRefused`` is a ``SystemExit`` and is a refusal through its exit code (5).
KIT_REFUSAL_CLASS_NAMES: frozenset[str] = frozenset({"ActivationError", "NotLoaded", "OpenModeError"})


def is_kit_refusal_exception(error: BaseException) -> bool:
    """Return whether ``error`` is one of the kits' refusal exceptions (:data:`KIT_REFUSAL_CLASS_NAMES`, by MRO name).

    Parameters
    ----------
    error : BaseException
        The exception a kit call raised.

    Returns
    -------
    bool
        ``True`` when a class in ``type(error).__mro__`` is named in :data:`KIT_REFUSAL_CLASS_NAMES`.
    """
    return any(cls.__name__ in KIT_REFUSAL_CLASS_NAMES for cls in type(error).__mro__)


def _is_kit_refusal_code(code: object) -> bool:
    """Whether a ``SystemExit`` code is one of :data:`KIT_REFUSAL_EXIT_CODES` (a code may be any object, unhashable too)."""
    return isinstance(code, int) and code in KIT_REFUSAL_EXIT_CODES


CoreT = TypeVar("CoreT")


class OptimizationUnavailableError(RuntimeError):
    """The requested optimization mode cannot run on this GPU or stack."""


def is_refusal(error: BaseException) -> bool:
    """Return whether ``error`` refuses the optimization mode rather than reporting a failure.

    A refusal is an exception whose class is named ``OptimizationUnavailableError`` (matched by name, because a kit
    runtime cannot import boileroom and raises a local class of that name) or a kit's in-process refusal ``SystemExit``
    (a code in :data:`KIT_REFUSAL_EXIT_CODES`). The model worker's child keeps its own copy of this rule, since it
    never imports boileroom.

    Parameters
    ----------
    error : BaseException
        The exception to classify.

    Returns
    -------
    bool
        ``True`` for a refusal.
    """
    if isinstance(error, SystemExit):
        return _is_kit_refusal_code(error.code)
    return type(error).__name__ == OptimizationUnavailableError.__name__


def initialize_core(core: Any) -> Exception | None:
    """Run ``core._initialize()`` and return the failure instead of raising it.

    A Modal wrapper calls this (through :class:`GuardedCore`) from ``@modal.enter()``. An exception raised there crashes
    the container and Modal restarts it until the call times out, so the caller never sees why a load failed (wrong GPU,
    no kit in the image, kit inactive, weights that do not load). The wrapper keeps the returned error and raises it
    from the call instead; see :func:`retry_initialize`.

    A ``SystemExit`` is returned as an exception chained from it: a kit's refusal code (:data:`KIT_REFUSAL_EXIT_CODES`)
    as a permanent :class:`OptimizationUnavailableError`, any other code as a ``RuntimeError`` naming it.
    ``KeyboardInterrupt`` propagates.
    """
    try:
        core._initialize()
    except SystemExit as exit_:
        return _exit_failure(type(core).__name__, exit_)
    except Exception as error:
        return error
    return None


def _exit_failure(owner: str, exit_: SystemExit) -> Exception:
    """Turn a ``SystemExit`` raised while ``owner`` loaded into the exception the caller sees, chained from it."""
    code = exit_.code
    if _is_kit_refusal_code(code):
        failure: Exception = OptimizationUnavailableError(
            f"{owner} load refused: the optimization kit exited with code {code}"
        )
    else:
        detail = f"exited with code {code}" if code is None or isinstance(code, int) else f"exited: {code}"
        failure = RuntimeError(f"{owner} load failed: it {detail}")
    failure.__cause__ = exit_
    return failure


def retry_initialize(core: Any, failure: Exception | None) -> Exception | None:
    """Return the standing load failure of ``core``, trying the load again unless the failure is a permanent one.

    A refusal (:class:`OptimizationUnavailableError`) or a bad config (``ValueError``) cannot change within a container,
    so it stands. Anything else (a download that dropped, a full disk) gets another attempt on the next call.
    """
    if failure is None or isinstance(failure, OptimizationUnavailableError | ValueError):
        return failure
    return initialize_core(core)


class GuardedCore(Generic[CoreT]):
    """A core built and loaded in ``@modal.enter()`` whose failure is raised from the calls instead.

    Anything raised in ``@modal.enter()`` puts the container into Modal's silent restart loop, so construction (where
    the config is validated) and the load both run here and only keep their failure. A construction failure always
    stands; a load failure is retried as :func:`retry_initialize` decides.

    Parameters
    ----------
    factory : Callable[[], CoreT]
        Builds the core, e.g. ``lambda: ESMFold2Core(config)``.

    Attributes
    ----------
    core : CoreT | None
        The core, or ``None`` if its construction failed.
    failure : Exception | None
        The standing construction or load failure.
    """

    def __init__(self, factory: Callable[[], CoreT]) -> None:
        self.core: CoreT | None = None
        self.failure: Exception | None = None
        try:
            self.core = factory()
        except SystemExit as exit_:
            self.failure = _exit_failure("core construction", exit_)
        except Exception as error:
            self.failure = error
        else:
            self.failure = initialize_core(self.core)

    def get(self) -> CoreT:
        """Return the loaded core, retrying a transient load failure; raise the failure that stands."""
        if self.core is not None:
            self.failure = retry_initialize(self.core, self.failure)
        if self.failure is not None:
            raise self.failure
        assert self.core is not None
        return self.core

    def close(self) -> None:
        """Close the core if it was built and has a ``close()``."""
        close = getattr(self.core, "close", None)
        if callable(close):
            close()


@dataclass(frozen=True)
class GpuInfo:
    """The device an optimization mode is resolved against."""

    name: str
    capability: tuple[int, int]

    @property
    def sm(self) -> str:
        """Compute capability as ``"sm80"``."""
        return f"sm{self.capability[0]}{self.capability[1]}"


@dataclass(frozen=True)
class OptimizationResolution:
    """Outcome of resolving an ``optimization`` option against a GPU.

    A mode that cannot be served raises instead of downgrading, so ``mode`` is both what was asked for and what runs.
    """

    mode: str
    kit_config: str | None
    gpu_name: str | None
    capability: str | None

    @property
    def kit(self) -> bool:
        """Whether a kit mode (``exact`` or ``fast``) runs."""
        return self.mode != "vanilla"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record for output metadata."""
        return asdict(self)


def validate_optimization(mode: object) -> str:
    """Return ``mode`` if it is a known optimization mode, else raise ``ValueError``."""
    if mode not in OPTIMIZATION_MODES:
        raise ValueError(f"optimization must be one of {list(OPTIMIZATION_MODES)}, got {mode!r}")
    return str(mode)


def refuse_without_kit(name: str, mode: object) -> str:
    """Return ``mode`` for a model without an optimization kit, which runs only ``vanilla``.

    The one wording of this refusal, shared by the wrapper (before a backend starts) and the core.

    Parameters
    ----------
    name : str
        The model's display name, for the message.
    mode : object
        The requested ``optimization``.

    Returns
    -------
    str
        ``mode`` (always ``"vanilla"``).

    Raises
    ------
    ValueError
        If ``mode`` is not a known optimization mode.
    OptimizationUnavailableError
        If ``mode`` is a kit mode (``exact`` or ``fast``).
    """
    mode = validate_optimization(mode)
    if mode != DEFAULT_OPTIMIZATION:
        raise OptimizationUnavailableError(
            f"{name} has no optimization kit, so optimization={mode!r} cannot run; "
            f"use optimization={DEFAULT_OPTIMIZATION!r} or leave it unset"
        )
    return mode


def detect_gpu(device: str | None = None) -> GpuInfo:
    """Read name and compute capability of the CUDA device the model will run on.

    Raises
    ------
    OptimizationUnavailableError
        If ``device`` is not a CUDA device, no GPU is visible, or the GPU cannot be read.
    """
    device = None if device is None else str(device)
    if device is not None and not device.startswith("cuda"):
        raise OptimizationUnavailableError(f"optimization exact/fast runs on a CUDA GPU, not on device={device!r}")
    try:
        index = int(device.split(":")[1]) if device and device.startswith("cuda:") else 0
    except ValueError as error:
        raise OptimizationUnavailableError(f"cannot read a CUDA device index from device={device!r}") from error
    try:
        import torch
    except ImportError:
        return _detect_gpu_nvidia_smi(index)
    if not torch.cuda.is_available():
        raise OptimizationUnavailableError("optimization exact/fast needs a CUDA GPU; none is visible")
    try:
        return GpuInfo(name=torch.cuda.get_device_name(index), capability=torch.cuda.get_device_capability(index))
    except (AssertionError, RuntimeError) as error:
        raise OptimizationUnavailableError(f"could not read CUDA device {index}: {error}") from error


def describe_gpu(device: str | None = None) -> GpuInfo | None:
    """Return the CUDA device ``device`` runs on, or ``None`` when there is none or it cannot be read.

    Best effort, for provenance: never raises. ``None`` for a non-CUDA device, without CUDA, and when neither torch nor
    ``nvidia-smi`` can read the GPU.
    """
    if device is not None and not str(device).startswith("cuda"):
        return None
    try:
        return detect_gpu(device)
    except Exception:
        return None


def _detect_gpu_nvidia_smi(index: int) -> GpuInfo:
    """Read logical CUDA device ``index`` through ``nvidia-smi`` (for interpreters without torch).

    ``nvidia-smi`` lists physical GPUs, so the logical index is mapped through ``CUDA_VISIBLE_DEVICES`` (integer indices
    or ``GPU-`` UUID prefixes) the way the CUDA runtime does.
    """
    try:
        lines = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,name,compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        ).stdout.splitlines()
        rows = [_parse_nvidia_smi_row(line) for line in lines if line.strip()]
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        raise OptimizationUnavailableError(
            "optimization exact/fast could not read the GPU (nvidia-smi failed)"
        ) from error

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        selector = str(index)
    else:
        entries = [entry.strip() for entry in visible.split(",")]
        # CUDA reads the list up to its first invalid entry; "" and "-1" hide every GPU.
        valid = []
        for entry in entries:
            if not entry or entry.startswith("-"):
                break
            valid.append(entry)
        if index >= len(valid):
            raise OptimizationUnavailableError(
                f"CUDA device {index} is not visible (CUDA_VISIBLE_DEVICES={visible!r}); no GPU to resolve against"
            )
        selector = valid[index]
    if selector.startswith("MIG-"):
        raise OptimizationUnavailableError(
            f"CUDA device {selector} is a MIG slice, which the kits were not validated on"
        )
    matches = [row for row in rows if _matches_selector(row, selector)]
    if len(matches) != 1:
        raise OptimizationUnavailableError(
            f"nvidia-smi lists no single GPU for CUDA device {selector!r} (CUDA_VISIBLE_DEVICES={visible!r})"
        )
    # nvidia-smi numbers GPUs in PCI bus order. CUDA numbers them (CUDA_VISIBLE_DEVICES integers included) over every
    # GPU on the node, fastest first unless CUDA_DEVICE_ORDER=PCI_BUS_ID, so on a node with mixed cards an integer
    # selector may name another card than nvidia-smi's row of that index. A UUID selector is unambiguous.
    if (
        selector.isdigit()
        and os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID"
        and len({(row[2], row[3]) for row in rows}) > 1
    ):
        raise OptimizationUnavailableError(
            "this node mixes GPU models, so CUDA's device order may differ from nvidia-smi's; "
            "set CUDA_DEVICE_ORDER=PCI_BUS_ID or select the GPU by UUID in CUDA_VISIBLE_DEVICES"
        )
    _, _, name, capability = matches[0]
    return GpuInfo(name=name, capability=capability)


def _parse_nvidia_smi_row(line: str) -> tuple[str, str, str, tuple[int, int]]:
    """Parse one ``index, uuid, name, compute_cap`` row; the name may itself contain commas."""
    index, uuid, rest = (part.strip() for part in line.split(",", 2))
    name, cap = (part.strip() for part in rest.rsplit(",", 1))
    major, minor = cap.split(".")
    return index, uuid, name, (int(major), int(minor))


def _matches_selector(row: tuple[str, str, str, tuple[int, int]], selector: str) -> bool:
    """Whether a CUDA_VISIBLE_DEVICES entry (integer index or UUID prefix) names this nvidia-smi row."""
    index, uuid = row[0], row[1]
    return selector == index if selector.isdigit() else uuid.startswith(selector)


def _served_by(config: str, gpu: GpuInfo) -> bool:
    """Whether ``gpu`` is a card ``config`` was validated on, matched by whole name tokens (GH200 is not H200)."""
    tokens = set(re.findall(r"[a-z0-9]+", gpu.name.lower()))
    return any(card in tokens for card in _KIT_CARD_NAMES[config])


def resolve_optimization(family: str, mode: str, gpu: GpuInfo | None = None) -> OptimizationResolution:
    """Resolve ``mode`` for a model family on ``gpu``; raise by name when it cannot be served."""
    validate_optimization(mode)
    if mode == "vanilla":
        return OptimizationResolution("vanilla", None, gpu.name if gpu else None, gpu.sm if gpu else None)
    if family not in KIT_FAMILIES:
        raise OptimizationUnavailableError(f"optimization={mode!r} has no kit for model family {family!r}")
    if gpu is None:
        raise OptimizationUnavailableError(f"optimization={mode!r} for {family} needs a GPU to resolve against")
    config = _KIT_CONFIG_BY_CAPABILITY.get(gpu.capability)
    if config is None or not _served_by(config, gpu):
        # A card of a served capability that is not a validated card (A30 is sm80; H800 and GH200 are sm90) is refused
        # too: the kit was not validated on it.
        why = (
            _UNSERVED_REASON[family]
            if gpu.capability == (8, 9)
            else f"the kit is only validated on {_SERVED_CARDS} cards"
        )
        raise OptimizationUnavailableError(
            f"optimization={mode!r} cannot run {family} on {gpu.name} ({gpu.sm}): {why}. "
            f"Use optimization='vanilla' here, or an {_SERVED_CARDS} GPU."
        )
    return OptimizationResolution(mode, config, gpu.name, gpu.sm)
