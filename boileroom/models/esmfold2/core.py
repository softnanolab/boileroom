"""Core ESMFold2 implementation without Modal dependencies."""

from __future__ import annotations

import dataclasses
import importlib
import logging
import math
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any, ClassVar, cast

import numpy as np

from ...base import FoldingAlgorithm, PredictionMetadata
from ...inputs import a3m_rows
from ...optimization import (
    DEFAULT_OPTIMIZATION,
    GpuInfo,
    OptimizationResolution,
    OptimizationUnavailableError,
    describe_gpu,
    detect_gpu,
    is_kit_refusal_exception,
    is_refusal,
    resolve_optimization,
)
from ...provenance import _ABSENT, _UNKNOWN, _joined, kit_provenance, runtime_provenance
from ...utils import MODAL_MODEL_DIR, Timer, safe_mkdir, validate_sequence
from .._worker import release_evicted_kit_caches
from .payloads import decode_structure_input
from .types import (
    CovalentBond,
    DistogramConditioning,
    DNAInput,
    ESMFold2Output,
    LigandInput,
    Modification,
    MSAInput,
    PocketConditioning,
    ProteinInput,
    RNAInput,
    StructurePredictionInput,
)

logger = logging.getLogger(__name__)

SequenceInput = ProteinInput | RNAInput | DNAInput | LigandInput
ESMFold2FoldInput = (
    str
    | Sequence[str]
    | StructurePredictionInput
    | Sequence[StructurePredictionInput]
    | Sequence[SequenceInput]
    | Mapping[str, Any]
    | Sequence[Mapping[str, Any]]
)


@dataclass(frozen=True)
class _FoldRequest:
    """One normalized fold request plus its sequence-length metadata."""

    input: StructurePredictionInput
    sequence_length: int


# Biohub re-bundled biohub/ESMFold2 in place on 2026-09-14, so unpinned loads silently
# changed checkpoint layout under existing images. Pin the default model's snapshot.
ESMFOLD2_HF_REPO = "biohub/ESMFold2"
ESMFOLD2_HF_REVISION = "69869f737beffec5294845ede23db5fc0b4f509e"
#: Under the model directory: the HF_HOME of the kit modes, which load their own pinned snapshots (not ESMFOLD2_HF_REVISION).
KIT_HF_SUBDIR = "esmfold2/kit-hf"
#: The kit's fail-loud switch: ``1`` makes it refuse (exit 3) instead of falling back to its slow attention and MLP paths.
#: Kit modes always set it before importing the kit; the kit image sets it too. Vanilla never sets it.
KIT_REQUIRE_FAST_ENV = "ESMFOLD2_OPT_REQUIRE_FAST_ENV"
#: Where the kit memoizes its weight digests, keyed by file path, size, mtime and inode; a miss only re-hashes. Kit modes
#: default it to :data:`KIT_WEIGHTS_MEMO_SUBDIR` under the model directory so a cold start does not re-hash ~27 GB.
KIT_WEIGHTS_MEMO_ENV = "ESMFOLD2_OPT_WEIGHTS_MEMO_DIR"
KIT_WEIGHTS_MEMO_SUBDIR = "esmfold2/kit-digests"
#: Set by the kit image (``kit/Dockerfile``): the kit commit and the compiled stack, recorded in ``metadata.runtime``.
#: ``protenix/runtime.py`` keeps its own copy of the commit name because it must not import boileroom (Python 3.10
#: worker); tests/contracts/test_kit_dockerfiles.py asserts the two are equal.
KIT_COMMIT_ENV = "BOILEROOM_KIT_COMMIT"
KIT_STACK_ENV = "BOILEROOM_KIT_STACK"
#: Attributes of the installed ``esmfold2_opt`` that may carry its source commit, read before :data:`KIT_COMMIT_ENV`
#: in the order ``protenix/runtime.py`` reads them.
KIT_COMMIT_ATTRIBUTES = ("__commit__", "KIT_COMMIT", "COMMIT")
#: The compute capabilities each kit stack (``kit/kit_wheels.sh``'s ``STACK``) compiles its CUDA kernels for. A known
#: stack on a card outside its set is refused before any weight is fetched: the kernels would not launch there.
KIT_STACK_CAPABILITIES: dict[str, frozenset[tuple[int, int]]] = {
    "img_ef2_fa": frozenset({(9, 0)}),
    "img_esmfold2_a100": frozenset({(8, 0), (9, 0)}),
}
#: Sampler controls that the kit's esm 3.3.0 ``fold()`` does not apply the way vanilla esm 3.4.1 does; refused in kit modes.
KIT_REFUSED_SAMPLER_KEYS = ("noise_scale", "step_scale", "max_inference_sigma")
#: Distributions recorded in ``metadata.runtime`` (their installed versions, or ``"absent"``).
VANILLA_RUNTIME_PACKAGES = ("esm", "transformers", "flash-attn", "transformer-engine", "xformers")
KIT_RUNTIME_PACKAGES = ("esmfold2-opt", *VANILLA_RUNTIME_PACKAGES)
#: esm 3.4.1's import-time kernel switches, recorded (not enforced) for vanilla as ``esm.<FLAG>``: the stock image runs
#: whatever these say, which differs between environments that install the optional kernels and those that do not.
VANILLA_KERNEL_FLAGS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("esm.models.esmfold2.layers", ("FLASH_ATTN_AVAILABLE", "CUE_AVAILABLE", "TRITON_KERNELS_AVAILABLE")),
    ("esm.models.esmfold2.model", ("TE_AVAILABLE",)),
    (
        "esm.models.esmc.kernels",
        ("TE_INSTALLED", "XFORMERS_INSTALLED", "FLASH_ATTN_INSTALLED", "FLASH_ATTN_ROTARY_INSTALLED"),
    ),
)
#: The attention / MLP words a kit report carries, each recorded in ``metadata.runtime`` as ``kit.attn.<word>``.
KIT_ATTN_WORDS = ("atom_attn", "esmc_mlp", "esmc_attn", "esmc_rope")
#: Heavy atoms of an unmodified residue, as both esm builders (3.3.0 and 3.4.1) tokenize it: ``PROTEIN_HEAVY_ATOMS``,
#: ``DNA_HEAVY_ATOMS`` and ``RNA_HEAVY_ATOMS`` in ``esm.models.esmfold2.constants``, by one-letter code. A covalent bond's
#: atom index is bounded by these before either builder sees it (esm 3.3.0 drops an out-of-range bond silently). A
#: protein letter outside the table becomes ``UNK``, whose four backbone atoms esm builds; an unknown nucleotide letter
#: has no static count.
# fmt: off
PROTEIN_RESIDUE_ATOM_COUNTS: dict[str, int] = {
    "A": 5, "R": 11, "N": 8, "D": 8, "C": 6, "Q": 9, "E": 9, "G": 4, "H": 10, "I": 8,
    "L": 8, "K": 9, "M": 8, "F": 11, "P": 7, "S": 6, "T": 7, "W": 14, "Y": 12, "V": 7,
}
# fmt: on
UNKNOWN_PROTEIN_RESIDUE_ATOMS = 4
DNA_RESIDUE_ATOM_COUNTS: dict[str, int] = {"A": 21, "G": 22, "C": 19, "T": 20}
RNA_RESIDUE_ATOM_COUNTS: dict[str, int] = {"A": 22, "G": 23, "C": 20, "U": 20}
#: Hugging Face's offline switches: the environment variables, and the module attributes they are frozen into on import.
HF_OFFLINE_ENV = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
HF_OFFLINE_ATTRIBUTES = (
    ("huggingface_hub.constants", "HF_HUB_OFFLINE"),
    ("transformers.utils.hub", "_is_offline_mode"),
)


def _polymer_residue_atoms(item: SequenceInput, res_idx: int) -> tuple[str, int] | None:
    """The name and heavy-atom count of an unmodified protein, DNA or RNA residue, or ``None`` without a static count."""
    if isinstance(item, LigandInput):
        return None
    if any(modification.position == res_idx for modification in item.modifications or []):
        return None
    letter = item.sequence[res_idx]
    if isinstance(item, ProteinInput):
        return f"protein {letter}", PROTEIN_RESIDUE_ATOM_COUNTS.get(letter, UNKNOWN_PROTEIN_RESIDUE_ATOMS)
    table = DNA_RESIDUE_ATOM_COUNTS if isinstance(item, DNAInput) else RNA_RESIDUE_ATOM_COUNTS
    if letter not in table:
        return None
    return f"{'DNA' if isinstance(item, DNAInput) else 'RNA'} {letter}", table[letter]


def _atom_past_residue(index: int, atom_idx: int, res_idx: int, chain_id: str, residue: str, count: int) -> ValueError:
    """The refusal of a covalent bond whose atom index is past the atoms esm builds for its residue."""
    return ValueError(
        f"ESMFold2 covalent bond {index}: atom index {atom_idx} is past residue {res_idx} of chain {chain_id!r} "
        f"({residue}: {count} atoms, 0-based 0-{count - 1})"
    )


def _kit_commit(kit: Any) -> str | None:
    """The kit's source commit: its own attribute (:data:`KIT_COMMIT_ATTRIBUTES`), else ``$BOILEROOM_KIT_COMMIT``.

    The same order as ``protenix/runtime.py``, without its ``.git`` lookup: the ESMFold2 kit image installs the kit
    without its repository, and a ``.git`` above an installed package would belong to some other project.

    Parameters
    ----------
    kit : Any
        The imported ``esmfold2_opt`` module, or ``None``.

    Returns
    -------
    str | None
        The commit, or ``None`` when neither source names one.
    """
    for attribute in KIT_COMMIT_ATTRIBUTES:
        value = getattr(kit, attribute, None)
        if isinstance(value, str) and value:
            return value
    return os.environ.get(KIT_COMMIT_ENV) or None


class ESMFold2Core(FoldingAlgorithm):
    """Biohub ESMFold2 all-atom structure prediction model.

    ``optimization="vanilla"`` (default) runs esm 3.4.1's ``EsmFold2Model`` from the stock image. ``"exact"`` and
    ``"fast"`` run the optimization kit (``esmfold2_opt``) on its own image: the transformers-fork ``ESMFold2Model``,
    folded through esm 3.3.0's ``ESMFold2InputBuilder.fold()``. Kit modes refuse (``OptimizationUnavailableError``)
    rather than fall back to slow attention / MLP kernels, and every output records what ran in ``metadata.runtime``.

    Config notes
    ------------
    kit_msa : bool
        Kit modes only, fixed at construction: ``True`` arms the ``full_msa`` kernel variant (full checkpoint), the
        default ``False`` arms ``full_nomsa``; the Fast checkpoint always arms ``fast``. Both full variants consume a
        user MSA, so ``options["msa"]`` is accepted with either; ``kit_msa`` selects the kernels, not whether an MSA is
        used.

    An MSA (inline on an input entry or from ``options["msa"]``) is refused in every mode when the loaded checkpoint has
    no MSA encoder (ESMFold2-Fast), which would otherwise ignore it.

    Kit-mode differences from vanilla (esm 3.3.0 builder versus esm 3.4.1):

    - ``noise_scale``, ``step_scale`` and ``max_inference_sigma`` are refused; ``lm_mask_pct`` is forwarded.
    - ``msa_max_depth`` must be an integer: ``None`` means the checkpoint's own depth under esm 3.4.1 but every row
      (no subsampling) under the kit's esm 3.3.0, so it is refused rather than read two ways.
    - Covalent bonds are validated here before either builder sees them (esm 3.3.0 silently drops a bond that names
      a missing chain, residue or atom). Atom indices are bounded for unmodified protein / DNA / RNA residues and CCD
      ligands; on a modified residue, an unknown nucleotide or a SMILES ligand, a past-the-end atom index is still
      only caught by esm 3.4.1.
    - esm 3.3.0 deduplicates entities by sequence only; 3.4.1 also keys them on modifications, so two copies of a
      sequence with different modifications are one entity under the kit and two under vanilla.
    - Timing: the kit's ``fold()`` prepares, runs and decodes in one call, so ``inference_time`` covers the whole fold
      and ``preprocessing_time`` / ``postprocessing_time`` are ``0.0``.
    """

    DEFAULT_CONFIG: ClassVar[dict[str, Any]] = {
        "device": "cuda:0",
        "model_name": ESMFOLD2_HF_REPO,
        # None pins biohub/ESMFold2 to ESMFOLD2_HF_REVISION; other checkpoints load their latest snapshot.
        "revision": None,
        "cache_dir": None,
        "ccd_cache_dir": None,
        "dtype": None,
        "optimization": "vanilla",
        "kit_msa": False,
        "num_loops": 3,
        "num_sampling_steps": 50,
        "num_diffusion_samples": 1,
        "seed": None,
        "noise_scale": None,
        "step_scale": None,
        "max_inference_sigma": None,
        "lm_mask_pct": None,
        "msa_max_depth": 1024,
        "msa_column_mask_rate": 0.1,
        "complex_id": "pred",
        "include_fields": None,
        "msa": None,
        "templates": None,
    }
    STATIC_CONFIG_KEYS: ClassVar[frozenset[str]] = frozenset(
        {"device", "model_name", "revision", "cache_dir", "ccd_cache_dir", "dtype", "optimization", "kit_msa"}
    )
    #: ``options["msa"]`` is one A3M text (or None) per entry of the input's ``sequences``; templates are not supported.
    SUPPORTS_USER_MSA: ClassVar[bool] = True
    SUPPORTS_USER_TEMPLATES: ClassVar[bool] = False

    def __init__(self, config: dict | None = None) -> None:
        """Create an ESMFold2 core instance."""
        super().__init__(config)
        self.optimization: OptimizationResolution | None = None
        self._metadata_template = self._initialize_metadata(
            model_name="ESMFold2",
            model_version=str(self.config["model_name"]),
        )
        self.model_dir: str | None = os.environ.get("MODEL_DIR", MODAL_MODEL_DIR)
        self._device: Any | None = None
        self.model: Any | None = None
        self.input_builder: Any | None = None
        self._gpu: GpuInfo | None = None
        #: Provenance of the loaded runtime, attached to every output as ``metadata.runtime``.
        self._runtime: dict[str, str] | None = None
        #: Set only once the armed kit reported its accelerated kernels on the loaded model; the kit fold needs it.
        self._kernel_gate_passed = False
        #: The attention paths lost after a fold. That refusal stands, so later folds raise it before reaching the GPU.
        self._fold_refusal: OptimizationUnavailableError | None = None

    def _initialize(self) -> None:
        """Load the ESMFold2 model and input builder."""
        self._load()

    def _validate_config(self, config: Mapping[str, Any]) -> None:
        """Refuse config this core cannot honour, at construction and on every merged per-call config.

        Parameters
        ----------
        config : Mapping[str, Any]
            A full (merged) configuration.

        Raises
        ------
        ValueError
            On the shared checks (unknown keys), an unknown ``optimization``, a non-bool ``kit_msa``, invalid inference
            options, or, in kit modes, a sampler override the kit would not apply, ``msa_max_depth=None``, a
            ``revision`` / ``cache_dir`` / ``ccd_cache_dir`` (the kit loads its own pinned snapshots) or a
            ``model_name`` the kit does not serve.
        """
        super()._validate_config(config)
        mode = config.get("optimization", DEFAULT_OPTIMIZATION)
        if not isinstance(config.get("kit_msa", False), bool):
            raise ValueError(f"ESMFold2 config 'kit_msa' must be True or False, got {config.get('kit_msa')!r}.")
        self._validate_effective_config(config)
        if mode == "vanilla":
            return
        for key in ("revision", "cache_dir", "ccd_cache_dir"):
            if config.get(key) is not None:
                raise ValueError(
                    f"ESMFold2 config {key!r} does not apply to optimization={mode!r}, which loads pinned snapshots "
                    f"from {KIT_HF_SUBDIR}; use optimization='vanilla' to choose your own."
                )
        served = (ESMFOLD2_HF_REPO, f"{ESMFOLD2_HF_REPO}-Fast")
        if config.get("model_name") not in served:
            raise ValueError(
                f"optimization={mode!r} serves {list(served)}, not model_name={config.get('model_name')!r}"
            )
        overridden = [key for key in KIT_REFUSED_SAMPLER_KEYS if config.get(key) is not None]
        if overridden:
            raise ValueError(
                f"ESMFold2 optimization={mode!r} folds with the kit's own sampler schedule and would not apply "
                f"{overridden}; leave them unset or use optimization='vanilla'."
            )
        if config.get("msa_max_depth") is None:
            raise ValueError(
                f"ESMFold2 optimization={mode!r} needs an integer 'msa_max_depth': None means the checkpoint's own "
                "depth under vanilla's esm 3.4.1 but every row (no subsampling) under the kit's esm 3.3.0. Pass a "
                "depth, or use optimization='vanilla'."
            )

    def _resolve_cache_dir(self, config_key: str, default_subdir: str) -> Path:
        """Resolve and create a configured or model-volume cache directory."""
        configured = self.config.get(config_key)
        if configured is not None:
            cache_dir = Path(str(configured))
        else:
            if self.model_dir is None:
                raise ValueError(f"model_dir must be set when {config_key} is not provided")
            cache_dir = Path(self.model_dir) / default_subdir
        safe_mkdir(cache_dir, parents=True)
        return cache_dir

    def _is_kit(self) -> bool:
        """Whether the resolved optimization runs the kit."""
        return self.optimization is not None and bool(self.optimization.kit)

    def _load(self) -> None:
        """Resolve the optimization mode, then load the model and input builder for it."""
        self._activate_optimization()
        if self._is_kit():
            self._load_kit()
        else:
            self._load_vanilla()
        self.ready = True

    def _dtype_kwargs(self) -> dict[str, Any]:
        """``from_pretrained`` dtype keyword for the configured ``dtype``, if any."""
        import torch

        dtype = self.config.get("dtype")
        if dtype is None:
            return {}
        return {"dtype": getattr(torch, dtype) if isinstance(dtype, str) else dtype}

    def _load_vanilla(self) -> None:
        """Load esm 3.4.1's ``EsmFold2Model`` at boileroom's pinned revision and record its kernel switches."""
        from esm.models.esmfold2 import ESMFold2InputBuilder, EsmFold2Model

        from .loading import load_pretrained

        cache_dir = self._resolve_cache_dir("cache_dir", "esmfold2")
        ccd_cache_dir = self._resolve_cache_dir("ccd_cache_dir", "esmfold2")
        model_name = str(self.config["model_name"])
        revision = self.config.get("revision")
        if revision is None and model_name == ESMFOLD2_HF_REPO:
            revision = ESMFOLD2_HF_REVISION
        if self.model is None:
            self.model = load_pretrained(
                EsmFold2Model, model_name, cache_dir=str(cache_dir), revision=revision, **self._dtype_kwargs()
            )
        self._device = self._resolve_device()
        self.model = self.model.to(self._device)
        self.model.eval()
        self.input_builder = ESMFold2InputBuilder(ccd_cache=self._ensure_ccd_cache(ccd_cache_dir))
        self._runtime = runtime_provenance(
            VANILLA_RUNTIME_PACKAGES,
            gpu=self._gpu,
            extra={"weights": f"{model_name}@{revision or 'latest'}", **self._vanilla_kernel_flags()},
        )

    @staticmethod
    def _vanilla_kernel_flags() -> dict[str, str]:
        """esm's import-time kernel switches as this process has them (``"not-loaded"`` / ``"absent"`` otherwise)."""
        flags: dict[str, str] = {}
        for module_name, names in VANILLA_KERNEL_FLAGS:
            module = sys.modules.get(module_name)
            for name in names:
                if module is None:
                    flags[f"esm.{name}"] = "not-loaded"
                else:
                    value = getattr(module, name, None)
                    flags[f"esm.{name}"] = "absent" if value is None else str(bool(value))
        return flags

    def _load_kit(self) -> None:
        """Load the transformers-fork ``ESMFold2Model`` from the kit's pinned snapshot and apply the armed levers."""
        from esm.models.esmfold2 import ESMFold2InputBuilder

        # The kit patches the transformers ESMFold2Model, not esm's EsmFold2Model.
        from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

        if self.model is None:
            # The kit's pinned snapshots load by repo id from HF_HOME, as its own CLI does. loading.py's pread
            # workaround deliberately does not apply: it patches esm's ``hub.load_file``, and transformers 5 reads
            # safetensors through ``safe_open`` with no ``load_file`` binding to patch.
            self.model = ESMFold2Model.from_pretrained(str(self.config["model_name"]), **self._dtype_kwargs())
        self._device = self._resolve_device()
        self.model = self.model.to(self._device)
        self.model.eval()
        self.input_builder = ESMFold2InputBuilder(ccd_cache=self._kit_ccd_dir())
        self._configure_optimization()

    def _kit_variant(self) -> str:
        """Kit variant for the loaded checkpoint.

        The kit arms one variant per process, before any request is seen, so the kernel variant is chosen at
        construction: ``kit_msa=True`` selects ``full_msa``; the default ``full_nomsa`` matches the MSA-free bakeoff
        workload. Both full variants consume a user MSA.
        """
        if str(self.config["model_name"]).endswith("ESMFold2-Fast"):
            return "fast"
        return "full_msa" if self.config["kit_msa"] else "full_nomsa"

    def _activate_optimization(self) -> None:
        """Resolve ``optimization`` on this GPU and arm the kit; runs before any weights load.

        Kit modes set :data:`KIT_REQUIRE_FAST_ENV` to ``1`` before importing the kit, refuse an image whose attention
        paths are slow before fetching any weight, and refuse an activation that does not report the accelerated paths.
        """
        self._kernel_gate_passed = False
        mode = str(self.config["optimization"])
        device = self.config.get("device")
        if mode == "vanilla":
            # Vanilla records the card for provenance when there is one and never touches the kit.
            self._gpu = describe_gpu(device)
            self.optimization = resolve_optimization("esmfold2", mode, self._gpu)
            return
        self._gpu = detect_gpu(device)
        self.optimization = resolve_optimization("esmfold2", mode, self._gpu)
        self._check_kit_stack(mode, self._gpu)
        # The kit reads its weights from HF_HOME, so it is set before anything imports the kit or huggingface_hub.
        model_dir = Path(self.model_dir or MODAL_MODEL_DIR)
        hf_home = Path(os.environ.setdefault("HF_HOME", str(model_dir / KIT_HF_SUBDIR)))
        # The kit's weight-digest memo, read when the weights are checked: on the persistent model directory, a cold
        # start reuses the digests instead of re-hashing every weight file. An explicit setting wins.
        os.environ.setdefault(KIT_WEIGHTS_MEMO_ENV, str(model_dir / KIT_WEIGHTS_MEMO_SUBDIR))
        os.environ[KIT_REQUIRE_FAST_ENV] = "1"
        try:
            import esmfold2_opt
            from esmfold2_opt import attn
        except ImportError as error:
            raise OptimizationUnavailableError(
                f"optimization={mode!r} needs the esmfold2 kit image (esmfold2_opt is not installed here)"
            ) from error
        if getattr(attn, "ENV_REQUIRE", None) != KIT_REQUIRE_FAST_ENV:
            raise OptimizationUnavailableError(
                f"optimization={mode!r}: this esmfold2_opt reads its fail-loud switch from "
                f"{getattr(attn, 'ENV_REQUIRE', None)!r}, not {KIT_REQUIRE_FAST_ENV!r}, so a slow-kernel fallback "
                "could not be refused"
            )
        self._kernel_pregate(attn)
        self._ensure_kit_weights(hf_home)
        self._go_offline()
        report = self._call_kit("enable", esmfold2_opt.enable, mode, variant=self._kit_variant())
        self._check_kit_report("at activation", report, attn)

    @staticmethod
    def _check_kit_stack(mode: str, gpu: GpuInfo) -> None:
        """Refuse a kit image whose compiled stack (:data:`KIT_STACK_ENV`) has no kernels for ``gpu``.

        Runs before the kit is imported or any weight is fetched. An unset or unknown stack is left to the kernel
        checks that follow and is recorded as such in ``metadata.runtime``.

        Raises
        ------
        OptimizationUnavailableError
            If the stack is known and ``gpu``'s compute capability is not one it compiles for.
        """
        stack = os.environ.get(KIT_STACK_ENV)
        capabilities = KIT_STACK_CAPABILITIES.get(stack or "")
        if capabilities is None or gpu.capability in capabilities:
            return
        served = ", ".join(f"sm_{major}{minor}" for major, minor in sorted(capabilities))
        major, minor = gpu.capability
        raise OptimizationUnavailableError(
            f"optimization={mode!r}: this kit image was compiled with {KIT_STACK_ENV}={stack}, which has kernels for "
            f"{served} only, not {gpu.name} (sm_{major}{minor}); use a kit image built with "
            "STACK=img_esmfold2_a100, or optimization='vanilla'"
        )

    def _attn_state(self, attn: Any, stage: str, **kwargs: Any) -> Mapping[str, Any]:
        """``attn.state(**kwargs)``, the kit's attention words as this process has them now; unreadable is refused."""
        try:
            return cast(Mapping[str, Any], attn.state(**kwargs))
        except Exception as error:
            raise OptimizationUnavailableError(
                f"optimization={self._mode()!r}: the kit's attention paths cannot be read in this image {stage} "
                f"({type(error).__name__}: {error})"
            ) from error

    def _kernel_pregate(self, attn: Any) -> None:
        """Refuse an image whose accelerated attention / MLP paths do not import, before ~27 GB of weights are fetched."""
        stage = "before the weight fetch"
        self._refuse_failing_words(stage, self._attn_state(attn, stage, load=True), attn)

    def _mode(self) -> str:
        """The resolved optimization mode (the configured one before resolution)."""
        return self.optimization.mode if self.optimization is not None else str(self.config["optimization"])

    def _gpu_name(self) -> str:
        return self._gpu.name if self._gpu is not None else "this GPU"

    def _call_kit(self, step: str, function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Call a kit entry point, turning its refusals into a typed :class:`OptimizationUnavailableError`.

        A refusal is a ``SystemExit`` whose code is in :data:`~boileroom.optimization.KIT_REFUSAL_EXIT_CODES`, or an
        exception with a class named in :data:`~boileroom.optimization.KIT_REFUSAL_CLASS_NAMES` in its MRO (the kit's
        ``ActivationError`` among them), the rule every kit family shares. A ``SystemExit`` with any other code
        (1 = failed, 2 = usage) is a failure and becomes a ``RuntimeError``; a ``SystemExit`` would otherwise escape
        ``initialize_core`` callers that do not map it and kill the server. Other exceptions (a GPU out-of-memory
        error, a bug) propagate unchanged.
        """
        try:
            return function(*args, **kwargs)
        except SystemExit as exit_:
            if not is_refusal(exit_):
                # 1 (failed) or 2 (usage): a failure, not a refusal, so it is not stored as a standing refusal.
                raise RuntimeError(
                    f"optimization={self._mode()!r}: esmfold2_opt {step} exited with code {exit_.code} on "
                    f"{self._gpu_name()} (a failure, not a refusal); the kit's reason is logged above"
                ) from exit_
            raise OptimizationUnavailableError(
                f"optimization={self._mode()!r}: esmfold2_opt {step} exited with code {exit_.code} on "
                f"{self._gpu_name()}; the kit's reason is logged above"
            ) from exit_
        except Exception as error:
            if is_kit_refusal_exception(error):
                raise OptimizationUnavailableError(
                    f"optimization={self._mode()!r}: esmfold2_opt {step} refused on {self._gpu_name()}: {error}"
                ) from error
            raise

    def _refuse_failing_words(self, stage: str, state: Mapping[str, Any], attn: Any) -> None:
        """Refuse when any of the kit's required kernel words (``attn.REQUIRED``) is not at its accelerated value."""
        failing = attn.failing_words(state)
        if failing:
            named = ", ".join(f"{word}={got} (expected {want}; needs {module})" for word, got, want, module in failing)
            raise OptimizationUnavailableError(
                f"optimization={self._mode()!r} would run slow kernels on {self._gpu_name()} {stage}: {named}. "
                "Refused rather than falling back; use optimization='vanilla' or a kit image with these kernels."
            )

    def _check_kit_report(self, stage: str, report: Mapping[str, Any], attn: Any) -> None:
        """Refuse a kit report that is inactive, partial, or lacks or fails the attention words."""
        mode = self._mode()
        if not report.get("active"):
            raise OptimizationUnavailableError(
                f"optimization={mode!r} is not active on {self._gpu_name()} {stage}: "
                f"{report.get('reason') or 'the kit gave no reason'}"
            )
        if report.get("partial"):
            raise OptimizationUnavailableError(
                f"optimization={mode!r} applied only part of its lever set {stage}: {report.get('partial')}"
            )
        state = report.get("attn")
        if not isinstance(state, Mapping):
            raise OptimizationUnavailableError(
                f"optimization={mode!r}: the kit reported no attention paths {stage}, so its kernels are unconfirmed"
            )
        self._refuse_failing_words(stage, state, attn)

    @staticmethod
    def _set_imported_hub_offline(offline: bool) -> None:
        """Flip the offline switch that already-imported ``huggingface_hub`` / ``transformers`` froze at import."""
        for module, attribute in HF_OFFLINE_ATTRIBUTES:
            if module in sys.modules and hasattr(sys.modules[module], attribute):
                setattr(sys.modules[module], attribute, offline)

    @staticmethod
    def _go_online() -> None:
        """Lift the offline switches for the kit's weight fetch, in the environment and in the imported modules.

        The kernel pre-gate imports transformers (and huggingface_hub with it) before the fetch, so the kit's own
        ``upstream_fetch``, which only pops the environment variables, would come too late if they were set.
        """
        for var in HF_OFFLINE_ENV:
            os.environ.pop(var, None)
        ESMFold2Core._set_imported_hub_offline(False)

    @staticmethod
    def _go_offline() -> None:
        """Serve the kit's frozen snapshots from disk only, as the kit's own launcher does.

        ``from_pretrained("biohub/ESMFold2")`` asks for ``main``; online it resolves upstream's newest commit, downloads
        it and repoints ``refs/main`` at it, which this kit's transformers fork cannot parse. ``huggingface_hub`` and
        ``transformers`` read the switches when they are imported, and the kernel pre-gate has already imported them,
        so the imported modules are flipped as well.
        """
        for var in HF_OFFLINE_ENV:
            os.environ[var] = "1"
        ESMFold2Core._set_imported_hub_offline(True)

    def _kit_ccd_dir(self) -> Path:
        """Directory holding the kit's pinned ``ccd.pkl`` (fetched with the kit weights)."""
        from esmfold2_opt import stack

        return (Path(os.environ["HF_HOME"]) / self._kit_ccd(stack.pins())[1]).parent

    @staticmethod
    def _kit_ccd(pins: dict[str, Any]) -> tuple[str, str]:
        """(repo, path under ``HF_HOME``) of the kit's pinned ``ccd.pkl``, which the Fast variant's own repo lacks."""
        repo, name = pins["ccd"]["repo"], pins["ccd"]["file"]
        snapshot = pins["weights"][repo]["snapshot_commit"]
        return repo, str(Path("hub") / f"models--{repo.replace('/', '--')}" / "snapshots" / snapshot / name)

    def _kit_scoped_pins(self, stack: Any) -> tuple[dict[str, Any], list[str]]:
        """The kit's pins narrowed to what this variant loads, and the paths (under ``HF_HOME``) of those files.

        Only the ESMC language model, the variant's own repository and the pinned ``ccd.pkl`` are in scope.
        """
        pins = stack.pins()
        paths = {(repo, rel) for repo, rel, _ in stack.pinned_weight_files(pins, self._kit_variant())}
        paths |= {self._kit_ccd(pins)}
        wanted: dict[str, set[str]] = {}
        for repo, rel in paths:
            wanted.setdefault(repo, set()).add(Path(rel).name)
        scoped = {
            **pins,
            "weights": {
                repo: {
                    **pins["weights"][repo],
                    "files": {n: d for n, d in pins["weights"][repo]["files"].items() if n in names},
                }
                for repo, names in wanted.items()
            },
        }
        return scoped, sorted(rel for _, rel in paths)

    def _ensure_kit_weights(self, hf_home: Path) -> None:
        """Fetch the kit's pinned checkpoints into ``hf_home`` unless the variant's files are already there.

        The kit loads frozen snapshots (not boileroom's pinned revision) and refuses to start without them; the image
        carries none. Present files are not fetched again, but their ``refs/main`` is re-pointed at the pin and their
        digests are checked. A failed transfer raises ``RuntimeError`` (retried by the next call); a file absent or off
        its pin after the fetch, or a present file off its pin, is a permanent ``OptimizationUnavailableError``.
        """
        from esmfold2_opt import stack, weights

        scoped, rels = self._kit_scoped_pins(stack)
        hf = str(hf_home)
        variant = self._kit_variant()
        if all((hf_home / rel).is_file() for rel in rels):
            for repo in sorted(scoped["weights"]):
                weights.point_ref(hf, repo, scoped["weights"][repo]["snapshot_commit"])
            absent, unknown = stack.weight_files_check(hf, scoped, None)
            if absent or unknown:
                raise OptimizationUnavailableError(
                    f"the pinned ESMFold2 kit weights under {hf_home} do not match their pins (absent: {absent}; off "
                    f"their pin: {unknown}); remove those files so that the next load fetches them again"
                )
            return
        logger.info(f"Fetching the pinned ESMFold2 kit weights for variant {variant!r} into {hf_home}")
        hf_home.mkdir(parents=True, exist_ok=True)
        transfer_errors: list[str] = []
        self._go_online()
        upstream = weights.upstream_fetch(hf)

        def fetch(repo: str, commit: str, name: str) -> str:
            try:
                return upstream(repo, commit, name)
            except Exception as error:
                transfer_errors.append(f"{repo} {name}: {type(error).__name__}: {error}")
                raise

        returncode = self._call_kit(
            "install_weights", weights.install_weights, hf, fetch=fetch, pins=scoped, log=logger.info
        )
        if transfer_errors:
            raise RuntimeError(
                f"fetching the pinned ESMFold2 kit weights into {hf_home} failed ({'; '.join(transfer_errors)}); "
                "the next load tries again"
            )
        if returncode != 0:
            raise OptimizationUnavailableError(
                f"the pinned ESMFold2 kit weights fetched into {hf_home} are absent or off their pins "
                f"(install_weights returned {returncode}); the files are logged above"
            )

    def _configure_optimization(self) -> None:
        """Install the armed kit's levers on the loaded model and refuse anything short of the accelerated stack."""
        if not self._is_kit():
            return
        import esmfold2_opt
        from esmfold2_opt import attn, stack

        self._call_kit(
            "apply_to",
            stack.apply_to,
            self.model,
            self.input_builder,
            trigger="boileroom",
            samples=int(self.config["num_diffusion_samples"]),
            out_dir=None,
        )
        report = self._call_kit("status", stack.status)
        self._check_kit_report("on the loaded model", report, attn)
        self._kernel_gate_passed = True
        self._runtime = self._kit_runtime(report, esmfold2_opt, attn, stack)

    def _kit_runtime(self, report: Mapping[str, Any], esmfold2_opt: Any, attn: Any, stack: Any) -> dict[str, str]:
        """``metadata.runtime`` of a kit mode: the shared provenance plus the kit, its weights and its kernels."""
        state = cast(Mapping[str, Any], report["attn"])
        scoped, _ = self._kit_scoped_pins(stack)
        extra: dict[str, object] = {
            "kit.stack": os.environ.get(KIT_STACK_ENV) or _UNKNOWN,
            "kit.package": getattr(esmfold2_opt, "__version__", None) or _UNKNOWN,
            "kit.variant": self._kit_variant(),
            "kit.weights": " ".join(
                f"{repo}@{entry['snapshot_commit']}" for repo, entry in sorted(scoped["weights"].items())
            ),
            "kit.require_fast_env": os.environ.get(KIT_REQUIRE_FAST_ENV) or _ABSENT,
            "kit.weights_memo": os.environ.get(KIT_WEIGHTS_MEMO_ENV) or _ABSENT,
            "kit.metadata_words": attn.metadata_words(),
            "kit.attn": attn.words(state),
        }
        for word in KIT_ATTN_WORDS:
            # A word the kit's report lacks is not established; one the kit could not read keeps the kit's own word.
            extra[f"kit.attn.{word}"] = str(state.get(word, _UNKNOWN))
        extra.update(self._kit_lever_words(report, int(self.config["num_diffusion_samples"]), stack))
        return runtime_provenance(KIT_RUNTIME_PACKAGES, gpu=self._gpu, extra=extra)

    @staticmethod
    def _kit_lever_words(report: Mapping[str, Any], num_diffusion_samples: int, stack: Any) -> dict[str, str]:
        """The ``metadata.runtime`` lever entries of a kit report, and the kit's scope notes at this sample count.

        The keys every kit family shares (:func:`~boileroom.provenance.kit_provenance`: ``kit.commit``,
        ``kit.levers_applied``, ``kit.levers_fallback``, ``kit.partial``), then ``kit.levers_gated`` / ``kit.gated``,
        which name levers whose declared guard sent calls to the previous chain (e.g. ``mk`` at
        ``num_diffusion_samples > 1``), ``kit.guards``, the verdict of every data-dependent guard the settle step read
        (``<lever>=<kind>: <note>``), and ``kit.scope``, the kit's own plan note for this call's sample count.
        """
        words = kit_provenance(
            commit=_kit_commit(sys.modules.get("esmfold2_opt")),
            levers_applied=report.get("levers_applied") or [],
            levers_fallback=report.get("levers_fallback") or [],
            partial=bool(report.get("partial")),
        )
        words["kit.levers_gated"] = _joined(report.get("levers_gated") or [])
        words["kit.gated"] = _joined(report.get("gated") or [], "; ")
        guards = report.get("guards") or {}
        words["kit.guards"] = _joined(
            (f"{lever}={guard.get('kind')}: {guard.get('note')}" for lever, guard in sorted(guards.items())), "; "
        )
        notes = [
            note(report.get("levers_planned"), num_diffusion_samples)
            for note in (getattr(stack, "mk_plan_note", None), getattr(stack, "guards_plan_note", None))
            if callable(note)
        ]
        words["kit.scope"] = _joined((note for note in notes if note), "; ")
        return words

    def _settle_kit_report(self, num_diffusion_samples: int) -> dict[str, str]:
        """After a kit fold: settle the levers from the kit's own counters, refuse a partial set, return the runtime.

        As the kit's CLI does after every run, ``stack.settle_mk`` and ``stack.settle_guards`` move levers that no fold
        reached into ``partial`` (refused, like the CLI's exit 3) and levers whose declared guard sent every call to the
        previous chain into ``levers_gated``. The counters are cumulative over the process, so ``kit.scope`` also states
        what the kit declares for this call's sample count. One deliberate departure: see :meth:`_gate_fallen_through_t16`.
        """
        from esmfold2_opt import attn, stack

        report = self._call_kit("status", stack.status)
        for step in ("settle_mk", "settle_guards"):
            settle = getattr(stack, step, None)
            if not callable(settle):
                raise OptimizationUnavailableError(
                    f"optimization={self._mode()!r}: this esmfold2_opt has no stack.{step}, so the levers that ran "
                    "cannot be confirmed after the fold"
                )
            report = self._call_kit(step, settle, report)
        report = self._gate_fallen_through_t16(report)
        # status() repeats the attention words the kit read when it applied the levers. Read them again on the model as
        # the fold left it, so that a path switched off since the load (a flash-attention flag cleared in the process)
        # is refused after the fold too, not only at the load. Those paths belong to the process, not to the input, so
        # that refusal stands: later folds raise it before reaching the GPU. A partial lever set depends on the call
        # (the kit scopes levers by sample count) and is refused per fold only.
        try:
            state = self._attn_state(attn, "after the fold", model=self.model)
            self._refuse_failing_words("after the fold", state, attn)
        except OptimizationUnavailableError as error:
            self._fold_refusal = error
            raise
        report = {**report, "attn": state}
        self._check_kit_report("after the fold", report, attn)
        assert self._runtime is not None, "kit runtime not recorded"
        return {**self._runtime, **self._kit_lever_words(report, num_diffusion_samples, stack)}

    @staticmethod
    def _gate_fallen_through_t16(report: Mapping[str, Any]) -> Mapping[str, Any]:
        """Settle ``t16`` as gated, not partial, when every call reached its wrapper and fell through by name.

        ``stack.guards_after_run`` calls ``t16`` "unreached" whenever no Transition / PairTransition call was served by
        the kernel, even when every call did reach the wrapper and fell through: a by-name refusal (the provider's
        cell table names the statement for that size, or the shape is outside its envelope) or an ineligible call.
        Each such call runs the previous statement, the fork's own fused forward, so the values are the previous
        chain's and only the speed differs: the same evidence the kit settles as "inactive" for other levers. Such a
        run is recorded (``t16`` in ``kit.levers_gated``, the counters and refusal reasons in ``kit.gated`` and
        ``kit.guards``) rather than refused. Zero calls at the wrapper (``fallthrough == 0``) stays "unreached" and is
        still refused: nothing shows the lever's code ever ran.
        """
        guard = (report.get("guards") or {}).get("t16") or {}
        if guard.get("kind") != "unreached" or "t16" not in (report.get("partial") or []):
            return report
        cute = sys.modules.get("ef2_transition_cute")
        stats = dict(getattr(cute, "STATS", None) or {})
        served = int(stats.get("transition_calls", 0) or 0) + int(stats.get("pair_transition_calls", 0) or 0)
        fell = int(stats.get("fallthrough_transition", 0) or 0) + int(stats.get("fallthrough_pair_transition", 0) or 0)
        if served or not fell:
            return report
        describe = getattr(cute, "describe", None)
        refused = (describe() or {}).get("refused") if callable(describe) else None
        reasons = ",".join(f"{reason}={count}" for reason, count in sorted((refused or {}).items())) or _UNKNOWN
        note = (
            f"all {fell} Transition / PairTransition call(s) took the previous statements, none served "
            f"(fallthrough_transition={int(stats.get('fallthrough_transition', 0) or 0)} "
            f"fallthrough_pair_transition={int(stats.get('fallthrough_pair_transition', 0) or 0)} "
            f"face_refused={int(stats.get('face_refused', 0) or 0)}; refused: {reasons}); "
            "gated by boileroom, not partial"
        )
        fallback_reasons = dict(report.get("fallback_reasons") or {})
        fallback_reasons.pop("t16", None)
        return {
            **report,
            "partial": [lever for lever in report.get("partial") or [] if lever != "t16"],
            "levers_fallback": [lever for lever in report.get("levers_fallback") or [] if lever != "t16"],
            "fallback_reasons": fallback_reasons,
            "levers_gated": sorted({*(report.get("levers_gated") or []), "t16"}),
            "gated": [*(report.get("gated") or []), f"t16: {note}"],
            "guards": {**(report.get("guards") or {}), "t16": {"kind": "inactive", "note": note}},
        }

    @staticmethod
    def _ensure_ccd_cache(ccd_cache_dir: Path) -> Path:
        """Return the pinned CCD directory, ignoring legacy unversioned caches."""
        ccd_cache_dir = ccd_cache_dir / ESMFOLD2_HF_REVISION
        ccd_path = ccd_cache_dir / "ccd.pkl"
        if ccd_path.is_file():
            return ccd_cache_dir

        from huggingface_hub import hf_hub_download

        hf_hub_download(
            repo_id=ESMFOLD2_HF_REPO,
            filename="ccd.pkl",
            revision=ESMFOLD2_HF_REVISION,
            local_dir=str(ccd_cache_dir),
        )
        return ccd_cache_dir

    def fold(self, sequences: ESMFold2FoldInput, options: dict | None = None) -> ESMFold2Output:
        """Predict one or more structures with ESMFold2."""
        effective_config = self._merge_options(options)
        requests = self._coerce_requests(sequences)
        user_msa = effective_config.get("msa")
        if self._has_user_msa(user_msa):
            if len(requests) != 1:
                raise ValueError("ESMFold2 option 'msa' applies to exactly one input structure, not a batch.")
            requests = [self._attach_user_msa(requests[0], user_msa)]
        for request in requests:
            self._check_covalent_bonds(request.input)

        if self.model is None or self.input_builder is None:
            logger.warning("Model not loaded. Forcing the model to load... Next time call _load() first.")
            self._load()
        assert self.model is not None and self.input_builder is not None, "Model not loaded"
        self._check_msa_is_read(requests)
        for request in requests:
            self._check_ligand_bond_atoms(request.input)

        results: list[Any] = []
        sequence_lengths: list[int] = []
        preprocessing_time = 0.0
        inference_time = 0.0
        postprocessing_time = 0.0

        for request_index, request in enumerate(requests):
            folded, timing = self._fold_one(request.input, effective_config, request_index)
            results.extend(folded)
            sequence_lengths.extend([request.sequence_length] * len(folded))
            preprocessing_time += timing["preprocessing"]
            inference_time += timing["inference"]
            postprocessing_time += timing["postprocessing"]

        runtime: dict[str, str] | None
        if self._is_kit():
            runtime = self._settle_kit_report(int(effective_config["num_diffusion_samples"]))
        else:
            runtime = dict(self._runtime) if self._runtime is not None else None
        metadata = dataclasses.replace(
            self._metadata_template,
            sequence_lengths=sequence_lengths,
            preprocessing_time=preprocessing_time,
            inference_time=inference_time,
            postprocessing_time=postprocessing_time,
            optimization=self.optimization.to_dict() if self.optimization else None,
            runtime=runtime,
        )
        return self._convert_results(results, metadata, effective_config)

    @staticmethod
    def _has_user_msa(msa: object) -> bool:
        """Whether ``options['msa']`` carries an MSA: ``None`` and an all-``None`` list do not."""
        if msa is None:
            return False
        if isinstance(msa, list | tuple):
            return any(entry is not None for entry in msa)
        return True

    def _check_msa_is_read(self, requests: list[_FoldRequest]) -> None:
        """Refuse an MSA (inline or from ``options['msa']``) that the loaded checkpoint has no encoder for.

        ESMFold2-Fast has no ``msa_encoder`` and both esm releases skip the MSA without one, so the fold would silently
        be single-sequence.
        """
        if getattr(self.model, "msa_encoder", True) is not None:
            return
        carrying = [
            f"{index}:{item.id}"
            for request in requests
            for index, item in enumerate(request.input.sequences)
            if isinstance(item, ProteinInput) and item.msa is not None
        ]
        if carrying:
            raise ValueError(
                f"ESMFold2 checkpoint {self.config['model_name']!r} has no MSA encoder and would ignore the MSA on "
                f"input entries {carrying}; fold with model_name={ESMFOLD2_HF_REPO!r} to use an MSA, or drop it."
            )

    def _fold_one(
        self,
        prediction_input: StructurePredictionInput,
        config: dict[str, Any],
        request_index: int,
    ) -> tuple[list[Any], dict[str, float]]:
        """Run one ESMFold2 structure input through the loaded runtime (vanilla or kit)."""
        assert self.model is not None and self.input_builder is not None, "Model not loaded"
        complex_id_base = str(config.get("complex_id") or "pred")
        complex_id = complex_id_base if request_index == 0 else f"{complex_id_base}_{request_index}"
        esm_input = self._to_esm_structure_prediction_input(prediction_input)
        if self._is_kit():
            return self._fold_kit(esm_input, config, complex_id)
        return self._fold_vanilla(esm_input, config, complex_id)

    def _fold_vanilla(
        self, esm_input: Any, config: dict[str, Any], complex_id: str
    ) -> tuple[list[Any], dict[str, float]]:
        """Prepare, run and decode one input with esm 3.4.1's model call."""
        import torch
        from esm.models.esmfold2.processor import _seed_context

        assert self.model is not None and self.input_builder is not None, "Model not loaded"
        seed = cast(int | None, config.get("seed"))
        num_diffusion_samples = int(config["num_diffusion_samples"])
        with Timer("ESMFold2 preprocessing") as preprocess_timer:
            features, chain_infos = self.input_builder.prepare_input(esm_input, seed=seed, device=self.model.device)

        with Timer("ESMFold2 inference") as inference_timer, torch.no_grad(), _seed_context(seed):
            output = self.model(
                **features,
                num_loops=int(config["num_loops"]),
                num_sampling_steps=int(config["num_sampling_steps"]),
                num_diffusion_samples=num_diffusion_samples,
                msa_max_depth=config.get("msa_max_depth"),
                msa_column_mask_rate=float(config.get("msa_column_mask_rate", 0.1)),
                **self._sampler_kwargs(config),
            )

        with Timer("ESMFold2 postprocessing") as postprocess_timer:
            decoded = self.input_builder.decode(
                output,
                features,
                chain_infos,
                num_diffusion_samples=num_diffusion_samples,
                complex_id=complex_id,
            )
        results = decoded if isinstance(decoded, list) else [decoded]
        return results, {
            "preprocessing": float(preprocess_timer.duration or 0.0),
            "inference": float(inference_timer.duration or 0.0),
            "postprocessing": float(postprocess_timer.duration or 0.0),
        }

    def _fold_kit(self, esm_input: Any, config: dict[str, Any], complex_id: str) -> tuple[list[Any], dict[str, float]]:
        """Fold one input through esm 3.3.0's ``ESMFold2InputBuilder.fold()``, which the kit hooks.

        The builder prepares, runs and decodes in one call, so the inference time covers the whole fold and the
        preprocessing / postprocessing times are ``0.0``.
        """
        assert self.model is not None and self.input_builder is not None, "Model not loaded"
        if self._fold_refusal is not None:
            raise self._fold_refusal
        if not self._kernel_gate_passed:
            raise OptimizationUnavailableError(
                f"optimization={self._mode()!r}: the kit's kernels were not confirmed on the loaded model; "
                "load the core (which runs the kernel gate) before folding"
            )
        try:
            with Timer("ESMFold2 inference") as inference_timer:
                decoded = self.input_builder.fold(self.model, esm_input, **self._kit_fold_kwargs(config, complex_id))
        finally:
            # fast's TriMul keeps every geometry its LRU evicted alive; folds of varied lengths would grow until OOM.
            release_evicted_kit_caches()
        results = decoded if isinstance(decoded, list) else [decoded]
        return results, {
            "preprocessing": 0.0,
            "inference": float(inference_timer.duration or 0.0),
            "postprocessing": 0.0,
        }

    @staticmethod
    def _kit_fold_kwargs(config: dict[str, Any], complex_id: str) -> dict[str, Any]:
        """Keyword arguments of the kit's ``input_builder.fold()``.

        ``lm_dropout=None`` keeps the checkpoint's own per-loop language-model dropout (p=0.25) live, which is what
        vanilla's direct model call does as well (esm 3.4.1 has no such argument). The sampler overrides are refused
        in kit modes by :meth:`_validate_config`; ``lm_mask_pct`` is forwarded.
        """
        kwargs: dict[str, Any] = {
            "num_loops": int(config["num_loops"]),
            "num_sampling_steps": int(config["num_sampling_steps"]),
            "num_diffusion_samples": int(config["num_diffusion_samples"]),
            "seed": cast(int | None, config.get("seed")),
            "lm_dropout": None,
            "msa_max_depth": int(config["msa_max_depth"]),
            "msa_column_mask_rate": float(config.get("msa_column_mask_rate", 0.1)),
            "complex_id": complex_id,
        }
        if config.get("lm_mask_pct") is not None:
            kwargs["lm_mask_pct"] = config["lm_mask_pct"]
        return kwargs

    @staticmethod
    def _validate_effective_config(config: Mapping[str, Any]) -> None:
        """Validate dynamic inference options before launching expensive model work."""
        for key in ("num_loops", "num_sampling_steps", "num_diffusion_samples"):
            value = config[key]
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"ESMFold2 option {key!r} must be a positive integer.")

        seed = config.get("seed")
        if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool) or seed < 0):
            raise ValueError("ESMFold2 option 'seed' must be a non-negative integer or None.")

        for key in (
            "noise_scale",
            "step_scale",
            "max_inference_sigma",
            "lm_mask_pct",
            "msa_column_mask_rate",
        ):
            value = config.get(key)
            if value is None:
                continue
            if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(float(value)):
                raise ValueError(f"ESMFold2 option {key!r} must be a finite number or None.")

        msa_max_depth = config.get("msa_max_depth")
        if msa_max_depth is not None and (
            not isinstance(msa_max_depth, int) or isinstance(msa_max_depth, bool) or msa_max_depth < 1
        ):
            raise ValueError("ESMFold2 option 'msa_max_depth' must be a positive integer or None.")

    def _attach_user_msa(self, request: _FoldRequest, msa: object) -> _FoldRequest:
        """Attach ``options['msa']`` (one A3M text or None per input entry) to the protein entries."""
        items = request.input.sequences
        if not isinstance(msa, list | tuple) or len(msa) != len(items):
            raise ValueError("ESMFold2 option 'msa' must have one A3M string or None per input sequence entry.")
        attached: list[SequenceInput] = []
        for index, (item, text) in enumerate(zip(items, msa, strict=True)):
            if text is None:
                attached.append(item)
                continue
            if not isinstance(item, ProteinInput):
                raise ValueError(f"ESMFold2 option 'msa' entry {index} targets a non-protein input; use None.")
            if item.msa is not None:
                raise ValueError(
                    f"ESMFold2 input entry {index} already carries an MSA; do not also pass options['msa']."
                )
            if ":" in item.sequence or "|" in item.sequence:
                raise ValueError(f"ESMFold2 option 'msa' entry {index} needs a single-chain input entry.")
            rows = a3m_rows(text, item.sequence)
            attached.append(dataclasses.replace(item, msa=MSAInput(sequences=rows)))
        return dataclasses.replace(request, input=dataclasses.replace(request.input, sequences=attached))

    @staticmethod
    def _check_covalent_bonds(prediction_input: StructurePredictionInput) -> None:
        """Validate covalent bonds before either esm builder sees them, and before any load.

        esm 3.4.1 raises on a bond naming a missing chain, residue or atom; the kit's esm 3.3.0 silently drops it (and
        reads a negative atom index from the end of the residue), folding the ligand unbonded. Residue and atom
        indices are 0-based, as in esm. Bonds cannot be combined with ``:`` / ``|`` chainbreaks (esm 3.4.1 refuses
        that too). The atom index of an unmodified protein, DNA or RNA residue is bounded here by its heavy-atom
        count (:data:`PROTEIN_RESIDUE_ATOM_COUNTS` and the nucleotide tables); a CCD ligand's is bounded after load
        by :meth:`_check_ligand_bond_atoms`. A modified residue, an unknown nucleotide and a SMILES ligand have no
        count without esm's own tokenizer and stay with esm.
        """
        bonds = prediction_input.covalent_bonds
        if not bonds:
            return
        chains: dict[str, tuple[SequenceInput, int]] = {}
        for item in prediction_input.sequences:
            if isinstance(item, ProteinInput | RNAInput | DNAInput):
                if ":" in item.sequence or "|" in item.sequence:
                    raise ValueError(
                        f"ESMFold2 covalent bonds cannot be combined with chainbreaks (':' or '|') in entry "
                        f"{item.id!r}; give each chain its own input entry."
                    )
                count = len(item.sequence)
            else:
                # A CCD list is one residue per code; a SMILES ligand is a single residue.
                count = len(item.ccd) if isinstance(item.ccd, list) and item.ccd else 1
            for chain_id in item.id if isinstance(item.id, list) else [item.id]:
                chains[chain_id] = (item, count)
        for index, bond in enumerate(bonds):
            for chain_id, res_idx, atom_idx in (
                (bond.chain_id1, bond.res_idx1, bond.atom_idx1),
                (bond.chain_id2, bond.res_idx2, bond.atom_idx2),
            ):
                if chain_id not in chains:
                    raise ValueError(
                        f"ESMFold2 covalent bond {index}: chain_id {chain_id!r} does not exist; "
                        f"available chain ids: {sorted(chains)}"
                    )
                item, count = chains[chain_id]
                if not isinstance(res_idx, int) or isinstance(res_idx, bool) or not 0 <= res_idx < count:
                    raise ValueError(
                        f"ESMFold2 covalent bond {index}: residue index {res_idx!r} is not in chain {chain_id!r} "
                        f"(0-based, 0-{count - 1})"
                    )
                if not isinstance(atom_idx, int) or isinstance(atom_idx, bool) or atom_idx < 0:
                    raise ValueError(
                        f"ESMFold2 covalent bond {index}: atom index {atom_idx!r} must be a non-negative integer "
                        "(0-based within the residue)"
                    )
                atoms = _polymer_residue_atoms(item, res_idx)
                if atoms is not None and atom_idx >= atoms[1]:
                    raise _atom_past_residue(index, atom_idx, res_idx, chain_id, *atoms)

    @staticmethod
    def _check_ligand_bond_atoms(prediction_input: StructurePredictionInput) -> None:
        """Bound the atom index of a covalent bond on a CCD ligand by the component's atoms in the loaded CCD.

        Runs after load, when the input builder has loaded the CCD. Both esm builders read the same
        ``esm.models.esmfold2.conformers`` (identical in esm 3.3.0 and 3.4.1) and tokenize a covalently bonded CCD
        ligand as one token per atom without its leaving atoms, so the bound is that count. A code missing from the
        CCD is left to esm, which refuses it.
        """
        bonds = prediction_input.covalent_bonds
        if not bonds:
            return
        ligands: dict[str, list[str]] = {}
        for item in prediction_input.sequences:
            if isinstance(item, LigandInput) and isinstance(item.ccd, list) and item.ccd:
                for chain_id in item.id if isinstance(item.id, list) else [item.id]:
                    ligands[chain_id] = item.ccd
        if not ligands:
            return
        conformers = importlib.import_module("esm.models.esmfold2.conformers")
        for index, bond in enumerate(bonds):
            for chain_id, res_idx, atom_idx in (
                (bond.chain_id1, bond.res_idx1, bond.atom_idx1),
                (bond.chain_id2, bond.res_idx2, bond.atom_idx2),
            ):
                if chain_id not in ligands:
                    continue
                code = ligands[chain_id][res_idx]
                ccd_atoms = conformers.get_ligand_ccd_atoms_with_charges(code)
                if ccd_atoms is None:
                    continue
                leaving = conformers.get_ccd_leaving_atoms(code)
                count = sum(1 for atom in ccd_atoms if atom[0] not in leaving)
                if atom_idx >= count:
                    raise _atom_past_residue(index, atom_idx, res_idx, chain_id, f"CCD {code}", count)

    @staticmethod
    def _sampler_kwargs(config: dict[str, Any]) -> dict[str, Any]:
        """Return optional sampler controls accepted by vanilla esm 3.4.1's model call."""
        kwargs = {}
        for key in ("noise_scale", "step_scale", "max_inference_sigma", "lm_mask_pct"):
            if config.get(key) is not None:
                kwargs[key] = config[key]
        return kwargs

    def _coerce_requests(self, sequences: ESMFold2FoldInput) -> list[_FoldRequest]:
        """Coerce public BoilerRoom inputs into one or more ESMFold2 structure inputs."""
        if isinstance(sequences, str):
            return [self._request_from_protein_string(sequences)]

        if isinstance(sequences, StructurePredictionInput):
            return [self._request_from_structure_input(sequences)]

        if isinstance(sequences, Mapping):
            return [self._request_from_structure_input(decode_structure_input(cast(Mapping[str, Any], sequences)))]

        if not isinstance(sequences, Sequence):
            raise TypeError("ESMFold2.fold expects a sequence string, structure input, or a sequence of those inputs.")

        items = list(sequences)
        if not items:
            raise ValueError("ESMFold2.fold received an empty input sequence.")

        if all(isinstance(item, str) for item in items):
            return [self._request_from_protein_string(cast(str, item)) for item in items]

        if all(isinstance(item, StructurePredictionInput) for item in items):
            return [self._request_from_structure_input(cast(StructurePredictionInput, item)) for item in items]

        if all(isinstance(item, Mapping) for item in items):
            return [
                self._request_from_structure_input(decode_structure_input(cast(Mapping[str, Any], item)))
                for item in items
            ]

        if all(self._is_sequence_input(item) for item in items):
            structure_input = StructurePredictionInput(sequences=cast(list[SequenceInput], items))
            return [self._request_from_structure_input(structure_input)]

        raise TypeError(
            "ESMFold2.fold input lists must contain only strings, only StructurePredictionInput objects, "
            "or only molecule input dataclasses."
        )

    @staticmethod
    def _is_sequence_input(value: object) -> bool:
        """Return whether a value is one lightweight molecule input dataclass."""
        return isinstance(value, ProteinInput | RNAInput | DNAInput | LigandInput)

    def _request_from_protein_string(self, sequence: str) -> _FoldRequest:
        """Convert a colon-delimited protein string into one structure request."""
        chains = sequence.replace("|", ":").split(":")
        if any(chain == "" for chain in chains):
            raise ValueError(f"Invalid ESMFold2 protein sequence {sequence!r}: empty chain near ':'.")
        for chain in chains:
            validate_sequence(chain)
        structure_input = StructurePredictionInput(
            sequences=[ProteinInput(id=self._chain_id(index), sequence=chain) for index, chain in enumerate(chains)]
        )
        return _FoldRequest(input=structure_input, sequence_length=sum(len(chain) for chain in chains))

    def _request_from_structure_input(self, prediction_input: StructurePredictionInput) -> _FoldRequest:
        """Validate a rich structure input and attach sequence-length metadata."""
        if not prediction_input.sequences:
            raise ValueError("StructurePredictionInput.sequences must not be empty.")
        return _FoldRequest(input=prediction_input, sequence_length=self._structure_input_length(prediction_input))

    @staticmethod
    def _chain_id(index: int) -> str:
        """Return spreadsheet-style chain IDs: A, B, ..., Z, AA, AB, ..."""
        index += 1
        parts = []
        while index:
            index, remainder = divmod(index - 1, 26)
            parts.append(chr(65 + remainder))
        return "".join(reversed(parts))

    def _structure_input_length(self, prediction_input: StructurePredictionInput) -> int:
        """Return the metadata length for all sequence-like entities in an input."""
        return sum(self._sequence_input_length(item) for item in prediction_input.sequences)

    def _sequence_input_length(self, item: SequenceInput) -> int:
        """Return the metadata length contribution for one chain or ligand input."""
        multiplier = self._id_count(item.id)
        if isinstance(item, ProteinInput | RNAInput | DNAInput):
            sequence = item.sequence.replace("|", ":")
            chains = sequence.split(":")
            if any(chain == "" for chain in chains):
                raise ValueError(f"Invalid ESMFold2 sequence {sequence!r}: empty chain near ':'.")
            if isinstance(item, ProteinInput):
                for chain in chains:
                    validate_sequence(chain)
            return sum(len(chain) for chain in chains) * multiplier
        ccd_count = len(item.ccd or [])
        ligand_count = ccd_count or (1 if item.smiles else 0)
        return max(multiplier, ligand_count)

    @staticmethod
    def _id_count(value: str | list[str]) -> int:
        """Return how many entity copies an ESMFold2 id represents."""
        return len(value) if isinstance(value, list) else 1

    def _to_esm_structure_prediction_input(self, prediction_input: StructurePredictionInput) -> Any:
        """Convert a lightweight structure input to Biohub's native dataclass."""
        from esm.models.esmfold2 import StructurePredictionInput as ESMStructurePredictionInput

        return ESMStructurePredictionInput(
            sequences=[self._to_esm_sequence_input(item) for item in prediction_input.sequences],
            pocket=self._to_esm_pocket_conditioning(prediction_input.pocket),
            distogram_conditioning=self._to_esm_distogram_conditioning(prediction_input.distogram_conditioning),
            covalent_bonds=self._to_esm_covalent_bonds(prediction_input.covalent_bonds),
        )

    def _to_esm_sequence_input(self, item: SequenceInput) -> Any:
        """Convert one lightweight chain or ligand input to Biohub's native type."""
        from esm.models.esmfold2 import DNAInput as ESMDNAInput
        from esm.models.esmfold2 import LigandInput as ESMLigandInput
        from esm.models.esmfold2 import ProteinInput as ESMProteinInput
        from esm.models.esmfold2 import RNAInput as ESMRNAInput

        if isinstance(item, ProteinInput):
            return ESMProteinInput(
                id=item.id,
                sequence=item.sequence,
                modifications=self._to_esm_modifications(item.modifications),
                msa=self._to_esm_msa(item.msa),
            )
        if isinstance(item, RNAInput):
            return ESMRNAInput(
                id=item.id,
                sequence=item.sequence,
                modifications=self._to_esm_modifications(item.modifications),
            )
        if isinstance(item, DNAInput):
            return ESMDNAInput(
                id=item.id,
                sequence=item.sequence,
                modifications=self._to_esm_modifications(item.modifications),
            )
        return ESMLigandInput(id=item.id, smiles=item.smiles, ccd=item.ccd)

    @staticmethod
    def _to_esm_modifications(modifications: list[Modification] | None) -> list[Any] | None:
        """Convert optional lightweight residue modifications to Biohub objects."""
        if modifications is None:
            return None
        from esm.models.esmfold2 import Modification as ESMModification

        return [
            ESMModification(position=modification.position, ccd=modification.ccd, smiles=modification.smiles)
            for modification in modifications
        ]

    @staticmethod
    def _to_esm_msa(msa: MSAInput | Any | None) -> Any | None:
        """Convert lightweight or native MSA inputs to Biohub's MSA object."""
        if msa is None:
            return None
        if isinstance(msa, MSAInput):
            if msa.sequences is None:
                raise ValueError(
                    "ESMFold2 currently requires in-memory MSA sequences; "
                    "file-backed MSA paths are reserved for other model adapters."
                )
            from esm.models.esmfold2 import MSA

            return MSA.from_sequences(msa.sequences, remove_insertions=msa.remove_insertions)
        from esm.models.esmfold2 import MSA

        if isinstance(msa, MSA):
            return msa
        if isinstance(msa, list):
            return MSA.from_sequences(msa)
        return msa

    @staticmethod
    def _to_esm_pocket_conditioning(pocket: PocketConditioning | None) -> Any | None:
        """Convert optional pocket conditioning to Biohub's native dataclass."""
        if pocket is None:
            return None
        from esm.utils.structure.input_builder import PocketConditioning as ESMPocketConditioning

        return ESMPocketConditioning(binder_chain_id=pocket.binder_chain_id, contacts=pocket.contacts)

    @staticmethod
    def _to_esm_distogram_conditioning(
        distograms: list[DistogramConditioning] | None,
    ) -> list[Any] | None:
        """Convert optional distogram conditioning entries to Biohub objects."""
        if distograms is None:
            return None
        from esm.models.esmfold2 import DistogramConditioning as ESMDistogramConditioning

        return [
            ESMDistogramConditioning(chain_id=distogram.chain_id, distogram=distogram.distogram)
            for distogram in distograms
        ]

    @staticmethod
    def _to_esm_covalent_bonds(bonds: list[CovalentBond] | None) -> list[Any] | None:
        """Convert optional covalent-bond constraints to Biohub objects."""
        if bonds is None:
            return None
        from esm.models.esmfold2 import CovalentBond as ESMCovalentBond

        return [
            ESMCovalentBond(
                chain_id1=bond.chain_id1,
                res_idx1=bond.res_idx1,
                atom_idx1=bond.atom_idx1,
                chain_id2=bond.chain_id2,
                res_idx2=bond.res_idx2,
                atom_idx2=bond.atom_idx2,
            )
            for bond in bonds
        ]

    def _convert_results(
        self,
        results: list[Any],
        metadata: PredictionMetadata,
        config: dict[str, Any],
    ) -> ESMFold2Output:
        """Convert Biohub decoded outputs to BoilerRoom's normalized output type."""
        include_fields = cast(list[str] | None, config.get("include_fields"))
        atom_arrays: list[Any] = []
        cif_values: list[str] | None = [] if self._wants_field(include_fields, "cif") else None
        pdb_values: list[str] | None = [] if self._wants_field(include_fields, "pdb") else None

        plddt: list[np.ndarray | None] | None = [] if self._wants_field(include_fields, "plddt") else None
        ptm: list[Any] | None = [] if self._wants_field(include_fields, "ptm") else None
        iptm: list[Any] | None = [] if self._wants_field(include_fields, "iptm") else None
        pae: list[np.ndarray | None] | None = [] if self._wants_field(include_fields, "pae") else None
        distogram: list[np.ndarray | None] | None = [] if self._wants_field(include_fields, "distogram") else None
        pair_chains_iptm: list[np.ndarray | None] | None = (
            [] if self._wants_field(include_fields, "pair_chains_iptm") else None
        )
        residue_index: list[np.ndarray | None] | None = (
            [] if self._wants_field(include_fields, "residue_index") else None
        )
        entity_id: list[np.ndarray | None] | None = [] if self._wants_field(include_fields, "entity_id") else None

        for result in results:
            cif_string = result.complex.to_mmcif()
            atom_array = self._cif_to_atom_array(cif_string)
            atom_arrays.append(atom_array)

            if cif_values is not None:
                cif_values.append(cif_string)
            if pdb_values is not None:
                pdb_values.append(self._atom_array_to_pdb(atom_array))
            if plddt is not None:
                plddt.append(self._tensor_to_numpy(result.plddt))
            if ptm is not None:
                ptm.append(result.ptm)
            if iptm is not None:
                iptm.append(result.iptm)
            if pae is not None:
                pae.append(self._tensor_to_numpy(result.pae))
            if distogram is not None:
                distogram.append(self._tensor_to_numpy(result.distogram))
            if pair_chains_iptm is not None:
                pair_chains_iptm.append(self._tensor_to_numpy(result.pair_chains_iptm))
            if residue_index is not None:
                residue_index.append(self._tensor_to_numpy(result.residue_index))
            if entity_id is not None:
                entity_id.append(self._tensor_to_numpy(result.entity_id))

        full_output = ESMFold2Output(
            metadata=metadata,
            atom_array=atom_arrays,
            plddt=plddt,
            ptm=ptm,
            iptm=iptm,
            pae=pae,
            distogram=distogram,
            pair_chains_iptm=pair_chains_iptm,
            residue_index=residue_index,
            entity_id=entity_id,
            pdb=pdb_values,
            cif=cif_values,
        )
        filtered = self._filter_include_fields(full_output, include_fields)
        return cast(ESMFold2Output, filtered)

    @staticmethod
    def _wants_field(include_fields: list[str] | None, field: str) -> bool:
        """Return whether a named optional output field was requested."""
        return include_fields is not None and ("*" in include_fields or field in include_fields)

    @staticmethod
    def _tensor_to_numpy(value: Any) -> np.ndarray | None:
        """Detach a tensor-like value and return it as a NumPy array."""
        if value is None:
            return None
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        return np.asarray(value)

    @staticmethod
    def _cif_to_atom_array(cif_string: str) -> Any:
        """Parse an mmCIF string into a Biotite atom array."""
        from biotite.structure.io.pdbx import CIFFile, get_structure

        return get_structure(CIFFile.read(StringIO(cif_string)), model=1)

    @staticmethod
    def _atom_array_to_pdb(atom_array: Any) -> str:
        """Serialize a Biotite atom array to PDB text."""
        from biotite.structure.io.pdb import PDBFile, set_structure

        pdb_file = PDBFile()
        set_structure(pdb_file, atom_array)
        buffer = StringIO()
        pdb_file.write(buffer)
        return buffer.getvalue()
