"""Core ESMFold2 implementation without Modal dependencies."""

from __future__ import annotations

import dataclasses
import logging
import math
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from io import StringIO
from pathlib import Path
from typing import Any, ClassVar, cast

import numpy as np

from ...base import FoldingAlgorithm, PredictionMetadata
from ...optimization import (
    OptimizationResolution,
    OptimizationUnavailableError,
    detect_gpu,
    resolve_optimization,
    validate_optimization,
)
from ...utils import MODAL_MODEL_DIR, Timer, safe_mkdir, validate_sequence
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


def _a3m_rows(text: object, sequence: str) -> list[str]:
    """Parse A3M text into aligned rows (insertions dropped), checking them against the query chain."""
    if not isinstance(text, str) or not text.lstrip().startswith(">"):
        raise ValueError("ESMFold2 option 'msa' entries must be A3M text or None")
    rows: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            rows.append("")
        elif rows:
            rows[-1] += line
    rows = ["".join(char for char in row if not char.islower() and char != ".") for row in rows]
    if not rows or rows[0] != sequence:
        raise ValueError("The first A3M row must match its input protein chain")
    if any(len(row) != len(sequence) for row in rows):
        raise ValueError("Every A3M row must have the input chain's aligned length")
    return rows


class ESMFold2Core(FoldingAlgorithm):
    """Biohub ESMFold2 all-atom structure prediction model."""

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
        validate_optimization(self.config["optimization"])
        self.optimization: OptimizationResolution | None = None
        self._metadata_template = self._initialize_metadata(
            model_name="ESMFold2",
            model_version=str(self.config["model_name"]),
        )
        self.model_dir: str | None = os.environ.get("MODEL_DIR", MODAL_MODEL_DIR)
        self._device: Any | None = None
        self.model: Any | None = None
        self.input_builder: Any | None = None

    def _initialize(self) -> None:
        """Load the ESMFold2 model and input builder."""
        self._load()

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

    def _load(self) -> None:
        """Load Biohub ESMFold2 from Hugging Face and prepare the input builder."""
        import torch

        self._activate_optimization()
        kit = self.optimization is not None and self.optimization.active != "vanilla"

        from esm.models.esmfold2 import ESMFold2InputBuilder

        from .loading import load_pretrained

        if kit:
            # The kit patches the transformers ESMFold2Model, not esm's EsmFold2Model.
            from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model as ModelClass
        else:
            from esm.models.esmfold2 import EsmFold2Model as ModelClass
        cache_dir = self._resolve_cache_dir("cache_dir", "esmfold2")
        ccd_cache_dir = self._resolve_cache_dir("ccd_cache_dir", "esmfold2")

        model_name = str(self.config["model_name"])
        revision = self.config.get("revision")
        if revision is None and model_name == ESMFOLD2_HF_REPO:
            revision = ESMFOLD2_HF_REVISION
        kwargs: dict[str, Any] = {"cache_dir": str(cache_dir), "revision": revision}
        dtype = self.config.get("dtype")
        if dtype is not None:
            kwargs["dtype"] = getattr(torch, dtype) if isinstance(dtype, str) else dtype

        if self.model is None:
            if kit:
                # The kit's pinned snapshots load by repo id from HF_HOME, as its own CLI does.
                kwargs = {key: value for key, value in kwargs.items() if key not in ("cache_dir", "revision")}
            loader = ModelClass.from_pretrained if kit else partial(load_pretrained, ModelClass)
            self.model = loader(model_name, **kwargs)

        self._device = self._resolve_device()
        self.model = self.model.to(self._device)
        self.model.eval()
        ccd_cache_dir = self._kit_ccd_dir() if kit else self._ensure_ccd_cache(ccd_cache_dir)
        self.input_builder = ESMFold2InputBuilder(ccd_cache=ccd_cache_dir)
        self._configure_optimization()
        self.ready = True

    def _kit_variant(self) -> str:
        """Kit variant for the loaded checkpoint.

        The kit arms one variant per process, before any request is seen, so MSA use is chosen at construction:
        ``kit_msa=True`` selects ``full_msa``; the default ``full_nomsa`` matches the MSA-free bakeoff workload.
        """
        if str(self.config["model_name"]).endswith("ESMFold2-Fast"):
            return "fast"
        return "full_msa" if self.config["kit_msa"] else "full_nomsa"

    def _activate_optimization(self) -> None:
        """Resolve ``optimization`` on this GPU and arm the kit; runs before any weights load."""
        mode = str(self.config["optimization"])
        gpu = None if mode == "vanilla" else detect_gpu(self.config.get("device"))
        self.optimization = resolve_optimization("esmfold2", mode, gpu)
        if mode == "vanilla":
            return
        # The kit reads its weights from HF_HOME, so it is set before anything imports the kit or huggingface_hub.
        hf_home = Path(os.environ.setdefault("HF_HOME", str(Path(self.model_dir or MODAL_MODEL_DIR) / KIT_HF_SUBDIR)))
        try:
            import esmfold2_opt
        except ImportError as error:
            raise OptimizationUnavailableError(
                f"optimization={mode!r} needs the esmfold2 kit image (esmfold2_opt is not installed here)"
            ) from error
        self._ensure_kit_weights(hf_home)
        self._go_offline()
        report = esmfold2_opt.enable(mode, variant=self._kit_variant())
        if not report.get("active"):
            raise OptimizationUnavailableError(
                f"optimization={mode!r} did not activate on {gpu.name if gpu else 'this GPU'}: {report.get('reason')}"
            )

    @staticmethod
    def _go_offline() -> None:
        """Serve the kit's frozen snapshots from disk only, as the kit's own launcher does.

        ``from_pretrained("biohub/ESMFold2")`` asks for ``main``; online it resolves upstream's newest commit, downloads
        it and repoints ``refs/main`` at it, which this kit's transformers fork cannot parse. ``huggingface_hub`` and
        ``transformers`` read the switches when they are imported, and the weight fetch above has imported them with the
        switches lifted, so the already-imported modules are flipped as well.
        """
        for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
            os.environ[var] = "1"
        for module, attribute in (
            ("huggingface_hub.constants", "HF_HUB_OFFLINE"),
            ("transformers.utils.hub", "_is_offline_mode"),
        ):
            if module in sys.modules and hasattr(sys.modules[module], attribute):
                setattr(sys.modules[module], attribute, True)

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

    def _ensure_kit_weights(self, hf_home: Path) -> None:
        """Fetch the kit's pinned checkpoints into ``hf_home`` unless the variant's files are already there.

        The kit loads frozen snapshots (not boileroom's pinned revision) and refuses to start without them; the image
        carries none. Only what the variant loads is fetched: the ESMC language model, the variant's own repository and
        the pinned ``ccd.pkl``. A present file is not fetched again, and a fetch that fails or leaves a file off its pin
        raises.
        """
        from esmfold2_opt import stack, weights

        pins = stack.pins()
        variant = self._kit_variant()
        paths = {(repo, rel) for repo, rel, _ in stack.pinned_weight_files(pins, variant)} | {self._kit_ccd(pins)}
        if all((hf_home / rel).is_file() for _, rel in paths):
            return
        wanted: dict[str, set[str]] = {}
        for repo, rel in paths:
            wanted.setdefault(repo, set()).add(Path(rel).name)
        logger.info(f"Fetching the pinned ESMFold2 kit weights for variant {variant!r} into {hf_home}")
        hf_home.mkdir(parents=True, exist_ok=True)
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
        if weights.install_weights(str(hf_home), pins=scoped) != 0:
            raise OptimizationUnavailableError(
                f"could not fetch the pinned ESMFold2 kit weights into {hf_home}; the files that failed are logged above"
            )

    def _configure_optimization(self) -> None:
        """Install the armed kit's levers on the loaded model and refuse a partial application."""
        if self.optimization is None or self.optimization.active == "vanilla":
            return
        from esmfold2_opt import stack

        stack.apply_to(
            self.model,
            self.input_builder,
            trigger="boileroom",
            samples=int(self.config["num_diffusion_samples"]),
            out_dir=None,
        )
        report = stack.status()
        if report.get("partial") or not report.get("active", False):
            raise OptimizationUnavailableError(
                f"optimization={self.optimization.requested!r} applied only part of its lever set: "
                f"{report.get('partial') or report.get('reason')}"
            )

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
        self._validate_effective_config(effective_config)
        requests = self._coerce_requests(sequences)
        user_msa = effective_config.get("msa")
        if user_msa:
            self._check_kit_consumes_msa(effective_config)
            if len(requests) != 1:
                raise ValueError("ESMFold2 option 'msa' applies to exactly one input structure, not a batch.")
            requests = [self._attach_user_msa(requests[0], user_msa)]

        if self.model is None or self.input_builder is None:
            logger.warning("Model not loaded. Forcing the model to load... Next time call _load() first.")
            self._load()
        assert self.model is not None and self.input_builder is not None, "Model not loaded"

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

        metadata = dataclasses.replace(
            self._metadata_template,
            sequence_lengths=sequence_lengths,
            preprocessing_time=preprocessing_time,
            inference_time=inference_time,
            postprocessing_time=postprocessing_time,
            optimization=self.optimization.to_dict() if self.optimization else None,
        )
        return self._convert_results(results, metadata, effective_config)

    def _fold_one(
        self,
        prediction_input: StructurePredictionInput,
        config: dict[str, Any],
        request_index: int,
    ) -> tuple[list[Any], dict[str, float]]:
        """Run one ESMFold2 structure input through preprocess, model, and decode."""
        import torch
        from esm.models.esmfold2.processor import _seed_context

        assert self.model is not None and self.input_builder is not None, "Model not loaded"

        esm_input = self._to_esm_structure_prediction_input(prediction_input)
        seed = cast(int | None, config.get("seed"))
        num_diffusion_samples = int(config["num_diffusion_samples"])
        complex_id_base = str(config.get("complex_id") or "pred")
        complex_id = complex_id_base if request_index == 0 else f"{complex_id_base}_{request_index}"

        sampler_kwargs = self._sampler_kwargs(config)
        if self.optimization is not None and self.optimization.active != "vanilla":
            # The kit hooks ESMFold2InputBuilder.fold(), so kit modes fold through it (lm_dropout off, as in the direct call).
            with Timer("ESMFold2 inference") as inference_timer:
                decoded = self.input_builder.fold(
                    self.model,
                    esm_input,
                    num_loops=int(config["num_loops"]),
                    num_sampling_steps=int(config["num_sampling_steps"]),
                    num_diffusion_samples=num_diffusion_samples,
                    seed=seed,
                    lm_dropout=None,
                    msa_max_depth=config.get("msa_max_depth"),
                    msa_column_mask_rate=float(config.get("msa_column_mask_rate", 0.1)),
                    complex_id=complex_id,
                    **sampler_kwargs,
                )
            preprocess_timer = postprocess_timer = Timer("unused")
            preprocess_timer.duration = postprocess_timer.duration = 0.0
        else:
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
                    **sampler_kwargs,
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
        timing = {
            "preprocessing": preprocess_timer.duration,
            "inference": inference_timer.duration,
            "postprocessing": postprocess_timer.duration,
        }
        return results, timing

    @staticmethod
    def _validate_effective_config(config: dict[str, Any]) -> None:
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

        if config.get("templates"):
            raise ValueError("ESMFold2 does not support user-supplied 'templates'")

        msa_max_depth = config.get("msa_max_depth")
        if msa_max_depth is not None and (
            not isinstance(msa_max_depth, int) or isinstance(msa_max_depth, bool) or msa_max_depth < 1
        ):
            raise ValueError("ESMFold2 option 'msa_max_depth' must be a positive integer or None.")

    def _check_kit_consumes_msa(self, config: dict[str, Any]) -> None:
        """Refuse a user MSA that the armed kit variant would silently ignore."""
        if str(config["optimization"]) == "vanilla":
            return
        variant = self._kit_variant()
        if variant != "full_msa":
            raise ValueError(
                f"optimization={config['optimization']!r} runs the kit's {variant!r} variant, which does not consume an "
                "MSA. Use a full-model checkpoint with config={'kit_msa': True}, or optimization='vanilla'."
            )

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
            rows = _a3m_rows(text, item.sequence)
            attached.append(dataclasses.replace(item, msa=MSAInput(sequences=rows)))
        return dataclasses.replace(request, input=dataclasses.replace(request.input, sequences=attached))

    @staticmethod
    def _sampler_kwargs(config: dict[str, Any]) -> dict[str, Any]:
        """Return optional sampler controls accepted by the Biohub model call."""
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
