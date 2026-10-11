"""Resident ProFam generation through the pinned upstream public Python API."""

from __future__ import annotations

import hashlib
import importlib
import math
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

from ...base import GenerationAlgorithm, PredictionMetadata
from ...optimization import describe_gpu
from ...provenance import runtime_provenance
from .types import ConditioningPrompt, ProFamOutput

UPSTREAM_COMMIT = "a19061606dc6469424035db3fdcaae6e63688017"
CHECKPOINT_REPO = "judewells/ProFam-1"
CHECKPOINT_REVISION = "fe4084f356480fd048a23097fcc133f1e5ba019d"
CHECKPOINT_SHA256 = "8ec54d80323f188d3454fb4457f63f12158fe55552b919554806b12730c30e08"
AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")


class ProFamCore(GenerationAlgorithm):
    """Load once, generate repeatedly, and discard model state on close.

    Input strings are raw amino-acid sequences, not FASTA paths, A3M alignments,
    or colon-separated chains. A sequence list is one family prompt.
    """

    DEFAULT_CONFIG = MappingProxyType(
        {
            "device": None,
            "cache_dir": None,
            "dtype": "auto",
            "attn_implementation": "sdpa",
            "num_samples": 8,
            "temperature": 1.0,
            "top_p": 0.95,
            "sampler": "single",
            "max_context_tokens": 4096,
            "max_generated_length": None,
            "max_sequence_length_multiplier": 1.2,
            "num_prompts_in_ensemble": 4,
            "reduction": "mean_probs",
            "minimum_sequence_length_proportion": 0.5,
            "maximum_retries": 5,
            "repeat_guard": True,
            "seed": None,
            "prompt_accessions": None,
        }
    )
    STATIC_CONFIG_KEYS = frozenset({"device", "cache_dir", "dtype", "attn_implementation"})

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        self.name = "ProFam"
        self.version = "ProFam-1"
        self.model: Any = None
        self._runtime: dict[str, str] = {}

    def _validate_config(self, config: Mapping[str, Any]) -> None:
        super()._validate_config(config)
        for name in ("num_samples", "max_context_tokens", "num_prompts_in_ensemble", "maximum_retries"):
            value = config[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if config["max_context_tokens"] > 8192:
            raise ValueError("max_context_tokens must be <= 8192 (the ProFam training context)")
        length = config["max_generated_length"]
        if length is not None and (isinstance(length, bool) or not isinstance(length, int) or length < 1):
            raise ValueError("max_generated_length must be a positive integer or None")
        seed = config["seed"]
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32):
            raise ValueError("seed must be an integer in [0, 2**32) or None")
        for name in ("temperature", "top_p", "max_sequence_length_multiplier", "minimum_sequence_length_proportion"):
            value = config[name]
            if value is None and name == "temperature":
                continue
            if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
            if name in {"top_p", "minimum_sequence_length_proportion"} and value > 1:
                raise ValueError(f"{name} must be <= 1")
        for name, choices in {
            "sampler": {"single", "ensemble"},
            "reduction": {"mean_probs", "sum_log_probs"},
            "dtype": {"auto", "float32", "float16", "bfloat16"},
            "attn_implementation": {"sdpa", "eager"},
        }.items():
            if not isinstance(config[name], str) or config[name] not in choices:
                raise ValueError(f"{name} must be one of {sorted(choices)}")
        if not isinstance(config["repeat_guard"], bool):
            raise ValueError("repeat_guard must be a boolean")
        for name in ("cache_dir", "device"):
            if config[name] is not None and (not isinstance(config[name], str) or not config[name]):
                raise ValueError(f"{name} must be a nonempty string or None")
        accessions = config["prompt_accessions"]
        if accessions is not None and (
            not isinstance(accessions, list)
            or any(not isinstance(x, str) or not x for x in accessions)
            or len(set(accessions)) != len(accessions)
        ):
            raise ValueError("prompt_accessions must be a list of unique nonempty strings or None")

    def _validate_sequences(self, sequences: str | Sequence[str]) -> list[str]:
        if isinstance(sequences, str):
            sequences = [sequences]
        if not isinstance(sequences, Sequence) or not sequences:
            raise ValueError("ProFam requires a nonempty family prompt of raw amino-acid sequences")
        for sequence in sequences:
            if not isinstance(sequence, str) or not sequence or not set(sequence) <= AMINO_ACIDS:
                raise ValueError(
                    "ProFam prompts must contain only uppercase canonical amino acids, without gaps or ':'"
                )
        return list(sequences)

    def validate_request(self, sequences: str | Sequence[str], options: dict | None = None) -> tuple[list[str], dict]:
        """Validate all caller-controlled generation settings before inference."""
        prompt = self._validate_sequences(sequences)
        merged = self._merge_options(options)
        generated_limit = max(map(len, prompt)) * merged["max_sequence_length_multiplier"]
        if not math.isfinite(generated_limit):
            raise ValueError("max_sequence_length_multiplier produces a nonfinite generation cap")
        max_length = int(generated_limit)
        if merged["max_generated_length"] is not None:
            max_length = min(max_length, merged["max_generated_length"])
        # Reserve [RAW], two [SEP] tokens and one complete shortest sequence.
        # Upstream truncates/subsamples the prompt to the remaining budget.
        if max_length < 1 or max_length + min(map(len, prompt)) + 3 > merged["max_context_tokens"]:
            raise ValueError(
                "max_context_tokens cannot fit a complete prompt sequence and the requested generation cap"
            )
        if merged["prompt_accessions"] is not None and len(merged["prompt_accessions"]) != len(prompt):
            raise ValueError("prompt_accessions must match the number of prompt sequences")
        return prompt, merged

    def _initialize(self) -> None:
        self._load()

    def _load(self) -> None:
        if self.ready:
            return
        import torch

        upstream_profam = importlib.import_module("profam").ProFam

        device = self._resolve_device()
        dtype = self.config["dtype"]
        if dtype == "auto":
            dtype = (
                "float32"
                if device.type != "cuda"
                else ("bfloat16" if torch.cuda.get_device_capability(device)[0] >= 8 else "float16")
            )
        from huggingface_hub import hf_hub_download

        cache = self.config["cache_dir"] or str(
            Path(os.environ.get("MODEL_DIR", str(Path.home() / ".cache/boileroom"))) / "profam"
        )
        path = Path(
            hf_hub_download(
                repo_id=CHECKPOINT_REPO,
                filename="checkpoints/last.ckpt",
                revision=CHECKPOINT_REVISION,
                cache_dir=cache,
                token=False,
            )
        )
        # Upstream loads a Lightning pickle (weights_only=False). Only this
        # independently pinned official artifact is allowed across that boundary.
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest != CHECKPOINT_SHA256:
            raise RuntimeError("ProFam checkpoint SHA256 mismatch; refusing unsafe checkpoint deserialization")
        model = upstream_profam(
            checkpoint=str(path),
            device=str(device),
            dtype=dtype,
            attn_implementation=self.config["attn_implementation"],
            auto_download=False,
        )
        self._runtime = runtime_provenance(
            packages=("profam", "transformers"),
            gpu=describe_gpu(str(device)),
            extra={
                "profam.commit": os.environ.get("BOILEROOM_PROFAM_COMMIT", "unknown"),
                "profam.checkpoint": CHECKPOINT_REPO,
                "profam.checkpoint_revision": CHECKPOINT_REVISION,
                "profam.checkpoint_sha256": digest,
                "profam.dtype": dtype,
                "profam.attention": self.config["attn_implementation"],
                "profam.device": str(device),
            },
        )
        self.model = model
        self.ready = True

    def generate(self, sequences: str | Sequence[str], options: dict | None = None) -> ProFamOutput:
        prompt, merged = self.validate_request(sequences, options)
        self._load()
        import torch

        kwargs = {key: merged[key] for key in self.DEFAULT_CONFIG if key not in self.STATIC_CONFIG_KEYS}
        kwargs["max_tokens"] = kwargs.pop("max_context_tokens")
        started = time.perf_counter()
        with torch.inference_mode():
            result = self.model.generate(prompt=prompt, **kwargs)
        sequences_out = list(result.sequences)
        scores = [float(value) for value in result.scores]
        if len(sequences_out) != merged["num_samples"] or len(scores) != len(sequences_out):
            raise RuntimeError(
                "ProFam returned fewer samples or scores than requested; increase maximum_retries or adjust filters"
            )
        if any(not sequence or not set(sequence) <= AMINO_ACIDS for sequence in sequences_out) or any(
            not math.isfinite(score) for score in scores
        ):
            raise RuntimeError("ProFam returned invalid amino-acid sequences or nonfinite sampling scores")
        effective_cap = int(max(map(len, prompt)) * merged["max_sequence_length_multiplier"])
        if merged["max_generated_length"] is not None:
            effective_cap = min(effective_cap, merged["max_generated_length"])
        sampling_facts = {
            f"profam.{key}": str(merged[key])
            for key in (
                "sampler",
                "seed",
                "num_samples",
                "temperature",
                "top_p",
                "reduction",
                "max_context_tokens",
                "maximum_retries",
                "repeat_guard",
                "minimum_sequence_length_proportion",
            )
        }
        sampling_facts["profam.generation_cap"] = str(effective_cap)
        sampling_facts["profam.num_conditioning_prompts"] = str(len(result.conditioning_prompts or []))
        return ProFamOutput(
            metadata=PredictionMetadata(
                model_name=self.name,
                model_version=self.version,
                sequence_lengths=[len(s) for s in sequences_out],
                inference_time=time.perf_counter() - started,
                runtime={**self._runtime, **sampling_facts},
            ),
            sequences=sequences_out,
            log_likelihoods=scores,
            conditioning_prompts=[
                ConditioningPrompt(sequences=list(item.sequences), accessions=list(item.accessions))
                for item in (result.conditioning_prompts or [])
            ],
        )

    def close(self) -> None:
        """Release resident weights; a later call explicitly loads a fresh model."""
        self.model = None
        self.ready = False
        self._runtime = {}
