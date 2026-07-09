"""Core ESM-C and ESM3 embedding implementations without Modal dependencies."""

from __future__ import annotations

import dataclasses
import importlib.metadata
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, cast

import numpy as np
import torch

from ...base import EmbeddingAlgorithm
from ...utils import Timer, get_model_cache_dir, validate_sequence
from .types import ESM3Output, ESMCOutput, ESMEmbeddingOutput

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ESM3ParsedSequence:
    """Residue and chain metadata for an ESM-C/ESM3 input sequence."""

    original: str
    sdk_sequence: str
    chain_index: np.ndarray
    residue_index: np.ndarray

    @property
    def residue_count(self) -> int:
        return int(self.chain_index.shape[0])


def parse_esm3_sequences(sequences: str | Sequence[str]) -> list[ESM3ParsedSequence]:
    """Parse ESM-C/ESM3 inputs and map colon chain separators to SDK breaks.

    ``AAA:BBB`` is submitted to the Biohub ESM SDK as ``AAA|BBB`` while
    residue indices remain aligned to the original residues only.
    """

    sequence_list = [sequences] if isinstance(sequences, str) else list(sequences)
    if not sequence_list:
        raise ValueError("ESM-C/ESM3 embedding input must contain at least one sequence.")

    parsed: list[ESM3ParsedSequence] = []
    for sequence in sequence_list:
        validate_sequence(sequence)
        chains = sequence.split(":")
        if any(chain == "" for chain in chains):
            raise ValueError("ESM-C/ESM3 sequences must not contain empty chains.")
        chain_index: list[int] = []
        residue_index: list[int] = []
        for chain_id, chain in enumerate(chains):
            chain_index.extend([chain_id] * len(chain))
            residue_index.extend(range(len(chain)))
        parsed.append(
            ESM3ParsedSequence(
                original=sequence,
                sdk_sequence="|".join(chains),
                chain_index=np.asarray(chain_index, dtype=np.int32),
                residue_index=np.asarray(residue_index, dtype=np.int32),
            )
        )
    return parsed


def _pad_and_stack(arrays: list[np.ndarray], residue_axis: int, pad_value: float | int) -> np.ndarray:
    if not arrays:
        raise ValueError("Cannot pad an empty batch of residue arrays.")
    max_residues = max(array.shape[residue_axis] for array in arrays) if arrays else 0
    padded: list[np.ndarray] = []
    for array in arrays:
        padding = [(0, 0)] * array.ndim
        padding[residue_axis] = (0, max_residues - array.shape[residue_axis])
        padded.append(np.pad(array, padding, mode="constant", constant_values=pad_value))
    return np.stack(padded, axis=0)


def pad_residue_arrays(
    embeddings: list[np.ndarray],
    chain_index: list[np.ndarray],
    residue_index: list[np.ndarray],
    hidden_states: list[np.ndarray] | None = None,
    lm_logits: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray, np.ndarray]:
    """Pad residue-aligned arrays to a common batch length."""

    padded_embeddings = _pad_and_stack(embeddings, residue_axis=0, pad_value=0)
    padded_hidden_states = (
        _pad_and_stack(hidden_states, residue_axis=1, pad_value=0).swapaxes(0, 1) if hidden_states else None
    )
    padded_lm_logits = _pad_and_stack(lm_logits, residue_axis=0, pad_value=0) if lm_logits else None
    padded_chain_index = _pad_and_stack(chain_index, residue_axis=0, pad_value=-1).astype(np.int32, copy=False)
    padded_residue_index = _pad_and_stack(residue_index, residue_axis=0, pad_value=-1).astype(np.int32, copy=False)
    return padded_embeddings, padded_hidden_states, padded_lm_logits, padded_chain_index, padded_residue_index


class _BaseESM3EmbeddingCore(EmbeddingAlgorithm):
    """Shared SDK-backed embedding logic for ESM-C and ESM3."""

    DEFAULT_CONFIG: ClassVar[dict[str, Any]] = {
        "device": "cuda:0",
        "model_name": "",
        "include_fields": None,
    }
    STATIC_CONFIG_KEYS: ClassVar[frozenset[str]] = frozenset({"device", "model_name"})
    MODEL_DISPLAY_NAME: ClassVar[str]
    MODEL_KIND: ClassVar[Literal["esmc", "esm3"]]
    VALID_MODEL_NAMES: ClassVar[frozenset[str]]
    MODEL_ALIASES: ClassVar[dict[str, str]] = {}
    SUPPORTED_OPTIONAL_FIELDS: ClassVar[frozenset[str]] = frozenset({"lm_logits"})

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        configured_model = str(self.config["model_name"])
        canonical_model = self._canonical_model_name(configured_model)
        self.config["model_name"] = canonical_model
        self._metadata_template = self._initialize_metadata(
            model_name=self.MODEL_DISPLAY_NAME,
            model_version=self._sdk_version_fallback(canonical_model),
        )
        self._device: str | torch.device | None = None
        self.model: Any | None = None
        self.assert_valid_model(canonical_model)

    @classmethod
    def _canonical_model_name(cls, model_name: str) -> str:
        return cls.MODEL_ALIASES.get(model_name, model_name)

    @classmethod
    def assert_valid_model(cls, model_name: str) -> None:
        canonical = cls._canonical_model_name(model_name)
        if canonical not in cls.VALID_MODEL_NAMES:
            supported = ", ".join(sorted(cls.VALID_MODEL_NAMES | frozenset(cls.MODEL_ALIASES)))
            raise ValueError(f"Unsupported {cls.MODEL_DISPLAY_NAME} model: {model_name}. Supported models: {supported}")

    @staticmethod
    def _sdk_version_fallback(model_name: str) -> str:
        try:
            return importlib.metadata.version("esm")
        except importlib.metadata.PackageNotFoundError:
            return model_name

    def _initialize(self) -> None:
        self._load()

    def _load(self) -> None:
        """Load the configured Biohub MIT ``esm`` SDK model."""

        if self.model is not None:
            return
        self._configure_huggingface_cache()
        if self.MODEL_KIND == "esmc":
            from esm.models.esmc import ESMC as SDKModel
        else:
            from esm.models.esm3 import ESM3 as SDKModel
        import esm

        self._device = self.config["device"]
        self.model = SDKModel.from_pretrained(self.config["model_name"]).to(self._device).eval()
        self._metadata_template.model_version = getattr(esm, "__version__", self._metadata_template.model_version)
        self.ready = True

    @staticmethod
    def _configure_huggingface_cache() -> None:
        """Route SDK/Hugging Face downloads through Boileroom's model cache."""

        cache_dir = get_model_cache_dir("esm3") / "huggingface"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(cache_dir))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_dir / "hub"))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir / "transformers"))

    def _include_requested(self, include_fields: list[str] | tuple[str, ...] | None, field: str) -> bool:
        if include_fields is None:
            return False
        return field in include_fields or ("*" in include_fields and field in self.SUPPORTED_OPTIONAL_FIELDS)

    def _validate_include_fields(self, include_fields: list[str] | tuple[str, ...] | None) -> None:
        if (
            include_fields is not None
            and "hidden_states" in include_fields
            and "hidden_states" not in self.SUPPORTED_OPTIONAL_FIELDS
        ):
            raise ValueError("hidden_states are not supported for ESM3 embeddings by the current SDK wrapper.")

    def embed(self, sequences: str | Sequence[str], options: dict | None = None) -> ESMEmbeddingOutput:
        """Compute residue-only embeddings using the Biohub MIT ESM SDK."""

        effective_config = self._merge_options(options)
        include_fields = effective_config.get("include_fields")
        self._validate_include_fields(include_fields)
        parsed_sequences = parse_esm3_sequences(sequences)

        if self.model is None:
            logger.warning("Model not loaded. Forcing the model to load... Next time call _load() first.")
            self._load()
        assert self.model is not None, "Model not loaded"

        metadata = dataclasses.replace(
            self._metadata_template,
            sequence_lengths=[parsed.residue_count for parsed in parsed_sequences],
        )

        compute_hidden_states = self._include_requested(include_fields, "hidden_states")
        compute_lm_logits = self._include_requested(include_fields, "lm_logits")

        with Timer(f"{self.MODEL_DISPLAY_NAME} preprocessing") as preprocess_timer:
            encoded_inputs = [self._encode_sequence(parsed.sdk_sequence) for parsed in parsed_sequences]

        embeddings: list[np.ndarray] = []
        hidden_states: list[np.ndarray] | None = [] if compute_hidden_states else None
        lm_logits: list[np.ndarray] | None = [] if compute_lm_logits else None

        with Timer("Model Inference") as inference_timer, torch.inference_mode():
            for parsed, encoded in zip(parsed_sequences, encoded_inputs, strict=True):
                raw_output = self._run_logits(encoded, compute_lm_logits, compute_hidden_states)
                keep_indices = self._residue_token_indices(parsed.sdk_sequence)
                embeddings.append(
                    self._select_residue_rows(self._extract_array(raw_output, "embeddings"), keep_indices)
                )
                if hidden_states is not None:
                    hidden_array = self._extract_optional_array(raw_output, "hidden_states")
                    if hidden_array is None:
                        raise ValueError("hidden_states were requested but the ESM SDK did not return them.")
                    hidden_states.append(self._select_hidden_residue_rows(hidden_array, keep_indices))
                if lm_logits is not None:
                    logits_array = self._extract_optional_array(
                        raw_output, "logits.sequence", "sequence_logits", "lm_logits"
                    )
                    if logits_array is None:
                        raise ValueError("lm_logits were requested but the ESM SDK did not return sequence logits.")
                    lm_logits.append(self._select_residue_rows(logits_array, keep_indices))

        with Timer(f"{self.MODEL_DISPLAY_NAME} postprocessing") as postprocess_timer:
            padded = pad_residue_arrays(
                embeddings=embeddings,
                hidden_states=hidden_states,
                lm_logits=lm_logits,
                chain_index=[parsed.chain_index for parsed in parsed_sequences],
                residue_index=[parsed.residue_index for parsed in parsed_sequences],
            )
            embeddings_out, hidden_out, logits_out, chain_out, residue_out = padded
            metadata.preprocessing_time = preprocess_timer.duration
            metadata.inference_time = inference_timer.duration
            metadata.postprocessing_time = postprocess_timer.duration
            full_output = ESMEmbeddingOutput(
                metadata=metadata,
                embeddings=embeddings_out,
                hidden_states=hidden_out,
                lm_logits=logits_out,
                chain_index=chain_out,
                residue_index=residue_out,
            )
            filtered = self._filter_include_fields(full_output, cast(list[str] | None, include_fields))
            return cast(ESMEmbeddingOutput, filtered)

    def _encode_sequence(self, sdk_sequence: str) -> Any:
        from esm.sdk.api import ESMProtein

        assert self.model is not None, "Model not loaded"
        return self.model.encode(ESMProtein(sequence=sdk_sequence))

    def _run_logits(self, encoded: Any, compute_lm_logits: bool, compute_hidden_states: bool) -> Any:
        from esm.sdk.api import LogitsConfig

        assert self.model is not None, "Model not loaded"
        config_kwargs = {
            "sequence": compute_lm_logits,
            "return_embeddings": True,
            "return_hidden_states": compute_hidden_states,
        }
        return self.model.logits(encoded, LogitsConfig(**config_kwargs))

    @staticmethod
    def _residue_token_indices(sdk_sequence: str) -> np.ndarray:
        # The SDK adds BOS and EOS around the sequence. Chain breaks are literal '|'
        # positions in the encoded stream and must be removed from public outputs.
        return np.asarray([index + 1 for index, token in enumerate(sdk_sequence) if token != "|"], dtype=np.int64)

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            # The Biohub ESM-C/ESM3 SDK runs in bfloat16 on CUDA; NumPy has no
            # bfloat16, so upcast floating tensors to float32 before conversion.
            if value.is_floating_point():
                value = value.to(torch.float32)
            return value.detach().cpu().numpy()
        return np.asarray(value)

    @classmethod
    def _extract_array(cls, output: Any, *names: str) -> np.ndarray:
        array = cls._extract_optional_array(output, *names)
        if array is None:
            raise ValueError(f"ESM SDK output did not include any of: {', '.join(names)}")
        return array

    @classmethod
    def _extract_optional_array(cls, output: Any, *names: str) -> np.ndarray | None:
        for name in names:
            value = cls._get_nested_value(output, name)
            if value is not None:
                return cls._to_numpy(value)
        return None

    @staticmethod
    def _get_nested_value(output: Any, name: str) -> Any | None:
        value: Any = output
        for part in name.split("."):
            value = value.get(part) if isinstance(value, dict) else getattr(value, part, None)
            if value is None:
                return None
        return value

    @staticmethod
    def _select_residue_rows(array: np.ndarray, keep_indices: np.ndarray) -> np.ndarray:
        if array.ndim == 3 and array.shape[0] == 1:
            array = array[0]
        if array.ndim != 2:
            raise ValueError(f"Expected token array with shape (tokens, features); got {array.shape}")
        return array[keep_indices]

    @staticmethod
    def _select_hidden_residue_rows(array: np.ndarray, keep_indices: np.ndarray) -> np.ndarray:
        if array.ndim == 4 and array.shape[0] == 1:
            array = array[0]
        if array.ndim == 4 and array.shape[1] == 1:
            array = array[:, 0]
        if array.ndim == 3:
            # Preferred public shape is (layers, residues, features).
            return array[:, keep_indices, :]
        if array.ndim == 2:
            return array[keep_indices][None, :, :]
        raise ValueError(f"Expected hidden states with shape (layers, tokens, features); got {array.shape}")


class ESMCCore(_BaseESM3EmbeddingCore):
    """ESM-C embedding model backed by the official ``esm`` SDK."""

    DEFAULT_CONFIG: ClassVar[dict[str, Any]] = {
        "device": "cuda:0",
        "model_name": "esmc_300m",
        "include_fields": None,
    }
    MODEL_DISPLAY_NAME: ClassVar[str] = "ESM-C"
    MODEL_KIND: ClassVar[Literal["esmc", "esm3"]] = "esmc"
    VALID_MODEL_NAMES: ClassVar[frozenset[str]] = frozenset({"esmc_300m", "esmc_600m"})
    SUPPORTED_OPTIONAL_FIELDS: ClassVar[frozenset[str]] = frozenset({"hidden_states", "lm_logits"})

    def embed(self, sequences: str | Sequence[str], options: dict | None = None) -> ESMCOutput:
        return cast(ESMCOutput, super().embed(sequences, options=options))


class ESM3Core(_BaseESM3EmbeddingCore):
    """ESM3 embedding-only model backed by the official ``esm`` SDK."""

    DEFAULT_CONFIG: ClassVar[dict[str, Any]] = {
        "device": "cuda:0",
        "model_name": "esm3_sm_open_v1",
        "include_fields": None,
    }
    MODEL_DISPLAY_NAME: ClassVar[str] = "ESM3"
    MODEL_KIND: ClassVar[Literal["esmc", "esm3"]] = "esm3"
    VALID_MODEL_NAMES: ClassVar[frozenset[str]] = frozenset({"esm3_sm_open_v1"})
    MODEL_ALIASES: ClassVar[dict[str, str]] = {
        "esm3-open": "esm3_sm_open_v1",
        "esm3-sm-open-v1": "esm3_sm_open_v1",
        "esm3-open-2024-03": "esm3_sm_open_v1",
    }
    SUPPORTED_OPTIONAL_FIELDS: ClassVar[frozenset[str]] = frozenset({"lm_logits"})

    def embed(self, sequences: str | Sequence[str], options: dict | None = None) -> ESM3Output:
        return cast(ESM3Output, super().embed(sequences, options=options))


__all__ = [
    "ESM3Core",
    "ESM3ParsedSequence",
    "ESMCCore",
    "pad_residue_arrays",
    "parse_esm3_sequences",
]
