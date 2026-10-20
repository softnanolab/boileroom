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
from .types import (
    ESM3InverseFoldingOutput,
    ESM3InverseGenerationOutput,
    ESM3Output,
    ESMCOutput,
    ESMEmbeddingOutput,
)

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


# ESM3-only optional per-residue "track logit" outputs. ESM3 is an all-to-all
# masked model, so any track can be predicted from sequence alone. Each entry maps
# the public output field to (LogitsConfig kwarg, source attribute paths on the SDK
# LogitsOutput to try in order). The structure/folding track is intentionally
# excluded. Residue-annotation logits live at the top level of LogitsOutput (they
# are multi-hot), the rest under ``logits.<track>``.
ESM3_TRACK_LOGIT_FIELDS: dict[str, tuple[str, tuple[str, ...]]] = {
    "sasa_logits": ("sasa", ("logits.sasa", "sasa")),
    "secondary_structure_logits": ("secondary_structure", ("logits.secondary_structure", "secondary_structure")),
    "function_logits": ("function", ("logits.function", "function")),
    "residue_annotation_logits": ("residue_annotations", ("residue_annotation_logits", "residue_annotations")),
}


# Standard amino acids, in the label order of ``ESM3Core.inverse_fold`` logits.
INVERSE_FOLDING_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
# Sequence-vocabulary tokens that are not one of the 20 standard amino acids but can still be sampled at a
# residue position: the ambiguity codes (X, B, Z), the two non-canonical residues (U, O) and the structural
# tokens (".", "-", "|"). Generation must never put one of these in a redesigned position.
NON_STANDARD_SEQUENCE_TOKENS = ("X", "B", "U", "Z", "O", ".", "-", "|")


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
    # Per-residue track-logit outputs this model can produce (empty by default;
    # ESM3 populates it). See ESM3_TRACK_LOGIT_FIELDS for the value shape.
    TRACK_LOGIT_FIELDS: ClassVar[dict[str, tuple[str, tuple[str, ...]]]] = {}

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

        # ESM3-only per-residue track logits (SASA, secondary structure, function,
        # residue annotations). Reject any requested on a model that lacks the track.
        for field in ESM3_TRACK_LOGIT_FIELDS:
            if self._include_requested(include_fields, field) and field not in self.TRACK_LOGIT_FIELDS:
                raise ValueError(f"{field} are only available for ESM3, not {self.MODEL_DISPLAY_NAME}.")
        requested_tracks = [
            field for field in self.TRACK_LOGIT_FIELDS if self._include_requested(include_fields, field)
        ]
        track_flags = {self.TRACK_LOGIT_FIELDS[field][0]: True for field in requested_tracks}

        with Timer(f"{self.MODEL_DISPLAY_NAME} preprocessing") as preprocess_timer:
            encoded_inputs = [self._encode_sequence(parsed.sdk_sequence) for parsed in parsed_sequences]

        embeddings: list[np.ndarray] = []
        hidden_states: list[np.ndarray] | None = [] if compute_hidden_states else None
        lm_logits: list[np.ndarray] | None = [] if compute_lm_logits else None
        track_arrays: dict[str, list[np.ndarray]] = {field: [] for field in requested_tracks}

        with Timer("Model Inference") as inference_timer, torch.inference_mode():
            for parsed, encoded in zip(parsed_sequences, encoded_inputs, strict=True):
                raw_output = self._run_logits(encoded, compute_lm_logits, compute_hidden_states, track_flags)
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
                for field in requested_tracks:
                    sources = self.TRACK_LOGIT_FIELDS[field][1]
                    track_array = self._extract_optional_array(raw_output, *sources)
                    if track_array is None:
                        raise ValueError(f"{field} were requested but the ESM SDK did not return them.")
                    track_arrays[field].append(self._select_residue_rows(track_array, keep_indices))

        with Timer(f"{self.MODEL_DISPLAY_NAME} postprocessing") as postprocess_timer:
            padded = pad_residue_arrays(
                embeddings=embeddings,
                hidden_states=hidden_states,
                lm_logits=lm_logits,
                chain_index=[parsed.chain_index for parsed in parsed_sequences],
                residue_index=[parsed.residue_index for parsed in parsed_sequences],
            )
            embeddings_out, hidden_out, logits_out, chain_out, residue_out = padded
            padded_tracks = {
                field: _pad_and_stack(arrays, residue_axis=0, pad_value=0) for field, arrays in track_arrays.items()
            }
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
                **padded_tracks,
            )
            filtered = self._filter_include_fields(full_output, cast(list[str] | None, include_fields))
            return cast(ESMEmbeddingOutput, filtered)

    def _encode_sequence(self, sdk_sequence: str) -> Any:
        from esm.sdk.api import ESMProtein

        assert self.model is not None, "Model not loaded"
        return self.model.encode(ESMProtein(sequence=sdk_sequence))

    def _run_logits(
        self,
        encoded: Any,
        compute_lm_logits: bool,
        compute_hidden_states: bool,
        track_flags: dict[str, bool] | None = None,
    ) -> Any:
        """Run the SDK ``logits`` call, requesting embeddings plus any enabled tracks.

        Parameters
        ----------
        encoded : Any
            The SDK-encoded protein to score.
        compute_lm_logits : bool
            Whether to request sequence (masked-LM) logits.
        compute_hidden_states : bool
            Whether to request per-layer hidden states.
        track_flags : dict[str, bool] | None
            Extra ``LogitsConfig`` track flags to enable (e.g. ``{"sasa": True}``).

        Returns
        -------
        Any
            The SDK ``LogitsOutput`` for the encoded input.
        """
        from esm.sdk.api import LogitsConfig

        assert self.model is not None, "Model not loaded"
        config_kwargs: dict[str, Any] = {
            "sequence": compute_lm_logits,
            "return_embeddings": True,
            "return_hidden_states": compute_hidden_states,
        }
        if track_flags:
            config_kwargs.update(track_flags)
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
    TRACK_LOGIT_FIELDS: ClassVar[dict[str, tuple[str, tuple[str, ...]]]] = ESM3_TRACK_LOGIT_FIELDS
    SUPPORTED_OPTIONAL_FIELDS: ClassVar[frozenset[str]] = frozenset({"lm_logits", *ESM3_TRACK_LOGIT_FIELDS})

    def embed(self, sequences: str | Sequence[str], options: dict | None = None) -> ESM3Output:
        return cast(ESM3Output, super().embed(sequences, options=options))

    def _prepare_masked_protein(
        self,
        sequence: str,
        backbone_coordinates: np.ndarray,
        positions: Sequence[int],
    ) -> tuple[ESM3ParsedSequence, np.ndarray, np.ndarray, str, np.ndarray]:
        """Validate the inputs and build the masked, structure-conditioned inputs for the SDK.

        Shared by :meth:`inverse_fold` and :meth:`inverse_generate`. Both mask the residues at
        ``positions`` and condition on ``backbone_coordinates``; they differ only in how the
        sequence track is read back out. Loads the model if it is not loaded yet.

        Parameters
        ----------
        sequence : str
            Amino-acid sequence; chains separated by ``:``.
        backbone_coordinates : np.ndarray
            ``(n_residues, 3, 3)`` N, CA, C coordinates in Angstrom, one row per residue (chain
            breaks excluded). Residues with missing coordinates may be NaN.
        positions : Sequence[int]
            Residue indices to mask, counted over all residues (chain breaks excluded).

        Returns
        -------
        tuple
            ``(parsed, position_array, keep_indices, masked_sdk_sequence, atom37)``, where
            ``keep_indices`` maps each residue index to its offset within ``parsed.sdk_sequence``
            and ``atom37`` holds the coordinates in the atom37 layout the SDK expects.

        Raises
        ------
        ValueError
            If the coordinate shape does not match the sequence, or ``positions`` is empty, out of
            range, or contains duplicates.
        """
        parsed = parse_esm3_sequences(sequence)[0]
        n_residues = parsed.residue_count
        coordinates = np.asarray(backbone_coordinates, dtype=np.float32)
        if coordinates.shape != (n_residues, 3, 3):
            raise ValueError(
                f"backbone_coordinates must have shape ({n_residues}, 3, 3) matching the sequence; "
                f"got {coordinates.shape}."
            )
        position_array = np.asarray(list(positions), dtype=np.int64)
        if position_array.ndim != 1 or position_array.size == 0:
            raise ValueError("positions must be a non-empty 1D sequence of residue indices.")
        if position_array.min() < 0 or position_array.max() >= n_residues:
            raise ValueError(f"positions must lie in [0, {n_residues}); got {position_array.tolist()}.")
        if len(set(position_array.tolist())) != position_array.size:
            raise ValueError("positions must not contain duplicates.")

        if self.model is None:
            logger.warning("Model not loaded. Forcing the model to load... Next time call _load() first.")
            self._load()
        assert self.model is not None, "Model not loaded"

        keep_indices = self._residue_token_indices(parsed.sdk_sequence) - 1  # positions within sdk_sequence
        masked_sdk = list(parsed.sdk_sequence)
        for position in position_array:
            masked_sdk[int(keep_indices[position])] = "_"

        # SDK expects atom37 coordinates aligned to sdk_sequence (NaN rows at chain breaks / unused atoms).
        atom37 = np.full((len(parsed.sdk_sequence), 37, 3), np.nan, dtype=np.float32)
        atom37[keep_indices, :3] = coordinates
        return parsed, position_array, keep_indices, "".join(masked_sdk), atom37

    def inverse_fold(
        self,
        sequence: str,
        backbone_coordinates: np.ndarray,
        positions: Sequence[int],
    ) -> ESM3InverseFoldingOutput:
        """Predict amino-acid logits at masked positions given the rest of the sequence and a backbone.

        The residues at ``positions`` are replaced by the SDK mask token; every other residue stays
        fixed. The structure is supplied as conditioning input, and ESM3's sequence track is read out
        at the masked positions (all positions are scored in a single forward pass, so the result for
        each masked position is conditioned on the other masked positions being masked as well).

        Parameters
        ----------
        sequence : str
            Amino-acid sequence; chains separated by ``:``.
        backbone_coordinates : np.ndarray
            ``(n_residues, 3, 3)`` N, CA, C coordinates in Angstrom, one row per residue (chain
            breaks excluded). Residues with missing coordinates may be NaN.
        positions : Sequence[int]
            Residue indices to mask, counted over all residues (chain breaks excluded).

        Returns
        -------
        ESM3InverseFoldingOutput
            Logits over the 20 standard amino acids at each masked position.
        """

        parsed, position_array, keep_indices, masked_sdk_sequence, atom37 = self._prepare_masked_protein(
            sequence, backbone_coordinates, positions
        )
        assert self.model is not None, "Model not loaded"
        from esm.sdk.api import ESMProtein, LogitsConfig

        metadata = dataclasses.replace(self._metadata_template, sequence_lengths=[parsed.residue_count])
        with Timer("ESM3 preprocessing") as preprocess_timer:
            protein = ESMProtein(sequence=masked_sdk_sequence, coordinates=torch.from_numpy(atom37))
            encoded = self.model.encode(protein)
        with Timer("Model Inference") as inference_timer, torch.inference_mode():
            raw_output = self.model.logits(encoded, LogitsConfig(sequence=True))
        with Timer("ESM3 postprocessing") as postprocess_timer:
            logits = self._extract_array(raw_output, "logits.sequence", "sequence_logits")
            if logits.ndim == 3 and logits.shape[0] == 1:
                logits = logits[0]
            token_ids = self.model.tokenizers.sequence.convert_tokens_to_ids(list(INVERSE_FOLDING_AMINO_ACIDS))
            # +1 skips the BOS token the SDK prepends to the encoded stream.
            token_rows = keep_indices[position_array] + 1
            selected = logits[token_rows][:, np.asarray(token_ids, dtype=np.int64)].astype(np.float32)
        metadata.preprocessing_time = preprocess_timer.duration
        metadata.inference_time = inference_timer.duration
        metadata.postprocessing_time = postprocess_timer.duration
        return ESM3InverseFoldingOutput(
            metadata=metadata,
            positions=position_array.astype(np.int32),
            logits=selected,
            amino_acids=INVERSE_FOLDING_AMINO_ACIDS,
        )

    def inverse_generate(
        self,
        sequence: str,
        backbone_coordinates: np.ndarray,
        positions: Sequence[int],
        num_steps: int | None = None,
        temperature: float = 0.1,
        seed: int | None = None,
        excluded_amino_acids: str = "",
    ) -> ESM3InverseGenerationOutput:
        """Jointly decode the residues at ``positions``, given the rest of the sequence and a backbone.

        Where :meth:`inverse_fold` scores every masked position in one forward pass, with the other
        masked positions still masked, this unmasks them over ``num_steps`` iterative rounds on a
        cosine schedule, so each residue is sampled in the context of those already decoded. Use it
        to redesign a whole stretch at once; use ``inverse_fold`` for per-position probabilities.

        Residues outside ``positions`` are held fixed, and the structure is conditioning input
        throughout, so the result is a redesign of the masked residues on the given backbone.

        Parameters
        ----------
        sequence : str
            Amino-acid sequence; chains separated by ``:``.
        backbone_coordinates : np.ndarray
            ``(n_residues, 3, 3)`` N, CA, C coordinates in Angstrom, one row per residue (chain
            breaks excluded). Residues with missing coordinates may be NaN.
        positions : Sequence[int]
            Residue indices to mask and decode, counted over all residues (chain breaks excluded).
        num_steps : int | None
            Number of decoding rounds, clamped to at most ``len(positions)`` because the SDK cannot
            take more steps than there are masked tokens. ``None`` uses one step per masked
            position. There is little gain beyond about 20 steps.
        temperature : float
            Sampling temperature. ESM3 advises staying at or below 0.1 for sequence design: above
            that the distribution flattens and the residues are close to randomly chosen.
        seed : int | None
            Seeds the global ``torch`` RNG before decoding, making the call reproducible. ``None``
            leaves the RNG untouched.
        excluded_amino_acids : str
            One-letter codes the model may not sample, passed to the SDK as invalid token ids alongside
            :data:`NON_STANDARD_SEQUENCE_TOKENS`, which are always forbidden. Empty (the default) allows
            all 20 standard amino acids.

        Returns
        -------
        ESM3InverseGenerationOutput
            The full decoded sequence and the residues sampled at ``positions``.

        Raises
        ------
        ValueError
            If ``temperature`` is not positive, ``num_steps`` is below 1, ``excluded_amino_acids``
            holds an unknown or every amino acid, or the inputs fail the checks in
            :meth:`_prepare_masked_protein`.
        RuntimeError
            If the SDK reports an error, leaves a masked residue undecoded, decodes a non-standard
            residue, or changes a residue that was meant to stay fixed.
        """
        if temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {temperature}.")
        if num_steps is not None and num_steps < 1:
            raise ValueError(f"num_steps must be at least 1, got {num_steps}.")
        excluded = sorted(set(excluded_amino_acids))
        unknown = [code for code in excluded if code not in INVERSE_FOLDING_AMINO_ACIDS]
        if unknown:
            raise ValueError(f"excluded_amino_acids holds codes outside the standard 20: {unknown}.")
        if len(excluded) == len(INVERSE_FOLDING_AMINO_ACIDS):
            raise ValueError("excluded_amino_acids cannot exclude every amino acid.")

        parsed, position_array, keep_indices, masked_sdk_sequence, atom37 = self._prepare_masked_protein(
            sequence, backbone_coordinates, positions
        )
        assert self.model is not None, "Model not loaded"
        from esm.sdk.api import ESMProtein, ESMProteinError, GenerationConfig

        # The SDK requires no more decoding steps than there are masked tokens.
        effective_steps = (
            int(position_array.size) if num_steps is None else min(int(num_steps), int(position_array.size))
        )
        # The caller's exclusions are not enough on their own: the sequence track can also emit ambiguity
        # codes and structural tokens, which are never a valid redesign of a residue.
        forbidden = sorted(set(NON_STANDARD_SEQUENCE_TOKENS) | set(excluded))
        invalid_ids = list(self.model.tokenizers.sequence.convert_tokens_to_ids(forbidden))

        metadata = dataclasses.replace(self._metadata_template, sequence_lengths=[parsed.residue_count])
        with Timer("ESM3 preprocessing") as preprocess_timer:
            protein = ESMProtein(sequence=masked_sdk_sequence, coordinates=torch.from_numpy(atom37))
            if seed is not None:
                # The SDK builds the structure encoder lazily, and constructing its modules draws from the
                # global RNG. Left until generate(), that would consume the seeded state on the first call
                # only, so the same seed would not reproduce. Force it before seeding.
                self.model.get_structure_encoder()
                torch.manual_seed(int(seed))
        # no_grad, not inference_mode: generate() tokenizes the structure internally, and the VQ-VAE encoder
        # feeds inference-mode tensors into autograd-tracked ops, which inference_mode forbids.
        with Timer("Model Inference") as inference_timer, torch.no_grad():
            generated = self.model.generate(
                protein,
                GenerationConfig(
                    track="sequence",
                    schedule="cosine",
                    num_steps=effective_steps,
                    temperature=float(temperature),
                    invalid_ids=invalid_ids,
                ),
            )
        with Timer("ESM3 postprocessing") as postprocess_timer:
            if isinstance(generated, ESMProteinError):
                raise RuntimeError(f"ESM3 sequence generation failed ({generated.error_code}): {generated.error_msg}")
            generated_sdk = str(generated.sequence)
            if len(generated_sdk) != len(masked_sdk_sequence):
                raise RuntimeError(
                    f"ESM3 returned {len(generated_sdk)} tokens for a {len(masked_sdk_sequence)}-token prompt."
                )
            residues = "".join(generated_sdk[int(keep_indices[position])] for position in position_array)
            if "_" in residues:
                raise RuntimeError("ESM3 left one or more masked residues undecoded.")
            # Backstop for the invalid_ids above, so a caller never receives a residue it cannot use.
            non_standard = sorted(set(residues) - set(INVERSE_FOLDING_AMINO_ACIDS))
            if non_standard:
                raise RuntimeError(f"ESM3 decoded non-standard residues {non_standard}.")
            # Every residue that was not masked must have survived generation unchanged.
            fixed = set(range(parsed.residue_count)) - set(position_array.tolist())
            changed = [
                index
                for index in sorted(fixed)
                if generated_sdk[int(keep_indices[index])] != parsed.sdk_sequence[int(keep_indices[index])]
            ]
            if changed:
                raise RuntimeError(f"ESM3 changed residues that were meant to stay fixed: {changed}.")
            decoded = generated_sdk.replace("|", ":")  # back to the chain separator boileroom uses
        metadata.preprocessing_time = preprocess_timer.duration
        metadata.inference_time = inference_timer.duration
        metadata.postprocessing_time = postprocess_timer.duration
        return ESM3InverseGenerationOutput(
            metadata=metadata,
            positions=position_array.astype(np.int32),
            sequence=decoded,
            residues=residues,
        )


__all__ = [
    "NON_STANDARD_SEQUENCE_TOKENS",
    "ESM3Core",
    "ESM3ParsedSequence",
    "INVERSE_FOLDING_AMINO_ACIDS",
    "ESMCCore",
    "pad_residue_arrays",
    "parse_esm3_sequences",
]
