import modal
import numpy as np
import os
from dataclasses import dataclass
from typing import List, Sequence, Union, Optional, TYPE_CHECKING, cast

import logging
import json

from ...base import ModelWrapper
from ...base import EmbeddingAlgorithm, EmbeddingPrediction, PredictionMetadata
from ...backend import LocalBackend, ModalBackend
from ...backend.base import Backend
from ...backend.modal import app
from ...utils import MINUTES, MODAL_MODEL_DIR, Timer

from .image import esm_image
from ...images.volumes import model_weights
from .linker import compute_position_ids, store_multimer_properties, replace_glycine_linkers

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

############################################################
# CORE ALGORITHM
############################################################


@dataclass
class ESM2Output(EmbeddingPrediction):
    """Output from ESM2 prediction including all model outputs."""

    embeddings: np.ndarray  # (batch_size, seq_len, embedding_dim)
    metadata: PredictionMetadata
    chain_index: np.ndarray  # (batch_size, seq_len)
    residue_index: np.ndarray  # (batch_size, seq_len)
    hidden_states: Optional[np.ndarray] = None  # (batch_size, hidden_state_iter, seq_len, embedding_dim)


with esm_image.imports():
    import torch
    from transformers import EsmModel, AutoTokenizer


class ESM2Core(EmbeddingAlgorithm):
    """ESM2 protein language model."""

    DEFAULT_CONFIG = {
        "device": "cuda:0",
        "model_name": "esm2_t33_650M_UR50D",
        "include_fields": None,  # Optional[List[str]] - controls which fields to include in output
        # Chain linking and positioning config
        "glycine_linker": "",
        "position_ids_skip": 512,
    }
    # Static config keys that can only be set at initialization
    STATIC_CONFIG_KEYS = {"device", "model_name"}

    def __init__(self, config: dict = {}) -> None:
        """
        Initialize the ESM2Core with the provided configuration and prepare internal state for model loading.
        
        Parameters:
            config (dict): Configuration overrides for the model (merged with defaults). Recognized keys include at least `model_name` and `device`.
        
        Description:
            - Sets up model metadata (name and version), resolves the model directory from the `MODEL_DIR` environment variable or a default, and initializes runtime attributes (`_device`, `tokenizer`, `model`) to None.
            - Validates that the configured `model_name` is supported.
        """
        super().__init__(config)
        self.metadata = self._initialize_metadata(
            model_name="ESM-2",
            model_version="v4.49.0",  # HuggingFace transformers version
        )
        self.model_dir: Optional[str] = os.environ.get("MODEL_DIR", MODAL_MODEL_DIR)
        self._device: torch.device | None = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[EsmModel] = None
        self.assert_valid_model(self.config["model_name"])

    @staticmethod
    def assert_valid_model(model_name: str) -> None:
        """
        Validate that the model name is supported.

        Available ESM-2 models:
        - esm2_t48_15B_UR50D: 48 layers, 5120 hidden size, 40 attention heads
        - esm2_t36_3B_UR50D: 36 layers, 2560 hidden size, 40 attention heads
        - esm2_t33_650M_UR50D: 33 layers, 1280 hidden size, 20 attention heads
        - esm2_t30_150M_UR50D: 30 layers, 640 hidden size, 12 attention heads
        - esm2_t12_35M_UR50D: 12 layers, 480 hidden size, 20 attention heads
        - esm2_t6_8M_UR50D: 6 layers, 320 hidden size, 20 attention heads
        """
        models_name = [
            "esm2_t48_15B_UR50D",
            "esm2_t36_3B_UR50D",
            "esm2_t33_650M_UR50D",
            "esm2_t30_150M_UR50D",
            "esm2_t12_35M_UR50D",
            "esm2_t6_8M_UR50D",
        ]
        assert model_name in models_name, f"Model {model_name} not supported"

    def _initialize(self) -> None:
        self._load()

    def _load(self) -> None:
        """
        Ensure the tokenizer and model are loaded, move the model to the resolved device, set it to evaluation mode, and mark the instance as ready.
        
        If the tokenizer or model are already present, they are not reloaded. This method also resolves and stores the target device and transfers the model there.
        """
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                f"facebook/{self.config['model_name']}", cache_dir=self.model_dir
            )
        if self.model is None:
            self.model = EsmModel.from_pretrained(f"facebook/{self.config['model_name']}", cache_dir=self.model_dir)
        self._device = self._resolve_device()
        self.model = self.model.to(self._device)
        self.model.eval()
        self.ready = True

    def embed(self, sequences: Union[str, Sequence[str]], options: Optional[dict] = None) -> ESM2Output:
        """
        Compute embeddings for one or more protein sequences using the configured ESM-2 model.
        
        This method accepts a single sequence or a list of sequences, merges per-call `options` with the model's static configuration, and returns an ESM2Output containing token-level embeddings and associated metadata. If any sequence contains ":" it is treated as a multimer (inter-chain separator) and multimer-specific preprocessing (glycine linker replacement, position id construction, and attention masking) will be applied. The method will load the model automatically if it is not already loaded and respects the `include_fields` option to determine whether hidden states are produced.
        
        Parameters:
            sequences (str | Sequence[str]): A single protein sequence string or an iterable of sequence strings. A sequence containing ":" is interpreted as a multimer.
            options (dict | None): Per-call configuration that is merged with the core config; can control model_name, glycine_linker, position_ids_skip, include_fields, and other runtime options.
        
        Returns:
            ESM2Output: Prediction container with embeddings, metadata, chain_index and residue_index arrays, and optional hidden_states depending on `include_fields`.
        """
        if self.tokenizer is None or self.model is None:
            logger.warning("Model not loaded. Forcing the model to load... Next time call _load() first.")
            self._load()
        assert self.tokenizer is not None and self.model is not None, "Model not loaded"

        # Merge static config with per-call options
        effective_config = self._merge_options(options)

        normalized_sequences: list[str]
        if isinstance(sequences, str):
            normalized_sequences = [sequences]
        else:
            normalized_sequences = list(sequences)

        logger.debug(f'Embedding {len(normalized_sequences)} sequences using {effective_config["model_name"]}')

        if any(":" in seq for seq in normalized_sequences):
            # Multimer logic
            glycine_linker = effective_config["glycine_linker"]
            multimer_properties = self._store_multimer_properties(normalized_sequences, glycine_linker)
            tokenized = self.tokenizer(
                replace_glycine_linkers(normalized_sequences, glycine_linker),
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            # Add position_ids and attention_mask
            tokenized["position_ids"] = compute_position_ids(
                normalized_sequences, glycine_linker, effective_config["position_ids_skip"]
            )
            tokenized["attention_mask"] = (multimer_properties["linker_map"] == 1).to(torch.int32)
        else:
            # Monomer logic
            tokenized = self.tokenizer(
                normalized_sequences,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            multimer_properties = None
        tokenized = tokenized.to(self._device)
        tokenized["output_hidden_states"] = self._should_compute_hidden_states(effective_config.get("include_fields"))

        with Timer("Model Inference") as timer:
            with torch.inference_mode():
                outputs = self.model(**tokenized)

        outputs = self._convert_outputs(outputs, multimer_properties, timer.duration, effective_config)

        return outputs

    def _should_compute_hidden_states(self, include_fields: Optional[list[str]]) -> bool:
        """
        Determine whether hidden states should be computed from the provided include_fields.
        
        Parameters:
            include_fields (Optional[list[str]]): List of field names to include in the output, or None to indicate no per-call includes.
        
        Returns:
            bool: `True` if `include_fields` contains `"hidden_states"` or `"*"`, `False` otherwise.
        """
        return include_fields is not None and ("hidden_states" in include_fields or "*" in include_fields)

    @staticmethod
    def _store_multimer_properties(sequences: List[str], glycine_linker: str) -> dict[str, torch.Tensor]:
        """
        Prepare multimer metadata tensors and pad them to account for special <cls> and <eos> tokens.
        
        Parameters:
            sequences (List[str]): List of input chain sequences comprising the multimer.
            glycine_linker (str): Linker string used to represent chain joins when tokenizing multimers.
        
        Returns:
            dict[str, torch.Tensor]: A dictionary with three tensors:
                - `linker_map`: per-position linker mask with `-1` padding inserted at the start and end.
                - `residue_index`: per-position residue indices with `-1` padding at start and end.
                - `chain_index`: per-position chain indices with `-1` padding at start and end.
        """
        linker_map, residue_index, chain_index = store_multimer_properties(sequences, glycine_linker)
        # Add <cls> and <eos> as effective padding
        batch_size = linker_map.shape[0]
        linker_map = torch.cat([-torch.ones(batch_size, 1), linker_map, -torch.ones(batch_size, 1)], dim=1)
        residue_index = torch.cat([-torch.ones(batch_size, 1), residue_index, -torch.ones(batch_size, 1)], dim=1)
        chain_index = torch.cat([-torch.ones(batch_size, 1), chain_index, -torch.ones(batch_size, 1)], dim=1)
        return {"linker_map": linker_map, "residue_index": residue_index, "chain_index": chain_index}

    def _convert_outputs(
        self,
        outputs: "BaseModelOutputWithPoolingAndCrossAttentions",
        multimer_properties: dict[str, torch.Tensor] | None,
        prediction_time: float,
        config: dict,
    ) -> ESM2Output:
        """
        Convert raw HuggingFace model outputs into a finalized ESM2Output, applying multimer linker masking when provided and filtering fields according to config.
        
        Parameters:
            outputs: The model output object containing `last_hidden_state` and optionally `hidden_states`.
            multimer_properties (dict | None): When present, provides tensors for `linker_map`, `residue_index`, and `chain_index` used to mask and reshape multimer embeddings.
            prediction_time (float): Elapsed time in seconds for the prediction; stored in the output metadata.
            config (dict): Effective configuration for this call; may include `include_fields` to control which fields are retained in the returned output.
        
        Returns:
            ESM2Output: Structured prediction containing `embeddings`, optional `hidden_states`, `chain_index`, `residue_index`, and updated `metadata` with `prediction_time`.
        """

        embeddings = outputs.last_hidden_state.cpu().numpy()

        if self._should_compute_hidden_states(config.get("include_fields")) and outputs.hidden_states is not None:
            assert torch.all(
                outputs.hidden_states[-1] == outputs.last_hidden_state
            ), "Last hidden state should be the same as the output of the model"
            hidden_states = np.stack([h.cpu().numpy() for h in outputs.hidden_states], axis=1)
        else:
            hidden_states = None

        if multimer_properties is not None:
            # TODO: maybe add a proper MULTIMER flag?
            result = self._mask_linker_region(embeddings, hidden_states, **multimer_properties)
            embeddings, hidden_states, chain_index_output, residue_index_output = result
        else:  # only MONOMERs
            if hidden_states is not None:
                hidden_states = hidden_states[:, :, 1:-1, :]  # remove the first and last token
            embeddings = embeddings[:, 1:-1, :]  # remove the first and last token
            # Generate chain_index and residue_index for monomers
            batch_size, seq_len, _ = embeddings.shape
            chain_index_output = np.zeros((batch_size, seq_len), dtype=np.int32)
            residue_index_output = np.tile(np.arange(seq_len, dtype=np.int32), (batch_size, 1))

        self.metadata.prediction_time = prediction_time

        # Build full output with all fields
        full_output = ESM2Output(
            metadata=self.metadata,
            embeddings=embeddings,
            hidden_states=hidden_states,
            chain_index=chain_index_output,
            residue_index=residue_index_output,
        )

        # Apply filtering based on include_fields
        include_fields = config.get("include_fields")
        filtered = self._filter_include_fields(full_output, include_fields)
        return cast(ESM2Output, filtered)

    def _mask_linker_region(
        self,
        embeddings: np.ndarray,
        hidden_states: np.ndarray,
        linker_map: torch.Tensor,
        residue_index: torch.Tensor,
        chain_index: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor]:
        """
        Mask linker regions from model outputs and return per-batch arrays padded to equal sequence lengths.
        
        Parameters:
            embeddings (np.ndarray): Model token embeddings with shape (batch, seq_len, embedding_dim).
            hidden_states (np.ndarray | None): Optional hidden states with shape (batch, num_layers, seq_len, embedding_dim) or None.
            linker_map (torch.Tensor): Per-batch mask with values 1 for residues to keep and -1 (or 0) for linker/padding positions.
            residue_index (torch.Tensor): Per-batch residue indices aligned to the input tokens.
            chain_index (torch.Tensor): Per-batch chain indices aligned to the input tokens.
        
        Returns:
            tuple:
                embeddings (np.ndarray): Filtered and padded embeddings with shape (batch, kept_seq_len_max, embedding_dim). Padded positions are zero.
                hidden_states (np.ndarray | None): If provided, filtered and padded hidden states with shape (batch, kept_seq_len_max, num_layers, embedding_dim); otherwise None. Padded positions are zero.
                chain_index (np.ndarray): Filtered and padded chain indices with shape (batch, kept_seq_len_max). Padded positions use -1.
                residue_index (np.ndarray): Filtered and padded residue indices with shape (batch, kept_seq_len_max). Padded positions use -1.
        """
        assert isinstance(linker_map, torch.Tensor), "linker_map must be a tensor"

        embeddings_list = []
        if hidden_states is not None:
            hidden_states_list = []
        chain_index_list = []
        residue_index_list = []

        for batch_idx, multimer in enumerate(linker_map):
            # Drop the -1 values, meaning 1s refer to residues we want to keep
            multimer = multimer.masked_fill(multimer == -1, 0).cpu().numpy()
            # Chain indices are the ones that were not masked, hence they were kept and are thus 1
            chain_indices = np.where(multimer == 1)[0]

            # Get embeddings for the residues we want to keep
            embeddings_list.append(embeddings[batch_idx, chain_indices])
            if hidden_states is not None:
                hidden_states_list.append(hidden_states[batch_idx, :, chain_indices, :])
            chain_index_list.append(chain_index[batch_idx, chain_indices])
            residue_index_list.append(residue_index[batch_idx, chain_indices])

        def pad_and_stack(
            arrays: list[np.ndarray], residue_dim: int, batch_dim: int, constant_value: int = 0
        ) -> np.ndarray:
            """Pad arrays to match the largest size in the residue dimension and stack them in the batch dimension.

            Args:
                arrays: List of NumPy arrays to pad and stack
                residue_dim: Dimension to pad to match sizes
                batch_dim: Dimension to stack the arrays along
                constant_value: Value to use for padding (default: 0)

            Returns:
                Stacked and padded NumPy array
            """
            max_size = max(arr.shape[residue_dim] for arr in arrays)
            padded_arrays = []
            for arr in arrays:
                padding = [(0, 0)] * arr.ndim
                padding[residue_dim] = (0, max_size - arr.shape[residue_dim])
                padded_arrays.append(np.pad(arr, padding, mode="constant", constant_values=constant_value))
            return np.stack(padded_arrays, axis=batch_dim)

        # Stack embeddings along batch dimension (0)
        embeddings = pad_and_stack(embeddings_list, residue_dim=0, batch_dim=0)
        if hidden_states is not None:
            hidden_states = pad_and_stack(hidden_states_list, residue_dim=0, batch_dim=0)
            # Transpose to get correct dimension order (batch, layers, seq_len, hidden_dim)
            hidden_states = np.transpose(hidden_states, (0, 2, 1, 3))
        chain_index = pad_and_stack(chain_index_list, residue_dim=0, batch_dim=0, constant_value=-1)
        residue_index = pad_and_stack(residue_index_list, residue_dim=0, batch_dim=0, constant_value=-1)

        return embeddings, hidden_states, chain_index, residue_index


############################################################
# MODAL-SPECIFIC WRAPPER
############################################################


@app.cls(
    image=esm_image,
    gpu="T4",
    timeout=20 * MINUTES,
    scaledown_window=10 * MINUTES,
    volumes={MODAL_MODEL_DIR: model_weights},
)
class ModalESM2:
    """Modal-specific wrapper around `ESM2`."""

    config: bytes = modal.parameter(default=b"{}")

    @modal.enter()
    def _initialize(self) -> None:
        """
        Create and initialize the ESM2Core backend instance from the encoded configuration.
        
        Decodes the JSON bytes stored in self.config, constructs an ESM2Core using that config, assigns it to self._core, and calls its initialization routine.
        """
        self._core = ESM2Core(config=json.loads(self.config.decode("utf-8")))
        self._core._initialize()

    @modal.method()
    def embed(self, sequences: Union[str, Sequence[str]], options: Optional[dict] = None) -> ESM2Output:
        """
        Compute embeddings for one or more protein sequences using the configured ESM-2 model.
        
        Parameters:
            sequences (str | Sequence[str]): A single protein sequence string or an iterable of sequence strings.
            options (dict | None): Per-call options that override the instance configuration (e.g., model selection, glycine_linker, position_ids_skip, include_fields). Only provided keys are merged with the static configuration.
        
        Returns:
            ESM2Output: Prediction container with fields including `embeddings`, `metadata`, `chain_index`, `residue_index`, and `hidden_states` when requested.
        """
        assert self._core is not None, "ModalESM2 has not been initialized"
        return self._core.embed(sequences, options=options)


############################################################
# HIGH-LEVEL INTERFACE
############################################################
class ESM2(ModelWrapper):
    """Interface for running ESM2 embeddings via Modal."""

    def __init__(self, backend: str = "modal", device: Optional[str] = None, config: Optional[dict] = None) -> None:
        """
        Initialize the ESM2 high-level interface and start the selected backend.
        
        Parameters:
            backend (str): Backend type to use; supported values are "modal" and "local".
            device (Optional[str]): Device identifier for model execution (for example "cuda:0" or "cpu").
            config (Optional[dict]): Configuration passed to the backend and underlying model.
        
        Raises:
            ValueError: If an unsupported backend string is provided.
        """
        if config is None:
            config = {}
        self.config = config
        self.device = device
        backend_instance: Backend
        if backend == "modal":
            backend_instance = ModalBackend(ModalESM2, config, device=device)
        elif backend == "local":
            backend_instance = LocalBackend(ESM2Core, config, device=device)
        else:
            raise ValueError(f"Backend {backend} not supported")
        self._backend = backend_instance
        self._backend.start()

    def embed(self, sequences: Union[str, Sequence[str]], options: Optional[dict] = None) -> ESM2Output:
        """
        Compute ESM-2 embeddings for one or more protein sequences using the configured backend.
        
        Parameters:
            sequences (str | Sequence[str]): A single protein sequence string or a sequence of protein sequences. Multimer inputs may be provided by including ":" characters to separate chains within a sequence.
            options (dict | None): Per-call options merged with the backend's static configuration to adjust behavior for this call (for example: model_name, include_fields, glycine_linker, position_ids_skip). Keys not present in `options` fall back to the configured defaults.
        
        Returns:
            ESM2Output: Embeddings and associated metadata (embeddings, chain_index, residue_index, metadata, and optional hidden_states) for the provided sequences.
        """
        return self._call_backend_method("embed", sequences, options=options)