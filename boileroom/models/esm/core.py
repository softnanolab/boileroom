"""Core ESM2 and ESMFold algorithm implementations without modal dependencies."""

import os
import logging
from typing import List, Optional, Sequence, Union, TYPE_CHECKING, cast

import numpy as np
import torch
from transformers import EsmModel, AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.modeling_esmfold import EsmFoldingTrunk
from biotite.structure import AtomArray

from ...base import EmbeddingAlgorithm, FoldingAlgorithm
from ...utils import Timer, get_model_dir, MODAL_MODEL_DIR

if TYPE_CHECKING:
    from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from .linker import compute_position_ids, store_multimer_properties, replace_glycine_linkers
from .types import ESM2Output, ESMFoldOutput

logger = logging.getLogger(__name__)


############################################################
# ESMFold-specific forward function patch
############################################################


def always_no_grad_forward(self, seq_feats, pair_feats, true_aa, residx, mask, no_recycles):
    """
    Inputs:
        seq_feats: B x L x C tensor of sequence features pair_feats: B x L x L x C tensor of pair features residx: B
        x L long tensor giving the position in the sequence mask: B x L boolean tensor indicating valid residues

    Output:
        predicted_structure: B x L x (num_atoms_per_residue * 3) tensor wrapped in a Coordinates object
    """

    device = seq_feats.device
    s_s_0 = seq_feats
    s_z_0 = pair_feats

    if no_recycles is None:
        no_recycles = self.config.max_recycles
    else:
        if no_recycles < 0:
            raise ValueError("Number of recycles must not be negative.")
        no_recycles += 1  # First 'recycle' is just the standard forward pass through the model.

    def trunk_iter(s, z, residx, mask):
        z = z + self.pairwise_positional_embedding(residx, mask=mask)

        for block in self.blocks:
            s, z = block(s, z, mask=mask, residue_index=residx, chunk_size=self.chunk_size)
        return s, z

    s_s = s_s_0
    s_z = s_z_0
    recycle_s = torch.zeros_like(s_s)
    recycle_z = torch.zeros_like(s_z)
    recycle_bins = torch.zeros(*s_z.shape[:-1], device=device, dtype=torch.int64)

    for recycle_idx in range(no_recycles):
        with torch.no_grad():
            # === Recycling ===
            recycle_s = self.recycle_s_norm(recycle_s.detach()).to(device)
            recycle_z = self.recycle_z_norm(recycle_z.detach()).to(device)
            recycle_z += self.recycle_disto(recycle_bins.detach()).to(device)

            s_s, s_z = trunk_iter(s_s_0 + recycle_s, s_z_0 + recycle_z, residx, mask)

            # === Structure module ===
            structure = self.structure_module(
                {"single": self.trunk2sm_s(s_s), "pair": self.trunk2sm_z(s_z)},
                true_aa,
                mask.float(),
            )

            recycle_s = s_s
            recycle_z = s_z
            # Distogram needs the N, CA, C coordinates, and bin constants same as alphafold.
            recycle_bins = EsmFoldingTrunk.distogram(
                structure["positions"][-1][:, :, :3],
                3.375,
                21.375,
                self.recycle_bins,
            )

    structure["s_s"] = s_s
    structure["s_z"] = s_z

    return structure


# Patch EsmFoldingTrunk to use always_no_grad_forward
EsmFoldingTrunk.forward = always_no_grad_forward


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
        """Initialize the ESM2Core with the provided configuration and prepare internal state for model loading.

        Sets up model metadata (name and version), resolves the model directory from the `MODEL_DIR` environment variable or XDG cache, and initializes runtime attributes (`_device`, `tokenizer`, `model`) to None. Validates that the configured `model_name` is supported.

        Parameters
        ----------
        config : dict
            Configuration overrides for the model (merged with defaults). Recognized keys include at least `model_name` and `device`.
        """
        super().__init__(config)
        self.metadata = self._initialize_metadata(
            model_name="ESM-2",
            model_version="v4.49.0",  # HuggingFace transformers version
        )
        self.model_dir: str = str(get_model_dir())
        self._device: torch.device | None = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[EsmModel] = None
        self.assert_valid_model(self.config["model_name"])

    @staticmethod
    def assert_valid_model(model_name: str) -> None:
        """Validate that the model name is supported.

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
        """Initialize the model by loading it."""
        self._load()

    def _load(self) -> None:
        """Ensure the tokenizer and model are loaded, move the model to the resolved device, set it to evaluation mode, and mark the instance as ready.

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
        """Compute embeddings for one or more protein sequences using the configured ESM-2 model.

        This method accepts a single sequence or a list of sequences, merges per-call `options` with the model's static configuration, and returns an ESM2Output containing token-level embeddings and associated metadata. If any sequence contains ":" it is treated as a multimer (inter-chain separator) and multimer-specific preprocessing (glycine linker replacement, position id construction, and attention masking) will be applied. The method will load the model automatically if it is not already loaded and respects the `include_fields` option to determine whether hidden states are produced.

        Parameters
        ----------
        sequences : str | Sequence[str]
            A single protein sequence string or an iterable of sequence strings. A sequence containing ":" is interpreted as a multimer.
        options : dict | None, optional
            Per-call configuration that is merged with the core config; can control model_name, glycine_linker, position_ids_skip, include_fields, and other runtime options.

        Returns
        -------
        ESM2Output
            Prediction container with embeddings, metadata, chain_index and residue_index arrays, and optional hidden_states depending on `include_fields`.
        """
        # Merge static config with per-call options (validate before loading model)
        effective_config = self._merge_options(options)

        if self.tokenizer is None or self.model is None:
            logger.warning("Model not loaded. Forcing the model to load... Next time call _load() first.")
            self._load()
        assert self.tokenizer is not None and self.model is not None, "Model not loaded"

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
        """Determine whether hidden states should be computed from the provided include_fields.

        Parameters
        ----------
        include_fields : Optional[list[str]]
            List of field names to include in the output, or None to indicate no per-call includes.

        Returns
        -------
        bool
            `True` if `include_fields` contains `"hidden_states"` or `"*"`, `False` otherwise.
        """
        return include_fields is not None and ("hidden_states" in include_fields or "*" in include_fields)

    @staticmethod
    def _store_multimer_properties(sequences: List[str], glycine_linker: str) -> dict[str, torch.Tensor]:
        """Prepare multimer metadata tensors and pad them to account for special <cls> and <eos> tokens.

        Parameters
        ----------
        sequences : List[str]
            List of input chain sequences comprising the multimer.
        glycine_linker : str
            Linker string used to represent chain joins when tokenizing multimers.

        Returns
        -------
        dict[str, torch.Tensor]
            A dictionary with three tensors:
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
        """Convert raw HuggingFace model outputs into a finalized ESM2Output, applying multimer linker masking when provided and filtering fields according to config.

        Parameters
        ----------
        outputs : BaseModelOutputWithPoolingAndCrossAttentions
            The model output object containing `last_hidden_state` and optionally `hidden_states`.
        multimer_properties : dict | None
            When present, provides tensors for `linker_map`, `residue_index`, and `chain_index` used to mask and reshape multimer embeddings.
        prediction_time : float
            Elapsed time in seconds for the prediction; stored in the output metadata.
        config : dict
            Effective configuration for this call; may include `include_fields` to control which fields are retained in the returned output.

        Returns
        -------
        ESM2Output
            Structured prediction containing `embeddings`, optional `hidden_states`, `chain_index`, `residue_index`, and updated `metadata` with `prediction_time`.
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
        hidden_states: Optional[np.ndarray],
        linker_map: torch.Tensor,
        residue_index: torch.Tensor,
        chain_index: torch.Tensor,
    ) -> tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray]:
        """Mask linker regions from model outputs and return per-batch arrays padded to equal sequence lengths.

        Parameters
        ----------
        embeddings : np.ndarray
            Model token embeddings with shape (batch, seq_len, embedding_dim).
        hidden_states : np.ndarray | None
            Optional hidden states with shape (batch, num_layers, seq_len, embedding_dim) or None.
        linker_map : torch.Tensor
            Per-batch mask with values 1 for residues to keep and -1 (or 0) for linker/padding positions.
        residue_index : torch.Tensor
            Per-batch residue indices aligned to the input tokens.
        chain_index : torch.Tensor
            Per-batch chain indices aligned to the input tokens.

        Returns
        -------
        tuple
            A tuple containing:
            - embeddings (np.ndarray): Filtered and padded embeddings with shape (batch, kept_seq_len_max, embedding_dim). Padded positions are zero.
            - hidden_states (np.ndarray | None): If provided, filtered and padded hidden states with shape (batch, kept_seq_len_max, num_layers, embedding_dim); otherwise None. Padded positions are zero.
            - chain_index (np.ndarray): Filtered and padded chain indices with shape (batch, kept_seq_len_max). Padded positions use -1.
            - residue_index (np.ndarray): Filtered and padded residue indices with shape (batch, kept_seq_len_max). Padded positions use -1.
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

            Parameters
            ----------
            arrays : list[np.ndarray]
                List of NumPy arrays to pad and stack.
            residue_dim : int
                Dimension to pad to match sizes.
            batch_dim : int
                Dimension to stack the arrays along.
            constant_value : int, optional
                Value to use for padding (default: 0).

            Returns
            -------
            np.ndarray
                Stacked and padded NumPy array.
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
# ESMFold Core Algorithm
############################################################


class ESMFoldCore(FoldingAlgorithm):
    """ESMFold protein structure prediction model."""

    DEFAULT_CONFIG = {
        "device": "cuda:0",
        # Chain linking and positioning config
        "glycine_linker": "",
        "position_ids_skip": 512,
        "include_fields": None,  # Optional[List[str]] - controls which fields to include in output
    }
    # Static config keys that can only be set at initialization
    STATIC_CONFIG_KEYS = {"device"}

    # We need to properly asses whether using this or the original ESMFold is better
    # based on speed, accuracy, bugs, etc.; as well as customizability
    # For instance, if we want to also allow differently sized structure modules, than this would be good
    # TODO: we should add a settings dictionary or something, that would make it easier to add new options
    # TODO: maybe use OmegaConf instead to make it easier instead of config
    def __init__(self, config: dict = {}) -> None:
        """Create an ESMFold core instance and initialize runtime fields and placeholders.

        Initializes model metadata (name and version). Resolves model_dir from the environment variable `MODEL_DIR` falling back to the packaged `MODAL_MODEL_DIR`. Prepares placeholders for device (`_device`), tokenizer (`tokenizer`), and model (`model`) that will be populated during lazy loading.

        Parameters
        ----------
        config : dict
            Configuration overrides for the predictor (merged with DEFAULT_CONFIG at runtime).
        """
        super().__init__(config)
        self.metadata = self._initialize_metadata(
            model_name="ESMFold",
            model_version="v4.49.0",  # HuggingFace transformers version
        )
        self.model_dir: Optional[str] = os.environ.get("MODEL_DIR", MODAL_MODEL_DIR)
        self._device: torch.device | None = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[EsmForProteinFolding] = None

    def _initialize(self) -> None:
        """Initialize the model during container startup. This helps us determine whether we run locally or remotely."""
        self._load()

    def _load(self) -> None:
        """Ensure the tokenizer and folding model are loaded and prepare the model for inference.

        Loads the tokenizer and model into the instance if absent, resolves and sets the device,
        moves the model to that device, switches the model to evaluation mode, configures the
        trunk chunk size, and marks the predictor as ready for use.
        """
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1", cache_dir=self.model_dir)
        if self.model is None:
            self.model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", cache_dir=self.model_dir)
        self._device = self._resolve_device()
        self.model = self.model.to(self._device)
        self.model.eval()
        self.model.trunk.set_chunk_size(64)
        self.ready = True

    def fold(self, sequences: Union[str, Sequence[str]], options: Optional[dict] = None) -> ESMFoldOutput:
        """Predict protein structure(s) from one or more amino acid sequence(s) using the ESMFold model.

        If the model is not loaded, this method will load it lazily before running inference. Updates self.metadata.sequence_lengths with the processed sequence lengths.

        Parameters
        ----------
        sequences : str | Sequence[str]
            A single amino acid sequence or an iterable of sequences to predict.
        options : dict, optional
            Per-call configuration merged with the instance's static config (for example, `include_fields` to select which output fields to return).

        Returns
        -------
        ESMFoldOutput
            Predicted structure(s) and optional auxiliary outputs. The set of returned fields honors the merged configuration (e.g., `include_fields` controls presence of pdb/cif and other model-specific outputs).
        """
        # Merge static config with per-call options (validate before loading model)
        effective_config = self._merge_options(options)

        if self.tokenizer is None or self.model is None:
            logger.warning("Model not loaded. Forcing the model to load... Next time call _load() first.")
            self._load()
        assert self.tokenizer is not None and self.model is not None, "Model not loaded"

        validated_sequences = self._validate_sequences(sequences)
        self.metadata.sequence_lengths = self._compute_sequence_lengths(validated_sequences)

        tokenized_input, multimer_properties = self._tokenize_sequences(validated_sequences, effective_config)

        with Timer("Model Inference") as timer:
            with torch.inference_mode():
                outputs = self.model(**tokenized_input)

        outputs = self._convert_outputs(outputs, multimer_properties, timer.duration, effective_config)
        return outputs

    def _tokenize_sequences(self, sequences: List[str], config: dict) -> tuple[dict, dict[str, torch.Tensor] | None]:
        """Tokenize one or more protein sequences for model input, handling multimer and monomer cases.

        Parameters
        ----------
        sequences : List[str]
            Protein sequences; presence of ':' in any sequence triggers multimer tokenization.
        config : dict
            Per-call configuration used for multimer tokenization (position ids, linker handling, etc.).

        Returns
        -------
        tuple
            A pair (tokenized, multimer_properties_or_none) where:
            - tokenized (dict): Tokenizer output tensors placed on the model device (keys like 'input_ids', 'attention_mask', etc.).
            - multimer_properties_or_none (dict[str, torch.Tensor] | None): Multimer-specific tensors (e.g., linker map, residue indices, chain indices) when multimer tokenization was used, otherwise `None`.
        """
        assert self.tokenizer is not None, "Tokenizer not loaded"
        if ":" in "".join(sequences):  # MULTIMER setting
            tokenized, multimer_properties = self._tokenize_multimer(sequences, config)
        else:  # MONOMER setting
            tokenized = self.tokenizer(
                sequences, return_tensors="pt", add_special_tokens=False, padding=True, truncation=True, max_length=1024
            )
            multimer_properties = None
        tokenized = {k: v.to(self._device) for k, v in tokenized.items()}

        return tokenized, multimer_properties

    def _tokenize_multimer(self, sequences: List[str], config: dict) -> tuple[dict, dict[str, torch.Tensor]]:
        """Prepare tokenized inputs and multimer metadata for sequences containing chain separators.

        Parameters
        ----------
        sequences : List[str]
            Input sequences where different chains are separated by ":".
        config : dict
            Configuration containing:
            - "glycine_linker": string used to replace ":" when constructing multimer inputs.
            - "position_ids_skip": integer flag/offset passed to position id computation.

        Returns
        -------
        tuple
            A tuple containing:
            - tokenized (Mapping[str, torch.Tensor]): Tokenizer output tensors including input ids, attention mask, and computed position_ids.
            - multimer_properties (dict): Dictionary with keys:
                - "linker_map" (torch.Tensor): Per-position mask indicating kept residues (1) vs linker/padding (-1 or 0).
                - "residue_index" (torch.Tensor): Residue indices mapping token positions to residue numbers.
                - "chain_index" (torch.Tensor): Chain indices mapping token positions to original chain ids.
        """
        assert self.tokenizer is not None, "Tokenizer not loaded"
        # Store multimer properties first
        glycine_linker = config["glycine_linker"]
        linker_map, residue_index, chain_index = store_multimer_properties(sequences, glycine_linker)

        # Create tokenized input using list comprehension directly
        tokenized = self.tokenizer(
            [seq.replace(":", glycine_linker) for seq in sequences],
            padding=True,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )

        # Add position IDs
        tokenized["position_ids"] = compute_position_ids(sequences, glycine_linker, config["position_ids_skip"])

        # Create attention mask (1 means keep, 0 means mask)
        # This also masks padding tokens, which are -1
        tokenized["attention_mask"] = (linker_map == 1).to(torch.int32)

        return tokenized, {"linker_map": linker_map, "residue_index": residue_index, "chain_index": chain_index}

    def _mask_linker_region(
        self,
        outputs: dict,
        linker_map: torch.Tensor,
        residue_index: torch.Tensor,
        chain_index: torch.Tensor,
    ) -> dict:
        """Mask the linker region in the outputs and track padding information.

        This includes all the metrics.

        Parameters
        ----------
        outputs : dict
            Dictionary containing model outputs.

        Returns
        -------
        dict
            Updated outputs with linker regions masked and padding information.
        """
        assert isinstance(linker_map, torch.Tensor), "linker_map must be a tensor"

        positions = []
        frames = []
        sidechain_frames = []
        unnormalized_angles = []
        angles = []
        states = []
        lddt_head = []

        s_s = []
        lm_logits = []
        aatype = []
        atom14_atom_exists = []
        residx_atom14_to_atom37 = []
        residx_atom37_to_atom14 = []
        atom37_atom_exists = []
        plddt = []

        s_z = []
        distogram_logits = []
        ptm_logits = []
        aligned_confidence_probs = []
        predicted_aligned_error = []

        _residue_index = []
        _chain_index = []

        for batch_idx, multimer in enumerate(linker_map):
            # Drop the -1 values, meaning 1s refer to residues we want to keep
            multimer = multimer.masked_fill(multimer == -1, 0).cpu().numpy()
            # Chain indices are the ones that were not masked, hence they were kept and are thus 1
            chain_indices = np.where(multimer == 1)[0]

            # 3rd dim is residue index
            positions.append(outputs["positions"][:, batch_idx, chain_indices])
            frames.append(outputs["frames"][:, batch_idx, chain_indices])
            sidechain_frames.append(outputs["sidechain_frames"][:, batch_idx, chain_indices])
            unnormalized_angles.append(outputs["unnormalized_angles"][:, batch_idx, chain_indices])
            angles.append(outputs["angles"][:, batch_idx, chain_indices])
            states.append(outputs["states"][:, batch_idx, chain_indices])
            lddt_head.append(outputs["lddt_head"][:, batch_idx, chain_indices])

            # 2nd dim is residue index
            s_s.append(outputs["s_s"][batch_idx, chain_indices])
            lm_logits.append(outputs["lm_logits"][batch_idx, chain_indices])
            aatype.append(outputs["aatype"][batch_idx, chain_indices])
            atom14_atom_exists.append(outputs["atom14_atom_exists"][batch_idx, chain_indices])
            residx_atom14_to_atom37.append(outputs["residx_atom14_to_atom37"][batch_idx, chain_indices])
            residx_atom37_to_atom14.append(outputs["residx_atom37_to_atom14"][batch_idx, chain_indices])
            atom37_atom_exists.append(outputs["atom37_atom_exists"][batch_idx, chain_indices])
            plddt.append(outputs["plddt"][batch_idx, chain_indices])

            # 2D properties that are per residue pair; thus residues is both the 2nd and 3rd dim
            s_z.append(outputs["s_z"][batch_idx, chain_indices][:, chain_indices])
            distogram_logits.append(outputs["distogram_logits"][batch_idx, chain_indices][:, chain_indices])
            ptm_logits.append(outputs["ptm_logits"][batch_idx, chain_indices][:, chain_indices])
            aligned_confidence_probs.append(
                outputs["aligned_confidence_probs"][batch_idx, chain_indices][:, chain_indices]
            )
            predicted_aligned_error.append(
                outputs["predicted_aligned_error"][batch_idx, chain_indices][:, chain_indices]
            )

            # Custom outputs, that also have 2nd dimension as residue index
            _residue_index.append(residue_index[batch_idx, chain_indices].cpu().numpy())
            _chain_index.append(chain_index[batch_idx, chain_indices].cpu().numpy())

        def pad_and_stack(
            arrays: list[np.ndarray], residue_dim: Union[int, List[int]], batch_dim: int, intermediate_dim: bool = False
        ) -> np.ndarray:
            """Pad arrays to match the largest size in the residue dimension and stack them in the batch dimension.

            Parameters
            ----------
            arrays : list[np.ndarray]
                List of NumPy arrays to pad and stack.
            residue_dim : Union[int, List[int]]
                Dimension(s) to pad to match sizes.
            batch_dim : int
                Dimension to stack the arrays along.
            intermediate_dim : bool, optional
                Whether the array has an intermediate dimension to preserve.

            Returns
            -------
            np.ndarray
                Stacked and padded NumPy array.
            """
            if isinstance(residue_dim, int):
                max_size = max(arr.shape[residue_dim] for arr in arrays)
                padded_arrays = []
                for arr in arrays:
                    padding = [(0, 0)] * arr.ndim
                    padding[residue_dim] = (0, max_size - arr.shape[residue_dim])
                    padded_arrays.append(np.pad(arr, padding, mode="constant", constant_values=-1))
            elif isinstance(residue_dim, list):
                # Multi-dimension padding (e.g., for 2D matrices)
                max_sizes = []
                for dim in residue_dim:
                    max_sizes.append(max(arr.shape[dim] for arr in arrays))

                padded_arrays = []
                for arr in arrays:
                    padding = [(0, 0)] * arr.ndim
                    for dim, max_size in zip(residue_dim, max_sizes):
                        padding[dim] = (0, max_size - arr.shape[dim])
                    padded_arrays.append(np.pad(arr, padding, mode="constant", constant_values=-1))

            # Handle intermediate dimensions differently
            if intermediate_dim:
                # Stack along axis=1 to preserve intermediate dim as first dimension
                return np.stack(padded_arrays, axis=1)
            else:
                return np.stack(padded_arrays, axis=batch_dim)

        # 2nd dimension is the batch size, 3rd dimension was the residue index (without batch it's the 2nd dim)
        # These are not done same as below is because of getting the 8 intermediate outputs from StructureModule
        outputs["positions"] = pad_and_stack(positions, residue_dim=1, batch_dim=0, intermediate_dim=True)
        outputs["frames"] = pad_and_stack(frames, residue_dim=1, batch_dim=0, intermediate_dim=True)
        outputs["sidechain_frames"] = pad_and_stack(sidechain_frames, residue_dim=1, batch_dim=0, intermediate_dim=True)
        outputs["unnormalized_angles"] = pad_and_stack(
            unnormalized_angles, residue_dim=1, batch_dim=0, intermediate_dim=True
        )
        outputs["angles"] = pad_and_stack(angles, residue_dim=1, batch_dim=0, intermediate_dim=True)
        outputs["states"] = pad_and_stack(states, residue_dim=1, batch_dim=0, intermediate_dim=True)
        outputs["lddt_head"] = pad_and_stack(lddt_head, residue_dim=1, batch_dim=0, intermediate_dim=True)

        # 1st dimension is the batch size, 2nd dimension was the residue index (without batch it's the 1st dim)
        outputs["s_s"] = pad_and_stack(s_s, residue_dim=0, batch_dim=0)
        outputs["lm_logits"] = pad_and_stack(lm_logits, residue_dim=0, batch_dim=0)
        outputs["aatype"] = pad_and_stack(aatype, residue_dim=0, batch_dim=0)
        outputs["atom14_atom_exists"] = pad_and_stack(atom14_atom_exists, residue_dim=0, batch_dim=0)
        outputs["residx_atom14_to_atom37"] = pad_and_stack(residx_atom14_to_atom37, residue_dim=0, batch_dim=0)
        outputs["residx_atom37_to_atom14"] = pad_and_stack(residx_atom37_to_atom14, residue_dim=0, batch_dim=0)
        outputs["atom37_atom_exists"] = pad_and_stack(atom37_atom_exists, residue_dim=0, batch_dim=0)
        outputs["plddt"] = pad_and_stack(plddt, residue_dim=0, batch_dim=0)

        # 2D properties, otherwise same as above
        outputs["s_z"] = pad_and_stack(s_z, residue_dim=[0, 1], batch_dim=0)
        outputs["distogram_logits"] = pad_and_stack(distogram_logits, residue_dim=[0, 1], batch_dim=0)
        outputs["ptm_logits"] = pad_and_stack(ptm_logits, residue_dim=[0, 1], batch_dim=0)
        outputs["aligned_confidence_probs"] = pad_and_stack(aligned_confidence_probs, residue_dim=[0, 1], batch_dim=0)
        outputs["predicted_aligned_error"] = pad_and_stack(predicted_aligned_error, residue_dim=[0, 1], batch_dim=0)

        # Custom
        outputs["chain_index"] = pad_and_stack(_chain_index, residue_dim=0, batch_dim=0)
        outputs["residue_index"] = pad_and_stack(_residue_index, residue_dim=0, batch_dim=0)

        return outputs

    def _convert_outputs(
        self,
        outputs: dict,
        multimer_properties: dict[str, torch.Tensor] | None,
        prediction_time: float,
        config: dict,
    ) -> ESMFoldOutput:
        """Convert raw model outputs and optional multimer metadata into a populated ESMFoldOutput.

        Parameters
        ----------
        outputs : dict
            Raw model outputs (tensor values) produced by the folding model.
        multimer_properties : dict[str, torch.Tensor] | None
            Multimer-related tensors (e.g., linker map, residue and chain indices). When provided, linker regions are masked and per-chain residue mappings are applied.
        prediction_time : float
            Elapsed time (seconds) for the prediction; recorded in the output metadata.
        config : dict
            Per-call configuration; the "include_fields" entry controls which optional fields (for example, "pdb" or "cif") are produced and which fields are retained in the returned output.

        Returns
        -------
        ESMFoldOutput
            Structured prediction containing metadata and outputs. An atom_array is always generated; PDB/CIF strings and other optional fields are included only if requested via config["include_fields"]. The returned output has internal-only fields (such as raw positions) removed and is filtered according to include_fields.
        """

        outputs = {k: v.cpu().numpy() for k, v in outputs.items()}
        if multimer_properties is not None:
            # TODO: maybe add a proper MULTIMER flag?
            outputs = self._mask_linker_region(outputs, **multimer_properties)
        else:  # only MONOMERs
            outputs["chain_index"] = np.zeros(outputs["residue_index"].shape, dtype=np.int32)

        self.metadata.prediction_time = prediction_time

        # Always generate atom_array
        atom_array = self._convert_outputs_to_atomarray(outputs)
        outputs["atom_array"] = atom_array

        # Generate PDB/CIF only if requested via include_fields
        include_fields = config.get("include_fields")
        if include_fields and ("*" in include_fields or "pdb" in include_fields):
            outputs["pdb"] = self._convert_outputs_to_pdb(atom_array)
        if include_fields and ("*" in include_fields or "cif" in include_fields):
            outputs["cif"] = self._convert_outputs_to_cif(atom_array)

        # Build full output with all fields (exclude positions as it's only used internally)
        outputs_without_positions = {k: v for k, v in outputs.items() if k != "positions"}
        full_output = ESMFoldOutput(metadata=self.metadata, **outputs_without_positions)

        # Apply filtering based on include_fields
        filtered = self._filter_include_fields(full_output, include_fields)
        return cast(ESMFoldOutput, filtered)

    def _convert_outputs_to_atomarray(self, outputs: dict) -> List[AtomArray]:
        """Create a list of Biotite AtomArray objects from model prediction tensors.

        Parameters
        ----------
        outputs : dict
            Model outputs containing at least the keys:
            - "positions": atom positions (used to derive atom37 coordinates),
            - "atom37_atom_exists": boolean mask of existing atom37 atoms,
            - "chain_index": per-residue chain indices,
            - "aatype": per-residue amino acid type ids,
            - "residue_index": per-residue residue indices,
            - "plddt": per-residue confidence scores.
            The function expects these tensors/arrays to be indexed by batch and residue.

        Returns
        -------
        List[AtomArray]
            One Biotite AtomArray per batch element. Each AtomArray contains only atoms marked as existing, with coordinates, chain_id, atom_name, three-letter residue name, residue index, chemical element, and b_factor populated from `plddt`.
        """
        from biotite.structure import Atom, array
        from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
        from transformers.models.esm.openfold_utils.residue_constants import atom_types, restypes, restype_1to3

        # Convert atom14 to atom37 format
        atom_positions = atom14_to_atom37(
            outputs["positions"][-1], outputs
        )  # (model_layer, batch, residue, atom37, xyz)
        atom_mask = outputs["atom37_atom_exists"]  # (batch, residue, atom37)

        assert len(atom_types) == atom_positions.shape[2] == 37, "Atom types must be 37"

        # Get batch and residue dimensions
        batch_size, n_residues, n_atoms = atom_mask.shape

        # Create list to store atoms
        arrays = []

        # Process each protein in the batch
        for b in range(batch_size):
            atoms = []  # clear out the atoms list
            # Process each residue
            for res_idx in range(n_residues):
                # Get chain ID (convert numeric index to letter A-Z)
                chain_id = chr(65 + outputs["chain_index"][b, res_idx])  # A=65 in ASCII

                # Get residue name (3-letter code)
                aa_type = outputs["aatype"][b, res_idx]  # id representing residue identity
                res_name = restypes[aa_type]  # 1-letter residue identity
                res_name = restype_1to3[res_name]  # 3-letter residue identity

                # Process each atom in the residue
                for atom_idx in range(n_atoms):  # loops through all 37 atom types
                    # Skip if atom doesn't exist
                    if not atom_mask[b, res_idx, atom_idx]:
                        continue

                    # Get atom coordinates
                    coord = atom_positions[b, res_idx, atom_idx]

                    # Create Atom object
                    atom = Atom(
                        coord=coord,
                        chain_id=chain_id,
                        atom_name=atom_types[atom_idx],
                        res_name=res_name,
                        res_id=outputs["residue_index"][b, res_idx],  # 0-indexed
                        element=atom_types[atom_idx][0],
                        # we only support C, N, O, S, [according to OpenFold Protein class]
                        # element is thus the first character of any atom name (according to PDB nomenclature)
                        b_factor=outputs["plddt"][b, res_idx, atom_idx],
                    )
                    atoms.append(atom)
            arrays.append(array(atoms))
        return arrays

    def _convert_outputs_to_pdb(self, atom_array: List[AtomArray]) -> list[str]:
        # TODO: this might make more sense to do locally, instead of doing it on the Modal instance
        from biotite.structure.io.pdb import PDBFile, set_structure
        from io import StringIO

        pdbs = []
        for a in atom_array:
            structure_file = PDBFile()
            set_structure(structure_file, a)
            string = StringIO()
            structure_file.write(string)
            pdbs.append(string.getvalue())
        return pdbs

    def _convert_outputs_to_cif(self, atom_array: List[AtomArray]) -> list[str]:
        # TODO: this might make more sense to do locally, instead of doing it on the Modal instance
        from biotite.structure.io.pdbx import CIFFile, set_structure
        from io import StringIO

        cifs = []
        for a in atom_array:
            structure_file = CIFFile()
            set_structure(structure_file, a)
            string = StringIO()
            structure_file.write(string)
            cifs.append(string.getvalue())
        return cifs
