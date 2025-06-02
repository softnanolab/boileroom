import modal
import numpy as np
from dataclasses import dataclass
from typing import List, Union, Optional, TYPE_CHECKING

import logging

from . import app
from .base import EmbeddingAlgorithm, EmbeddingPrediction, PredictionMetadata
from .images import esm_image
from .utils import MINUTES, MODEL_DIR
from .images.volumes import model_weights
from .utils import Timer
from .linker import compute_position_ids, store_multimer_properties, replace_glycine_linkers

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions


# TODO: turn this into a Pydantic model instead
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


@app.cls(
    image=esm_image,
    gpu="T4",
    timeout=20 * MINUTES,
    container_idle_timeout=10 * MINUTES,
    volumes={MODEL_DIR: model_weights},
)
class ESM2(EmbeddingAlgorithm):
    """ESM2 protein language model."""

    DEFAULT_CONFIG = {
        "model_name": "esm2_t33_650M_UR50D",
        "output_hidden_states": True,
        # Chain linking and positioning config
        "glycine_linker": "",
        "position_ids_skip": 512,
    }

    def __init__(self, config: dict = {}) -> None:
        super().__init__(config)
        self.metadata = self._initialize_metadata(
            model_name="ESM-2",
            model_version="v4.49.0",  # HuggingFace transformers version
        )
        self.model_dir: Optional[str] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[EsmModel] = None
        self.assert_valid_model(config)

    @staticmethod
    def assert_valid_model(config: dict) -> None:
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
        assert config["model_name"] in models_name, f"Model {config['model_name']} not supported"

    @modal.enter()
    def _initialize(self) -> None:
        self.model_dir = MODEL_DIR
        self._load()

    def _load(self) -> None:
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(f"facebook/{self.config['model_name']}")
        if self.model is None:
            self.model = EsmModel.from_pretrained(f"facebook/{self.config['model_name']}")
        self.device = "cuda"
        self.model = self.model.cuda()
        self.model.eval()
        self.ready = True

    @modal.method()
    def embed(self, sequences: Union[str, List[str]]) -> ESM2Output:
        if self.tokenizer is None and self.model is None:
            logger.warning("Model not loaded. Forcing the model to load... Next time call _load() first.")
            self._load()
        assert self.tokenizer is not None and self.model is not None, "Model not loaded"

        logger.debug(f'Embedding {len(sequences)} sequences using {self.config["model_name"]}')

        # Support for glycine linker and positional skip logic (multimer)
        if isinstance(sequences, str):
            sequences = [sequences]

        if any(":" in seq for seq in sequences):
            # Multimer logic
            glycine_linker = self.config["glycine_linker"]
            multimer_properties = self._store_multimer_properties(sequences, glycine_linker)
            tokenized = self.tokenizer(
                replace_glycine_linkers(sequences, glycine_linker),
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            # Add position_ids and attention_mask
            tokenized["position_ids"] = compute_position_ids(
                sequences, glycine_linker, self.config["position_ids_skip"]
            )
            tokenized["attention_mask"] = (multimer_properties["linker_map"] == 1).to(torch.int32)
        else:
            # Monomer logic
            tokenized = self.tokenizer(
                sequences,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            multimer_properties = None
        tokenized = tokenized.to(self.device)
        tokenized["output_hidden_states"] = self.config["output_hidden_states"]
        tokenized["use_cache"] = True

        with Timer("Model Inference") as timer:
            with torch.inference_mode():
                outputs = self.model(**tokenized)

        outputs = self._convert_outputs(outputs, multimer_properties, timer.duration)

        return outputs

    @staticmethod
    def _store_multimer_properties(sequences: List[str], glycine_linker: str) -> dict[str, torch.Tensor]:
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
    ) -> ESM2Output:
        """Convert model outputs to ESM2Output format."""

        embeddings = outputs.last_hidden_state.cpu().numpy()

        if self.config["output_hidden_states"]:
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
            chain_index_output = np.zeros((embeddings.shape[0], embeddings.shape[1]), dtype=np.int32)
            residue_index_output = None  # HACK: for now, but given it's only monomers, it is clear what the res ids are
            if hidden_states is not None:
                hidden_states = hidden_states[:, :, 1:-1, :]  # remove the first and last token
            embeddings = embeddings[:, 1:-1, :]  # remove the first and last token

        self.metadata.prediction_time = prediction_time

        return ESM2Output(
            metadata=self.metadata,
            embeddings=embeddings,
            hidden_states=hidden_states,
            chain_index=chain_index_output,
            residue_index=residue_index_output,
        )

    def _mask_linker_region(
        self,
        embeddings: np.ndarray,
        hidden_states: np.ndarray,
        linker_map: torch.Tensor,
        residue_index: torch.Tensor,
        chain_index: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor]:
        """
        Mask the linker region in the outputs and track padding information.

        Args:
            embeddings: Dictionary containing model outputs
            hidden_states: Dictionary containing model outputs
            chain_index: Dictionary containing model outputs
            residue_index: Dictionary containing model outputs

        Returns:
            dict: Updated outputs with linker regions masked and padding information
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


def get_esm2(gpu_type="T4", config: dict = {}):
    """
    Note that the app will still show that's using T4, but the actual method / function call will use the correct GPU,
    and display accordingly in the Modal dashboard.
    """
    Model = ESM2.with_options(gpu=gpu_type)  # type: ignore
    return Model(config=config)
