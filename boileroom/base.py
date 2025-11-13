"""Base classes and interfaces for BoilerRoom protein structure prediction models."""

import dataclasses
import logging
import torch

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Union, Sequence, Optional, Protocol, List, cast

import numpy as np

from .utils import validate_sequence

logger = logging.getLogger(__name__)


@dataclass
class PredictionMetadata:
    """Metadata about a protein structure prediction."""

    model_name: str
    model_version: str
    prediction_time: Optional[float]  # in seconds
    sequence_lengths: Optional[List[int]]


class StructurePrediction(Protocol):
    """Protocol defining the minimum interface for structure prediction outputs."""

    metadata: PredictionMetadata
    atom_array: Optional[List[Any]]  # Typically List[AtomArray]
    pdb: Optional[list[str]] = None
    cif: Optional[list[str]] = None


class EmbeddingPrediction(Protocol):
    """Protocol defining the minimum interface for embedding outputs."""

    metadata: PredictionMetadata
    embeddings: np.ndarray
    chain_index: np.ndarray
    residue_index: np.ndarray
    hidden_states: Optional[np.ndarray] = None


class Algorithm(ABC):
    """Abstract base class for algorithms."""

    DEFAULT_CONFIG: dict = {}
    # Static config keys that can only be set at initialization and cannot be overridden per-call
    STATIC_CONFIG_KEYS: set[str] = set()

    def __init__(self, config: dict = {}) -> None:
        """Initialize the algorithm."""
        self.config = {**self.DEFAULT_CONFIG, **config}
        self.name: str = self.__class__.__name__
        self.version: str = ""  # Should be overridden by implementations
        self.ready: bool = False

    @abstractmethod
    def _load(self) -> None:
        """Load the model and prepare it for prediction.

        This method should handle:
        - Loading model weights
        - Moving model to appropriate device
        - Setting up any necessary preprocessing

        Raises:
            RuntimeError: If model loading fails
        """
        raise NotImplementedError

    def update_config(self, config: dict) -> None:
        """
        Update the config with the default values.

        This does not work with Modal and remote execution. Create a new instance instead.
        """
        logger.warning("This does not work with Modal and remote execution. Create a new instance instead.")
        # TODO: Make this work smartly with remote Modal, calling _load() again, etc. and thus programmatically
        # updating the model if anything has changed
        self.config = {**self.config, **config}

    def _resolve_device(self) -> torch.device:
        requested = self.config.get("device")
        if requested is not None:
            return torch.device(requested)
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")

    def _merge_options(self, options: Optional[dict] = None) -> dict:
        """Merge per-call options with static config, enforcing that static keys cannot be overridden.

        Parameters
        ----------
        options : Optional[dict]
            Per-call options dictionary. Keys that are in STATIC_CONFIG_KEYS will raise ValueError.

        Returns
        -------
        dict
            Merged configuration dictionary with options overriding non-static config values.

        Raises
        ------
        ValueError
            If options contains any keys that are in STATIC_CONFIG_KEYS.
        """
        if options is None:
            options = {}

        static_keys: set[str] = getattr(self, "STATIC_CONFIG_KEYS", set())
        if not isinstance(static_keys, set):
            static_keys = set(static_keys)

        # Check for attempts to override static config keys
        conflicting_keys = set(options.keys()) & static_keys
        if conflicting_keys:
            raise ValueError(
                f"The following config keys can only be set at initialization and cannot be overridden per-call: {sorted(conflicting_keys)}"
            )

        # Merge: static config (from self.config) + dynamic options
        return {**self.config, **options}

    @staticmethod
    def _initialize_metadata(model_name: str, model_version: str) -> PredictionMetadata:
        """Initialize metadata for the prediction.

        Parameters
        ----------
        model_name : str
            Name of the model
        model_version : str
            Version of the model

        Returns
        -------
        PredictionMetadata
            Metadata for the prediction
        """
        return PredictionMetadata(
            model_name=model_name, model_version=model_version, prediction_time=None, sequence_lengths=None
        )


class FoldingAlgorithm(Algorithm):
    """Abstract base class for protein structure prediction algorithms.

    This class defines the interface that all protein structure prediction models must implement.
    Each implementation should handle model loading, prediction, and cleanup appropriately.

    Attributes:
        name (str): Name of the folding algorithm
        version (str): Version of the model being used
        ready (bool): Whether the model is loaded and ready for prediction
    """

    @abstractmethod
    def fold(self, sequences: Union[str, Sequence[str]], options: Optional[dict] = None) -> StructurePrediction:
        """Predict the structure for one or more protein sequences.

        Parameters
        ----------
        sequences : Union[str, Sequence[str]]
            A single sequence string or list of sequence strings
            containing valid amino acid characters
        options : Optional[dict]
            Optional per-call configuration overrides. Keys that are in STATIC_CONFIG_KEYS
            cannot be overridden and will raise ValueError.

        Returns
        -------
        StructurePrediction
            Structure prediction output implementing the StructurePrediction protocol

        Raises:
            ValueError: If sequences are invalid or if options contains static config keys
            RuntimeError: If prediction fails
        """
        raise NotImplementedError

    def _validate_sequences(self, sequences: Union[str, Sequence[str]]) -> list[str]:
        """Validate input sequences and convert to list format.

        Parameters
        ----------
        sequences : Union[str, Sequence[str]]
            Single sequence or list of sequences

        Returns
        -------
        list[str]
            List of validated sequences

        Raises:
            ValueError: If any sequence contains invalid amino acids
        """
        # Convert single sequence to list
        if isinstance(sequences, str):
            sequences = [sequences]

        # Validate each sequence and return as explicit list
        return [seq for seq in sequences if validate_sequence(seq)]

    def _compute_sequence_lengths(self, sequences: List[str]) -> List[int]:
        """
        Compute the sequence lengths for multimer sequences.
        """
        return [len(seq) - seq.count(":") for seq in sequences]

    @staticmethod
    def _filter_output_attributes(
        output: StructurePrediction,
        requested_attributes: Optional[List[str]],
    ) -> StructurePrediction:
        """Return a copy of ``output`` with attributes filtered per request."""
        if not dataclasses.is_dataclass(output) or (requested_attributes and "*" in requested_attributes):
            return output

        # Cast to Any to help mypy understand this is a dataclass instance
        dataclass_output: Any = output
        always_include = {"metadata", "atom_array"}
        available_fields = {field.name for field in dataclasses.fields(dataclass_output)}
        fields_to_keep = always_include | (set(requested_attributes or []) & available_fields)

        updates = {
            field.name: None for field in dataclasses.fields(dataclass_output) if field.name not in fields_to_keep
        }
        if not updates:
            return cast(StructurePrediction, output)

        filtered = dataclasses.replace(dataclass_output, **updates)
        return cast(StructurePrediction, filtered)

    def _prepare_multimer_sequences(self, sequences: List[str]) -> List[str]:
        """
        Prepare multimer sequences for prediction.
        This method is model-specific and how they handle multimers.

        Parameters
        ----------
        sequences : List[str]
            List of protein sequences

        Returns
        -------
        List[str]
            List of prepared sequences"
        """
        raise NotImplementedError


class EmbeddingAlgorithm(Algorithm):
    """Abstract base class for embedding algorithms."""

    @abstractmethod
    def embed(self, sequences: Union[str, Sequence[str]], options: Optional[dict] = None) -> EmbeddingPrediction:
        """Generate embeddings for one or more protein sequences.

        Parameters
        ----------
        sequences : Union[str, Sequence[str]]
            A single sequence string or list of sequence strings
            containing valid amino acid characters
        options : Optional[dict]
            Optional per-call configuration overrides. Keys that are in STATIC_CONFIG_KEYS
            cannot be overridden and will raise ValueError.

        Returns
        -------
        EmbeddingPrediction
            Embedding output implementing the EmbeddingPrediction protocol

        Raises:
            ValueError: If sequences are invalid or if options contains static config keys
            RuntimeError: If embedding generation fails
        """
        raise NotImplementedError

    @staticmethod
    def _filter_output_attributes(
        output: EmbeddingPrediction,
        requested_attributes: Optional[List[str]],
    ) -> EmbeddingPrediction:
        """Return a copy of ``output`` with attributes filtered per request."""
        if not dataclasses.is_dataclass(output) or (requested_attributes and "*" in requested_attributes):
            return output

        # Cast to Any to help mypy understand this is a dataclass instance
        dataclass_output: Any = output
        always_include = {"metadata", "embeddings", "chain_index", "residue_index"}
        available_fields = {field.name for field in dataclasses.fields(dataclass_output)}
        fields_to_keep = always_include | (set(requested_attributes or []) & available_fields)

        updates = {
            field.name: None for field in dataclasses.fields(dataclass_output) if field.name not in fields_to_keep
        }
        if not updates:
            return cast(EmbeddingPrediction, output)

        return cast(EmbeddingPrediction, dataclasses.replace(dataclass_output, **updates))


class ModelWrapper:
    def __init__(self, backend: str = "modal", device: str | None = None, config: dict | None = None) -> None:
        """Initialize the model wrapper."""
        self.backend = backend
        self.device = device
        self.config = config or {}

    def _call_backend_method(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        """Invoke a method on the underlying backend model, handling remote calls if needed."""
        backend = getattr(self, "_backend", None)
        if backend is None:
            raise RuntimeError("Backend is not initialized for this model wrapper.")

        backend_model = backend.get_model()
        method = getattr(backend_model, method_name, None)
        if method is None:
            raise AttributeError(f"Backend model does not have method '{method_name}'.")

        remote_callable = getattr(method, "remote", None)
        if callable(remote_callable):
            return remote_callable(*args, **kwargs)

        if callable(method):
            return method(*args, **kwargs)

        raise TypeError(f"Attribute '{method_name}' on backend model is not callable.")
