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
        """Initialize the algorithm instance and set its default runtime attributes.

        Parameters
        ----------
        config : dict
            Configuration overrides merged into the class DEFAULT_CONFIG; keys in this dict take precedence over defaults.
        """
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

        Raises
        ------
        RuntimeError
            If model loading fails.
        """
        raise NotImplementedError

    def update_config(self, config: dict) -> None:
        """Merge provided configuration entries into the algorithm's current configuration.

        This updates the instance's `config` in place; keys in `config` override existing values. This method is not compatible with Modal/remote execution and should not be relied on to update remote instances.

        Parameters
        ----------
        config : dict
            Configuration keys and values to merge into the existing config.
        """
        logger.warning("This does not work with Modal and remote execution. Create a new instance instead.")
        # TODO: Make this work smartly with remote Modal, calling _load() again, etc. and thus programmatically
        # updating the model if anything has changed
        self.config = {**self.config, **config}

    def _resolve_device(self) -> torch.device:
        """Select the computation device based on instance configuration and system capability.

        If the instance config contains a "device" entry, that device is returned; otherwise return "cuda:0" when CUDA is available, falling back to "cpu" when not.

        Returns
        -------
        torch.device
            The resolved device.
        """
        requested = self.config.get("device")
        if requested is not None:
            return torch.device(requested)
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")

    def _merge_options(self, options: Optional[dict] = None) -> dict:
        """Merge per-call options into the instance configuration while preventing overrides of static configuration keys.

        Parameters
        ----------
        options : Optional[dict]
            Per-call configuration that will override non-static keys in the instance config.

        Returns
        -------
        dict
            The merged configuration dictionary with values from `options` taking precedence for non-static keys.

        Raises
        ------
        ValueError
            If `options` contains any keys present in `STATIC_CONFIG_KEYS`.
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
        """Create PredictionMetadata for a model prediction.

        Parameters
        ----------
        model_name : str
            Name of the model.
        model_version : str
            Version of the model.

        Returns
        -------
        PredictionMetadata
            Instance with `model_name` and `model_version` set to the provided values, and `prediction_time` and `sequence_lengths` initialized to None.
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

        Raises
        ------
        ValueError
            If sequences are invalid or if options contains static config keys.
        RuntimeError
            If prediction fails.
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

        Raises
        ------
        ValueError
            If any sequence contains invalid amino acids.
        """
        # Convert single sequence to list
        if isinstance(sequences, str):
            sequences = [sequences]

        # Validate each sequence and return as explicit list
        return [seq for seq in sequences if validate_sequence(seq)]

    def _compute_sequence_lengths(self, sequences: List[str]) -> List[int]:
        """Compute residue counts for each multimer sequence.

        Parameters
        ----------
        sequences : List[str]
            Sequences where chains may be joined with ':' separators.

        Returns
        -------
        List[int]
            Number of residues for each sequence, treating ':' characters as chain separators and not counting them toward residue length.
        """
        return [len(seq) - seq.count(":") for seq in sequences]

    @staticmethod
    def _filter_include_fields(
        output: StructurePrediction,
        include_fields: Optional[List[str]],
    ) -> StructurePrediction:
        """Filter fields of a StructurePrediction dataclass according to an inclusion list.

        When `output` is a dataclass instance, returns a copy where any dataclass fields not listed in `include_fields`
        are set to `None`, while always preserving `metadata` and `atom_array`. If `include_fields` is `None`,
        empty, or contains `"*"`, or if `output` is not a dataclass, the original `output` is returned unchanged.

        Parameters
        ----------
        output : StructurePrediction
            The prediction dataclass to filter.
        include_fields : Optional[List[str]]
            Names of fields to retain in addition to the always-included fields.

        Returns
        -------
        StructurePrediction
            A dataclass instance with non-kept fields set to `None`, or the original `output`
            unchanged when filtering is not applicable.
        """
        if not dataclasses.is_dataclass(output) or (include_fields and "*" in include_fields):
            return output

        # Cast to Any to help mypy understand this is a dataclass instance
        dataclass_output: Any = output
        always_include = {"metadata", "atom_array"}
        available_fields = {field.name for field in dataclasses.fields(dataclass_output)}
        fields_to_keep = always_include | (set(include_fields or []) & available_fields)

        updates = {
            field.name: None for field in dataclasses.fields(dataclass_output) if field.name not in fields_to_keep
        }
        if not updates:
            return cast(StructurePrediction, output)

        filtered = dataclasses.replace(dataclass_output, **updates)
        return cast(StructurePrediction, filtered)

    def _prepare_multimer_sequences(self, sequences: List[str]) -> List[str]:
        """Prepare multimer sequences for model-specific prediction.

        Implementations should transform the provided input sequences into the sequence format required by the model for multimer (multi-chain) prediction.

        Parameters
        ----------
        sequences : List[str]
            Input protein sequences; each element may represent a single chain or a multimer component.

        Returns
        -------
        List[str]
            Sequences formatted for the model's multimer prediction pipeline.

        Raises
        ------
        NotImplementedError
            Always raised by the base implementation — subclasses must override this method.
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

        Raises
        ------
        ValueError
            If sequences are invalid or if options contains static config keys.
        RuntimeError
            If embedding generation fails.
        """
        raise NotImplementedError

    @staticmethod
    def _filter_include_fields(
        output: EmbeddingPrediction,
        include_fields: Optional[List[str]],
    ) -> EmbeddingPrediction:
        """Filter an EmbeddingPrediction dataclass to include only the requested fields.

        If `output` is not a dataclass or `include_fields` contains `"*"`, the original
        `output` is returned unchanged. Otherwise, the returned dataclass keeps the
        always-included fields `metadata`, `embeddings`, `chain_index`, and
        `residue_index`, plus any names present in `include_fields`. All other fields
        are set to `None`.

        Parameters
        ----------
        output : EmbeddingPrediction
            The EmbeddingPrediction dataclass to filter.
        include_fields : Optional[List[str]]
            Optional list of field names to include in addition to the
            always-included fields. If `None`, only the always-included fields are kept.

        Returns
        -------
        EmbeddingPrediction
            An EmbeddingPrediction with non-kept fields set to `None`, or the original
            `output` if no filtering is performed.
        """
        if not dataclasses.is_dataclass(output) or (include_fields and "*" in include_fields):
            return output

        # Cast to Any to help mypy understand this is a dataclass instance
        dataclass_output: Any = output
        always_include = {"metadata", "embeddings", "chain_index", "residue_index"}
        available_fields = {field.name for field in dataclasses.fields(dataclass_output)}
        fields_to_keep = always_include | (set(include_fields or []) & available_fields)

        updates = {
            field.name: None for field in dataclasses.fields(dataclass_output) if field.name not in fields_to_keep
        }
        if not updates:
            return cast(EmbeddingPrediction, output)

        return cast(EmbeddingPrediction, dataclasses.replace(dataclass_output, **updates))


class ModelWrapper:
    def __init__(self, backend: str = "modal", device: str | None = None, config: dict | None = None) -> None:
        """Create a ModelWrapper configured to use a specified backend, execution device, and optional configuration.

        Parameters
        ----------
        backend : str
            Backend identifier to use (default: "modal").
        device : str | None
            Preferred execution device (e.g., "cuda", "cpu"); None leaves device selection to the backend or later resolution.
        config : dict | None
            Optional backend-specific configuration; empty dict used if None.
        """
        self.backend = backend
        self.device = device
        self.config = config or {}

    def _call_backend_method(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        """Call a named method on the configured backend model, preferring a `remote` callable if present.

        Parameters
        ----------
        method_name : str
            Name of the method to invoke on the backend model.
        *args : Any
            Positional arguments forwarded to the backend method.
        **kwargs : Any
            Keyword arguments forwarded to the backend method.

        Returns
        -------
        Any
            The value returned by the backend method invocation.

        Raises
        ------
        RuntimeError
            If the wrapper has no initialized backend.
        AttributeError
            If the backend model does not expose `method_name`.
        TypeError
            If the attribute `method_name` exists but is not callable.
        """
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
