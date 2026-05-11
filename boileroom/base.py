"""Base classes and interfaces for BoilerRoom protein structure prediction models."""

import contextlib
import dataclasses
import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, ClassVar, Literal, Protocol, cast

import numpy as np
import torch

from .images.metadata import format_image_reference, get_default_image_tag
from .models.registry import ModelSpec, resolve_object
from .utils import validate_sequence

logger = logging.getLogger(__name__)


def _filter_dataclass_fields(output: Any, include_fields: list[str] | None, always_include: set[str]) -> Any:
    """Return a filtered dataclass copy when include_fields constrains output."""
    if not dataclasses.is_dataclass(output) or (include_fields and "*" in include_fields):
        return output

    dataclass_output: Any = output
    available_fields = {field.name for field in dataclasses.fields(dataclass_output)}
    fields_to_keep = always_include | (set(include_fields or []) & available_fields)
    updates = {field.name: None for field in dataclasses.fields(dataclass_output) if field.name not in fields_to_keep}
    if not updates:
        return output
    return dataclasses.replace(dataclass_output, **updates)


@dataclass
class PredictionMetadata:
    """Metadata about a protein structure prediction."""

    model_name: str
    model_version: str
    sequence_lengths: list[int] | None
    preprocessing_time: float | None = None  # in seconds
    inference_time: float | None = None  # in seconds
    postprocessing_time: float | None = None  # in seconds


class StructurePrediction(Protocol):
    """Protocol defining the minimum interface for structure prediction outputs."""

    metadata: PredictionMetadata
    atom_array: list[Any] | None  # Typically List[AtomArray]
    pdb: list[str] | None = None
    cif: list[str] | None = None


class EmbeddingPrediction(Protocol):
    """Protocol defining the minimum interface for embedding outputs."""

    metadata: PredictionMetadata
    embeddings: np.ndarray
    chain_index: np.ndarray
    residue_index: np.ndarray
    hidden_states: np.ndarray | None = None


class Algorithm(ABC):
    """Abstract base class for algorithms."""

    DEFAULT_CONFIG: ClassVar[Mapping[str, Any]] = MappingProxyType({})
    # Static config keys that can only be set at initialization and cannot be overridden per-call
    STATIC_CONFIG_KEYS: ClassVar[frozenset[str]] = frozenset()

    def __init__(self, config: dict | None = None) -> None:
        """Initialize the algorithm instance and set its default runtime attributes.

        Parameters
        ----------
        config : dict
            Configuration overrides merged into the class DEFAULT_CONFIG; keys in this dict take precedence over defaults.
        """
        if config is None:
            config = {}
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

    def _merge_options(self, options: dict | None = None) -> dict:
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
            Instance with `model_name` and `model_version` set to the provided values, and timing fields and `sequence_lengths` initialized to None.
        """
        return PredictionMetadata(
            model_name=model_name,
            model_version=model_version,
            sequence_lengths=None,
            preprocessing_time=None,
            inference_time=None,
            postprocessing_time=None,
        )

    def _validate_sequences(self, sequences: str | Sequence[str]) -> list[str]:
        """Validate input sequences and convert them to list format.

        Parameters
        ----------
        sequences : str | Sequence[str]
            Single sequence or list of sequences.

        Returns
        -------
        list[str]
            List of validated sequences.

        Raises
        ------
        ValueError
            If any sequence contains invalid amino acids.
        """
        if isinstance(sequences, str):
            sequences = [sequences]

        return [seq for seq in sequences if validate_sequence(seq)]


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
    def fold(self, sequences: str | Sequence[str], options: dict | None = None) -> StructurePrediction:
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

    def _compute_sequence_lengths(self, sequences: list[str]) -> list[int]:
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
        include_fields: list[str] | None,
    ) -> StructurePrediction:
        """Filter fields of a StructurePrediction dataclass according to an inclusion list."""
        return cast(StructurePrediction, _filter_dataclass_fields(output, include_fields, {"metadata", "atom_array"}))

    def _prepare_multimer_sequences(self, sequences: list[str]) -> list[str]:
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
    def embed(self, sequences: str | Sequence[str], options: dict | None = None) -> EmbeddingPrediction:
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
        include_fields: list[str] | None,
    ) -> EmbeddingPrediction:
        """Filter an EmbeddingPrediction dataclass to include only the requested fields."""
        return cast(
            EmbeddingPrediction,
            _filter_dataclass_fields(
                output, include_fields, {"metadata", "embeddings", "chain_index", "residue_index"}
            ),
        )


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
        self.model_spec: ModelSpec | None = None

    def _initialize_backend_from_spec(
        self,
        model_spec: ModelSpec,
        backend: str | None = None,
        device: str | None = None,
        config: dict | None = None,
    ) -> None:
        """Build and start a backend using shared model metadata.

        Parameters
        ----------
        model_spec : ModelSpec
            Registry entry describing how to construct backends for this model.
        backend : str | None
            Backend selector string. When omitted, uses the model spec default.
        device : str | None
            Optional device or GPU selector passed through to the backend.
        config : dict | None
            Optional backend/model configuration.

        Raises
        ------
        ValueError
            If the selected backend is unsupported or lacks the required spec metadata.
        """
        from .backend import ModalBackend
        from .backend.apptainer import ApptainerBackend

        resolved_backend = backend or model_spec.default_backend
        resolved_config = config or {}
        backend_type, backend_tag = self.parse_backend(resolved_backend)

        if backend_type not in model_spec.supported_backends:
            supported = ", ".join(model_spec.supported_backends)
            raise ValueError(
                f"Backend {backend_type} not supported for {model_spec.public_name}. Supported backends: {supported}"
            )

        backend_instance: Any
        if backend_type == "modal":
            if model_spec.modal_class_path is None:
                raise ValueError(f"Modal backend is not configured for {model_spec.public_name}")
            modal_cls = resolve_object(model_spec.modal_class_path)
            backend_instance = ModalBackend(modal_cls, resolved_config, device=device)
        elif backend_type == "apptainer":
            if model_spec.apptainer_core_class_path is None or model_spec.apptainer_image_name is None:
                raise ValueError(f"Apptainer backend is not configured for {model_spec.public_name}")
            image_uri = f"docker://{format_image_reference(model_spec.apptainer_image_name, backend_tag)}"
            backend_instance = ApptainerBackend(
                model_spec.apptainer_core_class_path,
                image_uri,
                resolved_config,
                device=device,
            )
        else:
            raise ValueError(f"Backend {backend_type} not supported")

        self.backend = resolved_backend
        self.device = device
        self.config = resolved_config
        self.model_spec = model_spec
        self._backend = backend_instance
        self._backend.start()

    @staticmethod
    def parse_backend(backend: str) -> tuple[str, str | None]:
        """Parse backend identifier into backend type and optional tag.

        This helper allows backends to be specified with an optional tag suffix
        using the syntax ``\"backend:tag\"``. Tags are currently only interpreted
        for the ``\"apptainer\"`` backend, where they map to Docker image tags.
        All other backends ignore any tag and treat the full string before the
        first colon as the backend identifier.

        Parameters
        ----------
        backend : str
            Backend identifier string, optionally including a tag suffix.

        Returns
        -------
        tuple[str, Optional[str]]
            A tuple of ``(backend_type, backend_tag)`` where ``backend_type`` is
            the base backend identifier (for example ``\"modal\"`` or
            ``\"apptainer\"``) and ``backend_tag`` is either the tag string for
            ``\"apptainer\"`` backends (defaulting to the installed package version
            when no tag is provided) or ``None`` for all other backends.
        """
        if ":" in backend:
            backend_type, backend_tag = backend.split(":", 1)
        else:
            backend_type, backend_tag = backend, None

        backend_type = backend_type.strip()
        if backend_type == "apptainer":
            backend_tag = (backend_tag or get_default_image_tag()).strip() or get_default_image_tag()
            return backend_type, backend_tag

        return backend_type, None

    def __del__(self) -> None:
        """Clean up backend when ModelWrapper is destroyed.

        This ensures that backend resources (such as image-backed subprocesses)
        are properly shut down when the ModelWrapper instance is garbage collected.
        """
        backend = getattr(self, "_backend", None)
        if backend is not None:
            # __del__ must not raise; exceptions during GC are problematic
            with contextlib.suppress(Exception):
                backend.stop()

    def __enter__(self) -> "ModelWrapper":
        """Context manager entry.

        Returns
        -------
        ModelWrapper
            Returns self to allow using the model in a with statement.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> Literal[False]:
        """Context manager exit - ensures backend is stopped.

        Parameters
        ----------
        exc_type : type | None
            Exception type if an exception was raised.
        exc_val : Exception | None
            Exception value if an exception was raised.
        exc_tb : TracebackType | None
            Exception traceback if an exception was raised.

        Returns
        -------
        bool
            Always returns False to not suppress exceptions.
        """
        backend = getattr(self, "_backend", None)
        if backend is not None:
            backend.stop()
        return False  # Don't suppress exceptions

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
