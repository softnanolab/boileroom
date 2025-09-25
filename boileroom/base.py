"""Base classes and interfaces for BoilerRoom protein structure prediction models."""

import logging

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union, Sequence, Optional, Protocol, List

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
    positions: np.ndarray  # Atom positions
    pdb: Optional[list[str]] = None
    cif: Optional[list[str]] = None


class EmbeddingPrediction(Protocol):
    """Protocol defining the minimum interface for embedding outputs."""

    metadata: PredictionMetadata
    embeddings: np.ndarray  # Atom positions


class Algorithm(ABC):
    """Abstract base class for algorithms."""

    DEFAULT_CONFIG: dict = {}

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
    def fold(self, sequences: Union[str, Sequence[str]]) -> StructurePrediction:
        """Predict the structure for one or more protein sequences.

        Parameters
        ----------
        sequences : Union[str, Sequence[str]]
            A single sequence string or list of sequence strings
            containing valid amino acid characters

        Returns
        -------
        StructurePrediction
            Structure prediction output implementing the StructurePrediction protocol

        Raises:
            ValueError: If sequences are invalid
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
    def embed(self, sequences: Union[str, Sequence[str]]) -> EmbeddingPrediction:
        """Generate embeddings for one or more protein sequences.

        Parameters
        ----------
        sequences : Union[str, Sequence[str]]
            A single sequence string or list of sequence strings
            containing valid amino acid characters

        Returns
        -------
        EmbeddingPrediction
            Embedding output implementing the EmbeddingPrediction protocol

        Raises:
            ValueError: If sequences are invalid
            RuntimeError: If embedding generation fails
        """
        raise NotImplementedError

from .backend.base import Backend

class ModelWrapper(Protocol):

    def __init__(self, backend: Backend, config: dict = {}) -> None:
        """Initialize the model wrapper."""
        self.backend = backend
        self.config = config

    @abstractmethod
    def __call__(self, sequences: Union[str, Sequence[str]]) -> StructurePrediction | EmbeddingPrediction:
        """Call the model wrapper."""
        raise NotImplementedError