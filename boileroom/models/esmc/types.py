"""Lightweight type definitions for ESM-C embedding outputs."""

from dataclasses import dataclass

import numpy as np

from ...base import EmbeddingPrediction, PredictionMetadata


@dataclass
class ESMEmbeddingOutput(EmbeddingPrediction):
    """Residue-aligned embedding output for ESM-C.

    Arrays are padded across the batch to the longest residue sequence. Padded
    embeddings/logits/hidden-state rows are zero; padded indices are ``-1``.
    """

    metadata: PredictionMetadata
    embeddings: np.ndarray
    chain_index: np.ndarray
    residue_index: np.ndarray
    hidden_states: np.ndarray | None = None
    lm_logits: np.ndarray | None = None


ESMCOutput = ESMEmbeddingOutput

__all__ = ["ESMCOutput", "ESMEmbeddingOutput"]
