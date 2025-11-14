"""Type definitions for ESM2 outputs without heavy dependencies."""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ...base import EmbeddingPrediction, PredictionMetadata


@dataclass
class ESM2Output(EmbeddingPrediction):
    """Output from ESM2 prediction including all model outputs."""

    embeddings: np.ndarray  # (batch_size, seq_len, embedding_dim)
    metadata: PredictionMetadata
    chain_index: np.ndarray  # (batch_size, seq_len)
    residue_index: np.ndarray  # (batch_size, seq_len)
    hidden_states: Optional[np.ndarray] = None  # (batch_size, hidden_state_iter, seq_len, embedding_dim)

