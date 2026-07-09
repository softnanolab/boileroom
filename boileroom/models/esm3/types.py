"""Lightweight type definitions for ESM-C and ESM3 embedding outputs."""

from dataclasses import dataclass

import numpy as np

from ...base import EmbeddingPrediction, PredictionMetadata


@dataclass
class ESMEmbeddingOutput(EmbeddingPrediction):
    """Residue-aligned embedding output for ESM-C and ESM3.

    Arrays are padded across the batch to the longest residue sequence. Padded
    embeddings/logits/hidden-state rows are zero; padded indices are ``-1``.
    """

    metadata: PredictionMetadata
    embeddings: np.ndarray
    chain_index: np.ndarray
    residue_index: np.ndarray
    hidden_states: np.ndarray | None = None
    lm_logits: np.ndarray | None = None
    # ESM3-only track logits, each per-residue and predicted from sequence alone
    # (structure input is optional, not required). Decode to estimates downstream
    # via the corresponding SDK track tokenizer. Each is None unless requested via
    # include_fields and the model is ESM3. The structure/folding track is not
    # exposed here.
    sasa_logits: np.ndarray | None = None  # over the discretized SASA token vocabulary
    secondary_structure_logits: np.ndarray | None = None  # over the SS8 token vocabulary
    function_logits: np.ndarray | None = None  # over the function-annotation vocabulary
    residue_annotation_logits: np.ndarray | None = None  # multi-hot residue-annotation logits


ESMCOutput = ESMEmbeddingOutput
ESM3Output = ESMEmbeddingOutput

__all__ = ["ESM3Output", "ESMCOutput", "ESMEmbeddingOutput"]
