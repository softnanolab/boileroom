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


@dataclass
class ESM3InverseFoldingOutput:
    """Structure-conditioned sequence logits at masked residue positions (ESM3 only).

    Attributes
    ----------
    metadata : PredictionMetadata
        Timing and model metadata.
    positions : np.ndarray
        ``(n_positions,)`` residue indices that were masked, counted over the residues of the
        whole (possibly multichain) input, excluding chain breaks.
    logits : np.ndarray
        ``(n_positions, 20)`` sequence-track logits restricted to the 20 standard amino acids,
        in the order given by ``amino_acids``.
    amino_acids : str
        Amino-acid one-letter codes labelling the last axis of ``logits``.
    """

    metadata: PredictionMetadata
    positions: np.ndarray
    logits: np.ndarray
    amino_acids: str


ESMCOutput = ESMEmbeddingOutput
ESM3Output = ESMEmbeddingOutput

__all__ = ["ESM3InverseFoldingOutput", "ESM3Output", "ESMCOutput", "ESMEmbeddingOutput"]
