"""Lightweight output types for SAE feature extraction.

Kept free of heavy dependencies (only ``numpy`` and base protocols) per the
repository's ``types.py`` policy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...base import PredictionMetadata


@dataclass
class SAEFeaturesOutput:
    """Sparse-autoencoder features for one batch of sequences.

    Arrays are padded across the batch to the longest residue sequence, matching
    the ESM-C embedding outputs they are derived from. Padded residue rows are
    zero and padded index entries are ``-1``.

    Attributes
    ----------
    metadata : PredictionMetadata
        Model / timing metadata.
    pooled_features : np.ndarray
        Per-protein feature vectors of shape ``(batch, num_features)``, obtained
        by max-pooling each feature across the (unpadded) residues.
    chain_index : np.ndarray
        Per-residue chain ids of shape ``(batch, residues)`` (``-1`` = padding).
    residue_index : np.ndarray
        Per-residue positions of shape ``(batch, residues)`` (``-1`` = padding).
    layer : int
        ESM-C transformer layer the SAE was applied to.
    num_features : int
        Size of the SAE feature (codebook) space.
    sae_model : str
        Identifier of the SAE weights used.
    features : np.ndarray | None
        Optional dense per-residue activations of shape
        ``(batch, residues, num_features)``. Only populated when explicitly
        requested via ``include_per_residue=True`` since it can be large.
    """

    metadata: PredictionMetadata
    pooled_features: np.ndarray
    chain_index: np.ndarray
    residue_index: np.ndarray
    layer: int
    num_features: int
    sae_model: str
    features: np.ndarray | None = None


__all__ = ["SAEFeaturesOutput"]
