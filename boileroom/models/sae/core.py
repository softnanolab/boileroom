"""Core SAE feature extraction: from an amino-acid sequence to SAE features.

Two feature sources are supported and selected by the ``feature_source`` config:

``"forge"`` (default)
    Call Biohub's hosted ``ESMCForgeInferenceClient`` with a ``SAEConfig`` and read
    back SAE feature activations. This is the only way to use the **ESMC-6B**
    layer-60 SAE featured in the paper (``k = 64``, codebook ``2**14``), which is
    too large to run locally. Requires a Biohub API token.

``"local"``
    Run ESM-C locally / on Modal to get per-layer hidden states, then apply a
    local :class:`~boileroom.models.sae.sae_module.SparseAutoencoder` loaded from
    the Biohub per-layer Hugging Face weights. Works for the 300M / 600M SAEs on a
    single GPU with no API token.

Either way, per-protein feature vectors are formed by **max-pooling** each feature
across the (unpadded) residues, following *Language Modeling Materializes a World
Model of Protein Biology* (Biohub, 2026).

The ESM-C embedder, the local SAE, and the Forge backend are all injectable so the
core can be unit-tested with fakes — no Modal, GPU, ``esm`` SDK, or API token.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Sequence
from typing import Any, ClassVar, Protocol, cast

import numpy as np
import torch

from ...base import EmbeddingAlgorithm
from ...utils import Timer
from .forge import ForgeSAEBackend
from .sae_module import SparseAutoencoder, max_pool_features
from .types import SAEFeaturesOutput

logger = logging.getLogger(__name__)

# ESM-C hidden-state dimensionality by model. Used to size the local SAE input.
ESMC_D_MODEL: dict[str, int] = {
    "esmc_300m": 960,
    "esmc_600m": 1152,
    "esmc_6b": 2560,
}

# Per-model defaults for the *local* backend: the released Biohub per-layer SAE
# repo and a representative transformer layer for each locally runnable ESM-C
# model. The base DEFAULT_CONFIG sae_layer/sae_repo_id target the default Forge
# ESMC-6B / layer-60 SAE, which is wrong for the local 300M/600M models (600M has
# no layer 60). The collection ships an SAE per layer; these defaults pick the
# layer at the same relative depth (~0.75) as the paper's featured 6B / layer-60
# SAE (60 of 6B's 80 layers), i.e. layer 27 for the 36-layer 600M model and layer
# 22 for the 30-layer 300M model. Override ``sae_layer`` to target another layer.
LOCAL_SAE_DEFAULTS: dict[str, dict[str, str | int]] = {
    "esmc_300m": {"sae_repo_id": "biohub/ESMC-300M-sae-k64-codebook16384", "sae_layer": 22},
    "esmc_600m": {"sae_repo_id": "biohub/ESMC-600M-sae-k64-codebook16384", "sae_layer": 27},
}


class _Embedder(Protocol):
    """Minimal interface the local SAE path needs from an ESM-C embedder."""

    def embed(self, sequences: str | Sequence[str], options: dict | None = None) -> Any: ...


class _ForgeFeatures(Protocol):
    """Minimal interface the forge SAE path needs from a Forge backend."""

    def features(self, sdk_sequence: str) -> np.ndarray: ...


class SAECore(EmbeddingAlgorithm):
    """Map protein sequences to sparse-autoencoder features.

    Parameters
    ----------
    config : dict | None
        Configuration overrides merged into :attr:`DEFAULT_CONFIG`. Key groups:

        Shared
            ``feature_source`` (``"forge"`` | ``"local"``), ``normalize_features``,
            ``include_per_residue``, ``num_features``, ``k``, ``sae_layer``,
            ``device``.
        Forge backend
            ``forge_model``, ``forge_sae_model``, ``forge_url``, ``forge_token``
            (falls back to the ``ESM_API_KEY`` env var).
        Local backend
            ``esmc_model_name``, ``sae_repo_id``, ``activation``.
    embedder : _Embedder | None
        Optional pre-built ESM-C embedder (local path). Injecting a fake makes the
        core testable without model dependencies.
    sae : SparseAutoencoder | None
        Optional pre-built local SAE. When ``None`` weights are loaded from
        ``sae_repo_id`` on load.
    forge_backend : _ForgeFeatures | None
        Optional pre-built Forge backend. When ``None`` one is constructed from the
        ``forge_*`` config on load.
    """

    DEFAULT_CONFIG: ClassVar[dict[str, Any]] = {
        "device": "cuda:0",
        # "forge" (hosted API, default) or "local" (ESM-C + local SAE).
        "feature_source": "forge",
        "normalize_features": True,
        "include_per_residue": False,
        "num_features": 16384,
        "k": 64,
        "sae_layer": 60,
        # Forge backend defaults: the paper's ESMC-6B layer-60 SAE.
        "forge_model": "esmc-6b-2024-12",
        "forge_sae_model": "esmc-6b-2024-12-sae-layer60-k64-codebook16384",
        "forge_url": "https://biohub.ai",
        "forge_token": None,
        # Local backend defaults (used when feature_source == "local").
        "esmc_model_name": "esmc_600m",
        "sae_repo_id": "biohub/ESMC-600M-sae-k64-codebook16384",
        "activation": "topk",
    }
    STATIC_CONFIG_KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "device",
            "feature_source",
            # normalize_features is baked into the Forge backend at construction and
            # applied at encode time locally; keep it init-only so both backends stay
            # consistent (it cannot be overridden per-call on the hosted path).
            "normalize_features",
            "num_features",
            "k",
            "sae_layer",
            "activation",
            "esmc_model_name",
            "sae_repo_id",
            "forge_model",
            "forge_sae_model",
            "forge_url",
            "forge_token",
        }
    )
    MODEL_DISPLAY_NAME: ClassVar[str] = "ESM-C SAE"

    def __init__(
        self,
        config: dict | None = None,
        embedder: _Embedder | None = None,
        sae: SparseAutoencoder | None = None,
        forge_backend: _ForgeFeatures | None = None,
    ) -> None:
        super().__init__(config)
        source = str(self.config["feature_source"])
        if source not in ("forge", "local"):
            raise ValueError(f"feature_source must be 'forge' or 'local'; got {source!r}.")
        self._source = source
        if source == "local":
            self._apply_local_defaults(config or {})
        self._embedder = embedder
        self._sae = sae
        self._forge = forge_backend
        model_version = str(self.config["forge_sae_model"]) if source == "forge" else str(self.config["sae_repo_id"])
        self._metadata_template = self._initialize_metadata(
            model_name=self.MODEL_DISPLAY_NAME, model_version=model_version
        )

    def _apply_local_defaults(self, user_config: dict[str, Any]) -> None:
        """Resolve model-appropriate local SAE defaults not set explicitly.

        The base :attr:`DEFAULT_CONFIG` ``sae_repo_id`` / ``sae_layer`` match the
        default Forge ESMC-6B / layer-60 SAE. For the local backend they must match
        the selected ESM-C model instead, so fill them from :data:`LOCAL_SAE_DEFAULTS`
        for ``esmc_model_name`` unless the caller provided them.

        Parameters
        ----------
        user_config : dict
            The raw, unmerged config passed to ``__init__`` — used to tell an
            explicit override apart from an inherited default.
        """
        defaults = LOCAL_SAE_DEFAULTS.get(str(self.config["esmc_model_name"]))
        if defaults is None:
            return
        for key, value in defaults.items():
            if key not in user_config:
                self.config[key] = value

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _initialize(self) -> None:
        self._load()

    def _load(self) -> None:
        """Build whichever backend ``feature_source`` selects (if not injected)."""
        if self._source == "forge":
            if self._forge is None:
                self._forge = ForgeSAEBackend(
                    model=str(self.config["forge_model"]),
                    sae_model=str(self.config["forge_sae_model"]),
                    url=str(self.config["forge_url"]),
                    token=self.config["forge_token"],
                    normalize_features=bool(self.config["normalize_features"]),
                )
        else:
            if self._embedder is None:
                from ..esm3.core import ESMCCore

                self._embedder = ESMCCore(
                    config={"device": self.config["device"], "model_name": self.config["esmc_model_name"]}
                )
                cast(Any, self._embedder)._initialize()
            if self._sae is None:
                self._sae = SparseAutoencoder.from_pretrained(
                    repo_id=str(self.config["sae_repo_id"]),
                    layer=int(self.config["sae_layer"]),
                    num_features=int(self.config["num_features"]),
                    d_model=self._infer_d_model(),
                    k=int(self.config["k"]),
                    activation=str(self.config["activation"]),  # type: ignore[arg-type]
                    device=self.config["device"],
                )
            self._sae.eval()
        self.ready = True

    def _infer_d_model(self) -> int:
        model_name = str(self.config["esmc_model_name"])
        try:
            return ESMC_D_MODEL[model_name]
        except KeyError as exc:
            raise ValueError(
                f"Unknown ESM-C model {model_name!r}; cannot infer d_model. Known: {sorted(ESMC_D_MODEL)}."
            ) from exc

    @property
    def sae(self) -> SparseAutoencoder:
        if self._sae is None:
            raise RuntimeError("Local SAE is not loaded. Call _load() first with feature_source='local'.")
        return self._sae

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    # SAE returns a ``SAEFeaturesOutput`` (always the pooled per-protein feature
    # vectors ``pooled_features``, plus optional dense per-residue activations in
    # ``features`` when ``include_per_residue=True``). That output intentionally
    # does not implement the base ``EmbeddingPrediction`` protocol -- it has no
    # ``embeddings``/``hidden_states`` fields -- so this override deliberately
    # narrows the ``EmbeddingAlgorithm.embed`` return type.
    def embed(  # type: ignore[override]
        self, sequences: str | Sequence[str], options: dict[str, Any] | None = None
    ) -> SAEFeaturesOutput:
        """Compute SAE features for one or more sequences.

        Parameters
        ----------
        sequences : str | Sequence[str]
            Amino-acid sequence(s); colon-separated chains are supported.
        options : dict | None
            Per-call overrides (non-static keys only), e.g.
            ``{"include_per_residue": True}``.

        Returns
        -------
        SAEFeaturesOutput
            Pooled per-protein features plus per-residue chain/residue indices.
        """
        effective = self._merge_options(options)
        include_per_residue = bool(effective.get("include_per_residue", False))

        if self._source == "forge":
            return self._embed_forge(sequences, include_per_residue)
        return self._embed_local(sequences, effective, include_per_residue)

    # ---- local path --------------------------------------------------
    def _embed_local(
        self, sequences: str | Sequence[str], effective: dict, include_per_residue: bool
    ) -> SAEFeaturesOutput:
        if self._embedder is None or self._sae is None:
            logger.warning("SAE core not loaded. Forcing load; call _load() first next time.")
            self._load()
        assert self._embedder is not None and self._sae is not None

        layer = int(effective["sae_layer"])
        normalize = bool(effective.get("normalize_features", False))

        with Timer("ESM-C hidden states") as embed_timer:
            emb = self._embedder.embed(sequences, options={"include_fields": ["hidden_states"]})
        hidden_states = getattr(emb, "hidden_states", None)
        if hidden_states is None:
            raise ValueError("The ESM-C embedder did not return hidden_states; cannot compute SAE features.")

        hidden = np.asarray(hidden_states)
        layer_states = self._select_layer(hidden, layer)  # (batch, residues, d_model)
        chain_index = np.asarray(emb.chain_index)
        residue_index = np.asarray(emb.residue_index)

        with Timer("SAE encode + pool") as sae_timer:
            pooled, per_residue = self._apply_sae(layer_states, chain_index, normalize, include_per_residue)

        metadata = dataclasses.replace(
            self._metadata_template,
            sequence_lengths=[int((row != -1).sum()) for row in chain_index],
            inference_time=embed_timer.duration,
            postprocessing_time=sae_timer.duration,
        )
        return SAEFeaturesOutput(
            metadata=metadata,
            pooled_features=pooled,
            chain_index=chain_index,
            residue_index=residue_index,
            layer=layer,
            num_features=int(self._sae.num_features),
            sae_model=str(self.config["sae_repo_id"]),
            features=per_residue,
        )

    @staticmethod
    def _select_layer(hidden: np.ndarray, layer: int) -> np.ndarray:
        """Return hidden states at ``layer`` with shape ``(batch, residues, d_model)``.

        ESM-C hidden states are returned with the layer axis first
        (``(layers, batch, residues, d_model)``). A 3-D array is treated as a
        single already-selected layer (``(batch, residues, d_model)``).
        """
        if hidden.ndim == 3:
            return hidden
        if hidden.ndim != 4:
            raise ValueError(
                f"Expected hidden states with 4 dims (layers, batch, residues, features); got shape {hidden.shape}."
            )
        n_layers = hidden.shape[0]
        if not -n_layers <= layer < n_layers:
            raise IndexError(f"sae_layer={layer} is out of range for {n_layers} available layers.")
        return hidden[layer]

    def _apply_sae(
        self,
        layer_states: np.ndarray,
        chain_index: np.ndarray,
        normalize: bool,
        include_per_residue: bool,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Encode residues and max-pool to per-protein vectors (local path)."""
        assert self._sae is not None
        device = next(self._sae.parameters()).device
        dtype = next(self._sae.parameters()).dtype
        batch = layer_states.shape[0]
        num_features = int(self._sae.num_features)
        pooled = np.zeros((batch, num_features), dtype=np.float32)
        per_residue = (
            np.zeros((batch, layer_states.shape[1], num_features), dtype=np.float32) if include_per_residue else None
        )
        with torch.inference_mode():
            for i in range(batch):
                valid = chain_index[i] != -1
                x = torch.as_tensor(layer_states[i], dtype=dtype, device=device)
                acts = self._sae.encode(x)  # (residues, num_features)
                if normalize:
                    acts = torch.nn.functional.normalize(acts, dim=-1)
                mask = torch.as_tensor(valid, dtype=torch.bool, device=device)
                pooled[i] = max_pool_features(acts, mask).float().cpu().numpy()
                if per_residue is not None:
                    masked = acts.clone()
                    masked[~mask] = 0.0
                    per_residue[i] = masked.float().cpu().numpy()
        return pooled, per_residue

    # ---- forge path --------------------------------------------------
    def _embed_forge(self, sequences: str | Sequence[str], include_per_residue: bool) -> SAEFeaturesOutput:
        if self._forge is None:
            logger.warning("SAE core not loaded. Forcing load; call _load() first next time.")
            self._load()
        assert self._forge is not None

        from ..esm3.core import parse_esm3_sequences

        parsed = parse_esm3_sequences(sequences)
        layer = int(self.config["sae_layer"])

        pooled_list: list[np.ndarray] = []
        residue_feature_list: list[np.ndarray] = []
        num_features = 0
        with Timer("Forge SAE features") as forge_timer:
            for item in parsed:
                dense = np.asarray(self._forge.features(item.sdk_sequence), dtype=np.float32)
                if dense.ndim != 2:
                    raise ValueError(f"Forge features must be 2-D (tokens, num_features); got shape {dense.shape}.")
                # Forge returns BOS + one row per SDK token (residues and chain-break
                # '|' tokens) + EOS. Verify that layout before indexing so a change in
                # SDK tokenization surfaces as an explicit error, not silent misalignment.
                expected_rows = len(item.sdk_sequence) + 2
                if dense.shape[0] != expected_rows:
                    raise ValueError(
                        f"Forge returned {dense.shape[0]} token rows for an SDK sequence of "
                        f"{len(item.sdk_sequence)} tokens; expected {expected_rows} (BOS + tokens + EOS)."
                    )
                # Residue token positions: BOS is index 0, chain-break '|' tokens are
                # dropped, EOS is the trailing token.
                keep = [i + 1 for i, tok in enumerate(item.sdk_sequence) if tok != "|"]
                residue_feats = dense[keep]  # (residues, num_features)
                num_features = residue_feats.shape[1]
                pooled_list.append(residue_feats.max(axis=0) if residue_feats.shape[0] else np.zeros(num_features))
                residue_feature_list.append(residue_feats)

        pooled = np.stack(pooled_list, axis=0).astype(np.float32)
        chain_index = _pad_index_arrays([item.chain_index for item in parsed])
        residue_index = _pad_index_arrays([item.residue_index for item in parsed])
        per_residue = _pad_feature_arrays(residue_feature_list) if include_per_residue else None

        metadata = dataclasses.replace(
            self._metadata_template,
            sequence_lengths=[item.residue_count for item in parsed],
            inference_time=forge_timer.duration,
        )
        return SAEFeaturesOutput(
            metadata=metadata,
            pooled_features=pooled,
            chain_index=chain_index,
            residue_index=residue_index,
            layer=layer,
            num_features=int(num_features),
            sae_model=str(self.config["forge_sae_model"]),
            features=per_residue,
        )


def _pad_index_arrays(arrays: list[np.ndarray]) -> np.ndarray:
    """Stack per-sequence 1-D index arrays into ``(batch, max_residues)`` (-1 pad)."""
    max_len = max((a.shape[0] for a in arrays), default=0)
    out = np.full((len(arrays), max_len), -1, dtype=np.int32)
    for i, a in enumerate(arrays):
        out[i, : a.shape[0]] = a
    return out


def _pad_feature_arrays(arrays: list[np.ndarray]) -> np.ndarray:
    """Stack per-sequence ``(residues, features)`` arrays into a padded batch."""
    max_len = max((a.shape[0] for a in arrays), default=0)
    num_features = arrays[0].shape[1] if arrays else 0
    out = np.zeros((len(arrays), max_len, num_features), dtype=np.float32)
    for i, a in enumerate(arrays):
        out[i, : a.shape[0]] = a
    return out


__all__ = ["SAECore", "SAEFeaturesOutput", "ESMC_D_MODEL"]
