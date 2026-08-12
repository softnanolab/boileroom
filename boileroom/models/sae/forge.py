"""Forge (Biohub API) backend for SAE features.

An alternative to running ESM-C + a local sparse autoencoder: call Biohub's
hosted ``ESMCForgeInferenceClient`` with a ``SAEConfig`` and read back the SAE
feature activations directly. This is the only way to use the ESMC-6B SAEs, which
are too large to run locally.

The heavy client is built lazily and the ``esm`` SDK is imported only inside
methods, so this module stays importable (and the SAE core stays unit-testable)
without the SDK or a Biohub token. Tests inject a fake backend exposing the same
``features`` method.
"""

from __future__ import annotations

import os

import numpy as np


class ForgeSAEBackend:
    """Fetch per-token SAE feature activations from the Biohub Forge API.

    Parameters
    ----------
    model : str
        Forge ESM-C model id, e.g. ``"esmc-6b-2024-12"``.
    sae_model : str
        Forge SAE model id, e.g. ``"esmc-6b-2024-12-sae-layer60-k64-codebook16384"``.
    url : str
        Forge base URL (default ``"https://biohub.ai"``).
    token : str | None
        API token. If ``None``, falls back to the ``ESM_API_KEY`` environment
        variable at first use.
    normalize_features : bool
        Whether to request TF-IDF normalized features from the API.
    """

    def __init__(
        self,
        model: str,
        sae_model: str,
        url: str = "https://biohub.ai",
        token: str | None = None,
        normalize_features: bool = True,
    ) -> None:
        self.model = model
        self.sae_model = sae_model
        self.url = url
        self.token = token
        self.normalize_features = normalize_features
        self._client: object | None = None

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        token = self.token or os.environ.get("ESM_API_KEY")
        if not token:
            raise ValueError(
                "Forge SAE backend requires an API token. Set config 'forge_token' or the ESM_API_KEY env var."
            )
        from esm.sdk.forge import ESMCForgeInferenceClient

        self._client = ESMCForgeInferenceClient(model=self.model, url=self.url, token=token)

    def features(self, sdk_sequence: str) -> np.ndarray:
        """Return dense per-token SAE activations of shape ``(tokens, num_features)``.

        The returned array still includes the BOS / EOS (and any chain-break)
        tokens; the caller selects residue rows.
        """
        self._ensure_client()
        from esm.sdk.api import ESMProtein, LogitsConfig, SAEConfig

        assert self._client is not None
        protein_tensor = self._client.encode(ESMProtein(sequence=sdk_sequence))  # type: ignore[attr-defined]
        output = self._client.logits(  # type: ignore[attr-defined]
            protein_tensor,
            config=LogitsConfig(
                sae_config=SAEConfig(models=[self.sae_model], normalize_features=self.normalize_features)
            ),
            return_bytes=False,
        )
        if getattr(output, "sae_outputs", None) is None:
            raise ValueError(f"Forge returned no sae_outputs for SAE model {self.sae_model!r}.")
        if self.sae_model not in output.sae_outputs:
            raise ValueError(
                f"Forge returned no activations for SAE model {self.sae_model!r}; "
                f"available keys: {sorted(output.sae_outputs)}."
            )
        sae_tensor = output.sae_outputs[self.sae_model]
        dense = sae_tensor.to_dense() if hasattr(sae_tensor, "to_dense") else sae_tensor
        array = dense.cpu().numpy() if hasattr(dense, "cpu") else np.asarray(dense)
        return np.asarray(array, dtype=np.float32)


__all__ = ["ForgeSAEBackend"]
