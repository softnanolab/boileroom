"""Sparse autoencoder (SAE) module for ESM-C representations.

This implements the sparse autoencoder used to decompose ESM-C residue
representations into a high-dimensional, sparse, and interpretable feature
space, as described in *Language Modeling Materializes a World Model of Protein
Biology* (Biohub, 2026). A separate SAE is trained per transformer layer; the
featured configuration in the paper uses ``k = 64`` active features per residue
and a codebook (feature) size of ``2**14 = 16384``.

The forward pass reproduces Biohub's reference implementation,
``esm.models.esmc.sae.EsmcSaeLayer`` (``esm==3.4.1.post1``), which the released
``biohub/ESMC-*-sae-*`` checkpoints are trained for::

    z         = (x - mean(x)) / (std(x) + 1e-5)    # per-residue z-score
    pre_acts  = (z - b_dec) @ W_enc                # no encoder bias
    acts      = scatter(topk(relu(pre_acts), k))   # TopK activation
    recon     = acts @ W_dec + b_dec               # reconstructs z, not x

A checkpoint holds exactly ``W_enc``, ``W_dec``, ``b_dec`` and two per-feature
normalization statistics, ``idf`` and ``max``. Normalized features are
``acts / max * idf`` (Biohub's ``normalize_sae`` / Forge ``normalize_features``);
the statistics are the maximum activation and ``log(N / f)`` inverse document
frequency over UniRef90. The released all-layer checkpoints ship placeholder
statistics (all ones), so :meth:`SparseAutoencoder.normalize_features` refuses to
run on them rather than return unnormalized features under that name.

The module is intentionally free of Modal / ``esm`` SDK dependencies so it can be
imported and unit-tested with only ``numpy`` and ``torch``. The heavier
``SAECore`` (which runs ESM-C to obtain hidden states) lives in ``core.py``.

Two encoder activations are supported:

``"topk"``
    Keep the ``k`` largest ReLU-ed pre-activations per residue and zero the
    rest. This is the activation the released Biohub SAEs are trained with.
``"relu"``
    A plain ``ReLU`` (no hard sparsity budget), for experimentation only: on the
    released checkpoints it does not reproduce the reference features.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor, nn

Activation = Literal["topk", "relu"]

#: Epsilon of the reference per-residue z-score (``EsmcSaeLayer._zscore_normalize_representation``).
ZSCORE_EPS = 1e-5
#: Tensors of a released Biohub SAE layer checkpoint; loading requires exactly these.
CHECKPOINT_KEYS: tuple[str, ...] = ("W_enc", "W_dec", "b_dec", "idf", "max")


@dataclass(frozen=True)
class SAEModuleConfig:
    """Static architecture description for a :class:`SparseAutoencoder`.

    Parameters
    ----------
    d_model : int
        Dimensionality of the ESM-C residue representation the SAE consumes
        (e.g. 960 for ESM-C 300M, 1152 for ESM-C 600M, 2560 for ESM-C 6B).
    num_features : int
        Size of the sparse feature (codebook) space, e.g. ``16384``.
    k : int
        Number of active features per residue for the ``"topk"`` activation.
        Ignored when ``activation == "relu"``.
    activation : {"topk", "relu"}
        Encoder sparsity mechanism.
    """

    d_model: int
    num_features: int
    k: int = 64
    activation: Activation = "topk"

    def __post_init__(self) -> None:
        if self.d_model <= 0 or self.num_features <= 0:
            raise ValueError("d_model and num_features must be positive.")
        if self.activation not in ("topk", "relu"):
            raise ValueError(f"Unsupported activation: {self.activation!r}. Use 'topk' or 'relu'.")
        if self.activation == "topk" and not (0 < self.k <= self.num_features):
            raise ValueError(f"k must satisfy 0 < k <= num_features; got k={self.k}, num_features={self.num_features}.")


def topk_activation(pre_acts: Tensor, k: int) -> Tensor:
    """Keep the ``k`` largest ReLU-ed pre-activations per row, zero the rest.

    Matches the reference ``relu`` -> ``topk`` -> ``scatter`` order of
    ``EsmcSaeLayer.forward``.

    Parameters
    ----------
    pre_acts : Tensor
        Pre-activation tensor of shape ``(..., num_features)``.
    k : int
        Number of features to keep per row.

    Returns
    -------
    Tensor
        Sparse activations with the same shape as ``pre_acts``; at most ``k``
        entries per row are non-zero and all are non-negative.
    """
    acts = torch.relu(pre_acts)
    if k >= acts.shape[-1]:
        return acts
    topk = torch.topk(acts, k=k, dim=-1)
    return torch.zeros_like(acts).scatter(-1, topk.indices, topk.values)


class SparseAutoencoder(nn.Module):
    """Biohub ESM-C sparse autoencoder over residue representations.

    Parameters
    ----------
    config : SAEModuleConfig
        Architecture description.

    Notes
    -----
    Parameter and buffer names match the released checkpoints, so
    ``state_dict()`` round-trips them unchanged: ``W_enc`` of shape
    ``(d_model, num_features)``, ``W_dec`` of shape ``(num_features, d_model)``,
    ``b_dec`` of shape ``(d_model,)``, and the ``idf`` / ``max`` buffers of shape
    ``(num_features,)``, which default to ones (no normalization statistics).
    """

    idf: Tensor
    max: Tensor

    def __init__(self, config: SAEModuleConfig) -> None:
        super().__init__()
        self.config = config
        d, f = config.d_model, config.num_features
        self.W_enc = nn.Parameter(torch.zeros(d, f))
        self.W_dec = nn.Parameter(torch.zeros(f, d))
        self.b_dec = nn.Parameter(torch.zeros(d))
        self.register_buffer("idf", torch.ones(f))
        self.register_buffer("max", torch.ones(f))
        self.reset_parameters()

    @property
    def d_model(self) -> int:
        return self.config.d_model

    @property
    def num_features(self) -> int:
        return self.config.num_features

    def reset_parameters(self) -> None:
        """Initialize weights so an untrained module is still well-conditioned."""
        nn.init.kaiming_uniform_(self.W_enc, a=5**0.5)
        with torch.no_grad():
            self.W_dec.copy_(self.W_enc.t())
            self.idf.fill_(1.0)
            self.max.fill_(1.0)
        nn.init.zeros_(self.b_dec)

    @staticmethod
    def standardize(x: Tensor) -> Tensor:
        """Z-score each representation over its feature axis, as the reference does before encoding."""
        x = x - x.mean(dim=-1, keepdim=True)
        return x / (x.std(dim=-1, keepdim=True) + ZSCORE_EPS)

    def pre_activations(self, x: Tensor) -> Tensor:
        """Return encoder pre-activations ``(standardize(x) - b_dec) @ W_enc``."""
        return (self.standardize(x) - self.b_dec) @ self.W_enc

    def encode(self, x: Tensor) -> Tensor:
        """Map representations to sparse feature activations.

        Parameters
        ----------
        x : Tensor
            Residue representations of shape ``(..., d_model)``, as produced by
            ESM-C (the module standardizes them itself).

        Returns
        -------
        Tensor
            Sparse, non-negative feature activations of shape
            ``(..., num_features)``.
        """
        pre = self.pre_activations(x)
        if self.config.activation == "topk":
            return topk_activation(pre, self.config.k)
        return torch.relu(pre)

    def decode(self, acts: Tensor) -> Tensor:
        """Reconstruct the *standardized* representation from feature activations."""
        return acts @ self.W_dec + self.b_dec

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(feature_activations, reconstruction_of_standardize(x))`` for ``x``."""
        acts = self.encode(x)
        return acts, self.decode(acts)

    # ------------------------------------------------------------------
    # Feature normalization
    # ------------------------------------------------------------------
    @property
    def has_normalization_stats(self) -> bool:
        """Whether ``idf`` / ``max`` hold real statistics rather than all-ones placeholders."""
        return not (bool(torch.all(self.idf == 1)) and bool(torch.all(self.max == 1)))

    def check_normalization_stats(self) -> None:
        """Raise unless ``idf`` / ``max`` can normalize features like the reference does.

        Raises
        ------
        ValueError
            If the statistics are the all-ones placeholders the released all-layer
            checkpoints ship (normalizing would silently return raw activations),
            or if ``max`` holds non-positive or non-finite values (the reference
            ``acts / max`` would produce ``inf`` / ``nan``).
        """
        if not self.has_normalization_stats:
            raise ValueError(
                "This SAE checkpoint ships placeholder normalization statistics (idf and max are all 1.0), "
                "so normalized features ((activation / max) * idf over UniRef90) are not available for it. "
                "Use raw activations (normalize_features=False)."
            )
        if not (bool(torch.isfinite(self.max).all()) and bool((self.max > 0).all())):
            raise ValueError("SAE normalization statistic 'max' must be finite and positive for every feature.")
        if not bool(torch.isfinite(self.idf).all()):
            raise ValueError("SAE normalization statistic 'idf' must be finite for every feature.")

    def normalize_features(self, acts: Tensor) -> Tensor:
        """Apply the reference feature normalization ``acts / max * idf``.

        Parameters
        ----------
        acts : Tensor
            Feature activations of shape ``(..., num_features)`` from :meth:`encode`.

        Returns
        -------
        Tensor
            Normalized activations with the same shape.

        Raises
        ------
        ValueError
            If the checkpoint has no usable statistics; see :meth:`check_normalization_stats`.
        """
        self.check_normalization_stats()
        return acts / self.max * self.idf

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, directory: str | Path) -> Path:
        """Save weights (``sae.pt``) and ``config.json`` to ``directory``.

        Returns
        -------
        Path
            The directory the checkpoint was written to.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), directory / "sae.pt")
        (directory / "config.json").write_text(json.dumps(asdict(self.config)))
        return directory

    @classmethod
    def load(cls, directory: str | Path, device: str | torch.device | None = None) -> SparseAutoencoder:
        """Load a checkpoint previously written by :meth:`save`."""
        directory = Path(directory)
        config = SAEModuleConfig(**json.loads((directory / "config.json").read_text()))
        state = torch.load(directory / "sae.pt", map_location=device or "cpu")
        return cls.from_state_dict(state, config, device=device)

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, Tensor],
        config: SAEModuleConfig,
        device: str | torch.device | None = None,
    ) -> SparseAutoencoder:
        """Build a module from a released Biohub SAE layer ``state_dict``.

        The checkpoint must hold exactly :data:`CHECKPOINT_KEYS` in this module's
        orientation. Nothing is renamed, transposed, defaulted or dropped: a
        missing tensor, an extra tensor (e.g. an encoder bias ``b_enc``, which the
        reference architecture does not have) or a wrong shape raises.

        Parameters
        ----------
        state_dict : dict[str, Tensor]
            Source tensors, e.g. ``safetensors.torch.load_file("layer_27.safetensors")``.
        config : SAEModuleConfig
            Target architecture. Must match the checkpoint's dimensions.
        device : str | torch.device | None
            Device to move the loaded module to.

        Returns
        -------
        SparseAutoencoder
            A module in eval mode with weights loaded.

        Raises
        ------
        KeyError
            If tensors are missing or unexpected.
        ValueError
            If a tensor's shape does not match ``config``.
        """
        keys = set(state_dict)
        missing = [key for key in CHECKPOINT_KEYS if key not in keys]
        unexpected = sorted(keys - set(CHECKPOINT_KEYS))
        if missing or unexpected:
            raise KeyError(
                f"SAE checkpoint does not match the Biohub EsmcSaeLayer layout {list(CHECKPOINT_KEYS)}: "
                f"missing {missing}, unexpected {unexpected} (checkpoint keys: {sorted(keys)})."
            )
        d, f = config.d_model, config.num_features
        expected_shapes = {"W_enc": (d, f), "W_dec": (f, d), "b_dec": (d,), "idf": (f,), "max": (f,)}
        for key, shape in expected_shapes.items():
            actual = tuple(state_dict[key].shape)
            if actual != shape:
                raise ValueError(
                    f"SAE checkpoint tensor {key!r} has shape {actual}; expected {shape} for "
                    f"d_model={d}, num_features={f}."
                )
        module = cls(config)
        module.load_state_dict(state_dict, strict=True)
        if device is not None:
            module = module.to(device)
        return module.eval()

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        *,
        layer: int,
        d_model: int | None = None,
        num_features: int | None = None,
        k: int | None = None,
        activation: Activation = "topk",
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        device: str | torch.device | None = None,
    ) -> SparseAutoencoder:
        """Download and load a released per-layer SAE from the Hugging Face Hub.

        A Biohub SAE repo such as ``biohub/ESMC-600M-sae-k64-codebook16384``
        holds a ``config.json`` (``d_model``, ``codebook_dim``, ``k``,
        ``available_layers``, ``use_residual_update_instead_of_states``) and one
        ``layer_{layer}.safetensors`` per backbone layer. The architecture is read
        from ``config.json``; any value the caller passes must agree with it.

        Parameters
        ----------
        repo_id : str
            Hugging Face repo id hosting the per-layer SAE weights.
        layer : int
            Backbone layer index; selects ``layer_{layer}.safetensors``. Biohub
            numbers layers by the native ESM-C hidden-state stack, where layer
            ``N`` is the input of transformer block ``N``.
        d_model, num_features, k : int | None
            Expected architecture; ``None`` takes the repo's value.
        activation : {"topk", "relu"}
            Encoder activation; see :class:`SAEModuleConfig`.
        revision, cache_dir, device
            Passed through to the Hub download / tensor placement.

        Returns
        -------
        SparseAutoencoder
            A module in eval mode with the downloaded weights loaded.

        Raises
        ------
        ValueError
            If the repo does not ship ``layer``, was trained on residual updates
            rather than hidden states, or disagrees with a requested dimension.
        """
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        def _download(filename: str) -> str:
            return hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                cache_dir=str(cache_dir) if cache_dir else None,
            )

        repo_config: dict[str, Any] = json.loads(Path(_download("config.json")).read_text())
        available = repo_config.get("available_layers")
        if available is not None and layer not in available:
            raise ValueError(f"{repo_id} has no SAE for layer {layer}; available layers: {sorted(available)}.")
        if repo_config.get("use_residual_update_instead_of_states", False):
            raise ValueError(
                f"{repo_id} was trained on residual updates (h[N] - h[N-1]), not hidden states; "
                "boileroom only feeds hidden states to the SAE."
            )
        resolved = {
            "d_model": repo_config["d_model"],
            "num_features": repo_config["codebook_dim"],
            "k": repo_config["k"],
        }
        requested = {"d_model": d_model, "num_features": num_features, "k": k}
        mismatched = {
            name: (value, resolved[name])
            for name, value in requested.items()
            if value is not None and int(value) != int(resolved[name])
        }
        if mismatched:
            details = ", ".join(f"{name}={want} (repo has {have})" for name, (want, have) in mismatched.items())
            raise ValueError(f"{repo_id} config.json disagrees with the requested SAE architecture: {details}.")
        config = SAEModuleConfig(
            d_model=int(resolved["d_model"]),
            num_features=int(resolved["num_features"]),
            k=int(resolved["k"]),
            activation=activation,
        )
        state_dict = load_file(_download(f"layer_{layer}.safetensors"), device="cpu")
        return cls.from_state_dict(state_dict, config, device=device)


def max_pool_features(feature_acts: Tensor, valid_mask: Tensor | None = None) -> Tensor:
    """Aggregate per-residue feature activations into one per-protein vector.

    Following the paper, a protein is represented by the maximum activation of
    each feature across its residues.

    Parameters
    ----------
    feature_acts : Tensor
        Per-residue activations of shape ``(residues, num_features)``.
    valid_mask : Tensor | None
        Optional boolean mask of shape ``(residues,)``; ``False`` rows (padding)
        are excluded from the max. If every row is masked out, a zero vector is
        returned.

    Returns
    -------
    Tensor
        Per-protein feature vector of shape ``(num_features,)``.
    """
    if feature_acts.ndim != 2:
        raise ValueError(f"feature_acts must be 2-D (residues, features); got shape {tuple(feature_acts.shape)}.")
    if valid_mask is not None:
        valid_mask = valid_mask.to(dtype=torch.bool)
        if valid_mask.shape[0] != feature_acts.shape[0]:
            raise ValueError("valid_mask length must match the residue dimension.")
        feature_acts = feature_acts[valid_mask]
    if feature_acts.shape[0] == 0:
        return torch.zeros(feature_acts.shape[-1], dtype=feature_acts.dtype, device=feature_acts.device)
    return feature_acts.max(dim=0).values


__all__ = [
    "CHECKPOINT_KEYS",
    "Activation",
    "SAEModuleConfig",
    "SparseAutoencoder",
    "max_pool_features",
    "topk_activation",
]
