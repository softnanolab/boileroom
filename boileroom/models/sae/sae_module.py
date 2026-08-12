"""Sparse autoencoder (SAE) module for ESM-C representations.

This implements the sparse-autoencoder used to decompose ESM-C residue
representations into a high-dimensional, sparse, and interpretable feature
space, as described in *Language Modeling Materializes a World Model of Protein
Biology* (Biohub, 2026). A separate SAE is trained per transformer layer; the
featured configuration in the paper uses ``k = 64`` active features per residue
and a codebook (feature) size of ``2**14 = 16384``.

The module is intentionally free of Modal / ``esm`` SDK dependencies so it can be
imported and unit-tested with only ``numpy`` and ``torch``. The heavier
``SAECore`` (which runs ESM-C to obtain hidden states) lives in ``core.py``.

Two encoder activations are supported:

``"topk"``
    Keep the ``k`` largest positive pre-activations per residue and zero the
    rest. This is the activation used for the released Biohub SAEs.
``"relu"``
    A plain ``ReLU`` sparse autoencoder (no hard sparsity budget), provided for
    experimentation and comparison.

The forward pass follows the standard tied-bias formulation::

    centered  = x - b_pre
    pre_acts  = centered @ W_enc + b_enc
    acts      = activation(pre_acts)          # TopK or ReLU
    recon     = acts @ W_dec + b_pre
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor, nn

Activation = Literal["topk", "relu"]

# Candidate parameter-name aliases seen across public SAE checkpoints. ``load_state_dict``
# maps whichever alias is present onto this module's canonical names.
_KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "W_enc": ("W_enc", "encoder.weight", "encoder.W", "w_enc"),
    "b_enc": ("b_enc", "encoder.bias", "b_encoder", "latent_bias"),
    "W_dec": ("W_dec", "decoder.weight", "decoder.W", "w_dec"),
    "b_pre": ("b_pre", "b_dec", "pre_bias", "decoder.bias"),
}


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
    normalize_decoder : bool
        If ``True``, decoder rows are unit-normalized on load, matching the
        common SAE training convention where dictionary atoms have unit norm.
    """

    d_model: int
    num_features: int
    k: int = 64
    activation: Activation = "topk"
    normalize_decoder: bool = False

    def __post_init__(self) -> None:
        if self.d_model <= 0 or self.num_features <= 0:
            raise ValueError("d_model and num_features must be positive.")
        if self.activation not in ("topk", "relu"):
            raise ValueError(f"Unsupported activation: {self.activation!r}. Use 'topk' or 'relu'.")
        if self.activation == "topk" and not (0 < self.k <= self.num_features):
            raise ValueError(f"k must satisfy 0 < k <= num_features; got k={self.k}, num_features={self.num_features}.")


def topk_activation(pre_acts: Tensor, k: int) -> Tensor:
    """Keep the ``k`` largest positive pre-activations per row, zero the rest.

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
    if k >= pre_acts.shape[-1]:
        return torch.relu(pre_acts)
    values, indices = torch.topk(pre_acts, k=k, dim=-1)
    values = torch.relu(values)
    out = torch.zeros_like(pre_acts)
    out.scatter_(-1, indices, values)
    return out


class SparseAutoencoder(nn.Module):
    """Tied-bias sparse autoencoder over ESM-C residue representations.

    Parameters
    ----------
    config : SAEModuleConfig
        Architecture description.

    Notes
    -----
    Weights are stored as ``W_enc`` of shape ``(d_model, num_features)`` and
    ``W_dec`` of shape ``(num_features, d_model)`` so that ``encode`` and
    ``decode`` are plain matrix multiplies without transposes at call time.
    """

    def __init__(self, config: SAEModuleConfig) -> None:
        super().__init__()
        self.config = config
        d, f = config.d_model, config.num_features
        self.W_enc = nn.Parameter(torch.zeros(d, f))
        self.b_enc = nn.Parameter(torch.zeros(f))
        self.W_dec = nn.Parameter(torch.zeros(f, d))
        self.b_pre = nn.Parameter(torch.zeros(d))
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
            if self.config.normalize_decoder:
                self._unit_normalize_decoder()
        nn.init.zeros_(self.b_enc)
        nn.init.zeros_(self.b_pre)

    def _unit_normalize_decoder(self) -> None:
        norms = self.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8)
        self.W_dec.div_(norms)

    def pre_activations(self, x: Tensor) -> Tensor:
        """Return encoder pre-activations ``(x - b_pre) @ W_enc + b_enc``."""
        return (x - self.b_pre) @ self.W_enc + self.b_enc

    def encode(self, x: Tensor) -> Tensor:
        """Map representations to sparse feature activations.

        Parameters
        ----------
        x : Tensor
            Residue representations of shape ``(..., d_model)``.

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
        """Reconstruct representations from feature activations."""
        return acts @ self.W_dec + self.b_pre

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(feature_activations, reconstruction)`` for ``x``."""
        acts = self.encode(x)
        return acts, self.decode(acts)

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
        module = cls(config)
        state = torch.load(directory / "sae.pt", map_location=device or "cpu")
        module.load_state_dict(state)
        if device is not None:
            module = module.to(device)
        return module.eval()

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, Tensor],
        config: SAEModuleConfig,
        strict: bool = False,
    ) -> SparseAutoencoder:
        """Build a module from a possibly foreign ``state_dict``.

        Parameter names are remapped through :data:`_KEY_ALIASES`, so checkpoints
        that call the encoder weight ``encoder.weight`` (etc.) still load. Encoder
        / decoder matrices are transposed automatically when their orientation is
        the transpose of this module's ``(d_model, num_features)`` convention.

        Parameters
        ----------
        state_dict : dict[str, Tensor]
            Source parameters.
        config : SAEModuleConfig
            Target architecture. Must match the checkpoint's dimensions.
        strict : bool
            If ``True``, raise when a canonical parameter cannot be resolved.

        Returns
        -------
        SparseAutoencoder
            A module in eval mode with weights loaded.
        """
        module = cls(config)
        remapped: dict[str, Tensor] = {}
        lowered = {key.lower(): value for key, value in state_dict.items()}
        for canonical, aliases in _KEY_ALIASES.items():
            tensor = None
            for alias in aliases:
                if alias in state_dict:
                    tensor = state_dict[alias]
                    break
                if alias.lower() in lowered:
                    tensor = lowered[alias.lower()]
                    break
            if tensor is None:
                if strict:
                    raise KeyError(
                        f"Could not resolve parameter {canonical!r} from state dict keys {sorted(state_dict)}."
                    )
                continue
            remapped[canonical] = cls._orient(canonical, torch.as_tensor(tensor), config)
        missing = module.load_state_dict(remapped, strict=False)
        if strict and missing.missing_keys:
            raise KeyError(f"Missing parameters after remap: {missing.missing_keys}.")
        return module.eval()

    @staticmethod
    def _orient(name: str, tensor: Tensor, config: SAEModuleConfig) -> Tensor:
        """Transpose 2-D weights when they arrive in the opposite orientation."""
        if name == "W_enc":
            target = (config.d_model, config.num_features)
        elif name == "W_dec":
            target = (config.num_features, config.d_model)
        else:
            return tensor
        if tuple(tensor.shape) == target:
            return tensor
        if tuple(tensor.shape) == (target[1], target[0]):
            return tensor.t().contiguous()
        raise ValueError(f"{name} has shape {tuple(tensor.shape)}, incompatible with target {target}.")

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        *,
        layer: int,
        num_features: int,
        d_model: int,
        k: int = 64,
        activation: Activation = "topk",
        filename: str | None = None,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        device: str | torch.device | None = None,
        normalize_decoder: bool = False,
    ) -> SparseAutoencoder:
        """Download and load a released per-layer SAE from the Hugging Face Hub.

        The Biohub SAE collection stores one ``safetensors`` file per transformer
        layer (e.g. ``layer_60.safetensors``) inside a repo such as
        ``biohub/ESMC-6B-sae-k64-codebook16384``.

        Parameters
        ----------
        repo_id : str
            Hugging Face repo id hosting the per-layer SAE weights.
        layer : int
            Transformer layer index; selects ``layer_{layer}.safetensors`` unless
            ``filename`` is given.
        num_features, d_model, k, activation, normalize_decoder
            Architecture description; see :class:`SAEModuleConfig`.
        filename : str | None
            Explicit weight filename, overriding the ``layer_{layer}`` default.
        revision, cache_dir, device
            Passed through to the Hub download / tensor placement.

        Returns
        -------
        SparseAutoencoder
            A module in eval mode with the downloaded weights loaded.
        """
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        weight_file = filename or f"layer_{layer}.safetensors"
        local_path = hf_hub_download(
            repo_id=repo_id, filename=weight_file, revision=revision, cache_dir=str(cache_dir) if cache_dir else None
        )
        state_dict = load_file(local_path, device="cpu")
        config = SAEModuleConfig(
            d_model=d_model,
            num_features=num_features,
            k=k,
            activation=activation,
            normalize_decoder=normalize_decoder,
        )
        module = cls.from_state_dict(state_dict, config)
        if device is not None:
            module = module.to(device)
        return module.eval()


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
    "Activation",
    "SAEModuleConfig",
    "SparseAutoencoder",
    "max_pool_features",
    "topk_activation",
]
