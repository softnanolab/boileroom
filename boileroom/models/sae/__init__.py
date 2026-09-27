"""ESM-C sparse-autoencoder feature model.

Public entry point is :class:`SAE`. The sparse-autoencoder math lives in
``sae_module`` and is free of Modal / ``esm`` dependencies so it can be imported
and tested in isolation.
"""

from typing import Any

from .sae_module import SAEModuleConfig, SparseAutoencoder, max_pool_features, topk_activation
from .types import SAEFeaturesOutput

__all__ = [
    "SAE",
    "SAEFeaturesOutput",
    "SAEModuleConfig",
    "SparseAutoencoder",
    "max_pool_features",
    "topk_activation",
]


def __getattr__(name: str) -> Any:
    # Lazy import so the public wrapper (which imports modal) is only pulled in on
    # demand, mirroring boileroom/__init__.py.
    if name == "SAE":
        from .sae import SAE

        return SAE
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
