"""Public ESM3-family exports with lazy Modal wrapper imports."""

from .types import ESM3Output, ESMCOutput, ESMEmbeddingOutput


def __getattr__(name: str):
    """Lazy import wrappers to keep types-only imports lightweight."""

    if name in {"ESM3", "ModalESM3"}:
        from .esm3 import ESM3, ModalESM3

        return {"ESM3": ESM3, "ModalESM3": ModalESM3}[name]
    if name in {"ESMC", "ModalESMC"}:
        from .esmc import ESMC, ModalESMC

        return {"ESMC": ESMC, "ModalESMC": ModalESMC}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["ESM3", "ESM3Output", "ESMC", "ESMCOutput", "ESMEmbeddingOutput", "ModalESM3", "ModalESMC"]
