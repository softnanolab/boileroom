"""Public ESM-C exports with lazy Modal wrapper imports."""

from typing import Any

from .types import ESMCOutput, ESMEmbeddingOutput


def __getattr__(name: str) -> Any:
    """Lazy import wrappers to keep types-only imports lightweight."""

    if name in {"ESMC", "ModalESMC"}:
        from .esmc import ESMC, ModalESMC

        return {"ESMC": ESMC, "ModalESMC": ModalESMC}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["ESMC", "ESMCOutput", "ESMEmbeddingOutput", "ModalESMC"]
