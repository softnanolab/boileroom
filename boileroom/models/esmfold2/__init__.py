"""Lazy imports for the ESMFold2 model family."""

from .types import (
    CovalentBond,
    DistogramConditioning,
    DNAInput,
    ESMFold2Output,
    LigandInput,
    Modification,
    MSAInput,
    PocketConditioning,
    ProteinInput,
    RNAInput,
    StructurePredictionInput,
)


def __getattr__(name: str):
    """Lazy import for the public wrapper to avoid importing modal eagerly."""
    if name == "ESMFold2":
        from .esmfold2 import ESMFold2

        return ESMFold2
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CovalentBond",
    "DNAInput",
    "DistogramConditioning",
    "ESMFold2",
    "ESMFold2Output",
    "LigandInput",
    "MSAInput",
    "Modification",
    "PocketConditioning",
    "ProteinInput",
    "RNAInput",
    "StructurePredictionInput",
]
