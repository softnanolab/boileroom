# Lazy imports to avoid importing modal when not needed (e.g., in conda backend)
def __getattr__(name: str):
    """Lazy import for model classes to avoid importing modal dependencies."""
    if name == "ESMFold":
        from .esm.esmfold import ESMFold

        return ESMFold
    if name == "ESM2":
        from .esm.esm2 import ESM2

        return ESM2
    if name == "Chai1":
        from .chai.chai1 import Chai1

        return Chai1
    if name == "Boltz2":
        from .boltz.boltz2 import Boltz2

        return Boltz2
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ESMFold",
    "ESM2",
    "Chai1",
    "Boltz2",
]
