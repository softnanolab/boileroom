# Lazy imports to avoid importing modal when not needed (e.g., in conda backend)
def __getattr__(name: str):
    """Lazy import for ESMFold and ESM2 to avoid importing modal dependencies."""
    if name == "ESMFold":
        from .esmfold import ESMFold

        return ESMFold
    if name == "ESM2":
        from .esm2 import ESM2

        return ESM2
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["ESMFold", "ESM2"]
