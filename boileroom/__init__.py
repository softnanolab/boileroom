# Lazy imports to avoid importing modal when not needed (e.g., in conda backend)
def __getattr__(name: str):
    """Lazy import for model classes to avoid importing modal dependencies."""
    if name == "ESMFold":
        from .models import ESMFold

        return ESMFold
    if name == "ESM2":
        from .models import ESM2

        return ESM2
    if name == "Chai1":
        from .models import Chai1

        return Chai1
    if name == "Boltz2":
        from .models import Boltz2

        return Boltz2
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["ESMFold", "ESM2", "Chai1", "Boltz2"]
