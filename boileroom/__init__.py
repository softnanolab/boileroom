# Lazy imports to avoid importing modal when not needed (e.g., in image-backed runtimes)
def __getattr__(name: str):
    """Lazy import for model classes to avoid importing modal dependencies."""
    if name == "ESMFold":
        from .models import ESMFold

        return ESMFold
    if name == "ESM2":
        from .models import ESM2

        return ESM2
    if name == "ESMC":
        from .models import ESMC

        return ESMC
    if name == "ESM3":
        from .models import ESM3

        return ESM3
    if name == "ESMFold2":
        from .models import ESMFold2

        return ESMFold2
    if name == "Chai1":
        from .models import Chai1

        return Chai1
    if name == "Boltz2":
        from .models import Boltz2

        return Boltz2
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["ESMFold", "ESM2", "ESMFold2", "ESMC", "ESM3", "Chai1", "Boltz2"]
