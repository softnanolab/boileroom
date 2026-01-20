# Lazy import to avoid importing modal when only core.py is needed
def __getattr__(name: str):
    if name == "Boltz2":
        from .boltz2 import Boltz2
        return Boltz2
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["Boltz2"]
