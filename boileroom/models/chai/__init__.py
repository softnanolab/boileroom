# Lazy import to avoid importing modal when only core.py is needed
def __getattr__(name: str):
    if name == "Chai1":
        from .chai1 import Chai1
        return Chai1
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["Chai1"]
