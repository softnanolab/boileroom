# Lazy import to avoid importing modal when only core.py is needed
def __getattr__(name: str):
    if name == "ProFam":
        from .profam import ProFam

        return ProFam
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["ProFam"]
