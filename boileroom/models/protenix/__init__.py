"""Protenix model package."""


def __getattr__(name: str):
    """Lazy imports avoid importing Modal when only core.py is needed."""
    if name == "Protenix":
        from .protenix import Protenix

        return Protenix
    if name == "ModalProtenix":
        from .protenix import ModalProtenix

        return ModalProtenix
    if name == "ProtenixOutput":
        from .types import ProtenixOutput

        return ProtenixOutput
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["ModalProtenix", "Protenix", "ProtenixOutput"]
