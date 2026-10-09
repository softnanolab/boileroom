"""RoseTTAFold 3 model package."""


def __getattr__(name: str):
    """Lazy imports avoid importing Modal when only core.py is needed."""
    if name == "RF3":
        from .rf3 import RF3

        return RF3
    if name == "ModalRF3":
        from .rf3 import ModalRF3

        return ModalRF3
    if name == "RF3Output":
        from .types import RF3Output

        return RF3Output
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["ModalRF3", "RF3", "RF3Output"]
