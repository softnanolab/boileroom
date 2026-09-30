"""OpenDDE model package."""


def __getattr__(name: str):
    """Lazy imports avoid importing Modal when only core.py is needed."""
    if name == "OpenDDE":
        from .opendde import OpenDDE

        return OpenDDE
    if name == "ModalOpenDDE":
        from .opendde import ModalOpenDDE

        return ModalOpenDDE
    if name == "OpenDDEOutput":
        from .types import OpenDDEOutput

        return OpenDDEOutput
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["ModalOpenDDE", "OpenDDE", "OpenDDEOutput"]
