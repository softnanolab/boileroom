"""AlphaFold model package."""


def __getattr__(name: str):
    """Lazy imports avoid importing Modal when only core.py is needed."""
    if name == "AlphaFold2Multimer":
        from .alphafold2_multimer import AlphaFold2Multimer

        return AlphaFold2Multimer
    if name == "ModalAlphaFold2Multimer":
        from .alphafold2_multimer import ModalAlphaFold2Multimer

        return ModalAlphaFold2Multimer
    if name == "AlphaFold2MultimerOutput":
        from .types import AlphaFold2MultimerOutput

        return AlphaFold2MultimerOutput
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["AlphaFold2Multimer", "AlphaFold2MultimerOutput", "ModalAlphaFold2Multimer"]
