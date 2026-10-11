"""Public ProFam exports with a lazy Modal wrapper import."""

from typing import Any

from .types import ConditioningPrompt, ProFamOutput


def __getattr__(name: str) -> Any:
    if name in {"ProFam", "ModalProFam"}:
        from .profam import ModalProFam, ProFam

        return {"ProFam": ProFam, "ModalProFam": ModalProFam}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["ProFam", "ModalProFam", "ProFamOutput", "ConditioningPrompt"]
