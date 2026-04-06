"""Lightweight fake model cores used for container smoke tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class HealthcheckOutput:
    """Minimal return payload for smoke-test endpoints."""

    status: str
    options: dict[str, Any] | None = None


class HealthcheckCore:
    """Fake core that exercises server startup without heavy ML dependencies."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Store configuration and mark the core as not yet initialized."""
        self.config = dict(config or {})
        self.ready = False

    def _initialize(self) -> None:
        """Mark the fake core as initialized."""
        self.ready = True

    def embed(self, sequences: str | list[str], options: dict[str, Any] | None = None) -> HealthcheckOutput:
        """Return a lightweight embedding placeholder."""
        return HealthcheckOutput(status="embed-ok", options=options)

    def fold(self, sequences: str | list[str], options: dict[str, Any] | None = None) -> HealthcheckOutput:
        """Return a lightweight folding placeholder."""
        return HealthcheckOutput(status="fold-ok", options=options)
