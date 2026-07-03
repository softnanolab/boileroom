"""Public and Modal wrappers for Protenix."""

import json
import logging
from collections.abc import Sequence

import modal

from ...backend.modal import get_modal_app
from ...base import ModelWrapper
from ...images.volumes import model_weights
from ...utils import MINUTES, MODAL_MODEL_DIR
from ..registry import PROTENIX_SPEC
from .image import protenix_image
from .types import ProtenixOutput

logger = logging.getLogger(__name__)
app = get_modal_app("protenix")


@app.cls(
    image=protenix_image,
    gpu="A100-40GB",
    timeout=60 * MINUTES,
    scaledown_window=10 * MINUTES,
    volumes={MODAL_MODEL_DIR: model_weights},
)
class ModalProtenix:
    """Modal entrypoint for Protenix."""

    config: bytes = modal.parameter(default=b"{}")

    @modal.enter()
    def _initialize(self) -> None:
        from .core import ProtenixCore

        self._core = ProtenixCore(json.loads(self.config.decode("utf-8")))
        self._core._initialize()

    @modal.method()
    def fold(self, sequences: str | Sequence[str], options: dict | None = None) -> "ProtenixOutput":
        return self._core.fold(sequences, options=options)


class Protenix(ModelWrapper):
    """Interface for Protenix structure prediction."""

    MODEL_SPEC = PROTENIX_SPEC

    def __init__(self, backend: str = "modal", device: str | None = None, config: dict | None = None) -> None:
        """Create a Protenix model wrapper."""
        super().__init__(backend=backend, device=device, config=config)
        self._initialize_backend_from_spec(self.MODEL_SPEC, backend=backend, device=device, config=config)

    def fold(self, sequences: str | Sequence[str], options: dict | None = None) -> "ProtenixOutput":
        """Run Protenix for a single sequence entry.

        Use ``:`` inside a sequence string to define multiple chains.
        """
        validated_sequences = [sequences] if isinstance(sequences, str) else list(sequences)
        if len(validated_sequences) != 1:
            raise ValueError(
                "Protenix currently supports exactly one top-level sequence per call; use ':' to join chains."
            )
        if options is not None:
            static_keys = self.MODEL_SPEC.contract.static_config_keys & set(options)
            if static_keys:
                raise ValueError(
                    "The following config keys can only be set at initialization and cannot be overridden per-call: "
                    f"{sorted(static_keys)}"
                )
        return self._call_backend_method("fold", sequences, options=options)
