"""Public and Modal wrappers for AlphaFold2-Multimer."""

import json
import logging
from collections.abc import Sequence

import modal

from ...backend.modal import get_modal_app
from ...base import ModelWrapper
from ...images.volumes import model_weights
from ...utils import HOURS, MINUTES, MODAL_MODEL_DIR
from ..registry import ALPHAFOLD2_MULTIMER_SPEC
from .image import alphafold2_multimer_image
from .types import AlphaFold2MultimerOutput

logger = logging.getLogger(__name__)
app = get_modal_app("alphafold2_multimer")


@app.cls(
    image=alphafold2_multimer_image,
    gpu="A100-80GB",
    timeout=6 * HOURS,
    scaledown_window=10 * MINUTES,
    volumes={MODAL_MODEL_DIR: model_weights},
)
class ModalAlphaFold2Multimer:
    """Modal entrypoint for AlphaFold2-Multimer."""

    config: bytes = modal.parameter(default=b"{}")

    @modal.enter()
    def _initialize(self) -> None:
        from .core import AlphaFold2MultimerCore

        self._core = AlphaFold2MultimerCore(json.loads(self.config.decode("utf-8")))
        self._core._initialize()

    @modal.method()
    def fold(self, sequences: str | Sequence[str], options: dict | None = None) -> "AlphaFold2MultimerOutput":
        return self._core.fold(sequences, options=options)


class AlphaFold2Multimer(ModelWrapper):
    """Interface for AlphaFold2-Multimer structure prediction."""

    MODEL_SPEC = ALPHAFOLD2_MULTIMER_SPEC

    def __init__(self, backend: str = "modal", device: str | None = None, config: dict | None = None) -> None:
        """Create an AlphaFold2-Multimer model wrapper."""
        super().__init__(backend=backend, device=device, config=config)
        self._initialize_backend_from_spec(self.MODEL_SPEC, backend=backend, device=device, config=config)

    def fold(self, sequences: str | Sequence[str], options: dict | None = None) -> "AlphaFold2MultimerOutput":
        """Run AlphaFold2-Multimer for a single sequence entry.

        Use ``:`` inside a sequence string to define multiple chains.
        """
        validated_sequences = [sequences] if isinstance(sequences, str) else list(sequences)
        if len(validated_sequences) != 1:
            raise ValueError(
                "AlphaFold2-Multimer currently supports exactly one top-level sequence per call; use ':' to join chains."
            )
        if options is not None:
            static_keys = self.MODEL_SPEC.contract.static_config_keys & set(options)
            if static_keys:
                raise ValueError(
                    "The following config keys can only be set at initialization and cannot be overridden per-call: "
                    f"{sorted(static_keys)}"
                )
        return self._call_backend_method("fold", sequences, options=options)
