import json
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import modal

from ...backend.modal import get_modal_app
from ...base import ModelWrapper
from ...images.volumes import model_weights
from ...utils import MINUTES, MODAL_MODEL_DIR
from ..registry import ESM3_SPEC
from .image import esm3_image

if TYPE_CHECKING:
    from .types import ESM3Output

logger = logging.getLogger(__name__)
app = get_modal_app("esm3")


@app.cls(
    image=esm3_image,
    gpu="T4",
    timeout=20 * MINUTES,
    scaledown_window=10 * MINUTES,
    volumes={MODAL_MODEL_DIR: model_weights},
)
class ModalESM3:
    """Modal wrapper around :class:`ESM3Core`."""

    config: bytes = modal.parameter(default=b"{}")

    @modal.enter()
    def _initialize(self) -> None:
        from .core import ESM3Core

        self._core = ESM3Core(config=json.loads(self.config.decode("utf-8")))
        self._core._initialize()

    @modal.method()
    def embed(self, sequences: str | Sequence[str], options: dict | None = None) -> "ESM3Output":
        assert self._core is not None, "ModalESM3 has not been initialized"
        return self._core.embed(sequences, options=options)


class ESM3(ModelWrapper):
    """Interface for ESM3 residue-level embeddings (embed-only)."""

    MODEL_SPEC = ESM3_SPEC

    def __init__(self, backend: str = "modal", device: str | None = None, config: dict | None = None) -> None:
        super().__init__(backend=backend, device=device, config=config)
        self._initialize_backend_from_spec(self.MODEL_SPEC, backend=backend, device=device, config=config)

    def embed(self, sequences: str | Sequence[str], options: dict | None = None) -> "ESM3Output":
        if options is not None:
            from .core import ESM3Core

            conflicting_keys = sorted(set(options) & ESM3Core.STATIC_CONFIG_KEYS)
            if conflicting_keys:
                raise ValueError(
                    "The following config keys can only be set at initialization and cannot be "
                    f"overridden per-call: {conflicting_keys}"
                )
        return self._call_backend_method("embed", sequences, options=options)
