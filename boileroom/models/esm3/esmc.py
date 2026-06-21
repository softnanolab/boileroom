import json
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import modal

from ...backend.modal import get_modal_app
from ...base import ModelWrapper
from ...images.volumes import model_weights
from ...utils import MINUTES, MODAL_MODEL_DIR
from ..registry import ESMC_SPEC
from .image import esm3_image

if TYPE_CHECKING:
    from .types import ESMCOutput

logger = logging.getLogger(__name__)
app = get_modal_app("esmc")


@app.cls(
    image=esm3_image,
    gpu="T4",
    timeout=20 * MINUTES,
    scaledown_window=10 * MINUTES,
    volumes={MODAL_MODEL_DIR: model_weights},
)
class ModalESMC:
    """Modal wrapper around :class:`ESMCCore`."""

    config: bytes = modal.parameter(default=b"{}")

    @modal.enter()
    def _initialize(self) -> None:
        from .core import ESMCCore

        self._core = ESMCCore(config=json.loads(self.config.decode("utf-8")))
        self._core._initialize()

    @modal.method()
    def embed(self, sequences: str | Sequence[str], options: dict | None = None) -> "ESMCOutput":
        assert self._core is not None, "ModalESMC has not been initialized"
        return self._core.embed(sequences, options=options)


class ESMC(ModelWrapper):
    """Interface for ESM-C residue-level embeddings."""

    MODEL_SPEC = ESMC_SPEC

    def __init__(self, backend: str = "modal", device: str | None = None, config: dict | None = None) -> None:
        super().__init__(backend=backend, device=device, config=config)
        self._initialize_backend_from_spec(self.MODEL_SPEC, backend=backend, device=device, config=config)

    def embed(self, sequences: str | Sequence[str], options: dict | None = None) -> "ESMCOutput":
        if options is not None:
            from .core import ESMCCore

            conflicting_keys = sorted(set(options) & ESMCCore.STATIC_CONFIG_KEYS)
            if conflicting_keys:
                raise ValueError(
                    "The following config keys can only be set at initialization and cannot be "
                    f"overridden per-call: {conflicting_keys}"
                )
        return self._call_backend_method("embed", sequences, options=options)
