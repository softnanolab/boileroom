import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import modal

from ...backend.modal import get_modal_app
from ...backend.modal_server import ModalEmbedServer
from ...base import ModelWrapper
from ...images.volumes import model_weights
from ...utils import MINUTES, MODAL_MODEL_DIR
from ..registry import ESMC_SPEC
from .image import esm3_image

if TYPE_CHECKING:
    from .core import ESMCCore
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
class ModalESMC(ModalEmbedServer):
    """Modal wrapper around :class:`ESMCCore`."""

    config: bytes = modal.parameter(default=b"{}")

    def _build_core(self, config: dict[str, Any]) -> "ESMCCore":
        from .core import ESMCCore

        return ESMCCore(config=config)


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
