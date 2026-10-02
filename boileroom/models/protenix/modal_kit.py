"""Modal entrypoint that runs Protenix on the optimization-kit image (``optimization="exact"`` and ``"fast"``)."""

import json
from collections.abc import Sequence
from typing import TYPE_CHECKING

import modal

from ...backend.modal import get_modal_app
from ...images.modal import get_modal_kit_image
from ...images.volumes import model_weights
from ...optimization import initialize_core
from ...utils import HOURS, MINUTES, MODAL_MODEL_DIR

if TYPE_CHECKING:
    from .types import ProtenixOutput

# A separate app: Modal builds every image registered on an app when it starts, and the kit image is not needed for the
# vanilla modes.
app = get_modal_app("protenix-kit")
protenix_kit_image = get_modal_kit_image("protenix")


@app.cls(
    image=protenix_kit_image,
    # The kit serves A100 and H100/H200; the checkpoint and data caches download on the first call.
    gpu="A100-40GB",
    timeout=1 * HOURS,
    scaledown_window=10 * MINUTES,
    volumes={MODAL_MODEL_DIR: model_weights},
)
class ModalProtenixKit:
    """Modal entrypoint for Protenix on the kit image."""

    config: bytes = modal.parameter(default=b"{}")

    @modal.enter()
    def _initialize(self) -> None:
        from .core import ProtenixCore

        self._core = ProtenixCore(json.loads(self.config.decode("utf-8")))
        self._refusal = initialize_core(self._core)

    @modal.exit()
    def _shutdown(self) -> None:
        self._core.close()

    @modal.method()
    def fold(self, sequences: str | Sequence[str], options: dict | None = None) -> "ProtenixOutput":
        if self._refusal is not None:
            raise self._refusal
        return self._core.fold(sequences, options=options)
