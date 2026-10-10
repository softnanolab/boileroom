"""Modal entrypoint that runs RF3 on the optimization-kit image (``optimization="exact"``)."""

import json
from collections.abc import Sequence
from typing import TYPE_CHECKING

import modal

from ...backend.modal import get_modal_app
from ...images.modal import get_modal_kit_image
from ...images.volumes import model_weights
from ...optimization import initialize_core, retry_initialize
from ...utils import HOURS, MINUTES, MODAL_MODEL_DIR

if TYPE_CHECKING:
    from .types import RF3Output

# A separate app: Modal builds every image registered on an app when it starts, and the kit image is not needed for the
# vanilla mode.
app = get_modal_app("rf3-kit")
rf3_kit_image = get_modal_kit_image("rf3")


@app.cls(
    image=rf3_kit_image,
    # The kit serves A100 and H100/H200 (its A100 configuration targets the 80 GB card); the checkpoint downloads on the
    # first call.
    gpu="A100-80GB",
    timeout=1 * HOURS,
    scaledown_window=10 * MINUTES,
    volumes={MODAL_MODEL_DIR: model_weights},
)
class ModalRF3Kit:
    """Modal entrypoint for RF3 on the kit image."""

    config: bytes = modal.parameter(default=b"{}")

    @modal.enter()
    def _initialize(self) -> None:
        from .core import RF3Core

        self._core = RF3Core(json.loads(self.config.decode("utf-8")))
        self._refusal = initialize_core(self._core)

    @modal.exit()
    def _shutdown(self) -> None:
        self._core.close()

    @modal.method()
    def fold(self, sequences: str | Sequence[str], options: dict | None = None) -> "RF3Output":
        self._refusal = retry_initialize(self._core, self._refusal)
        if self._refusal is not None:
            raise self._refusal
        return self._core.fold(sequences, options=options)
