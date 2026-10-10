"""Modal entrypoint that runs RF3 on the optimization-kit image (``optimization="exact"``)."""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import modal

from ...backend.modal import get_modal_app
from ...backend.modal_server import ModalFoldServer
from ...images.modal import get_modal_kit_image
from ...images.volumes import model_weights
from ...utils import HOURS, MINUTES, MODAL_MODEL_DIR

if TYPE_CHECKING:
    from .core import RF3Core

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
class ModalRF3Kit(ModalFoldServer):
    """Modal entrypoint for RF3 on the kit image."""

    config: bytes = modal.parameter(default=b"{}")

    def _build_core(self, config: dict[str, Any]) -> "RF3Core":
        from .core import RF3Core

        return RF3Core(config)
