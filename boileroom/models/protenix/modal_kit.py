"""Modal entrypoint that runs Protenix on the optimization-kit image (``optimization="exact"`` and ``"fast"``)."""

from typing import TYPE_CHECKING, Any

import modal

from ...backend.modal import get_modal_app
from ...backend.modal_server import ModalFoldServer
from ...images.modal import get_modal_kit_image
from ...images.volumes import model_weights
from ...utils import HOURS, MINUTES, MODAL_MODEL_DIR

if TYPE_CHECKING:
    from .core import ProtenixCore

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
class ModalProtenixKit(ModalFoldServer):
    """Modal entrypoint for Protenix on the kit image."""

    config: bytes = modal.parameter(default=b"{}")

    def _build_core(self, config: dict[str, Any]) -> "ProtenixCore":
        from .core import ProtenixCore

        return ProtenixCore(config)
