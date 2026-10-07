"""Modal entrypoint that runs ESMFold2 on the optimization-kit image (``optimization="exact"`` and ``"fast"``)."""

from typing import TYPE_CHECKING, Any

import modal

from ...backend.modal import get_modal_app
from ...backend.modal_server import ModalFoldServer
from ...images.modal import get_modal_kit_image
from ...images.volumes import model_weights
from ...utils import HOURS, MINUTES, MODAL_MODEL_DIR

if TYPE_CHECKING:
    from .core import ESMFold2Core

# A separate app: Modal builds every image registered on an app when it starts, and the kit image is not needed (and
# takes tens of minutes to build) for the vanilla modes.
app = get_modal_app("esmfold2-kit")
esmfold2_kit_image = get_modal_kit_image("esmfold2")


@app.cls(
    image=esmfold2_kit_image,
    # The kit serves A100 and H100/H200; the vanilla entrypoint's T4 would be refused.
    gpu="A100-80GB",
    # The first call downloads the pinned checkpoints (~27 GB) into the model volume and compiles the kernels; the
    # timeout covers that start-up too.
    timeout=1 * HOURS,
    scaledown_window=10 * MINUTES,
    volumes={MODAL_MODEL_DIR: model_weights},
)
class ModalESMFold2Kit(ModalFoldServer):
    """Modal-specific wrapper around `ESMFold2Core`, on the kit image."""

    config: bytes = modal.parameter(default=b"{}")

    def _build_core(self, config: dict[str, Any]) -> "ESMFold2Core":
        from .core import ESMFold2Core

        return ESMFold2Core(config)
