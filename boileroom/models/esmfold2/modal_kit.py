"""Modal entrypoint that runs ESMFold2 on the optimization-kit image (``optimization="exact"`` and ``"fast"``)."""

import json
from typing import TYPE_CHECKING

import modal

from ...backend.modal import get_modal_app
from ...images.modal import get_modal_kit_image
from ...images.volumes import model_weights
from ...optimization import initialize_core
from ...utils import HOURS, MINUTES, MODAL_MODEL_DIR
from .esmfold2 import ESMFold2FoldInput

if TYPE_CHECKING:
    from .types import ESMFold2Output

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
class ModalESMFold2Kit:
    """Modal-specific wrapper around `ESMFold2Core`, on the kit image."""

    config: bytes = modal.parameter(default=b"{}")

    @modal.enter()
    def _initialize(self) -> None:
        """Create and initialize the core ESMFold2 backend."""
        from .core import ESMFold2Core

        self._core = ESMFold2Core(json.loads(self.config.decode("utf-8")))
        self._refusal = initialize_core(self._core)

    @modal.method()
    def fold(self, sequences: ESMFold2FoldInput, options: dict | None = None) -> "ESMFold2Output":
        """Run ESMFold2 structure prediction."""
        if self._refusal is not None:
            raise self._refusal
        return self._core.fold(sequences, options=options)
