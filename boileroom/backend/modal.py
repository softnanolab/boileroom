
import modal
from modal import Image, gpu, MINUTES

from ..images import esm_image
from ..images.volumes import model_weights
from ..utils import MODEL_DIR
from .base import Backend
from ..base import Algorithm

class ModalBackend(Backend):
    """Backend for Modal."""

    def __init__(self, model: Algorithm) -> None:
        """Initialize the backend."""
        super().__init__()
        self._app = modal.App("boileroom")
        self._app_context = self._app.run()

        self.model.fold = modal.method(self.model.fold)
        self.model._initialize = modal.method(self.model._initialize)

        self.model = self._app.cls(
            image=esm_image,
            gpu="T4",
            timeout=20 * MINUTES,
            scaledown_window=10 * MINUTES,
            volumes={MODEL_DIR: model_weights},
        )(self.model)

    def startup(self) -> None:
        """Startup the backend."""
        self._app_context.__enter__()

    def shutdown(self) -> None:
        """Shutdown the backend."""
        self._app_context.__exit__(None, None, None)