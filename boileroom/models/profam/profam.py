"""ProFam public wrapper and thin Modal generation server."""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import modal

from ...backend.modal import get_modal_app
from ...backend.modal_server import ModalGenerateServer
from ...base import ModelWrapper
from ...images.volumes import model_weights
from ...utils import MINUTES, MODAL_MODEL_DIR
from ..registry import PROFAM_SPEC
from .image import profam_image
from .types import ProFamOutput

if TYPE_CHECKING:
    from .core import ProFamCore

app = get_modal_app("profam")


@app.cls(
    image=profam_image,
    gpu="T4",
    timeout=20 * MINUTES,
    scaledown_window=10 * MINUTES,
    memory=16384,
    volumes={MODAL_MODEL_DIR: model_weights},
)
class ModalProFam(ModalGenerateServer):
    """Serve one resident, guarded ProFam core."""

    config: bytes = modal.parameter(default=b"{}")

    def _build_core(self, config: dict[str, Any]) -> "ProFamCore":
        from .core import ProFamCore

        return ProFamCore(config)


class ProFam(ModelWrapper):
    """Generate proteins from one raw family prompt on Modal or Apptainer.

    A list contains homologs conditioning one generation job, not separate jobs.
    Scores are mean per-token sampling log-probabilities, not fitness predictions.
    """

    MODEL_SPEC = PROFAM_SPEC

    def __init__(self, backend: str = "modal", device: str | None = None, config: dict | None = None) -> None:
        from .core import ProFamCore

        config = dict(config or {})
        if self.parse_backend(backend)[0] == "apptainer":
            configured = config.get("device")
            if device is not None and configured is not None and device != configured:
                raise ValueError("ProFam device and config['device'] must select the same Apptainer device")
            device = device or configured
            if configured is not None:
                config["device"] = "cuda:0" if str(device).startswith("cuda") else device
        # Reject bad config before starting any paid runtime. Heavy dependencies
        # are lazy inside _load(), so importing the core here is lightweight.
        ProFamCore(config)
        super().__init__(backend=backend, device=device, config=config)
        self._initialize_backend_from_spec(self.MODEL_SPEC, backend=backend, device=device, config=config)

    def generate(self, sequences: str | Sequence[str], options: dict | None = None) -> ProFamOutput:
        from .core import ProFamCore

        ProFamCore(self.config).validate_request(sequences, options)
        return self._call_backend_method("generate", sequences, options=options)
