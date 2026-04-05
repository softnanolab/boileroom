import json
import logging
from collections.abc import Sequence

import modal

from ...backend import ModalBackend
from ...backend.base import Backend
from ...backend.modal import app
from ...base import ModelWrapper
from ...images.volumes import model_weights
from ...utils import MINUTES, MODAL_MODEL_DIR
from .image import chai_image
from .types import Chai1Output

logger = logging.getLogger(__name__)


############################################################
# MODAL BACKEND
############################################################
@app.cls(
    image=chai_image,
    gpu="T4",
    timeout=20 * MINUTES,
    scaledown_window=10 * MINUTES,
    volumes={MODAL_MODEL_DIR: model_weights},  # TODO: somehow link this to what Chai-1 actually uses
)
class ModalChai1:
    """
    Modal-specific wrapper around `Chai1Core`.
    """

    config: bytes = modal.parameter(default=b"{}")

    @modal.enter()
    def _initialize(self) -> None:
        """Instantiate Chai1Core from the JSON-encoded `self.config` bytes and perform its initialization.

        This decodes `self.config` as UTF-8 JSON, constructs a Chai1Core with the resulting dict, and calls its `_initialize` method.
        """
        from .core import Chai1Core

        self._core = Chai1Core(json.loads(self.config.decode("utf-8")))
        self._core._initialize()

    @modal.method()
    def fold(self, sequences: str | Sequence[str], options: dict | None = None) -> "Chai1Output":
        """Run structure prediction for the given sequence(s) and return the assembled prediction output.

        Parameters
        ----------
        sequences : str | Sequence[str]
            One sequence string or a sequence of sequence strings. Individual entries may contain multiple chains separated by ":"; when provided as a single string that contains ":" the string will be split into chains.
        options : dict, optional
            Per-call configuration overrides merged with the model's default configuration to control sampling, device, and which result fields to include.

        Returns
        -------
        Chai1Output
            Prediction results and associated metadata for the provided sequence(s).
        """
        return self._core.fold(sequences, options=options)


############################################################
# HIGH-LEVEL INTERFACE
############################################################


class Chai1(ModelWrapper):
    """
    Interface for Chai-1 protein structure prediction model.
    # TODO: This is the user-facing interface. It should give all the relevant details possible.
    # with proper documentation.
    """

    def __init__(self, backend: str = "modal", device: str | None = None, config: dict | None = None) -> None:
        """Create a Chai1 model wrapper and start the selected backend.

        Parameters
        ----------
        backend : str
            Backend type to use. Supported values:
            - "modal": Use Modal backend (default)
            - "apptainer": Use Apptainer backend (requires Apptainer installed)
        device : Optional[str]
            Optional device identifier to pass to the backend (e.g., "cuda:0" or None to let the backend choose).
        config : Optional[dict]
            Optional configuration dictionary forwarded to the underlying Chai1Core or backend.

        Raises
        ------
        ValueError
            If an unsupported backend string is provided.
        """
        if config is None:
            config = {}
        self.config = config
        self.device = device
        backend_type, backend_tag = ModelWrapper.parse_backend(backend)
        backend_instance: Backend
        if backend_type == "modal":
            backend_instance = ModalBackend(ModalChai1, config, device=device)
        elif backend_type == "apptainer":
            from ...backend.apptainer import ApptainerBackend

            # Pass Core class as string path to avoid importing it in main process
            core_class_path = "boileroom.models.chai.core.Chai1Core"
            image_uri = f"docker://docker.io/jakublala/boileroom-chai1:{backend_tag}"
            backend_instance = ApptainerBackend(core_class_path, image_uri, config or {}, device=device)
        else:
            raise ValueError(f"Backend {backend_type} not supported")
        self._backend = backend_instance
        self._backend.start()

    def fold(self, sequences: str | Sequence[str], options: dict | None = None) -> "Chai1Output":
        """Run structure prediction for the given sequence(s) using the configured backend.

        Parameters
        ----------
        sequences : str | Sequence[str]
            A single sequence or a sequence of sequences to predict. Each sequence may contain multiple chains separated by ":"; currently the implementation expects a single batch.
        options : dict | None, optional
            Per-call configuration overrides merged with the model's default config (e.g., include_fields, constraint_path, device-specific options).

        Returns
        -------
        Chai1Output
            Prediction results including metadata, generated atom arrays, and any requested confidence metrics or CIF output.
        """
        return self._call_backend_method("fold", sequences, options=options)
