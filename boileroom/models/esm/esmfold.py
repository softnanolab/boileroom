"""ESMFold implementation for protein structure prediction using Meta AI's ESM-2 model."""

import json
import logging
from typing import TYPE_CHECKING, Optional, Sequence, Union

import modal

from ...backend import LocalBackend, ModalBackend
from ...backend.base import Backend
from ...backend.modal import app
from ...base import ModelWrapper
from .image import esm_image
from ...images.volumes import model_weights
from ...utils import MINUTES, MODAL_MODEL_DIR

if TYPE_CHECKING:
    pass

from .types import ESMFoldOutput

logger = logging.getLogger(__name__)

############################################################
# CORE ALGORITHM
############################################################


############################################################
# MODAL-SPECIFIC WRAPPER
############################################################
@app.cls(
    image=esm_image,
    gpu="T4",
    timeout=20 * MINUTES,
    scaledown_window=10 * MINUTES,
    volumes={MODAL_MODEL_DIR: model_weights},
)
class ModalESMFold:
    """
    Modal-specific wrapper around `ESMFoldCore`.
    """

    config: bytes = modal.parameter(default=b"{}")

    @modal.enter()
    def _initialize(self) -> None:
        """Create an ESMFoldCore from this object's JSON-encoded config and initialize it.

        This sets the instance attribute `self._core` to the constructed ESMFoldCore and calls its initialization routine.
        """
        from .core import ESMFoldCore

        self._core = ESMFoldCore(json.loads(self.config.decode("utf-8")))
        self._core._initialize()

    @modal.method()
    def fold(self, sequences: Union[str, Sequence[str]], options: Optional[dict] = None) -> "ESMFoldOutput":
        """Run structure prediction for one or more protein sequences using the configured backend.

        Parameters
        ----------
        sequences : str | Sequence[str]
            A single amino-acid sequence or a sequence of sequences to predict.
        options : dict, optional
            Per-call configuration overrides (for example, include_fields to select which output fields to include). Keys in this dict override non-static entries of the model config for this prediction.

        Returns
        -------
        ESMFoldOutput
            Prediction results and associated metadata for each input sequence.
        """
        return self._core.fold(sequences, options=options)


############################################################
# HIGH-LEVEL INTERFACE
############################################################


class ESMFold(ModelWrapper):
    """
    Interface for ESMFold protein structure prediction model.
    # TODO: This is the user-facing interface. It should give all the relevant details possible.
    # with proper documentation.
    """

    def __init__(self, backend: str = "modal", device: Optional[str] = None, config: Optional[dict] = None) -> None:
        """Initialize the ESMFold high-level model wrapper and start the selected backend.

        Parameters
        ----------
        backend : str
            Backend type to use. Supported values:
            - "modal": Use Modal backend (default)
            - "local": Use local backend (requires dependencies in current environment)
            - "conda": Use conda backend with auto-detection (micromamba > mamba > conda)
            - "mamba": Use mamba explicitly
            - "micromamba": Use micromamba explicitly
            - "apptainer": Use Apptainer backend (requires Apptainer installed)
        device : Optional[str]
            Optional device specifier to pass to the backend (e.g., "cuda:0" or "cpu").
        config : Optional[dict]
            Optional configuration passed to the backend; if omitted an empty dict is used.

        Raises
        ------
        ValueError
            If an unsupported backend string is provided, or if conda backend is
            requested but no compatible tool (conda/mamba/micromamba) is available.
        """
        if config is None:
            config = {}
        self.config = config
        self.device = device
        backend_instance: Backend
        if backend == "modal":
            backend_instance = ModalBackend(ModalESMFold, config, device=device)
        elif backend == "local":
            from .core import ESMFoldCore

            backend_instance = LocalBackend(ESMFoldCore, config, device=device)
        elif backend in ("conda", "mamba", "micromamba"):
            from pathlib import Path
            from ...backend.conda import CondaBackend

            environment_yml = Path(__file__).parent / "environment.yml"
            # Pass Core class as string path to avoid importing it in main process
            # This keeps dependencies completely independent between Boiler Room and conda servers
            core_class_path = "boileroom.models.esm.core.ESMFoldCore"
            # Pass backend string directly as runner_command
            backend_instance = CondaBackend(
                core_class_path,
                config or {},
                device=device,
                environment_yml_path=environment_yml,
                runner_command=backend,
            )
        elif backend == "apptainer":
            from ...backend.apptainer import ApptainerBackend

            # Pass Core class as string path to avoid importing it in main process
            core_class_path = "boileroom.models.esm.core.ESMFoldCore"
            image_uri = "docker://docker.io/jakublala/boileroom-esm:latest"
            backend_instance = ApptainerBackend(core_class_path, image_uri, config or {}, device=device)
        else:
            raise ValueError(f"Backend {backend} not supported")
        self._backend = backend_instance
        self._backend.start()

    def fold(self, sequences: Union[str, Sequence[str]], options: Optional[dict] = None) -> "ESMFoldOutput":
        """Predict protein structure(s) for the provided sequence or sequences using the configured backend.

        Parameters
        ----------
        sequences : str | Sequence[str]
            A single amino-acid sequence or a sequence of amino-acid sequences to predict.
        options : dict, optional
            Per-call configuration overrides (for example `include_fields` to control which output fields are returned); keys override the instance's static config for this call.

        Returns
        -------
        ESMFoldOutput
            Prediction results and associated metadata, including generated atom arrays and any requested model outputs.
        """
        return self._call_backend_method("fold", sequences, options=options)
