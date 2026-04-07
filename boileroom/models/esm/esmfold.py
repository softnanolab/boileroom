"""ESMFold implementation for protein structure prediction using Meta AI's ESM-2 model."""

import json
import logging
from collections.abc import Sequence

import modal

from ...backend.modal import app
from ...base import ModelWrapper
from ...images.volumes import model_weights
from ...utils import MINUTES, MODAL_MODEL_DIR
from ..registry import ESMFOLD_SPEC
from .image import esm_image
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
    def fold(self, sequences: str | Sequence[str], options: dict | None = None) -> "ESMFoldOutput":
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

    MODEL_SPEC = ESMFOLD_SPEC

    def __init__(self, backend: str = "modal", device: str | None = None, config: dict | None = None) -> None:
        """Initialize the ESMFold high-level model wrapper and start the selected backend.

        Parameters
        ----------
        backend : str
            Backend type to use. Supported values:
            - "modal": Use Modal backend (default)
            - "apptainer": Use Apptainer backend (requires Apptainer installed)
        device : Optional[str]
            Optional device specifier to pass to the backend (e.g., "cuda:0" or "cpu").
        config : Optional[dict]
            Optional configuration passed to the backend; if omitted an empty dict is used.

        Raises
        ------
        ValueError
            If an unsupported backend string is provided.
        """
        super().__init__(backend=backend, device=device, config=config)
        self._initialize_backend_from_spec(self.MODEL_SPEC, backend=backend, device=device, config=config)

    def fold(self, sequences: str | Sequence[str], options: dict | None = None) -> "ESMFoldOutput":
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
