"""ESMFold implementation for protein structure prediction using Meta AI's ESM-2 model."""

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import modal

from ...backend.modal import get_modal_app
from ...backend.modal_server import ModalFoldServer
from ...base import ModelWrapper
from ...images.volumes import model_weights
from ...utils import MINUTES, MODAL_MODEL_DIR
from ..registry import ESMFOLD_SPEC
from .image import esm_image
from .types import ESMFoldOutput

if TYPE_CHECKING:
    from .core import ESMFoldCore

logger = logging.getLogger(__name__)
app = get_modal_app("esmfold")

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
class ModalESMFold(ModalFoldServer):
    """Modal-specific wrapper around `ESMFoldCore`."""

    config: bytes = modal.parameter(default=b"{}")

    def _build_core(self, config: dict[str, Any]) -> "ESMFoldCore":
        from .core import ESMFoldCore

        return ESMFoldCore(config)


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
