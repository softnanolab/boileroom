"""MiniFold implementation for protein structure prediction using pinned upstream MiniFold."""

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import modal

from ...backend.modal import get_modal_app
from ...backend.modal_server import ModalFoldServer
from ...base import ModelWrapper
from ...images.volumes import model_weights
from ...utils import MINUTES, MODAL_MODEL_DIR
from ..registry import MINIFOLD_SPEC
from .image import minifold_image
from .types import MiniFoldOutput

if TYPE_CHECKING:
    from .core import MiniFoldCore

logger = logging.getLogger(__name__)
app = get_modal_app("minifold")

############################################################
# CORE ALGORITHM
############################################################


############################################################
# MODAL-SPECIFIC WRAPPER
############################################################
@app.cls(
    image=minifold_image,
    gpu="L4",
    memory=32768,
    timeout=20 * MINUTES,
    scaledown_window=1 * MINUTES,
    volumes={MODAL_MODEL_DIR: model_weights},
)
class ModalMiniFold(ModalFoldServer):
    """Modal-specific wrapper around `MiniFoldCore`."""

    config: bytes = modal.parameter(default=b"{}")

    def _build_core(self, config: dict[str, Any]) -> "MiniFoldCore":
        from .core import MiniFoldCore

        return MiniFoldCore(config)


############################################################
# HIGH-LEVEL INTERFACE
############################################################


class MiniFold(ModelWrapper):
    """
    Interface for MiniFold protein structure prediction model.
    Single-chain proteins only. Lists run sequentially without padded attention.
    """

    MODEL_SPEC = MINIFOLD_SPEC

    def __init__(self, backend: str = "modal", device: str | None = None, config: dict | None = None) -> None:
        """Initialize the MiniFold high-level model wrapper and start the selected backend.

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
        config = dict(config or {})
        if self.parse_backend(backend)[0] == "apptainer":
            configured = config.get("device")
            if device is not None and configured is not None and device != configured:
                raise ValueError("MiniFold device and config['device'] must select the same Apptainer device")
            device = device or configured
            if configured is not None:
                assert device is not None
                config["device"] = "cuda:0" if device.startswith("cuda") else device
        super().__init__(backend=backend, device=device, config=config)
        self._initialize_backend_from_spec(self.MODEL_SPEC, backend=backend, device=device, config=config)

    def fold(self, sequences: str | Sequence[str], options: dict | None = None) -> "MiniFoldOutput":
        """Predict protein structure(s) for the provided sequence or sequences using the configured backend.

        Parameters
        ----------
        sequences : str | Sequence[str]
            A single amino-acid sequence or a sequence of amino-acid sequences to predict.
        options : dict, optional
            Per-call configuration overrides (for example `include_fields` to control which output fields are returned); keys override the instance's static config for this call.

        Returns
        -------
        MiniFoldOutput
            Prediction results and associated metadata, including generated atom arrays and any requested model outputs.
        """
        return self._call_backend_method("fold", sequences, options=options)
