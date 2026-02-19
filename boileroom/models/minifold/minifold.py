"""MiniFold implementation for protein structure prediction."""

import json
import logging
from typing import Optional, Sequence, Union

import modal

from ...backend import ModalBackend
from ...backend.modal import app
from .image import minifold_image
from ...base import ModelWrapper
from ...images.volumes import model_weights
from ...utils import MINUTES, MODAL_MODEL_DIR

from .types import MiniFoldOutput

logger = logging.getLogger(__name__)


############################################################
# MODAL-SPECIFIC WRAPPER
############################################################
@app.cls(
    image=minifold_image,
    gpu="T4",
    timeout=20 * MINUTES,
    scaledown_window=10 * MINUTES,
    volumes={MODAL_MODEL_DIR: model_weights},
)
class ModalMiniFold:
    """Modal-specific wrapper around MiniFoldCore."""

    config: bytes = modal.parameter(default=b"{}")

    @modal.enter()
    def _initialize(self) -> None:
        from .core import MiniFoldCore

        self._core = MiniFoldCore(json.loads(self.config.decode("utf-8")))
        self._core._initialize()

    @modal.method()
    def fold(self, sequences: Union[str, Sequence[str]], options: Optional[dict] = None) -> "MiniFoldOutput":
        return self._core.fold(sequences, options=options)


############################################################
# HIGH-LEVEL INTERFACE
############################################################


class MiniFold(ModelWrapper):
    """
    Interface for MiniFold protein structure prediction model.

    MiniFold is a lightweight, fast protein structure prediction model
    (10-20x speedup over ESMFold/AF2) that uses ESM2 embeddings.
    """

    def __init__(self, backend: str = "modal", device: Optional[str] = None, config: Optional[dict] = None) -> None:
        """Initialize the MiniFold model wrapper and start the selected backend.

        Parameters
        ----------
        backend : str
            Backend type to use. Supported values:
            - "modal": Use Modal backend (default)
        device : Optional[str]
            Optional device specifier to pass to the backend (e.g., "cuda:0" or "cpu").
        config : Optional[dict]
            Optional configuration passed to the backend; if omitted an empty dict is used.
        """
        if config is None:
            config = {}
        self.config = config
        self.device = device
        if backend == "modal":
            self._backend = ModalBackend(ModalMiniFold, config, device=device)
        else:
            raise ValueError(f"Backend {backend} not supported. MiniFold supports 'modal'.")
        self._backend.start()

    def fold(self, sequences: Union[str, Sequence[str]], options: Optional[dict] = None) -> "MiniFoldOutput":
        """Predict protein structure(s) for the provided sequence or sequences.

        Parameters
        ----------
        sequences : str | Sequence[str]
            A single amino-acid sequence or a sequence of amino-acid sequences to predict.
        options : dict, optional
            Per-call configuration overrides.

        Returns
        -------
        MiniFoldOutput
            Prediction results and associated metadata.
        """
        return self._call_backend_method("fold", sequences, options=options)
