import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import modal

from ...backend.modal import get_modal_app
from ...backend.modal_server import ModalFoldServer
from ...base import ModelWrapper
from ...images.volumes import model_weights
from ...utils import MINUTES, MODAL_MODEL_DIR
from ..registry import CHAI1_SPEC
from .image import chai_image
from .types import Chai1Output

if TYPE_CHECKING:
    from .core import Chai1Core

logger = logging.getLogger(__name__)
app = get_modal_app("chai1")


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
class ModalChai1(ModalFoldServer):
    """Modal-specific wrapper around `Chai1Core`."""

    config: bytes = modal.parameter(default=b"{}")

    def _build_core(self, config: dict[str, Any]) -> "Chai1Core":
        from .core import Chai1Core

        return Chai1Core(config)


############################################################
# HIGH-LEVEL INTERFACE
############################################################


class Chai1(ModelWrapper):
    """
    Interface for Chai-1 protein structure prediction model.
    # TODO: This is the user-facing interface. It should give all the relevant details possible.
    # with proper documentation.
    """

    MODEL_SPEC = CHAI1_SPEC

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
        super().__init__(backend=backend, device=device, config=config)
        self._initialize_backend_from_spec(self.MODEL_SPEC, backend=backend, device=device, config=config)

    def fold(self, sequences: str | Sequence[str], options: dict | None = None) -> "Chai1Output":
        """Run structure prediction for a single top-level sequence entry using the configured backend.

        Parameters
        ----------
        sequences : str | Sequence[str]
            A single sequence string or a one-item sequence containing a single sequence string. Use ":" inside that sequence to join multiple chains for multimer prediction.
        options : dict | None, optional
            Per-call configuration overrides merged with the model's default config (e.g., include_fields, constraint_path, device-specific options).

        Returns
        -------
        Chai1Output
            Prediction results including metadata, generated atom arrays, and any requested confidence metrics or CIF output.

        Raises
        ------
        ValueError
            If more than one top-level sequence entry is provided. Chai-1 currently supports a single input per call; use ":" to join chains for multimers.
        """
        if not isinstance(sequences, str) and len(sequences) != 1:
            raise ValueError(
                "Chai-1 currently supports exactly one top-level sequence per call; use ':' to join chains."
            )
        if options is not None:
            from .core import Chai1Core

            conflicting_keys = sorted(set(options) & Chai1Core.STATIC_CONFIG_KEYS)
            if conflicting_keys:
                raise ValueError(
                    "The following config keys can only be set at initialization and cannot be "
                    f"overridden per-call: {conflicting_keys}"
                )
        return self._call_backend_method("fold", sequences, options=options)
