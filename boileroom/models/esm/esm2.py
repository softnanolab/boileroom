import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import modal

from ...backend.modal import get_modal_app
from ...backend.modal_server import ModalEmbedServer
from ...base import ModelWrapper
from ...images.volumes import model_weights
from ...utils import MINUTES, MODAL_MODEL_DIR
from ..registry import ESM2_SPEC
from .image import esm_image

if TYPE_CHECKING:
    from .core import ESM2Core
    from .types import ESM2Output

logger = logging.getLogger(__name__)
app = get_modal_app("esm2")

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
class ModalESM2(ModalEmbedServer):
    """Modal-specific wrapper around `ESM2Core`."""

    config: bytes = modal.parameter(default=b"{}")

    def _build_core(self, config: dict[str, Any]) -> "ESM2Core":
        from .core import ESM2Core

        return ESM2Core(config=config)


############################################################
# HIGH-LEVEL INTERFACE
############################################################
class ESM2(ModelWrapper):
    """Interface for running ESM2 embeddings via Modal."""

    MODEL_SPEC = ESM2_SPEC

    def __init__(
        self,
        backend: str = "modal",
        device: str | None = None,
        config: dict | None = None,
    ) -> None:
        """Initialize the ESM2 high-level interface and start the selected backend.

        Parameters
        ----------
        backend : str
            Backend type to use. Supported values:
            - "modal": Use Modal backend (default)
            - "apptainer": Use Apptainer backend (requires Apptainer installed)
        device : Optional[str]
            Device identifier for model execution (for example "cuda:0" or "cpu").
        config : Optional[dict]
            Configuration passed to the backend and underlying model.

        Raises
        ------
        ValueError
            If an unsupported backend string is provided.
        """
        super().__init__(backend=backend, device=device, config=config)
        self._initialize_backend_from_spec(self.MODEL_SPEC, backend=backend, device=device, config=config)

    def embed(self, sequences: str | Sequence[str], options: dict | None = None) -> "ESM2Output":
        """Compute ESM-2 embeddings for one or more protein sequences using the configured backend.

        Parameters
        ----------
        sequences : str | Sequence[str]
            A single protein sequence string or a sequence of protein sequences. ESM2 inputs may include inline ``<mask>`` tokens, and multimer inputs may be provided by including ``:`` characters to separate chains within a sequence.
        options : dict | None, optional
            Per-call options merged with the backend's static configuration to adjust behavior for this call (for example: include_fields, glycine_linker, position_ids_skip). Request `include_fields=["lm_logits"]` for residue-aligned full-vocabulary logits, or combine it with `hidden_states` / `["*"]` to return both optional outputs.

        Returns
        -------
        ESM2Output
            Embeddings and associated metadata (`embeddings`, `chain_index`, `residue_index`, `metadata`, and optional `hidden_states` / `lm_logits`) for the provided sequences. Logit requests trigger automatic internal switching to an MLM-capable variant while keeping the public API unchanged; once upgraded, the instance continues reusing that model for subsequent calls.
        """
        return self._call_backend_method("embed", sequences, options=options)
