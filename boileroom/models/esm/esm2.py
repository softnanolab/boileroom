import json
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import modal

from ...backend.modal import app
from ...base import ModelWrapper
from ...images.volumes import model_weights
from ...utils import MINUTES, MODAL_MODEL_DIR
from ..registry import ESM2_SPEC
from .image import esm_image

if TYPE_CHECKING:
    from .types import ESM2Output

logger = logging.getLogger(__name__)

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
class ModalESM2:
    """Modal-specific wrapper around `ESM2`."""

    config: bytes = modal.parameter(default=b"{}")

    @modal.enter()
    def _initialize(self) -> None:
        """Create and initialize the ESM2Core backend instance from the encoded configuration.

        Decodes the JSON bytes stored in self.config, constructs an ESM2Core using that config, assigns it to self._core, and calls its initialization routine.
        """
        from .core import ESM2Core

        self._core = ESM2Core(config=json.loads(self.config.decode("utf-8")))
        self._core._initialize()

    @modal.method()
    def embed(self, sequences: str | Sequence[str], options: dict | None = None) -> "ESM2Output":
        """Compute embeddings for one or more protein sequences using the configured ESM-2 model.

        Parameters
        ----------
        sequences : str | Sequence[str]
            A single protein sequence string or an iterable of sequence strings.
        options : dict | None, optional
            Per-call options that override the instance configuration (e.g., model selection, glycine_linker, position_ids_skip, include_fields). Only provided keys are merged with the static configuration.

        Returns
        -------
        ESM2Output
            Prediction container with fields including `embeddings`, `metadata`, `chain_index`, `residue_index`, and `hidden_states` when requested.
        """
        assert self._core is not None, "ModalESM2 has not been initialized"
        return self._core.embed(sequences, options=options)


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
            A single protein sequence string or a sequence of protein sequences. Multimer inputs may be provided by including ":" characters to separate chains within a sequence.
        options : dict | None, optional
            Per-call options merged with the backend's static configuration to adjust behavior for this call (for example: model_name, include_fields, glycine_linker, position_ids_skip). Keys not present in `options` fall back to the configured defaults.

        Returns
        -------
        ESM2Output
            Embeddings and associated metadata (embeddings, chain_index, residue_index, metadata, and optional hidden_states) for the provided sequences.
        """
        return self._call_backend_method("embed", sequences, options=options)
