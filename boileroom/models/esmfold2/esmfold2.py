"""ESMFold2 wrapper for Biohub's all-atom structure prediction model."""

import json
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import modal

from ...backend.modal import get_modal_app
from ...base import ModelWrapper
from ...images.volumes import model_weights
from ...utils import MINUTES, MODAL_MODEL_DIR
from ..registry import ESMFOLD2_SPEC
from .image import esmfold2_image
from .payloads import encode_fold_input
from .types import DNAInput, LigandInput, ProteinInput, RNAInput, StructurePredictionInput

if TYPE_CHECKING:
    from .types import ESMFold2Output

logger = logging.getLogger(__name__)
app = get_modal_app("esmfold2")

ESMFold2FoldInput = (
    str
    | Sequence[str]
    | StructurePredictionInput
    | Sequence[StructurePredictionInput]
    | Sequence[ProteinInput | RNAInput | DNAInput | LigandInput]
    | dict[str, Any]
    | Sequence[dict[str, Any]]
)


@app.cls(
    image=esmfold2_image,
    gpu="T4",
    timeout=20 * MINUTES,
    scaledown_window=10 * MINUTES,
    volumes={MODAL_MODEL_DIR: model_weights},
)
class ModalESMFold2:
    """Modal-specific wrapper around `ESMFold2Core`."""

    config: bytes = modal.parameter(default=b"{}")

    @modal.enter()
    def _initialize(self) -> None:
        """Create and initialize the core ESMFold2 backend."""
        from .core import ESMFold2Core

        self._core = ESMFold2Core(json.loads(self.config.decode("utf-8")))
        self._core._initialize()

    @modal.method()
    def fold(self, sequences: ESMFold2FoldInput, options: dict | None = None) -> "ESMFold2Output":
        """Run ESMFold2 structure prediction."""
        return self._core.fold(sequences, options=options)


class ESMFold2(ModelWrapper):
    """Interface for Biohub ESMFold2 structure prediction."""

    MODEL_SPEC = ESMFOLD2_SPEC

    def __init__(self, backend: str = "modal", device: str | None = None, config: dict | None = None) -> None:
        """Initialize the ESMFold2 wrapper and selected backend."""
        super().__init__(backend=backend, device=device, config=config)
        self._initialize_backend_from_spec(self.MODEL_SPEC, backend=backend, device=device, config=config)

    def fold(self, sequences: ESMFold2FoldInput, options: dict | None = None) -> "ESMFold2Output":
        """Predict all-atom structure(s) using ESMFold2.

        Parameters
        ----------
        sequences : str or sequence
            A protein sequence, a list of independent protein sequences, a
            `StructurePredictionInput`, a list of `StructurePredictionInput`
            objects, or a sequence of molecule input dataclasses describing one
            complex. Use ":" in string protein inputs for multichain complexes.
        options : dict, optional
            Per-call overrides such as `include_fields`, `num_loops`,
            `num_sampling_steps`, `num_diffusion_samples`, and `seed`.

        Returns
        -------
        ESMFold2Output
            Predicted atom arrays and requested confidence/serialized outputs.
        """
        backend_input: Any = sequences
        if self.backend.split(":", 1)[0].strip() == "apptainer":
            backend_input = encode_fold_input(sequences)
        return self._call_backend_method("fold", backend_input, options=options)
