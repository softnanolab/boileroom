import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import modal

from ...backend.modal import get_modal_app
from ...backend.modal_server import ModalEmbedServer
from ...base import ModelWrapper
from ...images.volumes import model_weights
from ...provenance import record_gpu_memory
from ...utils import MINUTES, MODAL_MODEL_DIR
from ..registry import ESM3_SPEC
from .image import esm3_image

if TYPE_CHECKING:
    import numpy as np

    from .core import ESM3Core
    from .types import ESM3InverseFoldingOutput, ESM3Output

logger = logging.getLogger(__name__)
app = get_modal_app("esm3")


@app.cls(
    image=esm3_image,
    gpu="T4",
    timeout=20 * MINUTES,
    scaledown_window=10 * MINUTES,
    volumes={MODAL_MODEL_DIR: model_weights},
)
class ModalESM3(ModalEmbedServer):
    """Modal wrapper around :class:`ESM3Core`."""

    config: bytes = modal.parameter(default=b"{}")

    def _build_core(self, config: dict[str, Any]) -> "ESM3Core":
        from .core import ESM3Core

        return ESM3Core(config=config)

    @modal.method()
    def inverse_fold(
        self, sequence: str, backbone_coordinates: "np.ndarray", positions: Sequence[int]
    ) -> "ESM3InverseFoldingOutput":
        """Run :meth:`ESM3Core.inverse_fold` on the Modal worker (see that method for parameters).

        Like ``embed``, the output records the GPU memory in use after the call (see
        :func:`~boileroom.provenance.record_gpu_memory`).
        """
        output = self._loaded_core().inverse_fold(sequence, backbone_coordinates, positions)
        record_gpu_memory(output)
        return output


class ESM3(ModelWrapper):
    """Interface for ESM3 residue-level embeddings (embed-only)."""

    MODEL_SPEC = ESM3_SPEC

    def __init__(self, backend: str = "modal", device: str | None = None, config: dict | None = None) -> None:
        super().__init__(backend=backend, device=device, config=config)
        self._initialize_backend_from_spec(self.MODEL_SPEC, backend=backend, device=device, config=config)

    def embed(self, sequences: str | Sequence[str], options: dict | None = None) -> "ESM3Output":
        if options is not None:
            from .core import ESM3Core

            conflicting_keys = sorted(set(options) & ESM3Core.STATIC_CONFIG_KEYS)
            if conflicting_keys:
                raise ValueError(
                    "The following config keys can only be set at initialization and cannot be "
                    f"overridden per-call: {conflicting_keys}"
                )
        return self._call_backend_method("embed", sequences, options=options)

    def inverse_fold(
        self, sequence: str, backbone_coordinates: "np.ndarray", positions: Sequence[int]
    ) -> "ESM3InverseFoldingOutput":
        """Predict amino-acid logits at masked residues, conditioned on a backbone structure.

        Parameters
        ----------
        sequence : str
            Amino-acid sequence; chains separated by ``:``.
        backbone_coordinates : np.ndarray
            ``(n_residues, 3, 3)`` N, CA, C coordinates (Angstrom), chain breaks excluded.
        positions : Sequence[int]
            Residue indices to mask; all other residues are kept as given.

        Returns
        -------
        ESM3InverseFoldingOutput
            ``(n_positions, 20)`` logits over the standard amino acids at the masked positions.
        """
        return self._call_backend_method("inverse_fold", sequence, backbone_coordinates, positions)
