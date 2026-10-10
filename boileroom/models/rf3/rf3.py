"""Public and Modal wrappers for RoseTTAFold 3."""

import logging
from typing import TYPE_CHECKING, Any
from collections.abc import Sequence

import modal

from ...backend.modal import get_modal_app
from ...backend.modal_server import ModalFoldServer
from ...base import ModelWrapper
from ...images.volumes import model_weights
from ...utils import MINUTES, MODAL_MODEL_DIR
from ..registry import RF3_SPEC
from .image import rf3_image
from .types import RF3Output

if TYPE_CHECKING:
    from .core import RF3Core

logger = logging.getLogger(__name__)
app = get_modal_app("rf3")


@app.cls(
    image=rf3_image,
    gpu="A100-40GB",
    timeout=60 * MINUTES,
    scaledown_window=10 * MINUTES,
    volumes={MODAL_MODEL_DIR: model_weights},
)
class ModalRF3(ModalFoldServer):
    """Modal entrypoint for RF3."""

    config: bytes = modal.parameter(default=b"{}")

    def _build_core(self, config: dict[str, Any]) -> "RF3Core":
        from .core import RF3Core

        return RF3Core(config)


class RF3(ModelWrapper):
    """Interface for RoseTTAFold 3 structure prediction."""

    MODEL_SPEC = RF3_SPEC

    def __init__(self, backend: str = "modal", device: str | None = None, config: dict | None = None) -> None:
        """Create an RF3 model wrapper."""
        super().__init__(backend=backend, device=device, config=config)
        self._initialize_backend_from_spec(self.MODEL_SPEC, backend=backend, device=device, config=config)

    def fold(self, sequences: str | Sequence[str], options: dict | None = None) -> "RF3Output":
        """Run RF3 for a single sequence entry.

        Keep one instance open and call ``fold()`` for each job to reuse the
        loaded model. The backend context owns the worker's lifetime.
        Use ``:`` inside a sequence string to define multiple protein chains.
        RF3 runs no MSA search: pass ``options={"msa": [a3m_text_or_None, ...]}``
        (one entry per chain) for alignments, otherwise each chain is folded from
        its sequence alone. Each call returns ``diffusion_batch_size`` samples,
        best first by ``ranking_score``; ``seed`` and
        ``early_stopping_plddt_threshold`` may be set per call. Request ``pae``,
        ``token_chain_ids`` and ``token_res_ids`` to compute interface scores.
        Set ``config={"optimization": "exact" | "fast" | "big"}`` at
        initialization to run the Anthropic kit kernels (A100/H100/H200 only);
        ``big`` uses the least memory, for inputs that run ``fast`` out of memory.
        """
        validated_sequences = [sequences] if isinstance(sequences, str) else list(sequences)
        if len(validated_sequences) != 1:
            raise ValueError("RF3 currently supports exactly one top-level sequence per call; use ':' to join chains.")
        if options is not None:
            static_keys = self.MODEL_SPEC.contract.static_config_keys & set(options)
            if static_keys:
                raise ValueError(
                    "The following config keys can only be set at initialization and cannot be overridden per-call: "
                    f"{sorted(static_keys)}"
                )
        return self._call_backend_method("fold", sequences, options=options)
