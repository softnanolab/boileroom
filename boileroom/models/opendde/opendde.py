"""Public and Modal wrappers for OpenDDE."""

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import modal

from ...backend.modal import get_modal_app
from ...backend.modal_server import ModalFoldServer
from ...base import ModelWrapper
from ...images.volumes import model_weights
from ...utils import MINUTES, MODAL_MODEL_DIR
from ..registry import OPENDDE_SPEC
from .image import opendde_image
from .types import OpenDDEOutput

if TYPE_CHECKING:
    from .core import OpenDDECore

logger = logging.getLogger(__name__)
app = get_modal_app("opendde")


@app.cls(
    image=opendde_image,
    gpu="A100-40GB",
    timeout=60 * MINUTES,
    scaledown_window=10 * MINUTES,
    volumes={MODAL_MODEL_DIR: model_weights},
)
class ModalOpenDDE(ModalFoldServer):
    """Modal entrypoint for OpenDDE."""

    config: bytes = modal.parameter(default=b"{}")

    def _build_core(self, config: dict[str, Any]) -> "OpenDDECore":
        from .core import OpenDDECore

        return OpenDDECore(config)


class OpenDDE(ModelWrapper):
    """Interface for OpenDDE structure prediction."""

    MODEL_SPEC = OPENDDE_SPEC

    def __init__(self, backend: str = "modal", device: str | None = None, config: dict | None = None) -> None:
        """Create an OpenDDE model wrapper."""
        super().__init__(backend=backend, device=device, config=config)
        self._initialize_backend_from_spec(self.MODEL_SPEC, backend=backend, device=device, config=config)

    def fold(self, sequences: str | Sequence[str], options: dict | None = None) -> "OpenDDEOutput":
        """Run OpenDDE for a single sequence entry.

        Keep one instance open and call ``fold()`` for each job to reuse the
        loaded model. The backend context owns the worker's lifetime.
        Use ``:`` inside a sequence string to define multiple chains.
        Pass ``options={"msa": [target_a3m_text, None]}`` for an
        unpaired target alignment and a single-sequence binder; every ``msa`` entry is
        that chain's unpaired MSA, so cross-chain pairing is not available for a heteromer.
        Pass ``options={"templates": {name: mmcif_text}}`` (with ``templates_chain``
        selecting the chain) to supply mmCIF templates; the other chains get none. Results retain
        numeric seeds and within-seed confidence ranks. Request ``pae``,
        ``token_chain_ids`` and ``token_res_ids`` to compute interface scores.
        Set ``config={"optimization": "exact" | "fast"}`` at initialization to
        run the Anthropic kit kernels (A100/H100/H200 only).
        """
        validated_sequences = [sequences] if isinstance(sequences, str) else list(sequences)
        if len(validated_sequences) != 1:
            raise ValueError(
                "OpenDDE currently supports exactly one top-level sequence per call; use ':' to join chains."
            )
        if options is not None:
            # Refuse on the client, before a remote backend starts a container and loads weights; the core
            # refuses the same keys again for callers that reach it directly.
            static_keys = self.MODEL_SPEC.contract.static_config_keys & set(options)
            if static_keys:
                raise ValueError(
                    "The following config keys can only be set at initialization and cannot be overridden per-call: "
                    f"{sorted(static_keys)}"
                )
        return self._call_backend_method("fold", sequences, options=options)
