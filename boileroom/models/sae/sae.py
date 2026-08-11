"""Public and Modal wrappers for the ESM-C sparse-autoencoder feature model."""

import json
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import modal

from ...backend.modal import get_modal_app
from ...base import ModelWrapper
from ...images.volumes import model_weights
from ...utils import MINUTES, MODAL_MODEL_DIR
from ..registry import SAE_SPEC
from .image import sae_image

if TYPE_CHECKING:
    from .types import SAEFeaturesOutput

logger = logging.getLogger(__name__)
app = get_modal_app("sae")


@app.cls(
    image=sae_image,
    gpu="T4",
    timeout=20 * MINUTES,
    scaledown_window=10 * MINUTES,
    volumes={MODAL_MODEL_DIR: model_weights},
)
class ModalSAE:
    """Modal wrapper around :class:`~boileroom.models.sae.core.SAECore`."""

    config: bytes = modal.parameter(default=b"{}")

    @modal.enter()
    def _initialize(self) -> None:
        from .core import SAECore

        self._core = SAECore(config=json.loads(self.config.decode("utf-8")))
        self._core._initialize()

    @modal.method()
    def embed(self, sequences: str | Sequence[str], options: dict | None = None) -> "SAEFeaturesOutput":
        if getattr(self, "_core", None) is None:
            raise RuntimeError("ModalSAE has not been initialized")
        return self._core.embed(sequences, options=options)


class SAE(ModelWrapper):
    """Sequence-to-SAE-features model built on ESM-C representations.

    The model runs ESM-C to obtain per-residue hidden states at a configured
    transformer layer, then projects them through a trained sparse autoencoder to
    a high-dimensional, sparse, interpretable feature space. Each protein is
    summarized by max-pooling each feature across its residues.

    Parameters
    ----------
    backend : str
        BoilerRoom backend, normally ``"modal"`` or ``"apptainer"``.
    device : str | None
        Optional device passed to BoilerRoom.
    config : dict | None
        Configuration options; see
        :class:`~boileroom.models.sae.core.SAECore` for supported keys
        (``esmc_model_name``, ``sae_repo_id``, ``sae_layer``, ``num_features``,
        ``k``, ``activation``, ``normalize_features``, ``include_per_residue``).

    Examples
    --------
    >>> model = SAE(config={"esmc_model_name": "esmc_600m"})  # doctest: +SKIP
    >>> result = model.get_features("MKTAYIAKQR")             # doctest: +SKIP
    >>> result.pooled_features.shape                          # doctest: +SKIP
    (1, 16384)
    """

    MODEL_SPEC = SAE_SPEC

    def __init__(self, backend: str = "modal", device: str | None = None, config: dict | None = None) -> None:
        super().__init__(backend=backend, device=device, config=config)
        self._initialize_backend_from_spec(self.MODEL_SPEC, backend=backend, device=device, config=config)

    def embed(self, sequences: str | Sequence[str], options: dict | None = None) -> "SAEFeaturesOutput":
        """Compute SAE features for the given sequence(s)."""
        if options is not None:
            from .core import SAECore

            conflicting_keys = sorted(set(options) & SAECore.STATIC_CONFIG_KEYS)
            if conflicting_keys:
                raise ValueError(
                    "The following config keys can only be set at initialization and cannot be "
                    f"overridden per-call: {conflicting_keys}"
                )
        return self._call_backend_method("embed", sequences, options=options)

    def get_features(self, sequences: str | Sequence[str], options: dict | None = None) -> "SAEFeaturesOutput":
        """Alias for :meth:`embed` that reads naturally for feature extraction."""
        return self.embed(sequences, options=options)
