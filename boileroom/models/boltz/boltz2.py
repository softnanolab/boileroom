import json
import logging
from collections.abc import Sequence

import modal

from ...backend.modal import app
from ...base import ModelWrapper
from ...images.volumes import model_weights
from ...utils import MINUTES, MODAL_MODEL_DIR
from ..registry import BOLTZ2_SPEC
from .image import boltz_image
from .types import Boltz2Output

logger = logging.getLogger(__name__)


############################################################
# MODAL BACKEND
############################################################
@app.cls(
    image=boltz_image,
    gpu="T4",
    timeout=20 * MINUTES,
    scaledown_window=10 * MINUTES,
    volumes={
        MODAL_MODEL_DIR: model_weights
    },  # TODO: Volume is shared with MSA cache. Consider renaming volume to something more generic like "boileroom-data" in the future to reflect it's not just for model weights but all boileroom persistent data (models, MSA cache, etc.)
)
class ModalBoltz2:
    config: bytes = modal.parameter(default=b"{}")

    @modal.enter()
    def _initialize(self) -> None:
        from .core import Boltz2Core

        self._core = Boltz2Core(json.loads(self.config.decode("utf-8")))
        self._core._initialize()

    @modal.method()
    def fold(self, sequences: str | Sequence[str], options: dict | None = None) -> "Boltz2Output":
        return self._core.fold(sequences, options=options)


############################################################
# HIGH-LEVEL INTERFACE
############################################################


class Boltz2(ModelWrapper):
    """
    Interface for Boltz-2 protein structure prediction model.
    # TODO: This is the user-facing interface. It should give all the relevant details possible.
    # with proper documentation.

    # TODO: no support for multiple samples
    # TODO: no support for constraints
    # TODO: no support for templates
    # TODO: rewrite the output to be a list of Boltz2Output objects, not a single object with many entries from a batch
    """

    MODEL_SPEC = BOLTZ2_SPEC

    def __init__(self, backend: str = "modal", device: str | None = None, config: dict | None = None) -> None:
        """Create a Boltz-2 model wrapper that selects and starts a backend.

        Parameters
        ----------
        backend : str
            Backend type to use. Supported values:
            - "modal": Use Modal backend (default)
            - "apptainer": Use Apptainer backend (requires Apptainer installed)
        device : Optional[str]
            Device hint passed to the chosen backend (e.g., "cuda:0" or "cpu"); may be ignored by some backends.
        config : Optional[dict]
            Runtime configuration forwarded to the backend and underlying Boltz-2 core.

        Raises
        ------
        ValueError
            If an unsupported backend string is provided.
        """
        super().__init__(backend=backend, device=device, config=config)
        self._initialize_backend_from_spec(self.MODEL_SPEC, backend=backend, device=device, config=config)

    def fold(self, sequences: str | Sequence[str], options: dict | None = None) -> "Boltz2Output":
        """Run the Boltz-2 folding workflow for one or more protein sequences.

        Parameters
        ----------
        sequences : str | Sequence[str]
            A single amino-acid sequence or an iterable of sequences representing one or more chains/targets.
        options : dict, optional
            Per-call configuration overrides for the run, such as `seed`, `msa_cache_enabled`, MSA server options, `num_workers`, or `include_fields`. Load-bound settings such as sampling, diffusion, and MSA module parameters must be configured when the model is constructed.

        Returns
        -------
        Boltz2Output
            Prediction results and associated metadata, including per-sample atom arrays, confidence metrics (plddt, pae, pde), and optional PDB/MMCIF strings.
        """
        return self._call_backend_method("fold", sequences, options=options)
