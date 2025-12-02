import json
import logging
from typing import TYPE_CHECKING, Optional, Sequence, Union

import modal

from ...backend import LocalBackend, ModalBackend
from ...backend.base import Backend
from ...backend.modal import app
from .image import boltz_image
from ...base import ModelWrapper
from ...images.volumes import model_weights
from ...utils import MINUTES, MODAL_MODEL_DIR

if TYPE_CHECKING:
    pass

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
    def fold(self, sequences: Union[str, Sequence[str]], options: Optional[dict] = None) -> "Boltz2Output":
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

    def __init__(self, backend: str = "modal", device: Optional[str] = None, config: Optional[dict] = None) -> None:
        """Create a Boltz-2 model wrapper that selects and starts a backend.

        Parameters
        ----------
        backend : str
            Backend type to use. Supported values:
            - "modal": Use Modal backend (default)
            - "local": Use local backend (requires dependencies in current environment)
            - "conda": Use conda backend with auto-detection (micromamba > mamba > conda)
            - "mamba": Use mamba explicitly
            - "micromamba": Use micromamba explicitly
            - "apptainer": Use Apptainer backend (requires Apptainer installed)
        device : Optional[str]
            Device hint passed to the chosen backend (e.g., "cuda:0" or "cpu"); may be ignored by some backends.
        config : Optional[dict]
            Runtime configuration forwarded to the backend and underlying Boltz-2 core.

        Raises
        ------
        ValueError
            If an unsupported backend string is provided, or if conda backend is
            requested but no compatible tool (conda/mamba/micromamba) is available.
        """
        if config is None:
            config = {}
        self.config = config
        self.device = device
        backend_instance: Backend
        if backend == "modal":
            backend_instance = ModalBackend(ModalBoltz2, config, device=device)
        elif backend == "local":
            from .core import Boltz2Core

            backend_instance = LocalBackend(Boltz2Core, config, device=device)
        elif backend in ("conda", "mamba", "micromamba"):
            from pathlib import Path
            from ...backend.conda import CondaBackend

            environment_yml = Path(__file__).parent / "environment.yml"
            # Pass Core class as string path to avoid importing it in main process
            # This keeps dependencies completely independent between Boiler Room and conda servers
            core_class_path = "boileroom.models.boltz.core.Boltz2Core"
            # Pass backend string directly as runner_command
            backend_instance = CondaBackend(
                core_class_path,
                config or {},
                device=device,
                environment_yml_path=environment_yml,
                runner_command=backend,
            )
        elif backend == "apptainer":
            from ...backend.apptainer import ApptainerBackend

            # Pass Core class as string path to avoid importing it in main process
            core_class_path = "boileroom.models.boltz.core.Boltz2Core"
            
            # HACK
            image_uri = "docker://docker.io/jakublala/boileroom-boltz:dev"
            backend_instance = ApptainerBackend(core_class_path, image_uri, config or {}, device=device)
        else:
            raise ValueError(f"Backend {backend} not supported")
        self._backend = backend_instance
        self._backend.start()

    def fold(self, sequences: Union[str, Sequence[str]], options: Optional[dict] = None) -> "Boltz2Output":
        """Run the Boltz-2 folding workflow for one or more protein sequences.

        Parameters
        ----------
        sequences : str | Sequence[str]
            A single amino-acid sequence or an iterable of sequences representing one or more chains/targets.
        options : dict, optional
            Per-call configuration overrides for the run (e.g., sampling/recycling settings, device selection, include_fields). Keys mirror those in the core configuration.

        Returns
        -------
        Boltz2Output
            Prediction results and associated metadata, including per-sample atom arrays, confidence metrics (plddt, pae, pde), and optional PDB/MMCIF strings.
        """
        return self._call_backend_method("fold", sequences, options=options)
