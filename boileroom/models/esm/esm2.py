import json
import logging
from typing import TYPE_CHECKING, Optional, Sequence, Union

import modal

from ...base import ModelWrapper
from ...backend import LocalBackend, ModalBackend
from ...backend.base import Backend
from ...backend.modal import app
from ...utils import MINUTES, MODAL_MODEL_DIR

from .image import esm_image
from ...images.volumes import model_weights

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
    def embed(self, sequences: Union[str, Sequence[str]], options: Optional[dict] = None) -> "ESM2Output":
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

    def __init__(
        self,
        backend: str = "modal",
        device: Optional[str] = None,
        config: Optional[dict] = None,
    ) -> None:
        """Initialize the ESM2 high-level interface and start the selected backend.

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
            Device identifier for model execution (for example "cuda:0" or "cpu").
        config : Optional[dict]
            Configuration passed to the backend and underlying model.

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
            backend_instance = ModalBackend(ModalESM2, config, device=device)
        elif backend == "local":
            from .core import ESM2Core

            backend_instance = LocalBackend(ESM2Core, config, device=device)
        elif backend in ("conda", "mamba", "micromamba"):
            from pathlib import Path
            from ...backend.conda import CondaBackend

            environment_yml = Path(__file__).parent / "environment.yml"
            # Pass Core class as string path to avoid importing it in main process
            # This keeps dependencies completely independent between Boiler Room and conda servers
            core_class_path = "boileroom.models.esm.core.ESM2Core"
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
            core_class_path = "boileroom.models.esm.core.ESM2Core"
            image_uri = "docker://docker.io/jakublala/boileroom-esm:latest"
            backend_instance = ApptainerBackend(core_class_path, image_uri, config or {}, device=device)
        else:
            raise ValueError(f"Backend {backend} not supported")
        self._backend = backend_instance
        self._backend.start()

    def embed(self, sequences: Union[str, Sequence[str]], options: Optional[dict] = None) -> "ESM2Output":
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
