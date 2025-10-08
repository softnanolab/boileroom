
import os
import logging
import numpy as np
import modal
import json

from dataclasses import dataclass
from typing import Optional, Any, Union, Sequence


from ...backend.modal import ModalBackend, app
from ...images import boltz_image
from ...base import StructurePrediction, PredictionMetadata, FoldingAlgorithm, ModelWrapper
from ...images.volumes import model_weights
from ...utils import MODEL_DIR, MINUTES

with boltz_image.imports():
    # TODO: import the Boltz-2 model
    pass

# Set up basic logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

############################################################
# CORE ALGORITHM
############################################################


@dataclass
class Boltz2Output(StructurePrediction):
    """Output from Boltz-2 prediction including all model outputs."""
    # Required by StructurePrediction protocol
    positions: np.ndarray  # (model_layer, batch_size, residue, atom=14, xyz=3)
    metadata: PredictionMetadata



class Boltz2Core(FoldingAlgorithm):
    """Boltz-2 protein structure prediction model."""
    DEFAULT_CONFIG = {}

    def __init__(self, config: dict = {}) -> None:
        """Initialize Boltz-2."""
        super().__init__(config)
        self.metadata = self._initialize_metadata(
            model_name="Boltz-2",
            # TODO: add the version
        )
        self.model_dir: Optional[str] = os.environ.get("MODEL_DIR", MODEL_DIR)
        self.model: Optional[Any] = None # TODO: figure out the type

    
    def _initialize(self) -> None:
        """Initialize Boltz-2."""
        self._load()
    
    def _load(self) -> None:
        """Load Boltz-2."""
        raise NotImplementedError
        # self.model = boltz_lab.Boltz2(self.model_dir)
        self.model.eval()
        self.ready = True
    
    def fold(self, sequences: Union[str, Sequence[str]]) -> Boltz2Output:
        """Fold protein sequences."""
        raise NotImplementedError
        # return self.model.fold(sequences)



############################################################
# MODAL BACKEND
############################################################
@app.cls(
    image=boltz_image,
    gpu="T4",
    timeout=20 * MINUTES,
    scaledown_window=10 * MINUTES,
    volumes={MODEL_DIR: model_weights}, # TODO: somehow link this to what Boltz-2 actually uses
)
class ModalBoltz2:
    """
    Modal-specific wrapper around `Boltz2Core`.
    """

    config: bytes = modal.parameter(default=b"{}")

    @modal.enter()
    def _initialize(self) -> None:
        self._core = Boltz2Core(json.loads(self.config.decode("utf-8")))
        self._core._initialize()

    @modal.method()
    def fold(self, sequences: Union[str, Sequence[str]]) -> Boltz2Output:
        return self._core.fold(sequences)

############################################################
# HIGH-LEVEL INTERFACE
############################################################

class Boltz2(ModelWrapper):
    """
    Interface for Boltz-2 protein structure prediction model.
    # TODO: This is the user-facing interface. It should give all the relevant details possible.
    # with proper documentation.
    """

    def __init__(self, backend: str = "modal", device: Optional[str] = None, config: Optional[dict] = None) -> None:
        if config is None:
            config = {}
        self.config = config
        self.device = device
        if backend == "modal":
            self._backend = ModalBackend(ModalBoltz2, config, device=device)
        else:
            raise ValueError(f"Backend {backend} not supported")
        self._backend.start()

    def fold(self, sequences: Union[str, Sequence[str]]) -> Boltz2Output:
        if isinstance(self._backend, ModalBackend):
            backend_model = self._backend.get_model()
            return backend_model.fold.remote(sequences)
        else:
            raise ValueError(f"Backend {self._backend} not supported yet.")
