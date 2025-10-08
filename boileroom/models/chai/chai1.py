
import os
import logging
import numpy as np
import modal
import json

from dataclasses import dataclass
from typing import Optional, Any, Union, Sequence


from ...backend.modal import ModalBackend, app
from ...images import chai_image
from ...base import StructurePrediction, PredictionMetadata, FoldingAlgorithm, ModelWrapper
from ...images.volumes import model_weights
from ...utils import MODEL_DIR, MINUTES

with chai_image.imports():
    import chai_lab


# Set up basic logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

############################################################
# CORE ALGORITHM
############################################################


@dataclass
class Chai1Output(StructurePrediction):
    """Output from Chai-1 prediction including all model outputs."""
    # Required by StructurePrediction protocol
    positions: np.ndarray  # (model_layer, batch_size, residue, atom=14, xyz=3)
    metadata: PredictionMetadata



class Chai1Core(FoldingAlgorithm):
    """Chai-1 protein structure prediction model."""
    DEFAULT_CONFIG = {}

    def __init__(self, config: dict = {}) -> None:
        """Initialize Chai-1."""
        super().__init__(config)
        self.metadata = self._initialize_metadata(
            model_name="Chai-1",
            model_version="v0.6.1",
        )
        self.model_dir: Optional[str] = os.environ.get("MODEL_DIR", MODEL_DIR)
        self.model: Optional[Any] = None # TODO: figure out the type

    
    def _initialize(self) -> None:
        """Initialize Chai-1."""
        self._load()
    
    def _load(self) -> None:
        """Load Chai-1."""
        self.model = chai_lab.Chai1(self.model_dir)
        self.model.eval()
        self.ready = True
    
    def fold(self, sequences: Union[str, Sequence[str]]) -> Chai1Output:
        """Fold protein sequences."""
        return self.model.fold(sequences)



############################################################
# MODAL BACKEND
############################################################
@app.cls(
    image=chai_image,
    gpu="T4",
    timeout=20 * MINUTES,
    scaledown_window=10 * MINUTES,
    volumes={MODEL_DIR: model_weights}, # TODO: somehow link this to what Chai-1 actually uses
)
class ModalChai1:
    """
    Modal-specific wrapper around `ESMFoldCore`.
    """

    config: bytes = modal.parameter(default=b"{}")

    @modal.enter()
    def _initialize(self) -> None:
        self._core = Chai1Core(json.loads(self.config.decode("utf-8")))
        self._core._initialize()

    @modal.method()
    def fold(self, sequences: Union[str, Sequence[str]]) -> Chai1Output:
        return self._core.fold(sequences)

############################################################
# HIGH-LEVEL INTERFACE
############################################################

class Chai1(ModelWrapper):
    """
    Interface for Chai-1 protein structure prediction model.
    # TODO: This is the user-facing interface. It should give all the relevant details possible.
    # with proper documentation.
    """

    def __init__(self, backend: str = "modal", device: Optional[str] = None, config: Optional[dict] = None) -> None:
        if config is None:
            config = {}
        self.config = config
        self.device = device
        if backend == "modal":
            self._backend = ModalBackend(ModalChai1, config, device=device)
        else:
            raise ValueError(f"Backend {backend} not supported")
        self._backend.start()

    def fold(self, sequences: Union[str, Sequence[str]]) -> Chai1Output:
        if isinstance(self._backend, ModalBackend):
            backend_model = self._backend.get_model()
            return backend_model.fold.remote(sequences)
        else:
            raise ValueError(f"Backend {self._backend} not supported yet.")
