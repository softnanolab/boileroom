import json
import logging
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Optional, Sequence, Union

import modal
import numpy as np

from ...base import (
    FoldingAlgorithm,
    ModelWrapper,
    PredictionMetadata,
    StructurePrediction,
)
from ...backend import LocalBackend, ModalBackend
from ...backend.modal import app
from ...images import chai_image
from ...images.volumes import model_weights
from ...utils import MODEL_DIR, MINUTES

with chai_image.imports():
    import torch
    from biotite.structure.io.pdbx import CIFFile, get_structure

    from chai_lab.chai1 import run_inference, StructureCandidates
    from chai_lab.utils.paths import downloads_path

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

    DEFAULT_CONFIG: dict[str, Any] = {
        "device": "cuda:0",
        "num_trunk_recycles": 1,
        "num_diffn_timesteps": 20,
        "num_diffn_samples": 1,
        "num_trunk_samples": 1,
        "use_esm_embeddings": False,
        "use_msa_server": False,
        "use_templates_server": False,
    }

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config or {})
        self.metadata = self._initialize_metadata(
            model_name="Chai-1",
            model_version="v0.6.1",
        )
        self.model_dir: Optional[str] = self.config.get("model_dir", MODEL_DIR)
        self._device: torch.device | None = None
        self._trunk: Any | None = None

    def _initialize(self) -> None:
        self._load()

    def _load(self) -> None:
        self._device = self._resolve_device()
        self.ready = True

    def fold(self, sequences: Union[str, Sequence[str]], config: Optional[dict] = None) -> Chai1Output:
        # TODO: This config is not implemented across all models; and will require further synchronization.
        # TODO: ADD msa_directory: Path | None = None,
        # TODO: ADD constraint_path: Path | None = None,
        # TODO: ADD template_hits_path: Path | None = None,

        if config is not None:
            self.config.update(config)

        validated_sequences = self._validate_sequences(sequences)
        self.metadata.sequence_lengths = self._compute_sequence_lengths(validated_sequences)
        with TemporaryDirectory() as buffer_dir:
            fasta_path = self._write_fasta(validated_sequences, Path(buffer_dir))
            candidate = run_inference(
                fasta_file=fasta_path,
                output_dir=Path(buffer_dir) / "outputs",
                num_trunk_samples=self.config["num_trunk_samples"],
                num_trunk_recycles=self.config["num_trunk_recycles"],
                num_diffn_timesteps=self.config["num_diffn_timesteps"],
                num_diffn_samples=self.config["num_diffn_samples"],
                use_esm_embeddings=self.config["use_esm_embeddings"],
                use_msa_server=self.config["use_msa_server"],
                use_templates_server=self.config["use_templates_server"],
                device=str(self._device),
                low_memory=False,
            )
            positions = self._extract_positions(candidate)

        return Chai1Output(positions=positions, metadata=self.metadata)

    def _resolve_device(self) -> torch.device:
        requested = self.config.get("device")
        if requested is not None:
            return torch.device(requested)
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")
    
    def _write_fasta(self, sequences: list[str], buffer_dir: Path) -> Path:
        fasta_path = buffer_dir / "input.fasta"
        entries = [
            f">protein|name=chain_{index}\n{sequence}" for index, sequence in enumerate(sequences)
        ]
        fasta_path.write_text("\n".join(entries))
        return fasta_path

    def _extract_positions(self, candidates: "StructureCandidates") -> np.ndarray:
        cif_path = candidates.cif_paths[0]
        return self._positions_from_cif(cif_path)

    def _positions_from_cif(self, cif_path: Path) -> np.ndarray:
        cif_file = CIFFile.read(str(cif_path))
        structure = get_structure(cif_file)
        model = structure[0]
        coords = []
        for chain in model:
            atom_coords = [atom.coord for atom in chain]
            coords.append(np.array(atom_coords, dtype=np.float32))
        flattened = np.concatenate(coords, axis=0)
        return flattened


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
    Modal-specific wrapper around `Chai1Core`.
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
        elif backend == "local":
            self._backend = LocalBackend(Chai1Core, config, device=device)
        else:
            raise ValueError(f"Backend {backend} not supported")
        self._backend.start()

    def fold(self, sequences: Union[str, Sequence[str]]) -> Chai1Output:
        return self._call_backend_method("fold", sequences)
