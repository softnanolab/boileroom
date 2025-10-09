import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Mapping, Optional, Sequence, Union

import modal
import numpy as np
from biotite.structure import AtomArray

from ...base import (
    FoldingAlgorithm,
    ModelWrapper,
    PredictionMetadata,
    StructurePrediction,
)
from ...backend import LocalBackend, ModalBackend
from ...backend.modal import app
from .image import chai_image
from ...images.volumes import model_weights
from ...utils import MODEL_DIR, MINUTES, Timer

with chai_image.imports():
    import torch
    from biotite.structure.io.pdbx import CIFFile, get_structure
    from chai_lab.chai1 import run_inference, StructureCandidates

# Set up basic logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

############################################################
# CORE ALGORITHM
############################################################


@dataclass
class Chai1Output(StructurePrediction):
    """Output from Chai-1 prediction including all model outputs."""
    positions: np.ndarray  # (batch_size, residue, atom=14, xyz=3)
    metadata: PredictionMetadata
    
    pae: list[np.ndarray] = field(default_factory=list)
    pde: list[np.ndarray] = field(default_factory=list)
    plddt: list[np.ndarray] = field(default_factory=list)
    ptm: list[np.ndarray] = field(default_factory=list)
    iptm: list[np.ndarray] = field(default_factory=list)
    per_chain_iptm: list[np.ndarray] = field(default_factory=list)
    cif: Optional[list[str]] = None
    atom_array: Optional[AtomArray] = None



class Chai1Core(FoldingAlgorithm):
    """Chai-1 protein structure prediction model."""

    DEFAULT_CONFIG: dict[str, Any] = {
        "device": "cuda:0",
        "num_trunk_recycles": 3,
        "num_diffn_timesteps": 200,
        "num_diffn_samples": 5,
        "num_trunk_samples": 1,
        "use_esm_embeddings": False,
        "use_msa_server": False,
        "use_templates_server": False,
        "output_cif": False,
        "output_atomarray": False,
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
        # TODO: constraint path not tested properly
        # TODO: ADD template_hits_path: Path | None = None,

        if config is not None:
            self.config.update(config)

        validated_sequences = self._validate_sequences(sequences)
        self.metadata.sequence_lengths = self._compute_sequence_lengths(validated_sequences)

        with TemporaryDirectory() as buffer_dir:
            buffer_path = Path(buffer_dir)
            fasta_path = self._write_fasta(validated_sequences, buffer_path)
            constraint_path = self._write_constraint(buffer_path)
            output_dir = buffer_path / "outputs"
            output_dir.mkdir(parents=True, exist_ok=True)
            with Timer("Model Inference") as timer:
                candidate = run_inference(
                    fasta_file=fasta_path,
                    output_dir=output_dir,
                    constraint_path=constraint_path,
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
            output = self._convert_outputs(candidate, elapsed_time=timer.duration)
        
        return output

    def _write_constraint(self, buffer_path: Path) -> Optional[Path]:
        constraint_definition = self.config.get("constraint_path")
        if constraint_definition is None:
            return None

        constraint_path = buffer_path / "constraint.csv"

        if isinstance(constraint_definition, (str, Path)):
            source_path = Path(constraint_definition).expanduser().resolve()
            constraint_path.write_text(source_path.read_text())
            return constraint_path

        if not isinstance(constraint_definition, Sequence):
            raise TypeError("constraint_path must be a path or a sequence of restraint definitions")

        # Note: The model does not currently consume 'confidence' or 'min_distance_angstrom'.
        # They are included here for future-proofing and to keep the CSV schema stable.
        columns = [
            "restraint_id",
            "chainA",
            "res_idxA",
            "chainB",
            "res_idxB",
            "connection_type",
            "confidence",
            "min_distance_angstrom",
            "max_distance_angstrom",
            "comment",
        ]
        aliases = {
            "chainA": "chain_a",
            "res_idxA": "residue_token_a",
            "chainB": "chain_b",
            "res_idxB": "residue_token_b",
            "min_distance_angstrom": "minimum_distance_angstrom",
            "max_distance_angstrom": "maximum_distance_angstrom",
        }

        with constraint_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(columns)
            for entry in constraint_definition:
                if not isinstance(entry, Mapping):
                    raise TypeError("Each constraint must be provided as a mapping")

                row = []
                for column in columns:
                    source_key = aliases.get(column, column)
                    value = entry.get(source_key, "")
                    row.append("" if value is None else value)
                writer.writerow(row)

        return constraint_path

    def _convert_outputs(self, candidate: "StructureCandidates", elapsed_time: float) -> Chai1Output:
        pae, pde, plddt, ptm, iptm, per_chain_iptm = self._extract_confidence_metrics(candidate)
        positions, cif_string, atom_array = self._process_cif(candidate.cif_paths[0])
        self.metadata.prediction_time = elapsed_time

        return Chai1Output(
            positions=positions,
            metadata=self.metadata,
            pae=pae,
            pde=pde,
            plddt=plddt,
            ptm=ptm,
            iptm=iptm,
            per_chain_iptm=per_chain_iptm,
            cif=cif_string,
            atom_array=atom_array,
        )

    def _resolve_device(self) -> torch.device:
        requested = self.config.get("device")
        if requested is not None:
            return torch.device(requested)
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")
    
    def _write_fasta(self, sequences: list[str], buffer_dir: Path) -> Path:
        # assert that a single batch only
        assert len(sequences) == 1, "Chai-1 only supports a single batch for now."
        seqs = sequences[0].split(":") if ":" in sequences[0] else sequences[0]
        fasta_path = buffer_dir / "input.fasta"
        # TODO: naming of the chains should be synchronized with how ESMFold did that
        entries = [
            f">protein|name=chain_{index}\n{sequence}" for index, sequence in enumerate(seqs)
        ]
        fasta_path.write_text("\n".join(entries))
        return fasta_path

    def _process_cif(self, cif_path: Path) -> tuple[np.ndarray, Optional[str], AtomArray]:
        cif_file = CIFFile.read(str(cif_path))
        structure = get_structure(cif_file)
        coords = []
        for chain in structure:
            coords.append(np.array(chain.coord, dtype=np.float32))
        flattened = np.concatenate(coords, axis=0)
        
        cif_string = None
        if self.config["output_cif"]:
            with open(cif_path, "r") as f:
                cif_string = f.read()

        atom_array = None
        if self.config["output_atomarray"]:
            atom_array = structure
        
        return [flattened], [cif_string], [atom_array]

    def _extract_confidence_metrics(
        self,
        candidates: "StructureCandidates",
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        pae = self._collect_matrix(candidates, attribute_name="pae")
        pde = self._collect_matrix(candidates, attribute_name="pde")
        plddt = self._collect_matrix(candidates, attribute_name="plddt")

        ranking_data = candidates.ranking_data
        if len(ranking_data) != 1:
            logger.warning(f"Expected 1 ranking data, got {len(ranking_data)}. More not supported yet.")
            return pae, pde, plddt, None, None, None

        ptm = ranking_data[0].ptm_scores.complex_ptm
        iptm = ranking_data[0].ptm_scores.interface_ptm
        per_chain_iptm = ranking_data[0].ptm_scores.per_chain_pair_iptm
        return [pae], [pde], [plddt], [ptm], [iptm], [per_chain_iptm]

    def _collect_matrix(
        self,
        candidates: "StructureCandidates",
        attribute_name: str,
    ) -> list[np.ndarray]:
        matrices: list[np.ndarray] = []
        matrix_collection = getattr(candidates, attribute_name, None)
        if matrix_collection is None:
            return matrices

        for matrix in matrix_collection:
            matrices.append(np.asarray(matrix))
        return matrices


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
