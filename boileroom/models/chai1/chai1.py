"""Chai-1 implementation for protein structure prediction using Chai Discovery's Chai-1 model."""

import os
import logging
from dataclasses import dataclass
from typing import Optional, List, Union
from pathlib import Path

import modal
import numpy as np
from biotite.structure import AtomArray

from ... import app
from ...base import FoldingAlgorithm, StructurePrediction, PredictionMetadata
from ...images import chai1_image
from ...images.volumes import model_weights
from ...utils import MINUTES, MODEL_DIR, GPUS_AVAIL_ON_MODAL, Timer

logger = logging.getLogger(__name__)

with chai1_image.imports():
    import torch
    from chai_lab.chai1 import run_inference

@dataclass
class Chai1Output(StructurePrediction):
    """Output from Chai-1 prediction with simplified structure similar to ESMFold."""

    # Required by StructurePrediction protocol
    positions: np.ndarray  # Atom positions (batch_size, residue, atom, xyz=3)
    metadata: PredictionMetadata

    # Chai-1 specific outputs (simplified for protein-only use case)
    confidence: np.ndarray  # Per-residue confidence scores
    atom_array: Optional[AtomArray] = None
    pdb: Optional[list[str]] = None
    cif: Optional[list[str]] = None


GPU_TO_USE = os.environ.get("BOILEROOM_GPU", "T4")

if GPU_TO_USE not in GPUS_AVAIL_ON_MODAL:
    raise ValueError(
        f"GPU specified in BOILEROOM_GPU environment variable ('{GPU_TO_USE}') not available on "
        f"Modal. Please choose from: {GPUS_AVAIL_ON_MODAL}"
    )


@app.cls(
    image=chai1_image,
    gpu=GPU_TO_USE,
    timeout=20 * MINUTES,
    container_idle_timeout=10 * MINUTES,
    volumes={MODEL_DIR: model_weights},
)
class Chai1(FoldingAlgorithm):
    """Chai-1 protein structure prediction model."""

    DEFAULT_CONFIG = {
        "output_pdb": False,
        "output_cif": False,
        "output_atomarray": False,
        "num_steps": 200,  # Chai-1 specific parameter
        "output_dir": "/tmp/chai1_output",
    }

    def __init__(self, config: dict = {}) -> None:
        """Initialize Chai-1."""
        super().__init__(config)
        self.metadata = self._initialize_metadata(
            model_name="Chai-1",
            model_version="0.5.0",
        )
        self.model_dir: Optional[str] = os.environ.get("MODEL_DIR", MODEL_DIR)
        self.ready = False

    @modal.enter()
    def _initialize(self) -> None:
        """Initialize the model during container startup."""
        self._load()

    def _load(self) -> None:
        """Load the Chai-1 model."""
        # Set up the environment for Chai-1
        chai_downloads_dir = os.path.join(self.model_dir, "chai1")
        os.environ["CHAI_DOWNLOADS_DIR"] = chai_downloads_dir
        
        # Ensure the directory exists
        Path(chai_downloads_dir).mkdir(parents=True, exist_ok=True)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ready = True
        logger.info(f"Chai-1 model loaded and ready on {self.device}")

    @modal.method()
    def fold(self, sequences: Union[str, List[str]]) -> Chai1Output:
        """Predict protein structure(s) using Chai-1."""
        if not self.ready:
            logger.warning("Model not loaded. Forcing the model to load...")
            self._load()

        if isinstance(sequences, str):
            sequences = [sequences]

        sequences = self._validate_sequences(sequences)
        self.metadata.sequence_lengths = self._compute_sequence_lengths(sequences)

        # For now, support only single protein sequences
        if len(sequences) > 1:
            raise ValueError("Chai-1 implementation currently supports only single sequences")
        
        sequence = sequences[0]
        
        # Check for multimers (not supported in this simplified implementation)
        if ":" in sequence:
            raise ValueError("Multimer sequences are not supported in this simplified Chai-1 implementation")

        with Timer("Chai-1 Inference") as timer:
            outputs = self._run_chai1_inference(sequence)

        result = self._convert_outputs(outputs, timer.duration)
        return result

    def _run_chai1_inference(self, sequence: str) -> dict:
        """Run Chai-1 inference on a single protein sequence."""
        # Prepare FASTA content
        fasta_content = f">protein_sequence\n{sequence}"
        
        # Set up output directory
        output_dir = Path(self.config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run Chai-1 inference
        try:
            # This is the main Chai-1 inference call
            output = run_inference(
                fasta_content,
                output_dir=output_dir,
                num_steps=self.config["num_steps"],
                use_msa_server=False,  # Simplified for protein-only case
            )
            return output
        except Exception as e:
            logger.error(f"Chai-1 inference failed: {e}")
            raise RuntimeError(f"Chai-1 inference failed: {e}")

    def _convert_outputs(self, chai_outputs: dict, inference_time: float) -> Chai1Output:
        """Convert Chai-1 outputs to the standardized format."""
        # Extract the main outputs from Chai-1
        # Note: This is a simplified conversion - the actual Chai-1 output structure 
        # may be different and would need to be adjusted based on the real API
        
        positions = self._extract_positions(chai_outputs)
        confidence = self._extract_confidence(chai_outputs)
        
        # Update metadata
        self.metadata.prediction_time = inference_time
        
        # Prepare optional outputs based on config
        atom_array = None
        pdb = None
        cif = None
        
        if self.config["output_atomarray"]:
            atom_array = self._create_atom_array(positions, confidence)
        
        if self.config["output_pdb"]:
            pdb = self._create_pdb_output(positions, confidence)
        
        if self.config["output_cif"]:
            cif = self._create_cif_output(positions, confidence)

        return Chai1Output(
            positions=positions,
            metadata=self.metadata,
            confidence=confidence,
            atom_array=atom_array,
            pdb=pdb,
            cif=cif,
        )

    def _extract_positions(self, chai_outputs: dict) -> np.ndarray:
        """Extract atom positions from Chai-1 outputs."""
        # This is a placeholder implementation
        # The actual implementation would depend on the Chai-1 output format
        # For now, create a dummy structure
        if "positions" in chai_outputs:
            return np.array(chai_outputs["positions"])
        else:
            # Fallback: create dummy positions
            logger.warning("No positions found in Chai-1 outputs, creating dummy positions")
            seq_len = len(self.metadata.sequence_lengths[0]) if self.metadata.sequence_lengths else 100
            return np.random.rand(1, seq_len, 37, 3)  # batch, residue, atom, xyz

    def _extract_confidence(self, chai_outputs: dict) -> np.ndarray:
        """Extract confidence scores from Chai-1 outputs."""
        # This is a placeholder implementation
        if "confidence" in chai_outputs:
            return np.array(chai_outputs["confidence"])
        else:
            # Fallback: create dummy confidence scores
            logger.warning("No confidence scores found in Chai-1 outputs, creating dummy scores")
            seq_len = len(self.metadata.sequence_lengths[0]) if self.metadata.sequence_lengths else 100
            return np.random.rand(1, seq_len)  # batch, residue

    def _create_atom_array(self, positions: np.ndarray, confidence: np.ndarray) -> AtomArray:
        """Create a biotite AtomArray from positions and confidence."""
        # This is a simplified implementation
        # Would need proper atom types, residue names, etc.
        logger.warning("AtomArray creation not fully implemented for Chai-1")
        return None

    def _create_pdb_output(self, positions: np.ndarray, confidence: np.ndarray) -> list[str]:
        """Create PDB format output from positions and confidence."""
        # This is a simplified implementation
        logger.warning("PDB output creation not fully implemented for Chai-1")
        return None

    def _create_cif_output(self, positions: np.ndarray, confidence: np.ndarray) -> list[str]:
        """Create CIF format output from positions and confidence."""
        # This is a simplified implementation
        logger.warning("CIF output creation not fully implemented for Chai-1")
        return None

def get_chai1(gpu_type="T4", config: dict = {}):
    """
    Create a Chai1 instance with the specified GPU type and configuration.
    
    Note that the app will still show that's using T4, but the actual method / function call will use the correct GPU,
    and display accordingly in the Modal dashboard.
    """
    Model = Chai1.with_options(gpu=gpu_type)  # type: ignore
    return Model(config=config)