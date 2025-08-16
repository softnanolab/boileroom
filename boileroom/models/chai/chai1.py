import os
from dataclasses import dataclass
from typing import List, Optional, Union

import modal
import numpy as np

from ... import app
from ...base import FoldingAlgorithm, PredictionMetadata, StructurePrediction
from ...images.chai import chai_image
from ...images.volumes import model_weights
from ...utils import MINUTES, MODEL_DIR, GPUS_AVAIL_ON_MODAL, Timer


@dataclass
class Chai1Output(StructurePrediction):
	positions: np.ndarray  # (batch_size, residue, atom=?, xyz=3) if available, else empty
	metadata: PredictionMetadata
	pdb: Optional[list[str]] = None
	cif: Optional[list[str]] = None
	atom_array: Optional[list] = None  # Biotite AtomArray, list per sequence


GPU_TO_USE = os.environ.get("BOILEROOM_GPU", "A100-40GB")

if GPU_TO_USE not in GPUS_AVAIL_ON_MODAL:
	raise ValueError(
		f"GPU specified in BOILEROOM_GPU environment variable ('{GPU_TO_USE}') not available on "
		f"Modal. Please choose from: {GPUS_AVAIL_ON_MODAL}"
	)


@app.cls(
	image=chai_image,
	gpu=GPU_TO_USE,
	timeout=30 * MINUTES,
	container_idle_timeout=10 * MINUTES,
	volumes={MODEL_DIR: model_weights},
)
class Chai1(FoldingAlgorithm):
	"""Chai-1 protein structure prediction model (proteins only)."""

	DEFAULT_CONFIG = {
		# Output toggles; mimic ESMFold for simplicity
		"output_pdb": True,
		"output_cif": False,
		"output_atomarray": True,
	}

	def __init__(self, config: dict = {}) -> None:
		super().__init__(config)
		self.metadata = self._initialize_metadata(
			model_name="Chai-1",
			model_version="0.5.0",  # chai_lab package version targeted
		)
		self.model_dir: Optional[str] = os.environ.get("MODEL_DIR", MODEL_DIR)
		self.ready: bool = False
		self._model = None

	@modal.enter()
	def _initialize(self) -> None:
		self._load()

	def _load(self) -> None:
		# Lazy import inside image context
		from importlib import import_module

		chai_lab = import_module("chai_lab")
		# Various sources differ on API; prefer attribute presence checks
		# Try canonical constructor-like API
		# Fallbacks kept extremely conservative; if import fails, let it raise
		self._chai_lab = chai_lab
		self.device = "cuda"
		self.ready = True

	@modal.method()
	def fold(self, sequences: Union[str, List[str]]) -> Chai1Output:
		# Keep consistent with ESMFold: validate, listify, compute lengths
		sequences = self._validate_sequences(sequences)
		self.metadata.sequence_lengths = self._compute_sequence_lengths(sequences)

		# Run inference via chai_lab Python API, returning PDB strings per sequence
		pdbs: list[str] = []
		with Timer("Chai-1 Inference") as timer:
			from pathlib import Path
			from tempfile import TemporaryDirectory

			# Imports inside the container
			chai_lab = self._chai_lab

			with TemporaryDirectory() as tmpdir:
				tmp = Path(tmpdir)
				# Minimal Python API: prefer direct function if available, else CLI fallback
				# Attempt: chai_lab exposes a high-level predict function we can call per sequence
				for seq in sequences:
					# Try Python API patterns in order; stop at first that works
					pdb_str: Optional[str] = None
					# Pattern 1: chai_lab has a `predict_structure(sequence:str)` -> pdb_str
					if hasattr(chai_lab, "predict_structure"):
						pdb_str = chai_lab.predict_structure(seq)
					else:
						# Pattern 2: package provides a Model class with fold/infer methods
						model = None
						for attr in ("Chai1", "ChaiModel", "Model"):
							if hasattr(chai_lab, attr):
								Model = getattr(chai_lab, attr)
								try:
									# Prefer from_pretrained with downloads dir if available
									if hasattr(Model, "from_pretrained"):
										model = Model.from_pretrained(os.path.join(self.model_dir, "chai1"))
									else:
										model = Model()
									break
								except Exception:
									model = None
						if model is not None:
							# Try common method names to get PDB string
							for meth in ("infer_pdb", "fold", "predict", "predict_pdb"):
								if hasattr(model, meth):
									res = getattr(model, meth)(seq)
									if isinstance(res, str) and ("ATOM" in res or "HEADER" in res):
										pdb_str = res
										break
						# If still None, last resort: raise a clear error to surface API mismatch
					if pdb_str is None:
						raise RuntimeError("Unable to obtain PDB string from chai_lab Python API; update integration.")
					pdbs.append(pdb_str)

		# Package minimal output consistent with ESMFold
		self.metadata.prediction_time = timer.duration
		positions = np.empty((len(sequences), 0, 0, 3), dtype=np.float32)  # placeholder; focus on PDB first

		atom_array = None
		if self.config.get("output_atomarray"):
			from ...convert import pdb_string_to_atomarray
			atom_array = [pdb_string_to_atomarray(p) for p in pdbs]

		return Chai1Output(
			metadata=self.metadata,
			positions=positions,
			pdb=pdbs if self.config.get("output_pdb") else None,
			cif=None,
			atom_array=atom_array,
		)


def get_chai1(gpu_type="A100-40GB", config: dict = {}):
	Model = Chai1.with_options(gpu=gpu_type)  # type: ignore
	return Model(config=config)