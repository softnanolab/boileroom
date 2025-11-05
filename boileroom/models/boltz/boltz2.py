
import os
import logging
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import dataclasses

import numpy as np
import modal

from dataclasses import dataclass
from typing import Optional, Any, Union, Sequence, List, Dict


from ...backend import LocalBackend, ModalBackend
from ...backend.modal import app
from .image import boltz_image
from ...base import StructurePrediction, PredictionMetadata, FoldingAlgorithm, ModelWrapper
from ...images.volumes import model_weights
from ...utils import MINUTES, MODAL_MODEL_DIR, Timer

with boltz_image.imports():
    import torch
    from pytorch_lightning import Trainer
    from boltz.main import (
        download_boltz2,
        process_inputs,
        Boltz2DiffusionParams,
        PairformerArgsV2,
        MSAModuleArgs,
        BoltzSteeringParams,
    )
    from boltz.data.types import Manifest
    from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
    from boltz.model.models.boltz2 import Boltz2 as Boltz2Model

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
    DEFAULT_CONFIG = {
        "device": "cuda:0",
        "use_msa_server": True,
        "msa_server_url": "https://api.colabfold.com",
        "msa_pairing_strategy": "greedy",
        "msa_server_username": None,
        "msa_server_password": None,
        "api_key_header": None,
        "api_key_value": None,
        "recycling_steps": 3,
        "sampling_steps": 200,
        "diffusion_samples": 1,
        "max_parallel_samples": 5,
        "step_scale": None,  # will default to 1.5 for boltz2 if None
        "output_pdb": False,
        "output_cif": False,
        "write_full_pae": False,
        "write_full_pde": False,
        "num_workers": 2,
        "override": False,
        "seed": None,
        "no_kernels": False,
        "write_embeddings": False,
        "max_msa_seqs": 8192,
        "subsample_msa": True,
        "num_subsampled_msa": 1024,
        # Paths
        "cache_dir": None,  # defaults to Path(MODAL_MODEL_DIR)/"boltz"
    }

    def __init__(self, config: dict = {}) -> None:
        """Initialize Boltz-2."""
        super().__init__(config)
        self.metadata = self._initialize_metadata(
            model_name="Boltz-2",
            model_version="conf",  # matching ckpt naming; refine if needed
        )
        self.model_dir: Optional[str] = os.environ.get("MODEL_DIR", MODAL_MODEL_DIR)
        self.model: Optional[Any] = None
        self._trainer: Optional[Any] = None

    
    def _initialize(self) -> None:
        """Initialize Boltz-2."""
        self._load()
    
    def _load(self) -> None:
        """Load Boltz-2 once and prepare a persistent Trainer."""
        # Resolve cache and weights
        cache_dir = (
            Path(self.config.get("cache_dir"))
            if self.config.get("cache_dir")
            else Path(self.model_dir).resolve() / "boltz"
        )
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Download boltz2 weights and resources (mols)
        download_boltz2(cache_dir)

        checkpoint = cache_dir / "boltz2_conf.ckpt"

        # Build default args mirroring CLI
        diffusion_params = Boltz2DiffusionParams()
        diffusion_params.step_scale = 1.5 if self.config.get("step_scale") is None else float(self.config["step_scale"])

        pairformer_args = PairformerArgsV2()
        msa_args = MSAModuleArgs(
            subsample_msa=bool(self.config.get("subsample_msa", True)),
            num_subsampled_msa=int(self.config.get("num_subsampled_msa", 1024)),
            use_paired_feature=True,
        )

        predict_args = {
            "recycling_steps": int(self.config.get("recycling_steps", 3)),
            "sampling_steps": int(self.config.get("sampling_steps", 200)),
            "diffusion_samples": int(self.config.get("diffusion_samples", 1)),
            "max_parallel_samples": self.config.get("max_parallel_samples", 5),
            "write_confidence_summary": True,
            "write_full_pae": bool(self.config.get("write_full_pae", False)),
            "write_full_pde": bool(self.config.get("write_full_pde", False)),
        }

        steering_args = BoltzSteeringParams()
        steering_args.fk_steering = False
        steering_args.physical_guidance_update = False

        # Load LightningModule
        self.model = Boltz2Model.load_from_checkpoint(
            checkpoint,
            strict=True,
            predict_args=predict_args,
            map_location="cpu",
            diffusion_process_args=dataclasses.asdict(diffusion_params),
            ema=False,
            use_kernels=not bool(self.config.get("no_kernels", False)),
            pairformer_args=dataclasses.asdict(pairformer_args),
            msa_args=dataclasses.asdict(msa_args),
            steering_args=dataclasses.asdict(steering_args),
        )
        self.model.eval()

        # TODO: unnecessary to be this safe, and just require a GPU to be available
        # Move to device
        device = self.config.get("device", "cuda:0")
        try:
            self.model = self.model.to(device)
        except Exception:
            # Fallback to CPU
            device = "cpu"
            self.model = self.model.to(device)

        # Prepare a persistent Trainer (no callbacks by default)
        precision = "bf16-mixed" if device != "cpu" else 32
        self._trainer = Trainer(
            default_root_dir=str(cache_dir),
            accelerator=("gpu" if device != "cpu" else "cpu"),
            devices=1,
            precision=precision,
        )

        self.ready = True

    def _sequences_to_fasta(self, sequences: List[str]) -> str:
        """Convert sequences (monomer or multimer) into Boltz FASTA schema for proteins only.

        Supports inputs like ["SEQ"] or ["A:B"] or ["A", "B"]. Chain IDs assigned A, B, C...
        """
        chains: List[str] = []
        for entry in sequences:
            parts = entry.split(":") if ":" in entry else [entry]
            chains.extend([p.strip() for p in parts if p.strip()])

        headers = []
        for idx, seq in enumerate(chains):
            chain_name = chr(65 + idx)  # A, B, C...
            # TODO: last part could/should be the MSA path, if it is cached/available
            headers.append(f">{chain_name}|protein|")
            headers.append(seq)
        return "\n".join(headers)

    def _prepare_inputs(self, fasta_text: str, work_dir: Path, cache_dir: Path) -> Dict[str, Any]:
        """Run preprocessing using Boltz's pipeline to produce a Manifest and processed dirs."""
        data_dir = work_dir / "inputs"
        data_dir.mkdir(parents=True, exist_ok=True)
        fasta_path = data_dir / "input.fasta"
        fasta_path.write_text(fasta_text)

        # TODO: this might be completely ununsed, but is required, double check later
        out_dir = work_dir / f"boltz_results_{fasta_path.stem}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Ensure boltz2 resources are present
        download_boltz2(cache_dir)
        mol_dir = cache_dir / "mols"
        ccd_path = cache_dir / "ccd.pkl"  # unused for boltz2 but required by signature

        # MSA server config
        use_msa_server = bool(self.config.get("use_msa_server", True))
        msa_server_url = str(self.config.get("msa_server_url", "https://api.colabfold.com"))
        msa_pairing_strategy = str(self.config.get("msa_pairing_strategy", "greedy"))
        # Optional auth headers/params are intentionally omitted for broader compatibility

        # Process inputs (writes processed/manifest.json and NPZs)
        process_inputs(
            data=[fasta_path],
            out_dir=out_dir,
            ccd_path=ccd_path,
            mol_dir=mol_dir,
            use_msa_server=use_msa_server,
            msa_server_url=msa_server_url,
            msa_pairing_strategy=msa_pairing_strategy,
            boltz2=True,
            preprocessing_threads=1,
            max_msa_seqs=int(self.config.get("max_msa_seqs", 8192)),
        )

        manifest_path = out_dir / "processed" / "manifest.json"
        manifest = Manifest.load(manifest_path)
        processed = {
            "manifest": manifest,
            "targets_dir": out_dir / "processed" / "structures",
            "msa_dir": out_dir / "processed" / "msa",
            "constraints_dir": (out_dir / "processed" / "constraints"),
            "template_dir": (out_dir / "processed" / "templates"),
            "extra_mols_dir": (out_dir / "processed" / "mols"),
            "predictions_dir": out_dir / "predictions",
        }
        return processed

    # TODO: Potentially cache preprocessing for identical inputs to avoid repeated MSA/feature builds.

    def _build_datamodule(self, processed: Dict[str, Any], num_workers: int, cache_dir: Path, override_method: Optional[str] = None) -> Any:
        return Boltz2InferenceDataModule(
            manifest=processed["manifest"],
            target_dir=processed["targets_dir"],
            msa_dir=processed["msa_dir"],
            mol_dir=cache_dir / "mols",
            num_workers=num_workers,
            constraints_dir=processed["constraints_dir"],
            template_dir=processed["template_dir"],
            extra_mols_dir=processed["extra_mols_dir"],
            override_method=override_method,
            affinity=False,
        )

    def _predict_with_trainer(self, datamodule: Any) -> list:
        assert self._trainer is not None and self.model is not None
        with torch.inference_mode():
            return self._trainer.predict(self.model, datamodule=datamodule, return_predictions=True)
    
    def fold(self, sequences: Union[str, Sequence[str]]) -> Boltz2Output:
        """Fold protein sequences (proteins only), keeping model resident in memory."""
        if isinstance(sequences, str):
            sequences = [sequences]

        # Always use a temporary working directory on the machine
        cache_dir = (
            Path(self.config.get("cache_dir")).resolve()
            if self.config.get("cache_dir")
            else Path(self.model_dir).resolve() / "boltz"
        )

        with TemporaryDirectory() as tmp:
            work_path = Path(tmp).resolve()

            with Timer("Boltz-2 preprocessing"):
                fasta_text = self._sequences_to_fasta(list(sequences))
                processed = self._prepare_inputs(fasta_text, work_path, cache_dir)

            datamodule = self._build_datamodule(processed, int(self.config.get("num_workers", 2)), cache_dir)

            with Timer("Boltz-2 inference") as t:
                preds = self._predict_with_trainer(datamodule)

            # Minimal output mapping: collect positions/coords only
            positions: List[np.ndarray] = []
            for item in preds or []:
                if not isinstance(item, dict):
                    continue
                coords = item.get("sample_atom_coords") or item.get("coords")
                if coords is None:
                    continue
                try:
                    arr = coords.detach().cpu().numpy() if hasattr(coords, "detach") else np.array(coords)
                except Exception:
                    continue
                positions.append(arr)

            positions_np = np.array(positions, dtype=object)
            self.metadata.prediction_time = t.duration
            return Boltz2Output(positions=positions_np, metadata=self.metadata)



############################################################
# MODAL BACKEND
############################################################
@app.cls(
    image=boltz_image,
    gpu="T4",
    timeout=20 * MINUTES,
    scaledown_window=10 * MINUTES,
    volumes={MODAL_MODEL_DIR: model_weights}, # TODO: somehow link this to what Boltz-2 actually uses
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
        elif backend == "local":
            self._backend = LocalBackend(Boltz2Core, config, device=device)
        else:
            raise ValueError(f"Backend {backend} not supported")
        self._backend.start()

    def fold(self, sequences: Union[str, Sequence[str]]) -> Boltz2Output:
        return self._call_backend_method("fold", sequences)
