import os
import logging
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import dataclasses

import numpy as np
import modal

from dataclasses import dataclass
from typing import Optional, Any, Union, Sequence, List, Dict, cast


from ...backend import LocalBackend, ModalBackend
from ...backend.base import Backend
from ...backend.modal import app
from .image import boltz_image
from ...base import StructurePrediction, PredictionMetadata, FoldingAlgorithm, ModelWrapper
from ...images.volumes import model_weights
from ...utils import MINUTES, MODAL_MODEL_DIR, Timer

with boltz_image.imports():
    import torch
    from pytorch_lightning import Trainer, seed_everything
    from boltz.main import (
        download_boltz2,
        process_inputs,
        Boltz2DiffusionParams,
        PairformerArgsV2,
        MSAModuleArgs,
        BoltzSteeringParams,
    )
    from boltz.data.types import Manifest, StructureV2, Coords
    from boltz.data.write.mmcif import to_mmcif
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
    metadata: PredictionMetadata
    atom_array: Optional[List[Any]] = None  # Always generated, one AtomArray per sample

    # Confidence-related outputs (one entry per sample)
    confidence: Optional[List[Dict[str, Any]]] = None
    plddt: Optional[List[np.ndarray]] = None
    pae: Optional[List[np.ndarray]] = None
    pde: Optional[List[np.ndarray]] = None
    # Optional serialized structures (one string per sample)
    pdb: Optional[List[str]] = None
    cif: Optional[List[str]] = None


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
        "step_scale": 1.5,
        "write_full_pae": True,  # TODO: might remove this as an option to avoid confusion for the user
        "write_full_pde": True,  # TODO: might remove this as an option to avoid confusion for the user
        "include_fields": None,  # Optional[List[str]] - controls which fields to include in output
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
    # Static config keys that can only be set at initialization
    STATIC_CONFIG_KEYS = {"device", "cache_dir", "no_kernels"}

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
        cache_dir_str = self.config.get("cache_dir")
        if cache_dir_str is not None:
            cache_dir = Path(cache_dir_str)
        else:
            if self.model_dir is None:
                raise ValueError("model_dir must be set when cache_dir is not provided")
            cache_dir = Path(self.model_dir).resolve() / "boltz"
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

    def _prepare_inputs(self, fasta_text: str, work_dir: Path, cache_dir: Path, config: dict) -> Dict[str, Any]:
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
        use_msa_server = bool(config.get("use_msa_server", True))
        msa_server_url = str(config.get("msa_server_url", "https://api.colabfold.com"))
        msa_pairing_strategy = str(config.get("msa_pairing_strategy", "greedy"))
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
            max_msa_seqs=int(config.get("max_msa_seqs", 8192)),
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

    def _build_datamodule(
        self, processed: Dict[str, Any], num_workers: int, cache_dir: Path, override_method: Optional[str] = None
    ) -> Any:
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

    def _extract_sample_from_pred(
        self, item: Dict[str, Any]
    ) -> tuple[
        Optional[np.ndarray],
        Dict[str, Any],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        """Extract coordinates and confidence-related arrays/metrics from a single prediction dict.

        Returns: (coords, aggregated_confidence_dict, plddt, pae, pde)
        """
        coords_like = item.get("sample_atom_coords") or item.get("coords")
        coords = (
            (coords_like.detach().cpu().numpy() if hasattr(coords_like, "detach") else np.array(coords_like))
            if coords_like is not None
            else None
        )

        plddt_like = item.get("plddt")
        if plddt_like is not None:
            plddt = plddt_like.detach().cpu().numpy() if hasattr(plddt_like, "detach") else np.array(plddt_like)
            # Squeeze extra batch dimensions to ensure 1D array
            while plddt.ndim > 1:
                plddt = np.squeeze(plddt, axis=0)
        else:
            plddt = None

        pae_like = item.get("pae")
        if pae_like is not None:
            pae = pae_like.detach().cpu().numpy() if hasattr(pae_like, "detach") else np.array(pae_like)
            # Squeeze extra batch dimensions to ensure 2D square matrix
            while pae.ndim > 2:
                pae = np.squeeze(pae, axis=0)
        else:
            pae = None

        pde_like = item.get("pde")
        if pde_like is not None:
            pde = pde_like.detach().cpu().numpy() if hasattr(pde_like, "detach") else np.array(pde_like)
            # Squeeze extra batch dimensions to ensure 2D square matrix
            while pde.ndim > 2:
                pde = np.squeeze(pde, axis=0)
        else:
            pde = None

        aggregated: Dict[str, Any] = {}
        for key in [
            "confidence_score",
            "ptm",
            "iptm",
            "ligand_iptm",
            "protein_iptm",
            "complex_plddt",
            "complex_iplddt",
            "complex_pde",
            "complex_ipde",
        ]:
            if key in item and item[key] is not None:
                val = item[key].detach().cpu().numpy() if hasattr(item[key], "detach") else item[key]
                aggregated[key] = float(val) if np.ndim(val) == 0 else val

        if "pair_chains_iptm" in item and item["pair_chains_iptm"] is not None:
            pair_val = item["pair_chains_iptm"]
            if isinstance(pair_val, dict):
                aggregated["pair_chains_iptm"] = pair_val
            else:
                pair_np = pair_val.detach().cpu().numpy() if hasattr(pair_val, "detach") else np.array(pair_val)
                nested: Dict[str, Dict[str, float]] = {}
                for i in range(pair_np.shape[-2]):
                    nested[str(i)] = {str(j): float(pair_np[i, j]) for j in range(pair_np.shape[-1])}
                aggregated["pair_chains_iptm"] = nested

        if "pair_chains_iptm" in aggregated and isinstance(aggregated["pair_chains_iptm"], dict):
            chains_ptm: Dict[str, float] = {}
            for i_str, inner in aggregated["pair_chains_iptm"].items():
                if i_str in inner:
                    chains_ptm[i_str] = float(inner[i_str])
            if chains_ptm:
                aggregated["chains_ptm"] = chains_ptm

        return coords, aggregated, plddt, pae, pde

    def _validate_sample_arrays(
        self, plddt: Optional[np.ndarray], pae: Optional[np.ndarray], pde: Optional[np.ndarray]
    ) -> None:
        if plddt is not None and plddt.ndim != 1:
            raise ValueError(f"plddt expected 1D per-token array; got shape {plddt.shape}")
        if pae is not None and (pae.ndim != 2 or pae.shape[0] != pae.shape[1]):
            raise ValueError(f"pae expected square matrix; got shape {pae.shape}")
        if pde is not None and (pde.ndim != 2 or pde.shape[0] != pde.shape[1]):
            raise ValueError(f"pde expected square matrix; got shape {pde.shape}")

    def _convert_outputs_to_pdb(self, atom_array):
        from biotite.structure.io.pdb import PDBFile, set_structure
        from io import StringIO

        structure_file = PDBFile()
        set_structure(structure_file, atom_array)
        string = StringIO()
        structure_file.write(string)
        return string.getvalue()

    def _convert_outputs_using_boltz_structure(
        self,
        coords: np.ndarray,
        processed: Dict[str, Any],
        plddt: Optional[List[np.ndarray]] = None,
    ) -> tuple[List[Any], List[str]]:
        """Convert outputs using boltz's StructureV2 and to_mmcif, return AtomArrays and CIF strings."""
        from biotite.structure.io.pdbx import CIFFile, get_structure
        from io import StringIO
        from dataclasses import replace

        record = processed["manifest"].records[0]
        structure = StructureV2.load(processed["targets_dir"] / f"{record.id}.npz").remove_invalid_chains()

        if coords.dtype == object:
            coords_list = [np.asarray(c) for c in coords]
        elif coords.ndim == 4:
            coords_list = [coords[i] for i in range(coords.shape[0])]
        elif coords.ndim == 3:
            coords_list = [coords]
        else:
            raise ValueError(f"Unexpected coords shape: {coords.shape}")

        atom_arrays = []
        cif_strings = []

        for sample_idx, sample_coords in enumerate(coords_list):
            coord_unpad = sample_coords.reshape(-1, 3)[: len(structure.atoms)]

            atoms = structure.atoms.copy()
            atoms["coords"] = coord_unpad
            atoms["is_present"] = True

            updated_structure = replace(
                structure,
                atoms=atoms,
                residues=structure.residues.copy(),
                interfaces=np.array([], dtype=structure.interfaces.dtype),
                coords=np.array([(x,) for x in coord_unpad], dtype=Coords),
            )

            sample_plddt = None
            if plddt and sample_idx < len(plddt):
                sample_plddt = (
                    torch.from_numpy(plddt[sample_idx])
                    if isinstance(plddt[sample_idx], np.ndarray)
                    else plddt[sample_idx]
                )

            cif_string = to_mmcif(updated_structure, plddts=sample_plddt, boltz2=True)
            cif_strings.append(cif_string)

            atom_arrays.append(get_structure(CIFFile.read(StringIO(cif_string)), model=1))

        return atom_arrays, cif_strings

    def fold(self, sequences: Union[str, Sequence[str]], options: Optional[dict] = None) -> Boltz2Output:
        """Fold protein sequences (proteins only), keeping model resident in memory."""

        # Merge static config with per-call options
        effective_config = self._merge_options(options)

        # Set seed early for deterministic results (before any preprocessing or data loading)
        # TODO: It has not been tested yet if this is actually deterministic.
        seed = effective_config.get("seed")
        if seed is not None:
            seed_everything(seed)

        # Validate sequences and compute sequence lengths
        validated_sequences = self._validate_sequences(sequences)
        self.metadata.sequence_lengths = self._compute_sequence_lengths(validated_sequences)

        # Always use a temporary working directory on the machine
        cache_dir_str = effective_config.get("cache_dir")
        if cache_dir_str is not None:
            cache_dir = Path(cache_dir_str).resolve()
        else:
            if self.model_dir is None:
                raise ValueError("model_dir must be set when cache_dir is not provided")
            cache_dir = Path(self.model_dir).resolve() / "boltz"

        with TemporaryDirectory() as tmp:
            work_path = Path(tmp).resolve()

            with Timer("Boltz-2 preprocessing"):
                fasta_text = self._sequences_to_fasta(validated_sequences)
                processed = self._prepare_inputs(fasta_text, work_path, cache_dir, effective_config)

            datamodule = self._build_datamodule(processed, int(effective_config.get("num_workers", 2)), cache_dir)

            with Timer("Boltz-2 inference") as t:
                preds = self._predict_with_trainer(datamodule)

            # Extract and organize per-sample outputs succinctly
            extracted = [self._extract_sample_from_pred(item) for item in (preds or []) if isinstance(item, dict)]

            confidences: List[Dict[str, Any]] = [a for (_, a, _, _, _) in extracted if a]
            plddts: List[np.ndarray] = [p for (_, _, p, _, _) in extracted if p is not None]
            paes: List[np.ndarray] = [a for (_, _, _, a, _) in extracted if a is not None]
            pdes: List[np.ndarray] = [d for (_, _, _, _, d) in extracted if d is not None]

            # Validate shapes where present
            for arr in plddts:
                self._validate_sample_arrays(arr, None, None)
            for arr in paes:
                self._validate_sample_arrays(None, arr, None)
            for arr in pdes:
                self._validate_sample_arrays(None, None, arr)

            # Extract coords directly for atom_array generation
            coords_list = [c for (c, _, _, _, _) in extracted if c is not None]
            coords_np = np.array(coords_list, dtype=object)
            atom_array_list, cif_strings = self._convert_outputs_using_boltz_structure(
                coords_np, processed, plddt=plddts if plddts else None
            )

            include_fields = effective_config.get("include_fields")
            pdb_list = None
            if include_fields and ("*" in include_fields or "pdb" in include_fields):
                pdb_list = [self._convert_outputs_to_pdb(arr) for arr in atom_array_list]

            cif_list = None
            if include_fields and ("*" in include_fields or "cif" in include_fields):
                cif_list = cif_strings

            self.metadata.prediction_time = t.duration

            # Build full output with all fields
            full_output = Boltz2Output(
                metadata=self.metadata,
                confidence=(confidences if confidences else None),
                plddt=(plddts if plddts else None),
                pae=(paes if paes else None),
                pde=(pdes if pdes else None),
                pdb=pdb_list,
                cif=cif_list,
                atom_array=atom_array_list,
            )

            # Apply filtering based on include_fields
            filtered = self._filter_include_fields(full_output, include_fields)
            return cast(Boltz2Output, filtered)


############################################################
# MODAL BACKEND
############################################################
@app.cls(
    image=boltz_image,
    gpu="T4",
    timeout=20 * MINUTES,
    scaledown_window=10 * MINUTES,
    volumes={MODAL_MODEL_DIR: model_weights},  # TODO: somehow link this to what Boltz-2 actually uses
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
    def fold(self, sequences: Union[str, Sequence[str]], options: Optional[dict] = None) -> Boltz2Output:
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
        if config is None:
            config = {}
        self.config = config
        self.device = device
        backend_instance: Backend
        if backend == "modal":
            backend_instance = ModalBackend(ModalBoltz2, config, device=device)
        elif backend == "local":
            backend_instance = LocalBackend(Boltz2Core, config, device=device)
        else:
            raise ValueError(f"Backend {backend} not supported")
        self._backend = backend_instance
        self._backend.start()

    def fold(self, sequences: Union[str, Sequence[str]], options: Optional[dict] = None) -> Boltz2Output:
        return self._call_backend_method("fold", sequences, options=options)
