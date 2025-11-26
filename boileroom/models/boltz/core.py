"""Core Boltz2 algorithm implementation without modal dependencies."""

import os
import logging
import json
import hashlib
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from datetime import datetime
import dataclasses

import numpy as np
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

from typing import Optional, Any, Union, Sequence, List, Dict, cast

from ...base import FoldingAlgorithm
from ...utils import MODAL_MODEL_DIR, Timer
from .types import Boltz2Output

logger = logging.getLogger(__name__)


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
        "msa_cache_enabled": True,  # Enable/disable MSA caching
        # Paths
        "cache_dir": None,  # defaults to Path(MODAL_MODEL_DIR)/"boltz"
    }
    # Static config keys that can only be set at initialization
    STATIC_CONFIG_KEYS = {"device", "cache_dir", "no_kernels"}

    def __init__(self, config: dict = {}) -> None:
        """Create a Boltz-2 core instance configured for inference.

        Parameters
        ----------
        config : dict
            Runtime configuration options that override the class defaults.
            Keys listed in STATIC_CONFIG_KEYS are treated as fixed at initialization.
            The configuration is used later when loading model weights and preparing the trainer.
        """
        super().__init__(config)
        self.metadata = self._initialize_metadata(
            model_name="Boltz-2",
            model_version="conf",  # matching ckpt naming; refine if needed
        )
        self.model_dir: Optional[str] = os.environ.get("MODEL_DIR", MODAL_MODEL_DIR)
        self.model: Optional[Any] = None
        self._trainer: Optional[Any] = None

    def _initialize(self) -> None:
        """Prepare the core Boltz-2 runtime for inference by loading model weights and constructing the persistent trainer.

        After this call the core's model and trainer are initialized and ready for predictions (self.model, self._trainer), and the ready flag is set.
        """
        self._load()

    def _load(self) -> None:
        """Load Boltz-2 weights and resources, initialize the model on the configured device, and create a persistent PyTorch Lightning Trainer.

        This configures and populates the instance by:
        - Ensuring a cache directory exists and downloading required Boltz-2 assets into it.
        - Loading the Boltz2Model checkpoint into self.model with inference-related arguments.
        - Moving the model to the configured device, falling back to CPU if device assignment fails.
        - Creating and assigning a persistent Trainer to self._trainer.
        - Marking the core as ready by setting self.ready = True.

        Raises
        ------
        ValueError
            If neither a cache directory nor a model directory is available to resolve required assets.
        """
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

    def _get_sequence_hash(self, sequence: str) -> str:
        """Compute SHA256 hash of a protein sequence.

        Parameters
        ----------
        sequence : str
            Protein sequence string.

        Returns
        -------
        str
            Hexadecimal digest of the sequence hash.
        """
        return hashlib.sha256(sequence.encode()).hexdigest()

    def _get_msa_cache_dir(self) -> Path:
        """Get the MSA cache directory path.

        Returns the cache directory at {cache_dir}/boltz/msa_cache or {MODEL_DIR}/boltz/msa_cache.
        Handles both local and Modal paths (checks MODAL_MODEL_DIR if available).

        Returns
        -------
        Path
            Path to the MSA cache directory. Directory is created if it doesn't exist.
        """
        cache_dir_str = self.config.get("cache_dir")
        if cache_dir_str is not None:
            base_cache_dir = Path(cache_dir_str).resolve()
        else:
            if self.model_dir is None:
                raise ValueError("model_dir must be set when cache_dir is not provided")
            base_cache_dir = Path(self.model_dir).resolve() / "boltz"

        # Check if we're in Modal environment and adjust path
        modal_model_dir = os.environ.get("MODAL_MODEL_DIR")
        if modal_model_dir:
            base_cache_dir = Path(modal_model_dir).resolve() / "boltz"
        elif cache_dir_str is None and self.model_dir == MODAL_MODEL_DIR:
            # If model_dir was set to MODAL_MODEL_DIR but env var not set, use it directly
            base_cache_dir = Path(MODAL_MODEL_DIR).resolve() / "boltz"

        msa_cache_dir = base_cache_dir / "msa_cache"
        msa_cache_dir.mkdir(parents=True, exist_ok=True)
        return msa_cache_dir

    def _get_msa_cache_index_path(self) -> Path:
        """Get the path to the MSA cache index file.

        Returns
        -------
        Path
            Path to msa_index.json in the cache directory.
        """
        return self._get_msa_cache_dir() / "msa_index.json"

    def _load_msa_cache_index(self) -> Dict[str, Dict[str, Any]]:
        """Load the MSA cache index from disk.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Dictionary mapping sequence hash to cache metadata. Returns empty dict if file doesn't exist
            or if there's an error loading it.
        """
        index_path = self._get_msa_cache_index_path()
        if not index_path.exists():
            return {}

        try:
            with index_path.open("r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load MSA cache index from {index_path}: {e}. Recreating index.")
            # Backup corrupted index and create new one
            if index_path.exists():
                backup_path = index_path.with_suffix(".json.bak")
                try:
                    shutil.move(str(index_path), str(backup_path))
                    logger.info(f"Moved corrupted index to {backup_path}")
                except OSError:
                    pass
            return {}

    def _save_msa_cache_index(self, index: Dict[str, Dict[str, Any]]) -> None:
        """Save the MSA cache index to disk atomically.

        Writes to a temporary file first, then renames it to the final location to ensure atomic updates.

        Parameters
        ----------
        index : Dict[str, Dict[str, Any]]
            Dictionary mapping sequence hash to cache metadata to save.
        """
        index_path = self._get_msa_cache_index_path()
        cache_dir = index_path.parent
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Write to temporary file first for atomic update
        temp_path = index_path.with_suffix(".json.tmp")
        try:
            with temp_path.open("w") as f:
                json.dump(index, f, indent=2)
            # Atomic rename
            temp_path.replace(index_path)
        except OSError as e:
            logger.warning(f"Failed to save MSA cache index to {index_path}: {e}")
            # Clean up temp file if rename failed
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass

    def _check_msa_cache(self, sequences: List[str]) -> Dict[str, Path]:
        """Check the MSA cache for cached MSAs for the given sequences.

        Parameters
        ----------
        sequences : List[str]
            List of individual protein sequences (after splitting by colon for multimers).

        Returns
        -------
        Dict[str, Path]
            Dictionary mapping sequence string to cached MSA Path if found, empty dict otherwise.
            Only includes entries where the cached file actually exists.
        """
        if not self.config.get("msa_cache_enabled", True):
            return {}

        try:
            cache_dir = self._get_msa_cache_dir()
            index = self._load_msa_cache_index()
            cached_paths: Dict[str, Path] = {}
            index_updated = False
            now = datetime.utcnow().isoformat()

            for sequence in sequences:
                seq_hash = self._get_sequence_hash(sequence)
                if seq_hash in index:
                    entry = index[seq_hash]
                    # Index stores relative path, construct full path
                    msa_path_str = entry.get("msa_path", "")
                    if not msa_path_str:
                        # Fallback: try hash-based path structure
                        msa_path = cache_dir / seq_hash[:2] / seq_hash[2:4] / f"{seq_hash}.csv"
                    else:
                        msa_path = cache_dir / msa_path_str

                    # Check if file actually exists
                    if msa_path.exists() and msa_path.is_file():
                        cached_paths[sequence] = msa_path
                        # Update last_accessed timestamp
                        if entry.get("last_accessed") != now:
                            entry["last_accessed"] = now
                            index_updated = True
                    else:
                        # File doesn't exist, remove from index
                        logger.debug(f"Cached MSA file not found for hash {seq_hash}, removing from index")
                        del index[seq_hash]
                        index_updated = True

            # Save updated index if any timestamps were updated or entries removed
            if index_updated:
                self._save_msa_cache_index(index)

            return cached_paths
        except Exception as e:
            logger.warning(f"Error checking MSA cache: {e}. Continuing without cache.")
            return {}

    def _save_msa_to_cache(self, sequence: str, msa_source_path: Path) -> None:
        """Save a single MSA file to the cache.

        Parameters
        ----------
        sequence : str
            Protein sequence string.
        msa_source_path : Path
            Path to the source MSA file (typically a .csv file from preprocessing output).
        """
        if not self.config.get("msa_cache_enabled", True):
            return

        try:
            cache_dir = self._get_msa_cache_dir()
            seq_hash = self._get_sequence_hash(sequence)

            # Create hash-prefixed directory structure: {hash[:2]}/{hash[2:4]}/
            hash_prefix_dir = cache_dir / seq_hash[:2] / seq_hash[2:4]
            hash_prefix_dir.mkdir(parents=True, exist_ok=True)

            # Destination path
            msa_cache_path = hash_prefix_dir / f"{seq_hash}.csv"

            # Only copy if source file exists and destination doesn't already exist
            if not msa_source_path.exists():
                logger.warning(f"Source MSA file not found: {msa_source_path}")
                return

            if msa_cache_path.exists():
                # Already cached, skip
                logger.debug(f"MSA already cached for sequence hash {seq_hash}")
                return

            # Copy MSA file to cache
            shutil.copy2(msa_source_path, msa_cache_path)
            file_size = msa_cache_path.stat().st_size

            # Update index
            index = self._load_msa_cache_index()
            now = datetime.utcnow().isoformat()
            relative_path = f"{seq_hash[:2]}/{seq_hash[2:4]}/{seq_hash}.csv"
            index[seq_hash] = {
                "msa_path": relative_path,
                "created_at": now,
                "last_accessed": now,
                "file_size": file_size,
            }
            self._save_msa_cache_index(index)

            logger.debug(f"Cached MSA for sequence hash {seq_hash}")
        except Exception as e:
            logger.warning(f"Error saving MSA to cache: {e}. Continuing without caching.")

    def _save_msas_to_cache(
        self,
        processed: Dict[str, Any],
        individual_chains: List[str],
        out_dir: Path,
        cached_sequences: Optional[Dict[str, Path]] = None,
    ) -> None:
        """Extract and save MSAs from preprocessing output to cache.

        Parameters
        ----------
        processed : Dict[str, Any]
            Dictionary returned by _prepare_inputs containing manifest and directory paths.
        individual_chains : List[str]
            List of individual protein sequences (after splitting by colon for multimers).
            Order matches entity_id assignment (entity 0 = chains[0], entity 1 = chains[1], etc.).
        out_dir : Path
            Output directory where preprocessing wrote MSA files. Should contain msa/ subdirectory.
        cached_sequences : Optional[Dict[str, Path]]
            Optional dictionary of sequences that were already cached. These will be skipped.
        """
        if not self.config.get("msa_cache_enabled", True):
            return

        cached_sequences = cached_sequences or {}

        try:
            manifest = processed.get("manifest")
            if not manifest or not manifest.records:
                return

            # Get the target record
            record = manifest.records[0]
            target_id = record.id

            # Find MSA files in out_dir/msa/ directory
            msa_raw_dir = out_dir / "msa"
            if not msa_raw_dir.exists():
                return

            # Find all protein chains and their entity IDs
            # Entity IDs are assigned sequentially: first chain gets entity 0, second gets entity 1, etc.
            protein_chains = [chain for chain in record.chains if hasattr(chain, "entity_id")]
            protein_chains.sort(key=lambda c: c.entity_id)

            # Match sequences to entity IDs by order
            # Only cache MSAs for sequences that were actually generated (not from cache)
            for idx, chain in enumerate(protein_chains):
                entity_id = chain.entity_id
                if idx < len(individual_chains):
                    sequence = individual_chains[idx]
                    # Skip if this sequence was already cached
                    if sequence in cached_sequences:
                        continue

                    msa_filename = f"{target_id}_{entity_id}.csv"
                    msa_source_path = msa_raw_dir / msa_filename

                    if msa_source_path.exists() and msa_source_path.is_file():
                        self._save_msa_to_cache(sequence, msa_source_path)
        except Exception as e:
            logger.warning(f"Error saving MSAs to cache: {e}. Continuing without caching.")

    def _sequences_to_fasta(self, sequences: List[str], msa_paths: Optional[Dict[str, Path]] = None) -> str:
        """Convert input protein sequences into a Boltz-2 FASTA formatted string.

        Accepts a list of sequence entries where each entry is either a single sequence (e.g., "SEQ")
        or a colon-separated multimer entry (e.g., "A:B"). Whitespace is trimmed and empty parts are ignored.
        Chain identifiers are assigned sequentially as A, B, C, ... and used as FASTA headers.
        If msa_paths is provided, MSA paths are injected into FASTA headers as the third field.

        Parameters
        ----------
        sequences : List[str]
            Sequence entries or colon-separated multimer entries.
        msa_paths : Optional[Dict[str, Path]]
            Optional dictionary mapping sequence string to cached MSA Path. If provided, paths are injected
            into FASTA headers as the third field: `>X|protein|{msa_path}`.

        Returns
        -------
        str
            FASTA-formatted text where each chain has a header of the form `>X|protein|` (or `>X|protein|{msa_path}`
            if msa_paths is provided) followed by the corresponding sequence on the next line.
        """
        chains: List[str] = []
        for entry in sequences:
            parts = entry.split(":") if ":" in entry else [entry]
            chains.extend([p.strip() for p in parts if p.strip()])

        headers = []
        msa_paths_dict = msa_paths or {}
        for idx, seq in enumerate(chains):
            chain_name = chr(65 + idx)  # A, B, C...
            # If MSA path is provided for this sequence, inject it into the header
            if seq in msa_paths_dict:
                msa_path = msa_paths_dict[seq]
                # Use absolute path to ensure Boltz2 can resolve it
                msa_path_abs = msa_path.resolve() if isinstance(msa_path, Path) else Path(msa_path).resolve()
                headers.append(f">{chain_name}|protein|{msa_path_abs}")
            else:
                headers.append(f">{chain_name}|protein|")
            headers.append(seq)
        return "\n".join(headers)

    def _prepare_inputs(self, fasta_text: str, work_dir: Path, cache_dir: Path, config: dict) -> Dict[str, Any]:
        """Prepare input files and run Boltz-2 preprocessing to produce a processed manifest and standard input/output directories.

        Creates a FASTA file from `fasta_text` in `work_dir`, runs the Boltz-2 preprocessing pipeline (MSA generation and related data preparation), and returns paths and the loaded manifest needed for downstream inference.

        Parameters
        ----------
        fasta_text : str
            FASTA-formatted sequence data (single or multi-sequence).
        work_dir : Path
            Working directory where input and output subdirectories will be created.
        cache_dir : Path
            Directory containing or receiving Boltz-2 resources and auxiliary files.
        config : dict
            Preprocessing options. Recognized keys:
            - use_msa_server (bool): whether to use an external MSA server (default True).
            - msa_server_url (str): URL of the MSA server (default "https://api.colabfold.com").
            - msa_pairing_strategy (str): MSA pairing strategy (e.g., "greedy").
            - max_msa_seqs (int): maximum number of MSA sequences to request (default 8192).

        Returns
        -------
        Dict[str, Any]
            A mapping containing:
            - "manifest": Manifest object loaded from processed/manifest.json.
            - "targets_dir": Path to processed structures directory.
            - "msa_dir": Path to processed MSA directory.
            - "constraints_dir": Path to processed constraints directory.
            - "template_dir": Path to processed templates directory.
            - "extra_mols_dir": Path to processed extra molecules directory.
            - "predictions_dir": Path to predictions output directory.
        """
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

    # TODO: MSA Cache Cleanup
    # The MSA cache can grow over time. To clean it manually:
    # 1. Load msa_index.json from the cache directory ({cache_dir}/boltz/msa_cache/msa_index.json)
    # 2. Filter entries based on last_accessed timestamp (e.g., remove entries not accessed in 90+ days)
    # 3. Delete corresponding .csv files from hash-prefixed directories ({hash[:2]}/{hash[2:4]}/{hash}.csv)
    # 4. Update and save the cleaned index
    # Example: Remove entries where last_accessed < (datetime.now() - timedelta(days=90))
    # Example: Implement size-based limits (e.g., keep cache under 100GB total size using file_size field)
    # Future implementation could add a cleanup utility function or automated cleanup with LRU eviction

    def _build_datamodule(
        self, processed: Dict[str, Any], num_workers: int, cache_dir: Path, override_method: Optional[str] = None
    ) -> Any:
        """Create a Boltz2InferenceDataModule configured for inference from preprocessed inputs.

        Parameters
        ----------
        processed : Dict[str, Any]
            Dictionary produced by _prepare_inputs containing at least the keys
            "manifest", "targets_dir", "msa_dir", "constraints_dir", "template_dir", and "extra_mols_dir".
        num_workers : int
            Number of worker processes for data loading.
        cache_dir : Path
            Base cache directory; a "mols" subdirectory under this path will be used for molecule files.
        override_method : Optional[str]
            Optional override for the data loading/preprocessing method.

        Returns
        -------
        Boltz2InferenceDataModule
            A configured data module ready to be passed to the trainer for prediction.
        """
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
        """Run the initialized trainer to perform prediction over the provided datamodule.

        Parameters
        ----------
        datamodule : Any
            Data module supplying batches for prediction.

        Returns
        -------
        list
            Prediction results produced by the trainer, as returned by the Trainer.predict call.
        """
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
        """Extract per-sample atom coordinates and confidence metrics from a prediction dictionary.

        Parameters
        ----------
        item : Dict[str, Any]
            Prediction dictionary produced by the model inference step. Expected keys (optional) include:
            - "sample_atom_coords" or "coords": per-sample atom coordinates (tensor/array).
            - "plddt": per-residue confidence scores.
            - "pae", "pde": pairwise error/confidence matrices.
            - Any of: "confidence_score", "ptm", "iptm", "ligand_iptm", "protein_iptm",
              "complex_plddt", "complex_iplddt", "complex_pde", "complex_ipde" (scalars or arrays).
            - "pair_chains_iptm": either a dict or an array-like pairwise inter-chain scores.

        Returns
        -------
        tuple
            A tuple containing:
            - coords (Optional[np.ndarray]): Atom coordinates as a NumPy array or None if not present.
            - aggregated_confidence_dict (Dict[str, Any]): Collected scalar and array confidence metrics converted to
              NumPy/float form; includes processed "pair_chains_iptm" as a nested dict when present and a
              derived "chains_ptm" mapping of self-pair scores when available.
            - plddt (Optional[np.ndarray]): 1D per-residue pLDDT array or None.
            - pae (Optional[np.ndarray]): 2D pairwise alignment error (PAE) matrix or None.
            - pde (Optional[np.ndarray]): 2D pairwise distance error (PDE) matrix or None.
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
        """Validate shapes of per-sample confidence arrays.

        Raises a ValueError if:
        - `plddt` is provided and is not a 1-dimensional array.
        - `pae` is provided and is not a 2-dimensional square matrix.
        - `pde` is provided and is not a 2-dimensional square matrix.

        Parameters
        ----------
        plddt : Optional[np.ndarray]
            Per-residue confidence scores; expected shape (L,).
        pae : Optional[np.ndarray]
            Predicted aligned error matrix; expected shape (L, L).
        pde : Optional[np.ndarray]
            Predicted distance error matrix; expected shape (L, L).
        """
        if plddt is not None and plddt.ndim != 1:
            raise ValueError(f"plddt expected 1D per-token array; got shape {plddt.shape}")
        if pae is not None and (pae.ndim != 2 or pae.shape[0] != pae.shape[1]):
            raise ValueError(f"pae expected square matrix; got shape {pae.shape}")
        if pde is not None and (pde.ndim != 2 or pde.shape[0] != pde.shape[1]):
            raise ValueError(f"pde expected square matrix; got shape {pde.shape}")

    def _convert_outputs_to_pdb(self, atom_array):
        """Convert a Biotite atom array into a PDB-formatted string.

        Parameters
        ----------
        atom_array : Any
            A Biotite AtomArray or equivalent structure describing atomic coordinates and metadata.

        Returns
        -------
        str
            A string containing the PDB representation of the provided atom array.
        """
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
        """Convert model coordinates into Boltz-2 structure objects and MMCIF strings.

        Given per-sample coordinates and the processed input manifest, produce a list of Biotite AtomArray structures and corresponding MMCIF-formatted strings. If `plddt` is provided, the per-residue pLDDT values are attached to each MMCIF entry.

        Parameters
        ----------
        coords : np.ndarray
            Model output coordinates; may be a 3D array for a single sample, a 4D array for multiple samples, or an object-dtype sequence of per-sample arrays.
        processed : Dict[str, Any]
            Processed inputs dictionary produced by the preprocessing step; must include a manifest with a target record and a `targets_dir` containing the template structure.
        plddt : Optional[List[np.ndarray]]
            Optional list of per-sample pLDDT arrays to embed into the MMCIF output.

        Returns
        -------
        tuple[List[Any], List[str]]
            A tuple (atom_arrays, cif_strings) where `atom_arrays` is a list of Biotite AtomArray-like structures (one per sample) and `cif_strings` is a list of corresponding MMCIF-formatted strings.
        """
        from biotite.structure.io.pdbx import CIFFile, get_structure
        from io import StringIO
        from dataclasses import replace

        if not processed["manifest"].records or len(processed["manifest"].records) == 0:
            raise ValueError("Manifest has no records. Preprocessing may have failed.")

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
        """Predicts 3D structures for the given protein sequences using the resident Boltz-2 model.

        Parameters
        ----------
        sequences : Union[str, Sequence[str]]
            One or more protein sequences. Accepted formats include a single sequence string, a chain-delimited string like "A:B" for multiple chains, or a sequence/list of individual chain strings.
        options : Optional[dict]
            Optional runtime configuration overrides for this call. Supported keys include `seed` (int) to control randomness, `cache_dir` (str) for resource caching, `num_workers` (int) for data loading, and `include_fields` (iterable) to request additional output fields such as `"pdb"` or `"cif"`.

        Returns
        -------
        Boltz2Output
            Prediction results including metadata and per-sample outputs such as atom arrays, confidence metrics, `plddt`, `pae`, `pde`, and optional PDB/MMCIF strings depending on `include_fields`.
        """

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

        # Split sequences into individual chains for cache checking
        individual_chains: List[str] = []
        for entry in validated_sequences:
            parts = entry.split(":") if ":" in entry else [entry]
            individual_chains.extend([p.strip() for p in parts if p.strip()])

        # Check MSA cache before preprocessing
        cached_msa_paths: Dict[str, Path] = {}
        if self.config.get("msa_cache_enabled", True):
            try:
                cached_msa_paths = self._check_msa_cache(individual_chains)
                if cached_msa_paths:
                    logger.info(
                        f"Found {len(cached_msa_paths)} cached MSA(s) out of {len(individual_chains)} sequence(s)"
                    )
            except Exception as e:
                logger.warning(f"Error checking MSA cache: {e}. Continuing without cache.")

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
                # Pass cached MSA paths to FASTA generation
                fasta_text = self._sequences_to_fasta(
                    validated_sequences, msa_paths=cached_msa_paths if cached_msa_paths else None
                )
                processed = self._prepare_inputs(fasta_text, work_path, cache_dir, effective_config)

                # Save newly generated MSAs to cache (before temp cleanup)
                if self.config.get("msa_cache_enabled", True):
                    try:
                        # Extract out_dir from processed dict (targets_dir is out_dir/processed/structures)
                        targets_dir = processed.get("targets_dir")
                        if targets_dir:
                            # out_dir is the parent of "processed" directory
                            out_dir = targets_dir.parent.parent
                            self._save_msas_to_cache(
                                processed, individual_chains, out_dir, cached_sequences=cached_msa_paths
                            )
                    except Exception as e:
                        logger.warning(f"Error saving MSAs to cache: {e}. Continuing without caching.")

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
