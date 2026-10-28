"""Core Boltz2 algorithm implementation without modal dependencies."""

import dataclasses
import logging
import os
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, ClassVar, cast

import numpy as np
import torch
from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
from boltz.data.types import Coords, Manifest, StructureV2
from boltz.data.write.mmcif import to_mmcif
from boltz.main import (
    Boltz2DiffusionParams,
    BoltzSteeringParams,
    MSAModuleArgs,
    PairformerArgsV2,
    compute_msa,
    download_boltz2,
    process_inputs,
)
from boltz.model.models.boltz2 import Boltz2 as Boltz2Model
from pytorch_lightning import Trainer, seed_everything

from ...base import FoldingAlgorithm
from ...msa_cache import MSACache
from ...utils import MODAL_MODEL_DIR, Timer, safe_mkdir
from .msa import (
    cache_key,
    chains_to_fasta,
    provided_alignments,
    resolve_chain_msas,
    split_chains,
    stage_provided_alignments,
)
from .types import Boltz2Output

logger = logging.getLogger(__name__)


class Boltz2Core(FoldingAlgorithm):
    """Boltz-2 protein structure prediction model."""

    DEFAULT_CONFIG: ClassVar[dict[str, Any]] = {
        "device": "cuda:0",
        "use_msa_server": True,  # setting to False automatically sets to no MSA mode
        "msa_server_url": "https://api.colabfold.com",
        "msa_pairing_strategy": "greedy",
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
        "msa_cache_enabled": True,  # cache the MSA fetched from the server for each chain, by sequence
        "msa": None,  # caller-supplied alignments: one A3M text (or None) per chain, used for the sequence each is of
        # Paths
        "cache_dir": None,  # defaults to Path(MODAL_MODEL_DIR)/"boltz"
    }
    #: ``options["msa"]`` is one A3M text (or ``None``) per chain; see :mod:`boileroom.models.boltz.msa`.
    SUPPORTS_USER_MSA: ClassVar[bool] = True
    # Static config keys that can only be set at initialization
    STATIC_CONFIG_KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "device",
            "cache_dir",
            "no_kernels",
            "recycling_steps",
            "sampling_steps",
            "diffusion_samples",
            "max_parallel_samples",
            "step_scale",
            "subsample_msa",
            "num_subsampled_msa",
            "write_full_pae",
            "write_full_pde",
        }
    )

    def __init__(self, config: dict | None = None) -> None:
        """Create a Boltz-2 core instance configured for inference.

        Parameters
        ----------
        config : dict
            Runtime configuration options that override the class defaults.
            Keys listed in STATIC_CONFIG_KEYS are treated as fixed at initialization.
            The configuration is used later when loading model weights and preparing the trainer.
        """
        super().__init__(config)
        self._metadata_template = self._initialize_metadata(
            model_name="Boltz-2",
            model_version="conf",  # matching ckpt naming; refine if needed
        )
        self.model_dir: str | None = os.environ.get("MODEL_DIR", MODAL_MODEL_DIR)
        self.model: Any | None = None
        self._trainer: Any | None = None

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
            cache_dir = Path(self.model_dir) / "boltz"
        # Use safe_mkdir to avoid creating parent directories that don't exist in containers
        safe_mkdir(cache_dir, parents=False)

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

    def _msa_cache_base_dir(self) -> Path:
        """Return the base directory that houses the Boltz MSA cache.

        Resolves to ``{cache_dir}`` when configured, otherwise ``{MODEL_DIR}/boltz``,
        with Modal's mounted model directory taking precedence when present.
        """
        cache_dir_str = self.config.get("cache_dir")
        if cache_dir_str is not None:
            base_cache_dir = Path(cache_dir_str)
        else:
            if self.model_dir is None:
                raise ValueError("model_dir must be set when cache_dir is not provided")
            base_cache_dir = Path(self.model_dir) / "boltz"

        # Prefer Modal's mounted model directory when running on Modal.
        modal_model_dir = os.environ.get("MODAL_MODEL_DIR")
        if modal_model_dir:
            base_cache_dir = Path(modal_model_dir) / "boltz"
        elif cache_dir_str is None and self.model_dir == MODAL_MODEL_DIR:
            base_cache_dir = Path(MODAL_MODEL_DIR) / "boltz"
        return base_cache_dir

    def _msa_cache(self) -> MSACache:
        """Return the shared content-addressed cache for Boltz ``.csv`` MSAs."""
        return MSACache(self._msa_cache_base_dir(), suffix=".csv")

    def _msa_cache_key(self, sequence: str, config: dict[str, Any] | None = None) -> str:
        """Cache key of the MSA of ``sequence`` under ``config`` (the effective configuration of the call)."""
        config = self.config if config is None else config
        return cache_key(
            sequence,
            str(config.get("msa_server_url", "https://api.colabfold.com")),
            int(config.get("max_msa_seqs", 8192)),
        )

    def _check_msa_cache(
        self, sequences: list[str], *, cache_enabled: bool, config: dict[str, Any] | None = None
    ) -> dict[str, Path]:
        """Return cached MSA paths for the sequences present in the cache.

        Parameters
        ----------
        sequences : list[str]
            Individual protein sequences (after splitting multimers on ``:``).
        cache_enabled : bool
            When ``False`` the cache is bypassed and an empty mapping is returned.
        config : dict, optional
            The effective configuration of the call, which keys the cache entries (default: the instance config).

        Returns
        -------
        dict[str, Path]
            Mapping from sequence to its cached MSA file, for cache hits only.
        """
        if not cache_enabled:
            return {}
        cache = self._msa_cache()
        cached_paths: dict[str, Path] = {}
        for sequence in sequences:
            path = cache.get(self._msa_cache_key(sequence, config))
            if path is not None:
                cached_paths[sequence] = path
        return cached_paths

    def _save_msa_to_cache(
        self, sequence: str, msa_source_path: Path, *, cache_enabled: bool, config: dict[str, Any] | None = None
    ) -> None:
        """Copy the ``.csv`` MSA fetched for ``sequence`` into the shared cache."""
        if not cache_enabled:
            return
        self._msa_cache().put(self._msa_cache_key(sequence, config), msa_source_path)

    def _lookup_cached_msas(self, chains: Sequence[str], config: dict[str, Any]) -> dict[str, Path]:
        """Cached MSA files of ``chains``; a broken cache is reported and treated as empty."""
        try:
            cached = self._check_msa_cache(list(chains), cache_enabled=True, config=config)
        except Exception as e:
            logger.warning(f"Error checking MSA cache: {e}. Continuing without cache.")
            return {}
        if cached:
            logger.info(f"Found {len(cached)} cached MSA(s) out of {len(chains)} sequence(s)")
        return cached

    def _store_msa(self, sequence: str, path: Path, config: dict[str, Any]) -> None:
        """Cache a fetched MSA; a failure to cache is reported and does not fail the fold."""
        try:
            self._save_msa_to_cache(sequence, path, cache_enabled=True, config=config)
        except Exception as e:
            logger.warning(f"Error saving the MSA to the cache: {e}. Continuing without caching.")

    def _fetch_chain_msa(self, sequence: str, directory: Path, config: dict[str, Any]) -> Path:
        """Fetch the MSA of one chain on its own (unpaired) from the MSA server and return its ``.csv`` file.

        This is the function Boltz itself uses to generate MSAs, called for a single sequence, so that the result
        depends on the sequence alone and can be cached.
        """
        directory.mkdir(parents=True, exist_ok=True)
        name = MSACache.hash_key(sequence)[:16]
        compute_msa(
            data={name: sequence},
            target_id=name,
            msa_dir=directory,
            msa_server_url=str(config.get("msa_server_url", "https://api.colabfold.com")),
            msa_pairing_strategy=str(config.get("msa_pairing_strategy", "greedy")),
        )
        path = directory / f"{name}.csv"
        if not path.is_file():
            raise RuntimeError(f"The MSA server returned no MSA for a chain of {len(sequence)} residues")
        return path

    def _sequences_to_fasta(
        self, sequences: list[str], msa_paths: dict[str, Path] | None = None, config: dict | None = None
    ) -> str:
        """Convert input protein sequences into a Boltz-2 FASTA formatted string.

        Accepts a list of sequence entries where each entry is either a single sequence (e.g., "SEQ")
        or a colon-separated multimer entry (e.g., "A:B"). Whitespace is trimmed and empty parts are ignored.
        Chain identifiers are assigned sequentially as A, B, C, ... and used as FASTA headers.
        A chain whose sequence is in ``msa_paths`` gets that MSA file as the third header field; otherwise it gets
        ``empty`` (a single-sequence run) if ``use_msa_server=False`` and nothing (Boltz generates its MSA) if not.
        :meth:`fold` assigns the MSA of every chain with :func:`~boileroom.models.boltz.msa.resolve_chain_msas`
        instead, which also handles caller alignments and never mixes custom and generated MSAs.

        Parameters
        ----------
        sequences : List[str]
            Sequence entries or colon-separated multimer entries.
        msa_paths : Optional[Dict[str, Path]]
            Optional mapping from sequence to the MSA file of that sequence.
        config : Optional[dict]
            Optional configuration dictionary; ``use_msa_server`` selects ``empty`` for chains without an MSA file.

        Returns
        -------
        str
            FASTA-formatted text where each chain has a header of the form ``>X|protein|<msa field>`` followed by the
            corresponding sequence on the next line.
        """
        chains = split_chains(sequences)
        use_msa_server = True if config is None else bool(config.get("use_msa_server", True))
        paths = msa_paths or {}
        fields: dict[int, Any] = {
            index: paths.get(sequence, "" if use_msa_server else "empty") for index, sequence in enumerate(chains)
        }
        return chains_to_fasta(chains, fields)

    def _prepare_inputs(self, fasta_text: str, work_dir: Path, cache_dir: Path, config: dict) -> dict[str, Any]:
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
        # Upstream Boltz preprocessing currently does not expose auth/header parameters.

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
        self, processed: dict[str, Any], num_workers: int, cache_dir: Path, override_method: str | None = None
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
        self, item: dict[str, Any]
    ) -> tuple[
        np.ndarray | None,
        dict[str, Any],
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
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
        coords_like = item.get("sample_atom_coords")
        if coords_like is None:
            coords_like = item.get("coords")
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

        aggregated: dict[str, Any] = {}
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
                nested: dict[str, dict[str, float]] = {}
                for i in range(pair_np.shape[-2]):
                    nested[str(i)] = {str(j): float(pair_np[i, j]) for j in range(pair_np.shape[-1])}
                aggregated["pair_chains_iptm"] = nested

        if "pair_chains_iptm" in aggregated and isinstance(aggregated["pair_chains_iptm"], dict):
            chains_ptm: dict[str, float] = {}
            for i_str, inner in aggregated["pair_chains_iptm"].items():
                if i_str in inner:
                    chains_ptm[i_str] = float(inner[i_str])
            if chains_ptm:
                aggregated["chains_ptm"] = chains_ptm

        return coords, aggregated, plddt, pae, pde

    def _validate_sample_arrays(self, plddt: np.ndarray | None, pae: np.ndarray | None, pde: np.ndarray | None) -> None:
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
        from io import StringIO

        from biotite.structure.io.pdb import PDBFile, set_structure

        structure_file = PDBFile()
        set_structure(structure_file, atom_array)
        string = StringIO()
        structure_file.write(string)
        return string.getvalue()

    def _convert_outputs_using_boltz_structure(
        self,
        coords: np.ndarray,
        processed: dict[str, Any],
        plddt: list[np.ndarray | None] | None = None,
    ) -> tuple[list[Any], list[str]]:
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
            Entries may be None when a sample lacks pLDDT output.

        Returns
        -------
        tuple[List[Any], List[str]]
            A tuple (atom_arrays, cif_strings) where `atom_arrays` is a list of Biotite AtomArray-like structures (one per sample) and `cif_strings` is a list of corresponding MMCIF-formatted strings.
        """
        from dataclasses import replace
        from io import StringIO

        from biotite.structure.io.pdbx import CIFFile, get_structure

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
            if plddt and sample_idx < len(plddt) and plddt[sample_idx] is not None:
                sample_plddt_value = plddt[sample_idx]
                sample_plddt = (
                    torch.from_numpy(sample_plddt_value)
                    if isinstance(sample_plddt_value, np.ndarray)
                    else sample_plddt_value
                )

            cif_string = to_mmcif(updated_structure, plddts=sample_plddt, boltz2=True)
            cif_strings.append(cif_string)

            atom_arrays.append(get_structure(CIFFile.read(StringIO(cif_string)), model=1))

        return atom_arrays, cif_strings

    def fold(self, sequences: str | Sequence[str], options: dict | None = None) -> Boltz2Output:
        """Predicts 3D structures for the given protein sequences using the resident Boltz-2 model.

        Parameters
        ----------
        sequences : Union[str, Sequence[str]]
            One or more protein sequences. Accepted formats include a single sequence string, a chain-delimited string like "A:B" for multiple chains, or a sequence/list of individual chain strings.
        options : Optional[dict]
            Optional runtime configuration overrides for this call. Supported keys include `seed` (int) to control randomness, `msa_cache_enabled` (bool), MSA server options such as `use_msa_server`, `msa_server_url`, `msa_pairing_strategy`, `max_msa_seqs`, `num_workers` (int) for data loading, and `include_fields` (iterable) to request additional output fields such as `"pdb"` or `"cif"`. Load-bound settings such as sampling and diffusion parameters must be configured when constructing the core.

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
        metadata = dataclasses.replace(self._metadata_template)
        metadata.sequence_lengths = self._compute_sequence_lengths(validated_sequences)

        # Individual chains: the MSA of each one is decided on its own
        individual_chains = split_chains(validated_sequences)
        use_msa_server = bool(effective_config.get("use_msa_server", True))
        msa_cache_enabled = bool(effective_config.get("msa_cache_enabled", True))

        # Always use a temporary working directory on the machine
        cache_dir_str = effective_config.get("cache_dir")
        if cache_dir_str is not None:
            cache_dir = Path(cache_dir_str)
        else:
            if self.model_dir is None:
                raise ValueError("model_dir must be set when cache_dir is not provided")
            cache_dir = Path(self.model_dir) / "boltz"

        with TemporaryDirectory() as tmp:
            work_path = Path(tmp).resolve()

            with Timer("Boltz-2 preprocessing") as preprocess_timer:
                alignments = provided_alignments(effective_config.get("msa"), len(individual_chains))
                provided = stage_provided_alignments(alignments, individual_chains, work_path / "provided_msa")
                sources: dict[int, str] = {}
                fields = resolve_chain_msas(
                    individual_chains,
                    provided=provided,
                    use_msa_server=use_msa_server,
                    cache_enabled=msa_cache_enabled,
                    lookup=lambda chains: self._lookup_cached_msas(chains, effective_config),
                    fetch=lambda sequence: self._fetch_chain_msa(sequence, work_path / "fetched_msa", effective_config),
                    store=lambda sequence, path: self._store_msa(sequence, path, effective_config),
                    sources=sources,
                )
                msa_report = ", ".join(f"{chr(65 + index)}={source}" for index, source in sorted(sources.items()))
                logger.info(f"Boltz-2 MSA source per chain: {msa_report}")
                fasta_text = chains_to_fasta(individual_chains, fields)
                processed = self._prepare_inputs(fasta_text, work_path, cache_dir, effective_config)

            datamodule = self._build_datamodule(processed, int(effective_config.get("num_workers", 2)), cache_dir)

            with Timer("Boltz-2 inference") as inference_timer:
                preds = self._predict_with_trainer(datamodule)

            with Timer("Boltz-2 postprocessing") as postprocess_timer:
                # Extract and organize per-sample outputs succinctly
                extracted = [self._extract_sample_from_pred(item) for item in (preds or []) if isinstance(item, dict)]

                coords_list: list[np.ndarray] = []
                confidences: list[dict[str, Any] | None] = []
                plddts: list[np.ndarray | None] = []
                paes: list[np.ndarray | None] = []
                pdes: list[np.ndarray | None] = []

                for coords, confidence, plddt, pae, pde in extracted:
                    if coords is None:
                        continue
                    coords_list.append(coords)
                    confidences.append(confidence or None)
                    plddts.append(plddt)
                    paes.append(pae)
                    pdes.append(pde)

                # Validate shapes where present
                for arr in plddts:
                    if arr is not None:
                        self._validate_sample_arrays(arr, None, None)
                for arr in paes:
                    if arr is not None:
                        self._validate_sample_arrays(None, arr, None)
                for arr in pdes:
                    if arr is not None:
                        self._validate_sample_arrays(None, None, arr)

                # Extract coords directly for atom_array generation
                coords_np = np.array(coords_list, dtype=object)
                atom_array_list, cif_strings = self._convert_outputs_using_boltz_structure(
                    coords_np,
                    processed,
                    plddt=plddts if any(arr is not None for arr in plddts) else None,
                )

                include_fields = effective_config.get("include_fields")
                pdb_list = None
                if include_fields and ("*" in include_fields or "pdb" in include_fields):
                    pdb_list = [self._convert_outputs_to_pdb(arr) for arr in atom_array_list]

                cif_list = None
                if include_fields and ("*" in include_fields or "cif" in include_fields):
                    cif_list = cif_strings

            metadata.preprocessing_time = preprocess_timer.duration
            metadata.inference_time = inference_timer.duration
            metadata.postprocessing_time = postprocess_timer.duration

            # Build full output with all fields
            full_output = Boltz2Output(
                metadata=metadata,
                confidence=(confidences if any(item is not None for item in confidences) else None),
                plddt=(plddts if any(item is not None for item in plddts) else None),
                pae=(paes if any(item is not None for item in paes) else None),
                pde=(pdes if any(item is not None for item in pdes) else None),
                pdb=pdb_list,
                cif=cif_list,
                atom_array=atom_array_list,
            )

            # Apply filtering based on include_fields
            filtered = self._filter_include_fields(full_output, include_fields)
            return cast(Boltz2Output, filtered)
