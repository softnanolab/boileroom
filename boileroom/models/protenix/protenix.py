"""Protenix (AlphaFold 3 reproduction) wrapper for protein structure prediction.

Protenix natively supports multi-chain complexes without requiring linker hacks
or positional encoding skips that ESMFold needs. It also provides rich confidence
metrics including per-atom pLDDT, PAE matrices, pTM, and ipTM scores.
"""

import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import List, Optional, Union

import modal
import numpy as np
from biotite.structure import AtomArray

from ... import app
from ...base import FoldingAlgorithm, PredictionMetadata, StructurePrediction
from ...images.protenix import protenix_image
from ...images.volumes import model_weights
from ...utils import MINUTES, MODEL_DIR, GPUS_AVAIL_ON_MODAL, Timer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProtenixOutput(StructurePrediction):
    """Output from Protenix prediction.

    Contains the predicted structure coordinates, confidence metrics (pLDDT, PAE, pTM, ipTM),
    and optionally the biotite AtomArray representation.

    Attributes
    ----------
    positions : np.ndarray
        Predicted atom coordinates, shape [N_sample, N_atom, 3].
    metadata : PredictionMetadata
        Prediction metadata (model name, version, timing, etc.).
    plddt : np.ndarray
        Per-atom pLDDT scores (0-1), shape [N_sample, N_atom].
    pae : np.ndarray
        Predicted Aligned Error matrix (per token pair), shape [N_sample, N_token, N_token].
    ptm : np.ndarray
        Predicted TM-score, shape [N_sample].
    iptm : np.ndarray
        Interface predicted TM-score, shape [N_sample].
    ranking_score : np.ndarray
        Overall ranking score, shape [N_sample].
    chain_index : np.ndarray
        Per-token chain/asymmetric unit index, shape [N_token].
    token_to_atom_map : np.ndarray
        Mapping from atoms to their token indices, shape [N_atom].
    atom_array : Optional[list[AtomArray]]
        Biotite AtomArray representation, one per sample. Only populated if
        config["output_atomarray"] is True.
    pdb : Optional[list[str]]
        PDB format strings. Only populated if config["output_pdb"] is True.
    cif : Optional[list[str]]
        CIF format strings. Only populated if config["output_cif"] is True.
    """

    # Required by StructurePrediction protocol
    positions: np.ndarray  # [N_sample, N_atom, 3]
    metadata: PredictionMetadata

    # Protenix-specific confidence outputs
    plddt: np.ndarray  # [N_sample, N_atom] per-atom pLDDT (0-1 scale)
    pae: np.ndarray  # [N_sample, N_token, N_token] predicted aligned error
    ptm: np.ndarray  # [N_sample] predicted TM-score
    iptm: np.ndarray  # [N_sample] interface predicted TM-score
    ranking_score: np.ndarray  # [N_sample] overall ranking score

    # Chain / residue mapping information
    chain_index: np.ndarray  # [N_token] asym_id per token
    token_to_atom_map: np.ndarray  # [N_atom] maps each atom to its token index

    # Optional output formats
    atom_array: Optional[list[AtomArray]] = None
    pdb: Optional[list[str]] = None
    cif: Optional[list[str]] = None


GPU_TO_USE = os.environ.get("BOILEROOM_GPU", "A100-80GB")

if GPU_TO_USE not in GPUS_AVAIL_ON_MODAL:
    logger.warning(
        f"GPU specified in BOILEROOM_GPU environment variable ('{GPU_TO_USE}') may not be suitable "
        f"for Protenix. Protenix requires significant GPU memory. Available GPUs: {GPUS_AVAIL_ON_MODAL}"
    )


DEFAULT_CONFIG = {
    # Output format config
    "output_pdb": False,
    "output_cif": False,
    "output_atomarray": False,
    # Protenix inference config
    "model_name": "protenix_base_default_v1.0.0",
    "n_cycle": 10,
    "n_step": 200,
    "n_sample": 1,
    "seed": 101,
    "use_msa": False,
    "use_template": False,
    "dtype": "bf16",
}


@app.cls(
    image=protenix_image,
    gpu=GPU_TO_USE,
    timeout=30 * MINUTES,
    scaledown_window=10 * MINUTES,
    volumes={MODEL_DIR: model_weights},
)
class ProtenixFold:
    """Protenix protein structure prediction model.

    Wraps the Protenix inference pipeline (an AlphaFold 3 reproduction by ByteDance)
    for use as a folding algorithm in boileroom.

    Unlike ESMFold, Protenix natively supports multi-chain protein complexes
    without needing glycine linkers or positional encoding skips.

    Parameters
    ----------
    config_json : str
        JSON-serialised configuration dictionary. Supported keys:

        - ``output_pdb`` (bool): Whether to produce PDB strings. Default False.
        - ``output_cif`` (bool): Whether to produce CIF strings. Default False.
        - ``output_atomarray`` (bool): Whether to produce biotite AtomArrays. Default False.
        - ``model_name`` (str): Name of the Protenix model checkpoint.
        - ``n_cycle`` (int): Number of pairformer recycling cycles. Default 10.
        - ``n_step`` (int): Number of diffusion steps. Default 200.
        - ``n_sample`` (int): Number of output samples per seed. Default 1.
        - ``seed`` (int): Random seed for reproducibility. Default 101.
        - ``use_msa`` (bool): Whether to use MSA features. Default False.
        - ``use_template`` (bool): Whether to use template features. Default False.
        - ``dtype`` (str): Precision for inference ("bf16" or "fp32"). Default "bf16".
    """

    # ---- Modal parameter: config is serialised as JSON string ----
    config_json: str = modal.parameter(default="{}")

    @modal.enter()
    def _initialize(self) -> None:
        """Initialize the model during container startup."""
        # Deserialise config and merge with defaults
        user_config = json.loads(self.config_json) if self.config_json else {}
        self.config = {**DEFAULT_CONFIG, **user_config}
        self.metadata = PredictionMetadata(
            model_name="Protenix",
            model_version="1.0.0",
            prediction_time=None,
            sequence_lengths=None,
        )
        self.model_dir: Optional[str] = os.environ.get("MODEL_DIR", MODEL_DIR)
        self.runner = None
        self._load()

    def _load(self) -> None:
        """Load the Protenix model and prepare it for inference."""
        import sys
        import torch

        # We need to add the protenix package paths to sys.path
        # since protenix uses relative imports from its runner/ and configs/ directories
        try:
            import protenix
            protenix_root = os.path.dirname(os.path.dirname(protenix.__file__))
            if protenix_root not in sys.path:
                sys.path.insert(0, protenix_root)
        except ImportError:
            logger.warning("Protenix not found as installed package, trying local paths...")

        from configs.configs_base import configs as configs_base
        from configs.configs_data import data_configs
        from configs.configs_inference import inference_configs
        from configs.configs_model_type import model_configs
        from protenix.config.config import parse_configs
        from runner.inference import InferenceRunner, download_inference_cache, update_gpu_compatible_configs

        # Build config
        model_name = self.config["model_name"]
        base_cfg = {**configs_base, **data_configs, **inference_configs}

        if model_name in model_configs:
            base_cfg.update(model_configs[model_name])

        base_cfg["model_name"] = model_name
        base_cfg["seeds"] = [self.config["seed"]]
        base_cfg["sample_diffusion.N_sample"] = self.config["n_sample"]
        base_cfg["sample_diffusion.N_step"] = self.config["n_step"]
        base_cfg["N_cycle"] = self.config["n_cycle"]
        base_cfg["dtype"] = self.config["dtype"]
        base_cfg["use_msa"] = self.config["use_msa"]
        base_cfg["use_template"] = self.config["use_template"]
        base_cfg["need_atom_confidence"] = True
        base_cfg["dump_dir"] = tempfile.mkdtemp()

        # Parse configs into a ConfigDict
        configs = parse_configs(base_cfg, fill_defaults=True)
        configs = update_gpu_compatible_configs(configs)

        # Download weights if needed
        download_inference_cache(configs)

        # Create the runner (this loads the model)
        self.runner = InferenceRunner(configs)
        self.configs = configs
        self.ready = True

    def _sequences_to_input_json(self, sequences: List[str]) -> List[dict]:
        """Convert a list of protein chain sequences into Protenix JSON input format.

        Each element of ``sequences`` is the amino acid sequence for one chain.
        Unlike ESMFold, chains are NOT concatenated with ":" -- instead, each chain
        becomes a separate ``proteinChain`` entity in the JSON input, allowing
        Protenix to handle multi-chain complexes natively.

        Parameters
        ----------
        sequences : List[str]
            List of protein chain sequences (one per chain).

        Returns
        -------
        List[dict]
            Protenix input JSON structure (a list containing one sample dict).
        """
        entities = []
        for seq in sequences:
            entities.append({
                "proteinChain": {
                    "sequence": seq,
                    "count": 1,
                }
            })

        sample = {
            "name": "bagel_prediction",
            "sequences": entities,
        }

        return [sample]

    def _validate_sequences(self, sequences: Union[str, List[str]]) -> list[str]:
        """Validate input sequences and convert to list format."""
        from ...utils import validate_sequence

        if isinstance(sequences, str):
            sequences = [sequences]
        return [seq for seq in sequences if validate_sequence(seq)]

    @modal.method()
    def fold(self, sequences: Union[str, List[str]]) -> ProtenixOutput:
        """Predict protein structure using Protenix.

        Parameters
        ----------
        sequences : Union[str, List[str]]
            Either a single sequence string, or a list of chain sequences.
            For multi-chain complexes, pass a list with one sequence per chain.
            Unlike ESMFold, do NOT use ":" to separate chains -- each chain
            should be a separate element in the list.

        Returns
        -------
        ProtenixOutput
            Prediction results including coordinates, confidence metrics, and
            optionally AtomArray/PDB/CIF representations.
        """
        import torch
        from protenix.data.inference.infer_dataloader import get_inference_dataloader
        from protenix.utils.seed import seed_everything

        if self.runner is None:
            logger.warning("Model not loaded. Loading now...")
            self._load()
        assert self.runner is not None, "Model not loaded"

        # Handle input: single sequence or list of chains
        if isinstance(sequences, str):
            # Could be a single chain or colon-separated chains
            if ":" in sequences:
                chain_sequences = sequences.split(":")
            else:
                chain_sequences = [sequences]
        else:
            # Already a list of chains -- flatten any colon-separated entries
            chain_sequences = []
            for seq in sequences:
                if ":" in seq:
                    chain_sequences.extend(seq.split(":"))
                else:
                    chain_sequences.append(seq)

        # Validate sequences
        chain_sequences = self._validate_sequences(chain_sequences)
        self.metadata.sequence_lengths = [len(s) for s in chain_sequences]

        # Create input JSON
        input_json_data = self._sequences_to_input_json(chain_sequences)

        # Write temporary JSON file for protenix
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(input_json_data, f, indent=2)
            input_json_path = f.name

        try:
            # Update configs for this specific prediction
            self.configs.input_json_path = input_json_path
            self.configs.seeds = [self.config["seed"]]

            # Seed everything
            seed_everything(seed=self.config["seed"], deterministic=False)

            # Create dataloader
            dataloader = get_inference_dataloader(configs=self.configs)

            with Timer("Protenix Inference") as timer:
                # Process the single batch
                for batch in dataloader:
                    data, atom_array, data_error_message = batch[0]

                    if len(data_error_message) > 0:
                        raise RuntimeError(f"Data processing error: {data_error_message}")

                    # Run prediction
                    prediction = self.runner.predict(data)
                    break  # Only one sample in our input

            # Extract results
            output = self._convert_outputs(
                prediction=prediction,
                atom_array=atom_array,
                input_feature_dict=data["input_feature_dict"],
                chain_sequences=chain_sequences,
                prediction_time=timer.duration,
            )

            return output

        finally:
            # Clean up temp file
            if os.path.exists(input_json_path):
                os.unlink(input_json_path)

    def _convert_outputs(
        self,
        prediction: dict,
        atom_array: AtomArray,
        input_feature_dict: dict,
        chain_sequences: List[str],
        prediction_time: float,
    ) -> ProtenixOutput:
        """Convert Protenix prediction outputs to ProtenixOutput format.

        Parameters
        ----------
        prediction : dict
            Raw prediction dictionary from Protenix model.
        atom_array : AtomArray
            Reference atom array from the dataloader.
        input_feature_dict : dict
            Input feature dictionary containing token/atom mappings.
        chain_sequences : List[str]
            List of chain sequences.
        prediction_time : float
            Time taken for inference in seconds.

        Returns
        -------
        ProtenixOutput
            Structured output with coordinates and confidence metrics.
        """
        import torch

        self.metadata.prediction_time = prediction_time

        # Extract coordinates: [N_sample, N_atom, 3]
        coordinates = prediction["coordinate"]
        if isinstance(coordinates, torch.Tensor):
            coordinates = coordinates.cpu().float().numpy()

        n_sample = coordinates.shape[0]

        # Extract per-atom pLDDT from full_data
        # full_data is a list of per-sample dicts, each containing "atom_plddt"
        plddt_per_sample = []
        for sample_idx in range(n_sample):
            if prediction["full_data"][sample_idx] and "atom_plddt" in prediction["full_data"][sample_idx]:
                atom_plddt = prediction["full_data"][sample_idx]["atom_plddt"]
                if isinstance(atom_plddt, torch.Tensor):
                    atom_plddt = atom_plddt.cpu().float().numpy()
                plddt_per_sample.append(atom_plddt)
            else:
                # Fallback: get from summary confidence
                plddt_val = prediction["summary_confidence"][sample_idx].get("plddt", 0.0)
                if isinstance(plddt_val, torch.Tensor):
                    plddt_val = plddt_val.cpu().float().numpy()
                # Create uniform pLDDT array
                n_atoms = coordinates.shape[1]
                plddt_per_sample.append(np.full(n_atoms, plddt_val / 100.0))

        plddt = np.stack(plddt_per_sample, axis=0)  # [N_sample, N_atom]

        # Extract PAE matrix from full_data
        # token_pair_pae: [N_sample, N_token, N_token]
        pae_per_sample = []
        for sample_idx in range(n_sample):
            if prediction["full_data"][sample_idx] and "token_pair_pae" in prediction["full_data"][sample_idx]:
                token_pae = prediction["full_data"][sample_idx]["token_pair_pae"]
                if isinstance(token_pae, torch.Tensor):
                    token_pae = token_pae.cpu().float().numpy()
                pae_per_sample.append(token_pae)
            else:
                # Fallback: zeros
                n_tokens = input_feature_dict["residue_index"].shape[-1] if "residue_index" in input_feature_dict else 0
                pae_per_sample.append(np.zeros((n_tokens, n_tokens)))

        pae = np.stack(pae_per_sample, axis=0)  # [N_sample, N_token, N_token]

        # Extract pTM and ipTM from summary_confidence
        ptm_values = []
        iptm_values = []
        ranking_values = []
        for sample_idx in range(n_sample):
            sc = prediction["summary_confidence"][sample_idx]

            ptm_val = sc.get("ptm", 0.0)
            if isinstance(ptm_val, torch.Tensor):
                ptm_val = ptm_val.cpu().float().item()
            ptm_values.append(ptm_val)

            iptm_val = sc.get("iptm", 0.0)
            if isinstance(iptm_val, torch.Tensor):
                iptm_val = iptm_val.cpu().float().item()
            iptm_values.append(iptm_val)

            rank_val = sc.get("ranking_score", 0.0)
            if isinstance(rank_val, torch.Tensor):
                rank_val = rank_val.cpu().float().item()
            ranking_values.append(rank_val)

        ptm = np.array(ptm_values)
        iptm = np.array(iptm_values)
        ranking_score = np.array(ranking_values)

        # Extract chain index (asym_id) and atom-to-token mapping
        asym_id = input_feature_dict.get("asym_id", None)
        if asym_id is not None:
            if isinstance(asym_id, torch.Tensor):
                asym_id = asym_id.cpu().numpy()
            chain_index = asym_id.flatten()
        else:
            chain_index = np.zeros(pae.shape[-1], dtype=np.int64)

        atom_to_token_idx = input_feature_dict.get("atom_to_token_idx", None)
        if atom_to_token_idx is not None:
            if isinstance(atom_to_token_idx, torch.Tensor):
                atom_to_token_idx = atom_to_token_idx.cpu().numpy()
            token_to_atom_map = atom_to_token_idx.flatten()
        else:
            token_to_atom_map = np.arange(coordinates.shape[1])

        # Build AtomArrays if requested
        output_atom_arrays = None
        output_pdbs = None
        output_cifs = None

        if self.config["output_atomarray"] or self.config["output_pdb"] or self.config["output_cif"]:
            output_atom_arrays = self._build_atom_arrays(
                coordinates=coordinates,
                atom_array=atom_array,
                plddt=plddt,
            )

            if self.config["output_pdb"]:
                output_pdbs = self._convert_to_pdb(output_atom_arrays)
            if self.config["output_cif"]:
                output_cifs = self._convert_to_cif(output_atom_arrays)
            if not self.config["output_atomarray"]:
                output_atom_arrays = None

        return ProtenixOutput(
            positions=coordinates,
            metadata=self.metadata,
            plddt=plddt,
            pae=pae,
            ptm=ptm,
            iptm=iptm,
            ranking_score=ranking_score,
            chain_index=chain_index,
            token_to_atom_map=token_to_atom_map,
            atom_array=output_atom_arrays,
            pdb=output_pdbs,
            cif=output_cifs,
        )

    def _build_atom_arrays(
        self,
        coordinates: np.ndarray,
        atom_array: AtomArray,
        plddt: np.ndarray,
    ) -> list[AtomArray]:
        """Build biotite AtomArray objects from predicted coordinates.

        Uses the reference atom_array from the protenix dataloader as a template,
        updating only the coordinates and B-factors (pLDDT).

        Parameters
        ----------
        coordinates : np.ndarray
            Predicted coordinates, shape [N_sample, N_atom, 3].
        atom_array : AtomArray
            Reference AtomArray from protenix's data pipeline.
        plddt : np.ndarray
            Per-atom pLDDT scores, shape [N_sample, N_atom].

        Returns
        -------
        list[AtomArray]
            One AtomArray per sample.
        """
        n_sample = coordinates.shape[0]
        arrays = []

        for sample_idx in range(n_sample):
            arr = atom_array.copy()
            # Update coordinates
            n_atoms = min(coordinates.shape[1], len(arr))
            arr.coord[:n_atoms] = coordinates[sample_idx, :n_atoms]
            # Update B-factors with pLDDT * 100
            if hasattr(arr, 'b_factor') or 'b_factor' in arr.get_annotation_categories():
                arr.set_annotation("b_factor", np.round(plddt[sample_idx, :n_atoms] * 100, 2))
            else:
                arr.add_annotation("b_factor", dtype=float)
                arr.b_factor[:n_atoms] = np.round(plddt[sample_idx, :n_atoms] * 100, 2)
            arrays.append(arr)

        return arrays

    def _convert_to_pdb(self, atom_arrays: list[AtomArray]) -> list[str]:
        """Convert AtomArrays to PDB format strings."""
        from biotite.structure.io.pdb import PDBFile, set_structure
        from io import StringIO

        pdbs = []
        for arr in atom_arrays:
            pdb_file = PDBFile()
            set_structure(pdb_file, arr)
            string = StringIO()
            pdb_file.write(string)
            pdbs.append(string.getvalue())
        return pdbs

    def _convert_to_cif(self, atom_arrays: list[AtomArray]) -> list[str]:
        """Convert AtomArrays to CIF format strings."""
        from biotite.structure.io.pdbx import CIFFile, set_structure
        from io import StringIO

        cifs = []
        for arr in atom_arrays:
            cif_file = CIFFile()
            set_structure(cif_file, arr)
            string = StringIO()
            cif_file.write(string)
            cifs.append(string.getvalue())
        return cifs


def get_protenix(gpu_type: str = "A100-80GB", config: dict = {}) -> ProtenixFold:
    """Create a ProtenixFold instance with a specific GPU type.

    Parameters
    ----------
    gpu_type : str
        GPU type to use on Modal (e.g., "A10G", "A100-40GB", "A100-80GB").
        Protenix requires significant GPU memory; A10G (24GB) is the minimum recommended.
    config : dict
        Configuration dictionary (see ProtenixFold docstring for keys).

    Returns
    -------
    ProtenixFold
        Configured ProtenixFold instance.
    """
    Model = ProtenixFold.with_options(gpu=gpu_type)  # type: ignore
    return Model(config_json=json.dumps(config))
