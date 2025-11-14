import os
import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Mapping, Optional, Sequence, Union, cast

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
from ...backend.base import Backend
from ...backend.modal import app
from .image import chai_image
from ...images.volumes import model_weights
from ...utils import MODAL_MODEL_DIR, MINUTES, Timer

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

    metadata: PredictionMetadata
    atom_array: Optional[list[AtomArray]] = None  # Always generated, one AtomArray per sample

    # Additional Chai-1-specific outputs (all optional, filtered by include_fields)
    pae: Optional[list[np.ndarray]] = None
    pde: Optional[list[np.ndarray]] = None
    plddt: Optional[list[np.ndarray]] = None
    ptm: Optional[list[np.ndarray]] = None
    iptm: Optional[list[np.ndarray]] = None
    per_chain_iptm: Optional[list[np.ndarray]] = None
    cif: Optional[list[str]] = None


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
        "include_fields": None,  # Optional[List[str]] - controls which fields to include in output
    }
    # Static config keys that can only be set at initialization
    STATIC_CONFIG_KEYS = {"device"}

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize the Chai1Core folding algorithm with the given configuration.

        Creates prediction metadata for "Chai-1" (version "v0.6.1"), resolves the model directory from the
        MODEL_DIR environment variable or a default modal model path, and initializes internal placeholders
        for the computation device and model trunk.

        Parameters
        ----------
        config : dict, optional
            Configuration overrides for the algorithm; keys merge with defaults and control runtime behavior (e.g., device, sampling, and output options).
        """
        super().__init__(config or {})
        self.metadata = self._initialize_metadata(
            model_name="Chai-1",
            model_version="v0.6.1",
        )
        self.model_dir: Optional[str] = os.environ.get("MODEL_DIR", MODAL_MODEL_DIR)
        self._device: torch.device | None = None
        self._trunk: Any | None = None

    def _initialize(self) -> None:
        """Initialize the core model for use by loading required resources.

        Prepares internal state so the instance is ready to run predictions.
        """
        self._load()

    def _load(self) -> None:
        """Initialize the core's runtime state by resolving the execution device and marking the model ready.

        Sets the instance's `_device` to the resolved device and sets `ready` to True.
        """
        self._device = self._resolve_device()
        self.ready = True

    def fold(self, sequences: Union[str, Sequence[str]], options: Optional[dict] = None) -> Chai1Output:
        # TODO: ADD msa_directory: Path | None = None,
        # TODO: constraint path not tested properly
        # TODO: ADD template_hits_path: Path | None = None,

        # Merge static config with per-call options
        """Run structure prediction for the given sequence(s) and return the assembled Chai1Output.

        Parameters
        ----------
        sequences : str | Sequence[str]
            A single amino-acid sequence or a sequence of sequences. A single sequence may contain multiple chains separated by ':'; otherwise provide one entry per sample.
        options : dict | None, optional
            Per-call configuration overrides merged with the algorithm's defaults (e.g., sampling, recycling, include_fields). Only keys recognized by the algorithm are applied.

        Returns
        -------
        Chai1Output
            Prediction result including metadata (with prediction time), a list of AtomArray structures, and any requested confidence metrics and CIF representations as dictated by `options` / include_fields.
        """
        effective_config = self._merge_options(options)

        validated_sequences = self._validate_sequences(sequences)
        self.metadata.sequence_lengths = self._compute_sequence_lengths(validated_sequences)

        with TemporaryDirectory() as buffer_dir:
            buffer_path = Path(buffer_dir)
            fasta_path = self._write_fasta(validated_sequences, buffer_path)
            constraint_path = self._write_constraint(buffer_path, effective_config)
            output_dir = buffer_path / "outputs"
            output_dir.mkdir(parents=True, exist_ok=True)
            with Timer("Model Inference") as timer:
                candidate = run_inference(
                    fasta_file=fasta_path,
                    output_dir=output_dir,
                    constraint_path=constraint_path,
                    num_trunk_samples=effective_config["num_trunk_samples"],
                    num_trunk_recycles=effective_config["num_trunk_recycles"],
                    num_diffn_timesteps=effective_config["num_diffn_timesteps"],
                    num_diffn_samples=effective_config["num_diffn_samples"],
                    use_esm_embeddings=effective_config["use_esm_embeddings"],
                    use_msa_server=effective_config["use_msa_server"],
                    use_templates_server=effective_config["use_templates_server"],
                    device=str(self._device),
                    low_memory=False,
                )
            output = self._convert_outputs(candidate, elapsed_time=timer.duration, effective_config=effective_config)

        return output

    def _write_constraint(self, buffer_path: Path, config: dict) -> Optional[Path]:
        """Write a constraint CSV file into buffer_path when constraint definitions are present in config.

        If config["constraint_path"] is a path-like string or Path, the file is copied to buffer_path/constraint.csv.
        If it is a sequence of mappings, a CSV is written with a fixed schema (columns: restraint_id, chainA, res_idxA, chainB, res_idxB, connection_type, confidence, min_distance_angstrom, max_distance_angstrom, comment). Several input keys are accepted via aliases (e.g., "chain_a" → "chainA", "residue_token_a" → "res_idxA"). Returns the Path to the written CSV, or None when no constraint was provided.

        Parameters
        ----------
        buffer_path : Path
            Directory where constraint.csv will be created.
        config : dict
            Configuration dictionary; expected to contain a "constraint_path" key whose value is either a path-like object (string/Path) or a sequence of mapping entries describing restraints.

        Returns
        -------
        Optional[Path]
            Path to the created constraint CSV, or `None` if no constraint definition is present.

        Raises
        ------
        TypeError
            If "constraint_path" is neither a path-like object nor a sequence, or if any entry in the sequence is not a mapping.
        """
        constraint_definition = config.get("constraint_path")
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

    def _convert_outputs(
        self, candidate: "StructureCandidates", elapsed_time: float, effective_config: dict
    ) -> Chai1Output:
        """Convert a raw inference candidate into a Chai1Output, applying include_fields filtering.

        Parameters
        ----------
        candidate : StructureCandidates
            The raw prediction candidate produced by the inference engine for a single sample.
        elapsed_time : float
            Total wall-clock time spent generating the prediction (seconds); stored in output metadata.
        effective_config : dict
            Merged configuration used for this call; may contain `include_fields` to limit returned fields.

        Returns
        -------
        Chai1Output
            A fully populated prediction result containing available confidence metrics, optional CIF strings,
            and the AtomArray structures; fields not produced are set to `None`.
        """
        pae, pde, plddt, ptm, iptm, per_chain_iptm = self._extract_confidence_metrics(candidate)
        cif_string_list, atom_array = self._process_cif(candidate.cif_paths[0], effective_config)
        self.metadata.prediction_time = elapsed_time

        # Build full output with all attributes
        # cif_string_list is a list with one element; convert [None] to None, keep [string] as [string]
        # Filter out None values to match type list[str]
        cif: Optional[list[str]] = None
        if cif_string_list and cif_string_list[0] is not None:
            cif = [s for s in cif_string_list if s is not None]

        full_output = Chai1Output(
            metadata=self.metadata,
            pae=pae if pae else None,
            pde=pde if pde else None,
            plddt=plddt if plddt else None,
            ptm=ptm if ptm else None,
            iptm=iptm if iptm else None,
            per_chain_iptm=per_chain_iptm if per_chain_iptm else None,
            cif=cif,
            atom_array=atom_array,
        )

        # Apply filtering based on include_fields
        include_fields = effective_config.get("include_fields")
        filtered = self._filter_include_fields(full_output, include_fields)
        return cast(Chai1Output, filtered)

    def _write_fasta(self, sequences: list[str], buffer_dir: Path) -> Path:
        # assert that a single batch only
        """Write a FASTA file for a single-batch sequence input, splitting multiple chains in the batch by ':'.

        Parameters
        ----------
        sequences : list[str]
            A single-item list containing the sequence batch; if that string contains ':' it is split into multiple chain sequences.
        buffer_dir : Path
            Directory where the file named "input.fasta" will be written.

        Returns
        -------
        Path
            The path to the created FASTA file ("input.fasta" in buffer_dir).
        """
        assert len(sequences) == 1, "Chai-1 only supports a single batch for now."
        seqs = sequences[0].split(":") if ":" in sequences[0] else [sequences[0]]
        fasta_path = buffer_dir / "input.fasta"
        # TODO: naming of the chains should be synchronized with how ESMFold did that
        entries = [f">protein|name=chain_{index}\n{sequence}" for index, sequence in enumerate(seqs)]
        fasta_path.write_text("\n".join(entries))
        return fasta_path

    def _process_cif(self, cif_path: Path, config: dict) -> tuple[list[Optional[str]], list[AtomArray]]:
        """Read a CIF file and produce its AtomArray representation, optionally including the CIF text.

        Parameters
        ----------
        cif_path : Path
            Path to the CIF file to read.
        config : dict
            Configuration dictionary; if its "include_fields" contains "*" or "cif",
            the CIF file content will be returned as a string.

        Returns
        -------
        tuple[list[Optional[str]], list[AtomArray]]
            A pair where the first element is a single-item
            list containing the CIF file content string if requested or None otherwise, and the second
            element is a list with the extracted AtomArray structure.
        """
        cif_file = CIFFile.read(str(cif_path))
        structure = get_structure(cif_file)

        # Always generate atom_array
        atom_array = [structure]

        # Generate CIF string only if requested via include_fields
        cif_string = None
        include_fields = config.get("include_fields")
        if include_fields and ("*" in include_fields or "cif" in include_fields):
            with open(cif_path, "r") as f:
                cif_string = f.read()

        return [cif_string], atom_array

    def _extract_confidence_metrics(
        self,
        candidates: "StructureCandidates",
    ) -> tuple[
        list[np.ndarray],
        list[np.ndarray],
        list[np.ndarray],
        Optional[list[np.ndarray]],
        Optional[list[np.ndarray]],
        Optional[list[np.ndarray]],
    ]:
        """Extract confidence metrics from a StructureCandidates object.

        Parameters
        ----------
        candidates : StructureCandidates
            Candidate prediction outputs containing per-model matrices and ranking data.

        Returns
        -------
        tuple
            A 6-tuple containing:
            - pae (list[np.ndarray]): List of per-model predicted aligned error (PAE) matrices.
            - pde (list[np.ndarray]): List of per-model predicted distance error (PDE) matrices.
            - plddt (list[np.ndarray]): List of per-model per-residue pLDDT arrays.
            - ptm (Optional[list[np.ndarray]]): List with complex pTM scores if ranking data is available, otherwise `None`.
            - iptm (Optional[list[np.ndarray]]): List with interface pTM scores if ranking data is available, otherwise `None`.
            - per_chain_iptm (Optional[list[np.ndarray]]): List with per-chain pair iPTM scores if ranking data is available, otherwise `None`.
        """
        pae = self._collect_matrix(candidates, attribute_name="pae")
        pde = self._collect_matrix(candidates, attribute_name="pde")
        plddt = self._collect_matrix(candidates, attribute_name="plddt")

        ranking_data = candidates.ranking_data
        if len(ranking_data) != 1:
            logger.warning(
                f"Expected exactly 1 ranking data entry; got {len(ranking_data)}. Skipping ptm/iptm/per_chain_iptm."
            )
            return pae, pde, plddt, None, None, None

        ptm = np.asarray(ranking_data[0].ptm_scores.complex_ptm)
        iptm = np.asarray(ranking_data[0].ptm_scores.interface_ptm)
        per_chain_iptm = np.asarray(ranking_data[0].ptm_scores.per_chain_pair_iptm)

        return pae, pde, plddt, [ptm], [iptm], [per_chain_iptm]

    def _collect_matrix(
        self,
        candidates: "StructureCandidates",
        attribute_name: str,
    ) -> list[np.ndarray]:
        """Collect a named matrix collection from a candidates object and return it as a list of NumPy arrays.

        Parameters
        ----------
        candidates : StructureCandidates
            The object (typically a StructureCandidates) that may expose matrix collections as attributes.
        attribute_name : str
            The attribute name on `candidates` to retrieve.

        Returns
        -------
        list[np.ndarray]
            A list of arrays converted with `np.asarray`. Returns an empty list if the attribute is missing or None.
        """
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
    volumes={MODAL_MODEL_DIR: model_weights},  # TODO: somehow link this to what Chai-1 actually uses
)
class ModalChai1:
    """
    Modal-specific wrapper around `Chai1Core`.
    """

    config: bytes = modal.parameter(default=b"{}")

    @modal.enter()
    def _initialize(self) -> None:
        """Instantiate Chai1Core from the JSON-encoded `self.config` bytes and perform its initialization.

        This decodes `self.config` as UTF-8 JSON, constructs a Chai1Core with the resulting dict, and calls its `_initialize` method.
        """
        self._core = Chai1Core(json.loads(self.config.decode("utf-8")))
        self._core._initialize()

    @modal.method()
    def fold(self, sequences: Union[str, Sequence[str]], options: Optional[dict] = None) -> Chai1Output:
        """Run structure prediction for the given sequence(s) and return the assembled prediction output.

        Parameters
        ----------
        sequences : str | Sequence[str]
            One sequence string or a sequence of sequence strings. Individual entries may contain multiple chains separated by ":"; when provided as a single string that contains ":" the string will be split into chains.
        options : dict, optional
            Per-call configuration overrides merged with the model's default configuration to control sampling, device, and which result fields to include.

        Returns
        -------
        Chai1Output
            Prediction results and associated metadata for the provided sequence(s).
        """
        return self._core.fold(sequences, options=options)


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
        """Create a Chai1 model wrapper and start the selected backend.

        Parameters
        ----------
        backend : str
            Backend type to use. Supported values are "modal" (deploy via Modal) and "local" (run using the local backend).
        device : Optional[str]
            Optional device identifier to pass to the backend (e.g., "cuda:0" or None to let the backend choose).
        config : Optional[dict]
            Optional configuration dictionary forwarded to the underlying Chai1Core or backend.

        Raises
        ------
        ValueError
            If an unsupported backend name is provided.
        """
        if config is None:
            config = {}
        self.config = config
        self.device = device
        backend_instance: Backend
        if backend == "modal":
            backend_instance = ModalBackend(ModalChai1, config, device=device)
        elif backend == "local":
            backend_instance = LocalBackend(Chai1Core, config, device=device)
        else:
            raise ValueError(f"Backend {backend} not supported")
        self._backend = backend_instance
        self._backend.start()

    def fold(self, sequences: Union[str, Sequence[str]], options: Optional[dict] = None) -> Chai1Output:
        """Run structure prediction for the given sequence(s) using the configured backend.

        Parameters
        ----------
        sequences : str | Sequence[str]
            A single sequence or a sequence of sequences to predict. Each sequence may contain multiple chains separated by ":"; currently the implementation expects a single batch.
        options : dict | None, optional
            Per-call configuration overrides merged with the model's default config (e.g., include_fields, constraint_path, device-specific options).

        Returns
        -------
        Chai1Output
            Prediction results including metadata, generated atom arrays, and any requested confidence metrics or CIF output.
        """
        return self._call_backend_method("fold", sequences, options=options)
