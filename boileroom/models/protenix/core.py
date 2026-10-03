"""Core Protenix implementation backed by a persistent inference runner."""

from __future__ import annotations

import dataclasses
import json
import logging
import sys
from collections.abc import Sequence
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, ClassVar, cast

from biotite.structure.io.pdb import PDBFile
from biotite.structure.io.pdbx import CIFFile, get_structure

from ...base import FoldingAlgorithm, PredictionMetadata
from ...inputs import a3m_rows
from ...optimization import OptimizationResolution, detect_gpu, resolve_optimization, validate_optimization
from ...utils import Timer, get_model_cache_dir
from .._runtime_utils import command_env, include_field
from .._worker import ModelWorker
from .outputs import read_json, read_token_confidence, sample_identity
from .templates import stage_templates
from .types import ProtenixOutput

logger = logging.getLogger(__name__)


class ProtenixCore(FoldingAlgorithm):
    """Protenix structure prediction model."""

    DEFAULT_CONFIG: ClassVar[dict[str, Any]] = {
        "device": None,
        "model_name": "protenix-v2",
        "seeds": "101",
        "cycle": 10,
        "step": 200,
        "sample": 5,
        "dtype": "bf16",
        "use_msa": True,
        "msa_server_url": "https://api.colabfold.com",
        "use_template": False,
        "use_default_params": False,
        "trimul_kernel": "cuequivariance",
        "triatt_kernel": "cuequivariance",
        "enable_cache": True,
        "enable_fusion": True,
        "enable_tf32": True,
        "optimization": "vanilla",
        "use_seeds_in_json": False,
        "use_tfg_guidance": False,
        "unpaired_msa": None,
        "msa": None,
        "templates": None,
        "templates_chain": 0,
        "include_fields": None,
        "timeout_seconds": 3500,
    }
    STATIC_CONFIG_KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "device",
            "model_name",
            "msa_server_url",
            "use_template",
            "trimul_kernel",
            "triatt_kernel",
            "enable_cache",
            "enable_fusion",
            "enable_tf32",
            "optimization",
        }
    )

    # Family hooks: AF3-style runners with Protenix's API (e.g. OpenDDE) subclass and override these.
    FAMILY: ClassVar[str] = "protenix"
    DISPLAY_NAME: ClassVar[str] = "Protenix"
    ROOT_ENV: ClassVar[str] = "PROTENIX_ROOT_DIR"
    RUNTIME_CLASS: ClassVar[str] = "ProtenixRuntime"
    RUNTIME_PATH: ClassVar[Path] = Path(__file__).with_name("runtime.py")
    OUTPUT_CLASS: ClassVar[type[ProtenixOutput]] = ProtenixOutput
    DTYPES: ClassVar[frozenset[str]] = frozenset({"bf16", "fp16", "fp32"})
    #: Whether this family's runtime honours ``templates``. A family that does not must
    #: refuse them: a prediction that quietly ignored a template is mislabelled.
    SUPPORTS_USER_MSA: ClassVar[bool] = True
    SUPPORTS_USER_TEMPLATES: ClassVar[bool] = True

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Create a Protenix core with one reusable model worker."""
        if config and "protenix_command" in config:
            raise ValueError("protenix_command is no longer supported; Protenix uses its Python runner")
        super().__init__(config or {})
        validate_optimization(self.config["optimization"])
        self.optimization: OptimizationResolution | None = None
        self._worker: ModelWorker | None = None
        self._metadata_template = self._initialize_metadata(
            model_name=self.DISPLAY_NAME,
            model_version=str(self.config["model_name"]),
        )

    def _initialize(self) -> None:
        """Load the Protenix model into its persistent worker."""
        self._load()

    def _load(self) -> None:
        """Validate configuration and load model weights once per core."""
        _validate_config(self.config, self.DTYPES)
        mode = str(self.config["optimization"])
        gpu = None if mode == "vanilla" else detect_gpu(self.config.get("device"))
        self.optimization = resolve_optimization(self.FAMILY, mode, gpu)
        if self._worker is None:
            self._worker = ModelWorker(
                self.config,
                self._worker_env(self.config, self.optimization),
                runtime_path=self.RUNTIME_PATH,
                runtime_class=self.RUNTIME_CLASS,
                label=self.DISPLAY_NAME,
                python_executable=self._worker_python(),
            )
        self._worker.start()
        self.ready = True

    def _worker_python(self) -> str:
        """Return the interpreter that runs the model runtime."""
        return sys.executable

    def _worker_env(self, config: dict[str, Any], optimization: OptimizationResolution | None) -> dict[str, str]:
        """Return the worker environment."""
        return _command_env(config, optimization, self.FAMILY, self.ROOT_ENV)

    def close(self) -> None:
        """Release the persistent worker and its model weights."""
        if self._worker is not None:
            self._worker.close()
        self.ready = False

    def fold(self, sequences: str | Sequence[str], options: dict | None = None) -> ProtenixOutput:
        """Run Protenix-style prediction for one sequence entry.

        Use ``:`` inside a sequence string to define multiple protein chains.
        """
        effective_config = self._merge_options(options)
        _validate_config(effective_config, self.DTYPES)
        validated_sequences = self._validate_sequences(sequences)
        if len(validated_sequences) != 1:
            raise ValueError(
                f"{self.DISPLAY_NAME} currently supports exactly one top-level sequence per call; use ':' to join chains."
            )

        metadata = dataclasses.replace(
            self._metadata_template,
            model_version=str(effective_config["model_name"]),
            sequence_lengths=self._compute_sequence_lengths(validated_sequences),
        )

        with TemporaryDirectory() as buffer_dir:
            buffer_path = Path(buffer_dir)
            with Timer(f"{self.DISPLAY_NAME} preprocessing") as preprocess_timer:
                staged = self._stage_templates(validated_sequences[0], buffer_path, effective_config)
                input_json = self._write_input_json(
                    validated_sequences[0],
                    buffer_path,
                    self._resolve_msa(effective_config),
                    staged,
                    effective_config["templates_chain"],
                )
                output_dir = buffer_path / "outputs"
                output_dir.mkdir(parents=True, exist_ok=True)

            with Timer(f"{self.DISPLAY_NAME} inference") as inference_timer:
                if not self.ready:
                    self._load()
                assert self._worker is not None
                self._worker.predict(
                    str(input_json),
                    str(output_dir),
                    # The runtime gets staged paths, never the structures themselves.
                    {**effective_config, "templates": None, "template_staging": staged},
                )

            # Resolved by the first load, so it is only known once inference has started the worker.
            metadata.optimization = self.optimization.to_dict() if self.optimization else None
            with Timer(f"{self.DISPLAY_NAME} postprocessing") as postprocess_timer:
                output = self._collect_outputs(output_dir, metadata, effective_config)

        output.metadata.preprocessing_time = preprocess_timer.duration
        output.metadata.inference_time = inference_timer.duration
        output.metadata.postprocessing_time = postprocess_timer.duration
        return output

    def _resolve_msa(self, config: dict[str, Any]) -> list[str | None] | None:
        """Return the caller's per-chain MSA from ``msa`` (or its older name ``unpaired_msa``), if any.

        The two names are the same option; setting both is ambiguous, and an MSA the run would not read
        (``use_msa=False``) is refused instead of silently ignored.
        """
        msa, unpaired = config.get("msa"), config.get("unpaired_msa")
        if msa and unpaired:
            raise ValueError("Pass either 'msa' or its older name 'unpaired_msa', not both")
        supplied = msa or unpaired
        if supplied and not config["use_msa"]:
            raise ValueError("A caller-supplied MSA needs use_msa=True; it would be ignored")
        return supplied

    def _stage_templates(self, sequence_entry: str, buffer_path: Path, config: dict[str, Any]) -> dict[str, str] | None:
        """Write caller-supplied template structures where Protenix reads them.

        ``templates`` maps a name to mmCIF text and applies to chain
        ``templates_chain`` only. Returns the staged paths, or ``None`` when no
        templates were supplied, in which case nothing about the request changes.
        """
        templates = config.get("templates")
        if not templates:
            return None
        if not self.SUPPORTS_USER_TEMPLATES:
            raise ValueError(f"{self.DISPLAY_NAME} does not support user-supplied templates")
        chains = sequence_entry.split(":")
        index = config["templates_chain"]
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(chains):
            raise ValueError("templates_chain must be the index of one of the input chains")
        return stage_templates(templates, chains[index], buffer_path / "templates")

    def _write_input_json(
        self,
        sequence_entry: str,
        buffer_path: Path,
        msa: list[str | None] | None = None,
        staged_templates: dict[str, str] | None = None,
        templates_chain: int = 0,
    ) -> Path:
        chains = sequence_entry.split(":")
        if not chains or any(not part for part in chains) or len(chains) > 26:
            raise ValueError(f"{self.DISPLAY_NAME} input requires 1 to 26 nonempty protein chains")
        if msa is not None and len(msa) != len(chains):
            raise ValueError("msa must have one A3M string or None per chain")

        sequence_records = []
        for index, sequence in enumerate(chains):
            sequence_records.append(
                {
                    "proteinChain": {
                        "sequence": sequence,
                        "count": 1,
                        "id": [chr(65 + index)],
                        "modifications": [],
                    }
                }
            )
            if msa is not None:
                # A query-only file for None suppresses upstream's automatic search
                # for the binder while allowing an unpaired target alignment.
                msa_text = msa[index] or f">query\n{sequence}\n"
                a3m_rows(msa_text, sequence)
                msa_path = buffer_path / f"chain_{index}.a3m"
                msa_path.write_text(msa_text, encoding="utf-8")
                sequence_records[-1]["proteinChain"]["unpairedMsaPath"] = str(msa_path)
            if staged_templates is not None and index == templates_chain:
                sequence_records[-1]["proteinChain"]["templatesPath"] = staged_templates["templates_path"]

        payload = [{"name": "boileroom_target", "sequences": sequence_records, "covalent_bonds": []}]
        input_json = buffer_path / "input.json"
        input_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return input_json

    def _collect_outputs(
        self,
        output_dir: Path,
        metadata: PredictionMetadata,
        config: dict[str, Any],
    ) -> ProtenixOutput:
        cif_paths = sorted(
            output_dir.glob("**/predictions/*_sample_*.cif"), key=lambda path: sample_identity(path, self.DISPLAY_NAME)
        )
        if not cif_paths:
            raise RuntimeError(f"{self.DISPLAY_NAME} produced no sample CIF files under {output_dir}")

        include_fields = config.get("include_fields")
        atom_arrays = []
        cif_strings: list[str] | None = [] if include_field(include_fields, "cif") else None
        identities = [sample_identity(path, self.DISPLAY_NAME) for path in cif_paths]
        expected = {(seed, rank) for seed in _parse_seeds(config["seeds"]) for rank in range(config["sample"])}
        if len(set(identities)) != len(identities) or set(identities) != expected:
            raise RuntimeError(
                f"{self.DISPLAY_NAME} returned incomplete or duplicate samples: expected {sorted(expected)}, got {identities}"
            )
        confidence: list[dict[str, Any] | None] = []
        token_confidence: dict[str, list[Any]] = {
            field: [] for field in ("pae", "token_chain_ids", "token_res_ids", "plddt", "atom_plddt")
        }
        pdb_strings: list[str] | None = [] if include_field(include_fields, "pdb") else None
        for cif_path in cif_paths:
            cif_file = CIFFile.read(str(cif_path))
            atoms = get_structure(cif_file, model=1, use_author_fields=False)
            atom_arrays.append(atoms)
            if cif_strings is not None:
                cif_strings.append(cif_path.read_text(encoding="utf-8"))

            prefix, rank = cif_path.stem.rsplit("_sample_", 1)
            confidence.append(
                read_json(cif_path.with_name(f"{prefix}_summary_confidence_sample_{rank}.json"), self.DISPLAY_NAME)
            )
            full = read_json(cif_path.with_name(f"{prefix}_full_data_sample_{rank}.json"), self.DISPLAY_NAME)
            for field, value in read_token_confidence(full, atoms, self.DISPLAY_NAME).items():
                token_confidence[field].append(value)
            if pdb_strings is not None:
                pdb_file = PDBFile()
                pdb_file.set_structure(atoms)
                buffer = StringIO()
                pdb_file.write(buffer)
                pdb_strings.append(buffer.getvalue())

        output = self.OUTPUT_CLASS(
            metadata=metadata,
            atom_array=atom_arrays,
            confidence=confidence,
            cif=cif_strings,
            pdb=pdb_strings,
            seeds=[seed for seed, _ in identities],
            sample_ranks=[rank for _, rank in identities],
            **token_confidence,
        )
        filtered = cast(ProtenixOutput, self._filter_include_fields(output, include_fields))
        filtered.seeds = output.seeds
        filtered.sample_ranks = output.sample_ranks
        return filtered


def _command_env(
    config: dict[str, Any],
    optimization: OptimizationResolution | None = None,
    family: str = "protenix",
    root_env: str = "PROTENIX_ROOT_DIR",
) -> dict[str, str]:
    env = command_env(config, {root_env: str(get_model_cache_dir(family))})
    if optimization is not None and optimization.active != "vanilla":
        env["MODEL_OPT_TARGET_GPU"] = str(optimization.kit_config).upper()
    # Protenix's MSA client speaks the ColabFold MMseqs2 API but defaults to its own
    # server, which can queue jobs for a long time; use the configured server instead.
    env["MMSEQS_SERVICE_HOST_URL"] = str(config["msa_server_url"])
    return env


def _parse_seeds(value: str) -> list[int]:
    error = "seeds must be comma-separated unique nonnegative integers"
    if not isinstance(value, str):
        raise ValueError(error)
    try:
        seeds = [int(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise ValueError(error) from exc
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError(error)
    return seeds


def _validate_config(config: dict[str, Any], dtypes: frozenset[str] = frozenset({"bf16", "fp16", "fp32"})) -> None:
    _parse_seeds(config["seeds"])
    if config["dtype"] not in dtypes:
        raise ValueError(f"dtype must be {', '.join(sorted(dtypes))}")
    timeout = config["timeout_seconds"]
    if timeout is not None and (
        not isinstance(timeout, int | float) or isinstance(timeout, bool) or not 0 < timeout < float("inf")
    ):
        raise ValueError("timeout_seconds must be a positive finite number or None")
    for field in ("sample", "cycle", "step"):
        if not isinstance(config[field], int) or isinstance(config[field], bool) or config[field] < 1:
            raise ValueError(f"{field} must be a positive integer")
    if config["use_seeds_in_json"] or config["use_default_params"]:
        raise ValueError("Explicit seeds and sampling settings require use_seeds_in_json/use_default_params=False")
