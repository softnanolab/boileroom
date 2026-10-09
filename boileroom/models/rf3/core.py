"""Core RoseTTAFold 3 implementation backed by a persistent inference worker."""

from __future__ import annotations

import dataclasses
import json
import logging
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
from .outputs import early_stop_message, read_json, read_token_confidence, sample_identity
from .types import RF3Output

logger = logging.getLogger(__name__)

# RF3 and its pinned stack need Python 3.12 with torch 2.7.1, so they live in their own virtualenv
# (like OpenDDE); the boileroom interpreter never imports them.
DEFAULT_RF3_PYTHON = "/opt/rf3/bin/python"
# The kit image's patched interpreter: the kit modes run there, never in the pristine /usr/local/bin/python.
KIT_RF3_PYTHON = "/kit/rosettafold3/opt/venv/bin/python"
# The torch release of the kit stack; with the GPU's compute capability it names the JIT cache directory.
KIT_TORCH_VERSION = "2.13.0"
RF3_MODEL_VERSION = "rf3_foundry_01_24"
MAX_CHAINS = 26
EXAMPLE_NAME = "boileroom_target"


class RF3Core(FoldingAlgorithm):
    """RoseTTAFold 3 structure prediction model (protein chains)."""

    DEFAULT_CONFIG: ClassVar[dict[str, Any]] = {
        "device": None,
        "n_recycles": 10,
        "diffusion_batch_size": 5,
        "num_steps": 50,
        "seed": 1,
        "early_stopping_plddt_threshold": None,
        "checkpoint_path": None,
        "rf3_python": DEFAULT_RF3_PYTHON,
        "optimization": "vanilla",
        "msa": None,
        "include_fields": None,
        "timeout_seconds": 3500,
    }
    # n_recycles, diffusion_batch_size and num_steps shape the loaded network and its pipeline, so they are
    # fixed when the worker starts; seed and the early-stop threshold are applied on every request.
    STATIC_CONFIG_KEYS: ClassVar[frozenset[str]] = frozenset(
        {"device", "n_recycles", "diffusion_batch_size", "num_steps", "checkpoint_path", "rf3_python", "optimization"}
    )
    SUPPORTS_USER_MSA: ClassVar[bool] = True
    SUPPORTS_USER_TEMPLATES: ClassVar[bool] = False
    FAMILY: ClassVar[str] = "rf3"
    DISPLAY_NAME: ClassVar[str] = "RF3"
    ROOT_ENV: ClassVar[str] = "RF3_ROOT_DIR"
    RUNTIME_CLASS: ClassVar[str] = "RF3Runtime"
    RUNTIME_PATH: ClassVar[Path] = Path(__file__).with_name("runtime.py")

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Create an RF3 core with one reusable model worker."""
        super().__init__(config or {})
        validate_optimization(self.config["optimization"])
        self.optimization: OptimizationResolution | None = None
        self._worker: ModelWorker | None = None
        self._metadata_template = self._initialize_metadata(
            model_name=self.DISPLAY_NAME, model_version=RF3_MODEL_VERSION
        )

    def _initialize(self) -> None:
        """Load the RF3 model into its persistent worker."""
        self._load()

    def _load(self) -> None:
        """Validate configuration and load model weights once per core."""
        _validate_config(self.config)
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
        """Return the interpreter that runs the RF3 runtime: the kit's patched one for the kit modes."""
        configured = str(self.config["rf3_python"])
        if self.config["optimization"] != "vanilla" and configured == DEFAULT_RF3_PYTHON:
            return KIT_RF3_PYTHON
        return configured

    def _worker_env(self, config: dict[str, Any], optimization: OptimizationResolution | None) -> dict[str, str]:
        """Return the worker environment."""
        return _command_env(config, optimization)

    def close(self) -> None:
        """Release the persistent worker and its model weights."""
        if self._worker is not None:
            self._worker.close()
        self.ready = False

    def fold(self, sequences: str | Sequence[str], options: dict | None = None) -> RF3Output:
        """Run RF3 prediction for one sequence entry.

        Use ``:`` inside a sequence string to define multiple protein chains.
        """
        effective_config = self._merge_options(options)
        _validate_config(effective_config)
        validated_sequences = self._validate_sequences(sequences)
        if len(validated_sequences) != 1:
            raise ValueError(
                f"{self.DISPLAY_NAME} currently supports exactly one top-level sequence per call; use ':' to join chains."
            )

        metadata = dataclasses.replace(
            self._metadata_template,
            sequence_lengths=self._compute_sequence_lengths(validated_sequences),
        )

        with TemporaryDirectory() as buffer_dir:
            buffer_path = Path(buffer_dir)
            with Timer(f"{self.DISPLAY_NAME} preprocessing") as preprocess_timer:
                input_json = self._write_input_json(validated_sequences[0], buffer_path, effective_config["msa"])
                output_dir = buffer_path / "outputs"
                output_dir.mkdir(parents=True, exist_ok=True)

            with Timer(f"{self.DISPLAY_NAME} inference") as inference_timer:
                if not self.ready:
                    self._load()
                assert self._worker is not None
                # The alignments are already staged as files the input JSON names.
                self._worker.predict(str(input_json), str(output_dir), {**effective_config, "msa": None})

            # Resolved by the first load, so it is only known once inference has started the worker.
            metadata.optimization = self.optimization.to_dict() if self.optimization else None
            with Timer(f"{self.DISPLAY_NAME} postprocessing") as postprocess_timer:
                output = self._collect_outputs(output_dir, metadata, effective_config)

        output.metadata.preprocessing_time = preprocess_timer.duration
        output.metadata.inference_time = inference_timer.duration
        output.metadata.postprocessing_time = postprocess_timer.duration
        return output

    def _write_input_json(self, sequence_entry: str, buffer_path: Path, msa: list[str | None] | None = None) -> Path:
        """Write the RF3 input file for ``sequence_entry``, staging caller-supplied alignments next to it.

        RF3 runs no MSA search of its own: a chain without ``msa_path`` is folded from its sequence alone.
        """
        chains = sequence_entry.split(":")
        if not chains or any(not part for part in chains) or len(chains) > MAX_CHAINS:
            raise ValueError(f"{self.DISPLAY_NAME} input requires 1 to {MAX_CHAINS} nonempty protein chains")
        if msa is not None and len(msa) != len(chains):
            raise ValueError("msa must have one A3M string or None per chain")

        components = []
        for index, sequence in enumerate(chains):
            component: dict[str, Any] = {"seq": sequence, "chain_id": chr(65 + index)}
            alignment = msa[index] if msa is not None else None
            if alignment:
                a3m_rows(alignment, sequence)
                msa_path = buffer_path / f"chain_{index}.a3m"
                msa_path.write_text(alignment, encoding="utf-8")
                component["msa_path"] = str(msa_path)
            components.append(component)

        input_json = buffer_path / "input.json"
        input_json.write_text(
            json.dumps([{"name": EXAMPLE_NAME, "components": components}], indent=2) + "\n", encoding="utf-8"
        )
        return input_json

    def _collect_outputs(
        self,
        output_dir: Path,
        metadata: PredictionMetadata,
        config: dict[str, Any],
    ) -> RF3Output:
        example_dir = output_dir / EXAMPLE_NAME
        cif_paths = list(example_dir.glob(f"seed-*_sample-*/{EXAMPLE_NAME}_seed-*_sample-*_model.cif"))
        if not cif_paths:
            if reason := early_stop_message(example_dir, EXAMPLE_NAME):
                raise RuntimeError(f"{self.DISPLAY_NAME} stopped early and wrote no structure: {reason}")
            raise RuntimeError(f"{self.DISPLAY_NAME} produced no sample CIF files under {output_dir}")

        identities = {path: sample_identity(path) for path in cif_paths}
        if len(set(identities.values())) != len(cif_paths) or {seed for seed, _ in identities.values()} != {
            int(config["seed"])
        }:
            raise RuntimeError(
                f"{self.DISPLAY_NAME} returned samples {sorted(identities.values())} for seed {config['seed']}"
            )
        if len(cif_paths) != self.config["diffusion_batch_size"]:
            raise RuntimeError(
                f"{self.DISPLAY_NAME} returned {len(cif_paths)} samples; expected {self.config['diffusion_batch_size']}"
            )

        include_fields = config.get("include_fields")
        samples = []
        for cif_path in cif_paths:
            prefix = cif_path.name.removesuffix("_model.cif")
            atoms = get_structure(CIFFile.read(str(cif_path)), model=1, use_author_fields=False)
            summary = read_json(cif_path.with_name(f"{prefix}_summary_confidences.json"), self.DISPLAY_NAME)
            full = read_json(cif_path.with_name(f"{prefix}_confidences.json"), self.DISPLAY_NAME)
            samples.append((cif_path, atoms, summary, read_token_confidence(full, atoms, self.DISPLAY_NAME)))
        # Best first, as RF3 ranks its own samples; the sample index breaks ties.
        samples.sort(key=lambda sample: (-_ranking_score(sample[2]), identities[sample[0]][1]))

        cif_strings: list[str] | None = [] if include_field(include_fields, "cif") else None
        pdb_strings: list[str] | None = [] if include_field(include_fields, "pdb") else None
        for cif_path, atoms, _, _ in samples:
            if cif_strings is not None:
                cif_strings.append(cif_path.read_text(encoding="utf-8"))
            if pdb_strings is not None:
                pdb_file = PDBFile()
                pdb_file.set_structure(atoms)
                buffer = StringIO()
                pdb_file.write(buffer)
                pdb_strings.append(buffer.getvalue())

        output = RF3Output(
            metadata=metadata,
            atom_array=[atoms for _, atoms, _, _ in samples],
            confidence=[summary for _, _, summary, _ in samples],
            plddt=[token["plddt"] for _, _, _, token in samples],
            pae=[token["pae"] for _, _, _, token in samples],
            token_chain_ids=[token["token_chain_ids"] for _, _, _, token in samples],
            token_res_ids=[token["token_res_ids"] for _, _, _, token in samples],
            atom_plddt=[token["atom_plddt"] for _, _, _, token in samples],
            seeds=[identities[path][0] for path, _, _, _ in samples],
            sample_ranks=list(range(len(samples))),
            sample_indices=[identities[path][1] for path, _, _, _ in samples],
            pdb=pdb_strings,
            cif=cif_strings,
        )
        filtered = cast(RF3Output, self._filter_include_fields(output, include_fields))
        filtered.seeds = output.seeds
        filtered.sample_ranks = output.sample_ranks
        filtered.sample_indices = output.sample_indices
        return filtered


def _ranking_score(summary: dict[str, Any]) -> float:
    score = summary.get("ranking_score")
    if not isinstance(score, int | float) or isinstance(score, bool):
        raise RuntimeError("RF3 summary confidence has no numeric ranking_score")
    return float(score)


def _command_env(config: dict[str, Any], optimization: OptimizationResolution | None = None) -> dict[str, str]:
    """Return the RF3 worker environment: weights root and, for the kit modes, its target GPU and JIT caches."""
    root = get_model_cache_dir("rf3")
    env = command_env(config, {"RF3_ROOT_DIR": str(root)})
    if optimization is not None and optimization.active != "vanilla":
        env["MODEL_OPT_TARGET_GPU"] = str(optimization.kit_config).upper()
        # Persist compiled kernels next to the weights so a restarted container skips the first-run compile. The
        # stack key (torch release and compute capability) names the cache directory, as the kit's configs/*.env do.
        jit = root / "jit"
        stack_key = f"torch{KIT_TORCH_VERSION}-{optimization.capability}"
        env.setdefault("MODEL_OPT_JIT_ROOT", str(jit))
        env.setdefault("MODEL_OPT_STACK_KEY", stack_key)
        env.setdefault("TRITON_CACHE_DIR", str(jit / stack_key / "triton"))
        env.setdefault("TORCHINDUCTOR_CACHE_DIR", str(jit / stack_key / "inductor"))
        env.setdefault("ROSETTAFOLD3_OPT_DIGEST_DIR", str(jit / "weights"))
        env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    return env


def _positive_int(config: dict[str, Any], field: str) -> None:
    value = config[field]
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{field} must be a positive integer")


def _validate_config(config: dict[str, Any]) -> None:
    for field in ("n_recycles", "diffusion_batch_size", "num_steps"):
        _positive_int(config, field)
    seed = config["seed"]
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    threshold = config["early_stopping_plddt_threshold"]
    if threshold is not None and (
        not isinstance(threshold, int | float) or isinstance(threshold, bool) or not 0 <= threshold <= 1
    ):
        raise ValueError("early_stopping_plddt_threshold must be None or a number between 0 and 1")
    timeout = config["timeout_seconds"]
    if timeout is not None and (
        not isinstance(timeout, int | float) or isinstance(timeout, bool) or not 0 < timeout < float("inf")
    ):
        raise ValueError("timeout_seconds must be a positive finite number or None")
    checkpoint = config["checkpoint_path"]
    if checkpoint is not None and (not isinstance(checkpoint, str) or not checkpoint):
        raise ValueError("checkpoint_path must be None or a nonempty path string")
    if not isinstance(config["rf3_python"], str) or not config["rf3_python"]:
        raise ValueError("rf3_python must be a nonempty path string")
    validate_optimization(config["optimization"])
