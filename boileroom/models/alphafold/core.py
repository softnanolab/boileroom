"""Core AlphaFold2-Multimer implementation backed by the official runner."""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import pickle
import subprocess
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, ClassVar, cast

import numpy as np
from biotite.structure.io.pdb import PDBFile
from biotite.structure.io.pdb import get_structure as get_pdb_structure
from biotite.structure.io.pdbx import CIFFile
from biotite.structure.io.pdbx import get_structure as get_cif_structure

from ...base import FoldingAlgorithm, PredictionMetadata
from ...utils import MODAL_MODEL_DIR, Timer
from .types import AlphaFold2MultimerOutput

logger = logging.getLogger(__name__)


class AlphaFold2MultimerCore(FoldingAlgorithm):
    """AlphaFold2-Multimer structure prediction model."""

    DEFAULT_CONFIG: ClassVar[dict[str, Any]] = {
        "device": None,
        "alphafold_command": "/app/run_alphafold.sh",
        "data_dir": None,
        "max_template_date": "2022-01-01",
        "db_preset": "full_dbs",
        "num_multimer_predictions_per_model": 5,
        "models_to_relax": "none",
        "use_gpu_relax": False,
        "use_precomputed_msas": False,
        "benchmark": False,
        "random_seed": None,
        "jackhmmer_binary_path": None,
        "hhblits_binary_path": None,
        "hhsearch_binary_path": None,
        "hmmsearch_binary_path": None,
        "hmmbuild_binary_path": None,
        "kalign_binary_path": None,
        "uniref90_database_path": None,
        "mgnify_database_path": None,
        "bfd_database_path": None,
        "small_bfd_database_path": None,
        "uniref30_database_path": None,
        "uniprot_database_path": None,
        "pdb_seqres_database_path": None,
        "template_mmcif_dir": None,
        "obsolete_pdbs_path": None,
        "include_fields": None,
        "timeout_seconds": None,
    }
    STATIC_CONFIG_KEYS: ClassVar[frozenset[str]] = frozenset({"device", "alphafold_command", "data_dir"})

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Create an AlphaFold2-Multimer CLI-backed core instance."""
        super().__init__(config or {})
        self._metadata_template = self._initialize_metadata(
            model_name="AlphaFold2-Multimer",
            model_version="v2",
        )

    def _initialize(self) -> None:
        """Mark the CLI-backed core ready."""
        self._load()

    def _load(self) -> None:
        """Validate static configuration and mark the core ready."""
        if not str(self.config.get("alphafold_command", "")).strip():
            raise ValueError("alphafold_command must not be empty")
        self.ready = True

    def fold(self, sequences: str | Sequence[str], options: dict | None = None) -> AlphaFold2MultimerOutput:
        """Run AlphaFold2-Multimer for one sequence entry.

        Use ``:`` inside a sequence string to define multiple protein chains.
        """
        effective_config = self._merge_options(options)
        validated_sequences = self._validate_sequences(sequences)
        if len(validated_sequences) != 1:
            raise ValueError(
                "AlphaFold2-Multimer currently supports exactly one top-level sequence per call; use ':' to join chains."
            )

        metadata = dataclasses.replace(
            self._metadata_template,
            sequence_lengths=self._compute_sequence_lengths(validated_sequences),
        )

        with TemporaryDirectory() as buffer_dir:
            buffer_path = Path(buffer_dir)
            with Timer("AlphaFold2-Multimer preprocessing") as preprocess_timer:
                fasta_path = self._write_fasta(validated_sequences[0], buffer_path)
                output_dir = buffer_path / "outputs"
                output_dir.mkdir(parents=True, exist_ok=True)
                command = self._build_command(fasta_path, output_dir, effective_config)

            with Timer("AlphaFold2-Multimer inference") as inference_timer:
                self._run_command(command, effective_config)

            with Timer("AlphaFold2-Multimer postprocessing") as postprocess_timer:
                output = self._collect_outputs(output_dir / fasta_path.stem, metadata, effective_config)

        output.metadata.preprocessing_time = preprocess_timer.duration
        output.metadata.inference_time = inference_timer.duration
        output.metadata.postprocessing_time = postprocess_timer.duration
        return output

    def _write_fasta(self, sequence_entry: str, buffer_path: Path) -> Path:
        chains = [part.strip() for part in sequence_entry.split(":") if part.strip()]
        if not chains:
            raise ValueError("AlphaFold2-Multimer input must contain at least one chain")

        fasta_path = buffer_path / "target.fasta"
        fasta_lines = []
        for index, sequence in enumerate(chains):
            fasta_lines.extend([f">chain_{index}", sequence])
        fasta_path.write_text("\n".join(fasta_lines) + "\n", encoding="utf-8")
        return fasta_path

    def _build_command(self, fasta_path: Path, output_dir: Path, config: dict[str, Any]) -> list[str]:
        data_dir = config.get("data_dir")
        if data_dir is None:
            data_dir = str(Path(os.environ.get("MODEL_DIR", MODAL_MODEL_DIR)) / "alphafold")

        database_paths = _resolve_multimer_database_paths(Path(str(data_dir)), config)
        command = [
            str(config["alphafold_command"]),
            f"--fasta_paths={fasta_path}",
            f"--output_dir={output_dir}",
            f"--data_dir={data_dir}",
            f"--max_template_date={config['max_template_date']}",
            f"--db_preset={config['db_preset']}",
            "--model_preset=multimer",
            f"--num_multimer_predictions_per_model={int(config['num_multimer_predictions_per_model'])}",
            f"--models_to_relax={config['models_to_relax']}",
            f"--use_gpu_relax={_bool_arg(config['use_gpu_relax'])}",
            f"--use_precomputed_msas={_bool_arg(config['use_precomputed_msas'])}",
            f"--benchmark={_bool_arg(config['benchmark'])}",
            "--logtostderr",
        ]
        for flag_name, path in database_paths.items():
            command.append(f"--{flag_name}={path}")
        for flag_name in (
            "jackhmmer_binary_path",
            "hhblits_binary_path",
            "hhsearch_binary_path",
            "hmmsearch_binary_path",
            "hmmbuild_binary_path",
            "kalign_binary_path",
        ):
            if config.get(flag_name) is not None:
                command.append(f"--{flag_name}={config[flag_name]}")
        if config.get("random_seed") is not None:
            command.append(f"--random_seed={int(config['random_seed'])}")
        return command

    def _run_command(self, command: list[str], config: dict[str, Any]) -> None:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=_command_env(config),
            timeout=config.get("timeout_seconds"),
        )
        if result.returncode != 0:
            tail = "\n".join((result.stdout + "\n" + result.stderr).splitlines()[-80:])
            raise RuntimeError(f"AlphaFold2-Multimer command failed with exit code {result.returncode}:\n{tail}")

    def _collect_outputs(
        self,
        target_dir: Path,
        metadata: PredictionMetadata,
        config: dict[str, Any],
    ) -> AlphaFold2MultimerOutput:
        if not target_dir.exists():
            raise RuntimeError(f"AlphaFold2-Multimer output directory not found: {target_dir}")

        ranking_path = target_dir / "ranking_debug.json"
        ranking = json.loads(ranking_path.read_text(encoding="utf-8")) if ranking_path.exists() else {}
        order = [str(item) for item in ranking.get("order", [])]

        pdb_paths = sorted(target_dir.glob("ranked_*.pdb"), key=_rank_from_ranked_path)
        cif_paths = {path.stem: path for path in target_dir.glob("ranked_*.cif")}
        if not pdb_paths and not cif_paths:
            raise RuntimeError(f"AlphaFold2-Multimer produced no ranked structures under {target_dir}")

        include_fields = config.get("include_fields")
        atom_arrays = []
        pdb_strings: list[str] | None = [] if _include_field(include_fields, "pdb") else None
        cif_strings: list[str] | None = [] if _include_field(include_fields, "cif") else None
        plddt_values: list[np.ndarray | None] = []
        ptm_values: list[np.ndarray | None] = []
        iptm_values: list[np.ndarray | None] = []
        pae_values: list[np.ndarray | None] = []

        ranked_indices = sorted(
            {_rank_from_ranked_path(path) for path in pdb_paths}
            | {_rank_from_ranked_path(path) for path in cif_paths.values()}
        )
        for rank in ranked_indices:
            stem = f"ranked_{rank}"
            cif_path = cif_paths.get(stem)
            pdb_path = target_dir / f"{stem}.pdb"
            if cif_path is not None and cif_path.exists():
                atom_arrays.append(get_cif_structure(CIFFile.read(str(cif_path)), model=1))
                if cif_strings is not None:
                    cif_strings.append(cif_path.read_text(encoding="utf-8"))
                if pdb_strings is not None and pdb_path.exists():
                    pdb_strings.append(pdb_path.read_text(encoding="utf-8"))
            elif pdb_path.exists():
                atom_arrays.append(get_pdb_structure(PDBFile.read(str(pdb_path)), model=1))
                if pdb_strings is not None:
                    pdb_strings.append(pdb_path.read_text(encoding="utf-8"))
            else:
                continue

            model_name = order[rank] if rank < len(order) else None
            result = _load_result_pickle(target_dir, model_name)
            plddt_values.append(_optional_array(result, "plddt"))
            ptm_values.append(_optional_scalar(result, "ptm"))
            iptm_values.append(_optional_scalar(result, "iptm"))
            pae_values.append(_optional_array(result, "predicted_aligned_error"))

        output = AlphaFold2MultimerOutput(
            metadata=metadata,
            atom_array=atom_arrays,
            ranking=ranking or None,
            plddt=plddt_values if any(value is not None for value in plddt_values) else None,
            ptm=ptm_values if any(value is not None for value in ptm_values) else None,
            iptm=iptm_values if any(value is not None for value in iptm_values) else None,
            pae=pae_values if any(value is not None for value in pae_values) else None,
            pdb=pdb_strings,
            cif=cif_strings,
        )
        return cast(AlphaFold2MultimerOutput, self._filter_include_fields(output, include_fields))


def _bool_arg(value: Any) -> str:
    return "true" if bool(value) else "false"


def _extract_device_number(device: str) -> str | None:
    if device.startswith("cuda:"):
        return device.split(":", 1)[1]
    return None


def _command_env(config: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    device = config.get("device")
    if device is None:
        return env
    device_name = str(device)
    device_number = _extract_device_number(device_name)
    if device_number is not None:
        env["CUDA_VISIBLE_DEVICES"] = device_number
    elif device_name == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def _resolve_multimer_database_paths(data_dir: Path, config: dict[str, Any]) -> dict[str, Path | str]:
    paths: dict[str, Path | str] = {
        "uniref90_database_path": data_dir / "uniref90" / "uniref90.fasta",
        "mgnify_database_path": data_dir / "mgnify" / "mgy_clusters_2022_05.fa",
        "template_mmcif_dir": data_dir / "pdb_mmcif" / "mmcif_files",
        "obsolete_pdbs_path": data_dir / "pdb_mmcif" / "obsolete.dat",
        "uniprot_database_path": data_dir / "uniprot" / "uniprot.fasta",
        "pdb_seqres_database_path": data_dir / "pdb_seqres" / "pdb_seqres.txt",
    }
    if config["db_preset"] == "reduced_dbs":
        paths["small_bfd_database_path"] = data_dir / "small_bfd" / "bfd-first_non_consensus_sequences.fasta"
    else:
        paths["bfd_database_path"] = data_dir / "bfd" / "bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt"
        paths["uniref30_database_path"] = data_dir / "uniref30" / "UniRef30_2021_03"

    for key in tuple(paths):
        if config.get(key) is not None:
            paths[key] = str(config[key])
    return paths


def _rank_from_ranked_path(path: Path) -> int:
    try:
        return int(path.stem.rsplit("_", 1)[1])
    except ValueError:
        return 0


def _load_result_pickle(target_dir: Path, model_name: str | None) -> dict[str, Any]:
    if model_name is None:
        return {}
    result_path = target_dir / f"result_{model_name}.pkl"
    if not result_path.exists():
        return {}
    with result_path.open("rb") as handle:
        result = pickle.load(handle)
    return result if isinstance(result, dict) else {}


def _optional_array(result: dict[str, Any], key: str) -> np.ndarray | None:
    if key not in result or result[key] is None:
        return None
    return np.asarray(result[key], dtype=np.float32)


def _optional_scalar(result: dict[str, Any], key: str) -> np.ndarray | None:
    if key not in result or result[key] is None:
        return None
    return np.asarray(result[key], dtype=np.float32).reshape(1)


def _include_field(include_fields: list[str] | None, field: str) -> bool:
    return include_fields is not None and ("*" in include_fields or field in include_fields)
