"""Core Protenix implementation backed by the official CLI."""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import subprocess
from collections.abc import Iterable, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, ClassVar, cast

from biotite.structure.io.pdbx import CIFFile, get_structure

from ...base import FoldingAlgorithm, PredictionMetadata
from ...utils import Timer
from .types import ProtenixOutput

logger = logging.getLogger(__name__)


class ProtenixCore(FoldingAlgorithm):
    """Protenix structure prediction model."""

    DEFAULT_CONFIG: ClassVar[dict[str, Any]] = {
        "device": None,
        "protenix_command": "protenix",
        "model_name": "protenix_base_default_v1.0.0",
        "seeds": "101",
        "cycle": 10,
        "step": 200,
        "sample": 5,
        "dtype": "bf16",
        "use_msa": True,
        "use_template": False,
        "use_default_params": False,
        "trimul_kernel": "cuequivariance",
        "triatt_kernel": "cuequivariance",
        "enable_cache": True,
        "enable_fusion": True,
        "enable_tf32": True,
        "use_seeds_in_json": False,
        "use_tfg_guidance": False,
        "include_fields": None,
        "timeout_seconds": None,
    }
    STATIC_CONFIG_KEYS: ClassVar[frozenset[str]] = frozenset({"device", "protenix_command"})

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Create a Protenix CLI-backed core instance."""
        super().__init__(config or {})
        self._metadata_template = self._initialize_metadata(
            model_name="Protenix",
            model_version=str(self.config["model_name"]),
        )

    def _initialize(self) -> None:
        """Mark the CLI-backed core ready."""
        self._load()

    def _load(self) -> None:
        """Validate static configuration and mark the core ready."""
        if not str(self.config.get("protenix_command", "")).strip():
            raise ValueError("protenix_command must not be empty")
        self.ready = True

    def fold(self, sequences: str | Sequence[str], options: dict | None = None) -> ProtenixOutput:
        """Run Protenix prediction for one sequence entry.

        Use ``:`` inside a sequence string to define multiple protein chains.
        """
        effective_config = self._merge_options(options)
        validated_sequences = self._validate_sequences(sequences)
        if len(validated_sequences) != 1:
            raise ValueError(
                "Protenix currently supports exactly one top-level sequence per call; use ':' to join chains."
            )

        metadata = dataclasses.replace(
            self._metadata_template,
            model_version=str(effective_config["model_name"]),
            sequence_lengths=self._compute_sequence_lengths(validated_sequences),
        )

        with TemporaryDirectory() as buffer_dir:
            buffer_path = Path(buffer_dir)
            with Timer("Protenix preprocessing") as preprocess_timer:
                input_json = self._write_input_json(validated_sequences[0], buffer_path)
                output_dir = buffer_path / "outputs"
                output_dir.mkdir(parents=True, exist_ok=True)
                command = self._build_command(input_json, output_dir, effective_config)

            with Timer("Protenix inference") as inference_timer:
                self._run_command(command, effective_config)

            with Timer("Protenix postprocessing") as postprocess_timer:
                output = self._collect_outputs(output_dir, metadata, effective_config)

        output.metadata.preprocessing_time = preprocess_timer.duration
        output.metadata.inference_time = inference_timer.duration
        output.metadata.postprocessing_time = postprocess_timer.duration
        return output

    def _write_input_json(self, sequence_entry: str, buffer_path: Path) -> Path:
        chains = [part.strip() for part in sequence_entry.split(":") if part.strip()]
        if not chains:
            raise ValueError("Protenix input must contain at least one chain")

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

        payload = [{"name": "boileroom_target", "sequences": sequence_records, "covalent_bonds": []}]
        input_json = buffer_path / "input.json"
        input_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return input_json

    def _build_command(self, input_json: Path, output_dir: Path, config: dict[str, Any]) -> list[str]:
        command = [
            str(config["protenix_command"]),
            "pred",
            "--input",
            str(input_json),
            "--out_dir",
            str(output_dir),
            "--seeds",
            str(config["seeds"]),
            "--model_name",
            str(config["model_name"]),
            "--cycle",
            str(int(config["cycle"])),
            "--step",
            str(int(config["step"])),
            "--sample",
            str(int(config["sample"])),
            "--dtype",
            str(config["dtype"]),
            "--use_msa",
            _bool_arg(config["use_msa"]),
            "--use_template",
            _bool_arg(config["use_template"]),
            "--use_default_params",
            _bool_arg(config["use_default_params"]),
            "--trimul_kernel",
            str(config["trimul_kernel"]),
            "--triatt_kernel",
            str(config["triatt_kernel"]),
            "--enable_cache",
            _bool_arg(config["enable_cache"]),
            "--enable_fusion",
            _bool_arg(config["enable_fusion"]),
            "--enable_tf32",
            _bool_arg(config["enable_tf32"]),
        ]
        if config.get("use_seeds_in_json"):
            command.extend(["--use_seeds_in_json", "true"])
        if config.get("use_tfg_guidance"):
            command.extend(["--use_tfg_guidance", "true"])
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
            raise RuntimeError(f"Protenix command failed with exit code {result.returncode}:\n{tail}")

    def _collect_outputs(
        self,
        output_dir: Path,
        metadata: PredictionMetadata,
        config: dict[str, Any],
    ) -> ProtenixOutput:
        prediction_dirs = _find_prediction_dirs(output_dir)
        cif_paths = sorted(_sample_cif_paths(prediction_dirs), key=_sample_sort_key)
        if not cif_paths:
            raise RuntimeError(f"Protenix produced no sample CIF files under {output_dir}")

        include_fields = config.get("include_fields")
        atom_arrays = []
        cif_strings: list[str] | None = [] if _include_field(include_fields, "cif") else None
        confidence: list[dict[str, Any] | None] = []
        for cif_path in cif_paths:
            cif_file = CIFFile.read(str(cif_path))
            atom_arrays.append(get_structure(cif_file, model=1))
            if cif_strings is not None:
                cif_strings.append(cif_path.read_text(encoding="utf-8"))

            confidence_path = _confidence_path_for_cif(cif_path)
            item = json.loads(confidence_path.read_text(encoding="utf-8")) if confidence_path.exists() else None
            confidence.append(item)

        output = ProtenixOutput(
            metadata=metadata,
            atom_array=atom_arrays,
            confidence=confidence if any(item is not None for item in confidence) else None,
            cif=cif_strings,
        )
        return cast(ProtenixOutput, self._filter_include_fields(output, include_fields))


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


def _find_prediction_dirs(output_dir: Path) -> list[Path]:
    prediction_dirs = sorted(path for path in output_dir.glob("**/predictions") if path.is_dir())
    if not prediction_dirs:
        raise RuntimeError(f"Could not find Protenix predictions directory under {output_dir}")
    return prediction_dirs


def _sample_cif_paths(prediction_dirs: Iterable[Path]) -> Iterable[Path]:
    for prediction_dir in prediction_dirs:
        yield from prediction_dir.glob("*_sample_*.cif")


def _sample_sort_key(path: Path) -> tuple[str, int]:
    return (str(path.parent), _rank_from_sample_path(path))


def _rank_from_sample_path(path: Path) -> int:
    stem = path.stem
    marker = "_sample_"
    if marker not in stem:
        return 0
    try:
        return int(stem.rsplit(marker, 1)[1])
    except ValueError:
        return 0


def _confidence_path_for_cif(cif_path: Path) -> Path:
    sample_name, rank = cif_path.stem.rsplit("_sample_", 1)
    return cif_path.with_name(f"{sample_name}_summary_confidence_sample_{rank}.json")


def _include_field(include_fields: list[str] | None, field: str) -> bool:
    return include_fields is not None and ("*" in include_fields or field in include_fields)
