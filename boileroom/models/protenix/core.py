"""Core Protenix implementation backed by the official CLI."""

from __future__ import annotations

import dataclasses
import json
import logging
from collections.abc import Sequence
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, ClassVar, cast

from biotite.sequence.io.fasta import FastaFile
from biotite.structure.io.pdb import PDBFile
from biotite.structure.io.pdbx import CIFFile, get_structure

from ...base import FoldingAlgorithm, PredictionMetadata
from ...utils import Timer, get_model_cache_dir
from .._cli import bool_arg, command_env, include_field, run_command
from .outputs import read_json, read_token_confidence, sample_identity
from .types import ProtenixOutput

logger = logging.getLogger(__name__)


class ProtenixCore(FoldingAlgorithm):
    """Protenix structure prediction model."""

    DEFAULT_CONFIG: ClassVar[dict[str, Any]] = {
        "device": None,
        "protenix_command": "protenix",
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
        "use_seeds_in_json": False,
        "use_tfg_guidance": False,
        "unpaired_msa": None,
        "include_fields": None,
        "timeout_seconds": 3500,
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
        _validate_config(effective_config)
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
                input_json = self._write_input_json(
                    validated_sequences[0], buffer_path, effective_config.get("unpaired_msa")
                )
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

    def _write_input_json(
        self, sequence_entry: str, buffer_path: Path, unpaired_msa: list[str | None] | None = None
    ) -> Path:
        chains = sequence_entry.split(":")
        if not chains or any(not part for part in chains) or len(chains) > 26:
            raise ValueError("Protenix input requires 1 to 26 nonempty protein chains")
        if unpaired_msa is not None and len(unpaired_msa) != len(chains):
            raise ValueError("unpaired_msa must have one A3M string or None per chain")

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
            if unpaired_msa is not None:
                # A query-only file for None suppresses upstream's automatic search
                # for the binder while allowing an unpaired target alignment.
                msa_text = unpaired_msa[index] or f">query\n{sequence}\n"
                _validate_msa(msa_text, sequence)
                msa_path = buffer_path / f"chain_{index}.a3m"
                msa_path.write_text(msa_text, encoding="utf-8")
                sequence_records[-1]["proteinChain"]["unpairedMsaPath"] = str(msa_path)

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
            bool_arg(config["use_msa"]),
            "--use_template",
            bool_arg(config["use_template"]),
            "--use_default_params",
            bool_arg(config["use_default_params"]),
            "--trimul_kernel",
            str(config["trimul_kernel"]),
            "--triatt_kernel",
            str(config["triatt_kernel"]),
            "--enable_cache",
            bool_arg(config["enable_cache"]),
            "--enable_fusion",
            bool_arg(config["enable_fusion"]),
            "--enable_tf32",
            bool_arg(config["enable_tf32"]),
            "--need_atom_confidence",
            "true",
        ]
        if config.get("use_seeds_in_json"):
            command.extend(["--use_seeds_in_json", "true"])
        if config.get("use_tfg_guidance"):
            command.extend(["--use_tfg_guidance", "true"])
        return command

    def _run_command(self, command: list[str], config: dict[str, Any]) -> None:
        run_command(command, "Protenix", _command_env(config), config.get("timeout_seconds"))

    def _collect_outputs(
        self,
        output_dir: Path,
        metadata: PredictionMetadata,
        config: dict[str, Any],
    ) -> ProtenixOutput:
        cif_paths = sorted(output_dir.glob("**/predictions/*_sample_*.cif"), key=sample_identity)
        if not cif_paths:
            raise RuntimeError(f"Protenix produced no sample CIF files under {output_dir}")

        include_fields = config.get("include_fields")
        atom_arrays = []
        cif_strings: list[str] | None = [] if include_field(include_fields, "cif") else None
        identities = [sample_identity(path) for path in cif_paths]
        expected = {(seed, rank) for seed in _parse_seeds(config["seeds"]) for rank in range(config["sample"])}
        if len(set(identities)) != len(identities) or set(identities) != expected:
            raise RuntimeError(
                f"Protenix returned incomplete or duplicate samples: expected {sorted(expected)}, got {identities}"
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
            confidence.append(read_json(cif_path.with_name(f"{prefix}_summary_confidence_sample_{rank}.json")))
            full = read_json(cif_path.with_name(f"{prefix}_full_data_sample_{rank}.json"))
            for field, value in read_token_confidence(full, atoms).items():
                token_confidence[field].append(value)
            if pdb_strings is not None:
                pdb_file = PDBFile()
                pdb_file.set_structure(atoms)
                buffer = StringIO()
                pdb_file.write(buffer)
                pdb_strings.append(buffer.getvalue())

        output = ProtenixOutput(
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


def _command_env(config: dict[str, Any]) -> dict[str, str]:
    env = command_env(config, {"PROTENIX_ROOT_DIR": str(get_model_cache_dir("protenix"))})
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


def _validate_config(config: dict[str, Any]) -> None:
    _parse_seeds(config["seeds"])
    for field in ("sample", "cycle", "step"):
        if not isinstance(config[field], int) or isinstance(config[field], bool) or config[field] < 1:
            raise ValueError(f"{field} must be a positive integer")
    if config["use_seeds_in_json"] or config["use_default_params"]:
        raise ValueError("Explicit seeds and sampling settings require use_seeds_in_json/use_default_params=False")


def _validate_msa(text: str, sequence: str) -> None:
    if not isinstance(text, str) or not text.lstrip().startswith(">"):
        raise ValueError("unpaired_msa entries must be A3M text or None")
    rows = list(FastaFile.read(StringIO(text)).values())
    if not rows or rows[0] != sequence:
        raise ValueError("The first A3M row must match its input protein chain")
    for row in rows:
        aligned = "".join(char for char in row if not char.islower() and char != ".")
        if len(aligned) != len(sequence):
            raise ValueError("Every A3M row must have the input chain's aligned length")
