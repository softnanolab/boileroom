"""Core AlphaFold2-Multimer implementation backed by resident ColabFold runners.

MSAs come from the public ColabFold MMseqs2 server (no local genetic databases),
are content-addressed and cached across folds, and can also be supplied directly
by the caller. A persistent Python 3.10 worker owns ColabFold's loaded models;
the core maps each request's output tree onto :class:`AlphaFold2MultimerOutput`.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from collections.abc import Sequence
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, ClassVar, cast

import numpy as np
from biotite.structure.io.pdb import PDBFile
from biotite.structure.io.pdb import get_structure as get_pdb_structure
from biotite.structure.io.pdbx import CIFFile, set_structure

from ...base import FoldingAlgorithm, PredictionMetadata
from ...inputs import MSAInput
from ...msa_cache import MSACache
from ...utils import Timer, get_model_cache_dir
from .._runtime_utils import command_env, include_field
from .._worker import ModelWorker
from .types import AlphaFold2MultimerOutput

logger = logging.getLogger(__name__)


class AlphaFold2MultimerCore(FoldingAlgorithm):
    """AlphaFold2-Multimer structure prediction backed by ColabFold."""

    DEFAULT_CONFIG: ClassVar[dict[str, Any]] = {
        "device": None,
        "colabfold_python": "/opt/colabfold/bin/python",
        "data_dir": None,
        "model_type": "alphafold2_multimer_v3",
        "num_recycle": 3,
        "num_models": 5,
        "num_seeds": 1,
        "random_seed": 0,
        "use_msa_server": True,
        "msa_mode": "mmseqs2_uniref_env",
        "msa_server_url": "https://api.colabfold.com",
        "pair_mode": "unpaired_paired",
        "rank_by": "multimer",
        "use_templates": False,
        "use_amber": False,
        "use_gpu_relax": False,
        "msa_cache_enabled": True,
        "include_fields": None,
        "timeout_seconds": None,
    }
    STATIC_CONFIG_KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "device",
            "colabfold_python",
            "data_dir",
            "model_type",
            "num_models",
            "num_recycle",
            "use_templates",
            "rank_by",
        }
    )

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Create an AlphaFold2-Multimer (ColabFold) core instance."""
        if config and "colabfold_command" in config:
            raise ValueError("colabfold_command is no longer supported; ColabFold uses a resident Python runner")
        super().__init__(config or {})
        self._worker: ModelWorker | None = None
        self._metadata_template = self._initialize_metadata(
            model_name="AlphaFold2-Multimer",
            model_version=str(self.config["model_type"]).removeprefix("alphafold2_multimer_"),
        )

    def _initialize(self) -> None:
        """Load ColabFold's model runners into the persistent worker."""
        self._load()

    def _load(self) -> None:
        """Validate configuration and load model parameters once per core."""
        _validate_config(self.config)
        if self._worker is None:
            config = {**self.config, "data_dir": str(self._data_dir())}
            self._worker = ModelWorker(
                config,
                _command_env(config),
                runtime_path=Path(__file__).with_name("runtime.py"),
                runtime_class="AlphaFold2MultimerRuntime",
                label="AlphaFold2-Multimer",
                python_executable=config["colabfold_python"],
            )
        self._worker.start()
        self.ready = True

    def close(self) -> None:
        """Release the loaded model and its isolated interpreter."""
        if self._worker is not None:
            self._worker.close()
        self.ready = False

    def fold(self, sequences: str | Sequence[str], options: dict | None = None) -> AlphaFold2MultimerOutput:
        """Run AlphaFold2-Multimer for one sequence entry.

        Use ``:`` inside a sequence string to define multiple protein chains.
        An optional :class:`~boileroom.inputs.MSAInput` may be passed via
        ``options["msa"]`` to supply an alignment instead of querying the server.
        """
        effective_config = self._merge_options(options)
        _validate_config(effective_config)
        validated_sequences = self._validate_sequences(sequences)
        if len(validated_sequences) != 1:
            raise ValueError(
                "AlphaFold2-Multimer currently supports exactly one top-level sequence per call; use ':' to join chains."
            )
        provided_msa = (options or {}).get("msa")
        if provided_msa is not None and not isinstance(provided_msa, MSAInput):
            raise ValueError("options['msa'] must be an MSAInput instance")

        chains = _split_chains(validated_sequences[0])
        joined = ":".join(chains)

        metadata = dataclasses.replace(
            self._metadata_template,
            sequence_lengths=self._compute_sequence_lengths(validated_sequences),
        )

        with TemporaryDirectory() as buffer_dir:
            buffer_path = Path(buffer_dir)
            output_dir = buffer_path / "outputs"
            output_dir.mkdir(parents=True, exist_ok=True)

            with Timer("AlphaFold2-Multimer preprocessing") as preprocess_timer:
                input_path, msa_mode, cache_key = self._resolve_msa_input(
                    joined, chains, provided_msa, buffer_path, effective_config
                )

            with Timer("AlphaFold2-Multimer inference") as inference_timer:
                if not self.ready:
                    self._load()
                assert self._worker is not None
                # Only primitive inference settings cross the Python-version
                # boundary; the MSAInput has already been materialized to disk.
                worker_options = {
                    "msa_mode": msa_mode or "mmseqs2_uniref_env",
                    "pair_mode": effective_config["pair_mode"],
                    "msa_server_url": effective_config["msa_server_url"],
                    "random_seed": effective_config["random_seed"],
                    "num_seeds": effective_config["num_seeds"],
                    "use_amber": effective_config["use_amber"],
                    "use_gpu_relax": effective_config["use_gpu_relax"],
                    "timeout_seconds": effective_config["timeout_seconds"],
                }
                self._worker.predict(str(input_path), str(output_dir), worker_options)

            with Timer("AlphaFold2-Multimer postprocessing") as postprocess_timer:
                if cache_key is not None:
                    self._cache_generated_msa(output_dir, cache_key, effective_config)
                output = self._collect_outputs(output_dir, metadata, effective_config)

        output.metadata.preprocessing_time = preprocess_timer.duration
        output.metadata.inference_time = inference_timer.duration
        output.metadata.postprocessing_time = postprocess_timer.duration
        return output

    # -- MSA resolution -------------------------------------------------------

    def _resolve_msa_input(
        self,
        joined: str,
        chains: list[str],
        provided_msa: MSAInput | None,
        buffer_path: Path,
        config: dict[str, Any],
    ) -> tuple[Path, str | None, str | None]:
        """Resolve the ColabFold input file and MSA strategy.

        Returns a tuple ``(input_path, msa_mode, cache_key)`` where ``msa_mode``
        is the value for ``--msa-mode`` (``None`` means the input is an ``.a3m``
        and the flag is omitted), and ``cache_key`` is set only when the server
        will be queried and the resulting alignment should be cached.
        """
        if provided_msa is not None:
            a3m_text = self._materialize_msa(provided_msa, chains)
            a3m_path = buffer_path / "provided.a3m"
            a3m_path.write_text(a3m_text, encoding="utf-8")
            return a3m_path, None, None

        if not config["use_msa_server"]:
            return self._write_fasta(joined, buffer_path), "single_sequence", None

        cache_key = self._cache_key(joined, config)
        if config["msa_cache_enabled"]:
            cached = self._msa_cache().get(cache_key)
            if cached is not None:
                a3m_path = buffer_path / "cached.a3m"
                a3m_path.write_text(cached.read_text(encoding="utf-8"), encoding="utf-8")
                return a3m_path, None, None

        return self._write_fasta(joined, buffer_path), str(config["msa_mode"]), cache_key

    def _materialize_msa(self, msa: MSAInput, chains: list[str]) -> str:
        """Render a provided MSA into ColabFold a3m text, honoring ``remove_insertions``.

        File-backed MSAs are passed through and may be plain a3m or ColabFold's complex
        a3m (first line ``#<lengths>\t<cardinalities>``). Sequence lists for a complex
        are ``:``-joined rows, one segment per chain, and get the complex header.
        """
        if msa.path is not None:
            text = Path(msa.path).read_text(encoding="utf-8")
        elif len(chains) > 1:
            text = _complex_a3m(msa.sequences or [], chains)
        else:
            rows = msa.sequences or []
            text = "\n".join(f">seq_{index}\n{row}" for index, row in enumerate(rows)) + "\n"
        if not text.lstrip().startswith((">", "#")):
            raise ValueError("Provided MSA must be in a3m/FASTA format (first line starts with '>' or a '#' header)")
        if msa.remove_insertions:
            text = _strip_insertions(text)
        return text

    def _cache_key(self, joined: str, config: dict[str, Any]) -> str:
        signature = f"{joined}|{config['msa_mode']}|{config['pair_mode']}|{config['model_type']}"
        return MSACache.hash_key(signature)

    def _cache_generated_msa(self, output_dir: Path, cache_key: str, config: dict[str, Any]) -> None:
        if not config["msa_cache_enabled"]:
            return
        a3m_files = sorted(output_dir.glob("*.a3m"))
        if a3m_files:
            self._msa_cache().put(cache_key, a3m_files[0])

    def _msa_cache(self) -> MSACache:
        return MSACache(self._data_dir(), suffix=".a3m")

    # -- Input files and cache paths ------------------------------------------

    def _write_fasta(self, joined: str, buffer_path: Path) -> Path:
        """Write the query as a single ColabFold record (chains colon-joined)."""
        fasta_path = buffer_path / "target.fasta"
        fasta_path.write_text(f">query\n{joined}\n", encoding="utf-8")
        return fasta_path

    def _data_dir(self, config: dict[str, Any] | None = None) -> Path:
        data_dir = (config or self.config).get("data_dir")
        if data_dir is not None:
            return Path(str(data_dir))
        return get_model_cache_dir("alphafold")

    # -- Output collection ----------------------------------------------------

    def _collect_outputs(
        self,
        output_dir: Path,
        metadata: PredictionMetadata,
        config: dict[str, Any],
    ) -> AlphaFold2MultimerOutput:
        if not any(output_dir.glob("*.done.txt")):
            raise RuntimeError(f"AlphaFold2-Multimer did not complete (no done.txt) under {output_dir}")

        score_paths = sorted(output_dir.glob("*_scores_rank_*.json"), key=_rank_from_scores_path)
        if not score_paths:
            raise RuntimeError(f"AlphaFold2-Multimer produced no ranked scores under {output_dir}")

        include_fields = config.get("include_fields")
        relaxed = bool(config["use_amber"])
        atom_arrays: list[Any] = []
        pdb_strings: list[str] | None = [] if include_field(include_fields, "pdb") else None
        cif_strings: list[str] | None = [] if include_field(include_fields, "cif") else None
        plddt_values: list[np.ndarray | None] = []
        ptm_values: list[np.ndarray | None] = []
        iptm_values: list[np.ndarray | None] = []
        pae_values: list[np.ndarray | None] = []
        ranking_scores: dict[str, float] = {}

        for score_path in score_paths:
            pdb_path = _pdb_for_scores(score_path, relaxed=relaxed)
            if not pdb_path.exists():
                raise RuntimeError(f"AlphaFold2-Multimer scores {score_path.name} has no matching structure")
            scores = json.loads(score_path.read_text(encoding="utf-8"))

            atoms = get_pdb_structure(PDBFile.read(str(pdb_path)), model=1)
            atom_arrays.append(atoms)
            if pdb_strings is not None:
                pdb_strings.append(pdb_path.read_text(encoding="utf-8"))
            if cif_strings is not None:
                cif_strings.append(_atoms_to_cif(atoms))

            plddt_values.append(_optional_array(scores, "plddt"))
            ptm = _optional_scalar(scores, "ptm")
            iptm = _optional_scalar(scores, "iptm")
            ptm_values.append(ptm)
            iptm_values.append(iptm)
            pae_values.append(_optional_array(scores, "pae"))
            ranking_scores[_model_key(score_path)] = _rank_score(config["rank_by"], ptm, iptm, plddt_values[-1])

        ranking = {
            config["rank_by"]: ranking_scores,
            "order": [_model_key(path) for path in score_paths],
        }

        output = AlphaFold2MultimerOutput(
            metadata=metadata,
            atom_array=atom_arrays,
            ranking=ranking,
            plddt=plddt_values if any(value is not None for value in plddt_values) else None,
            ptm=ptm_values if any(value is not None for value in ptm_values) else None,
            iptm=iptm_values if any(value is not None for value in iptm_values) else None,
            pae=pae_values if any(value is not None for value in pae_values) else None,
            pdb=pdb_strings,
            cif=cif_strings,
        )
        return cast(AlphaFold2MultimerOutput, self._filter_include_fields(output, include_fields))


def _split_chains(sequence_entry: str) -> list[str]:
    chains = [part.strip() for part in sequence_entry.split(":")]
    if not any(chains):
        raise ValueError("AlphaFold2-Multimer input must contain at least one chain")
    if not all(chains):
        raise ValueError("AlphaFold2-Multimer input must not contain empty chains (check for stray ':')")
    return chains


def _complex_a3m(rows: list[str], chains: list[str]) -> str:
    """Serialize ``:``-joined complex MSA rows into ColabFold's complex a3m format.

    ColabFold expects a ``#<lengths>\t<cardinalities>`` header over the unique chains,
    followed by the concatenated unique query and then one concatenated row per hit.
    Each hit contributes the segment of the first copy of every unique chain.
    """
    unique = list(dict.fromkeys(chains))
    first_index = [chains.index(chain) for chain in unique]
    header = f"#{','.join(str(len(c)) for c in unique)}\t{','.join(str(chains.count(c)) for c in unique)}"
    labels = "\t".join(str(101 + index) for index in range(len(unique)))
    lines = [header, f">{labels}", "".join(unique)]
    for index, row in enumerate(rows):
        segments = row.split(":")
        if len(segments) != len(chains):
            raise ValueError(
                f"MSA row {index} has {len(segments)} ':'-separated segments; expected one per chain ({len(chains)})"
            )
        lines.extend([f">seq_{index}", "".join(segments[i] for i in first_index)])
    return "\n".join(lines) + "\n"


def _validate_config(config: dict[str, Any]) -> None:
    if not str(config["colabfold_python"]).strip():
        raise ValueError("colabfold_python must not be empty")
    if config["model_type"] not in {"alphafold2_multimer_v1", "alphafold2_multimer_v2", "alphafold2_multimer_v3"}:
        raise ValueError("model_type must be alphafold2_multimer_v1, v2 or v3")
    if config["rank_by"] not in {"multimer", "plddt", "ptm", "iptm"}:
        raise ValueError("rank_by must be multimer, plddt, ptm or iptm")
    timeout = config["timeout_seconds"]
    if timeout is not None and (
        not isinstance(timeout, int | float) or isinstance(timeout, bool) or not 0 < timeout < float("inf")
    ):
        raise ValueError("timeout_seconds must be a positive finite number or None")
    for field in ("num_models", "num_seeds", "num_recycle"):
        value = config[field]
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{field} must be a positive integer")
    if not 1 <= config["num_models"] <= 5:
        raise ValueError("num_models must be between 1 and 5")


def _strip_insertions(a3m_text: str) -> str:
    lines = []
    for line in a3m_text.splitlines():
        if line.startswith((">", "#")) or not line:
            lines.append(line)
        else:
            lines.append("".join(char for char in line if not char.islower() and char != "."))
    return "\n".join(lines) + "\n"


def _rank_from_scores_path(path: Path) -> int:
    marker = "_rank_"
    stem = path.stem
    try:
        return int(stem.split(marker, 1)[1].split("_", 1)[0])
    except (IndexError, ValueError):
        return 0


def _model_key(score_path: Path) -> str:
    """Derive a stable per-prediction key such as ``rank_001_model_1_seed_000``."""
    stem = score_path.stem
    index = stem.find("_rank_")
    return stem[index + 1 :] if index != -1 else stem


def _pdb_for_scores(score_path: Path, *, relaxed: bool) -> Path:
    tag = "_relaxed_" if relaxed else "_unrelaxed_"
    name = score_path.name.replace("_scores_", tag).replace(".json", ".pdb")
    return score_path.with_name(name)


# ColabFold defaults to CUDA unified memory (TF_FORCE_UNIFIED_MEMORY=1, fraction 4.0)
# unless these are already set. Managed allocations stall for hours on Modal's
# sandboxed runtime, so use ordinary device memory instead.
_JAX_MEMORY_ENV = {"TF_FORCE_UNIFIED_MEMORY": "0", "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.9"}


def _command_env(config: dict[str, Any]) -> dict[str, str]:
    return command_env(config, _JAX_MEMORY_ENV)


def _rank_score(rank_by: str, ptm: np.ndarray | None, iptm: np.ndarray | None, plddt: np.ndarray | None) -> float:
    if rank_by == "plddt" and plddt is not None:
        return float(np.mean(plddt))
    p = float(ptm[0]) if ptm is not None else 0.0
    i = float(iptm[0]) if iptm is not None else 0.0
    if rank_by == "multimer" and iptm is not None:
        return 0.8 * i + 0.2 * p
    if rank_by == "iptm" and iptm is not None:
        return i
    return p


def _atoms_to_cif(atoms: Any) -> str:
    cif_file = CIFFile()
    set_structure(cif_file, atoms)
    buffer = StringIO()
    cif_file.write(buffer)
    return buffer.getvalue()


def _optional_array(scores: dict[str, Any], key: str) -> np.ndarray | None:
    if key not in scores or scores[key] is None:
        return None
    return np.asarray(scores[key], dtype=np.float32)


def _optional_scalar(scores: dict[str, Any], key: str) -> np.ndarray | None:
    if key not in scores or scores[key] is None:
        return None
    return np.asarray(scores[key], dtype=np.float32).reshape(1)
