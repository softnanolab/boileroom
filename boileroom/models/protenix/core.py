"""Core Protenix implementation backed by a persistent inference runner."""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import string
import sys
from collections.abc import Mapping, Sequence
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, ClassVar, Final, NoReturn, cast

from biotite.structure.io.pdb import PDBFile
from biotite.structure.io.pdbx import CIFFile, get_structure

from ...base import FoldingAlgorithm, PredictionMetadata
from ...inputs import a3m_rows, parse_a3m
from ...optimization import (
    GpuInfo,
    OptimizationResolution,
    OptimizationUnavailableError,
    describe_gpu,
    detect_gpu,
    resolve_optimization,
)
from ...provenance import kit_provenance, runtime_provenance
from ...utils import Timer, get_model_cache_dir
from .._runtime_utils import command_env, include_field
from .._worker import ModelWorker
from .outputs import read_json, read_token_confidence, sample_identity
from .templates import StagedTemplates, stage_templates, write_query_only_hits
from .types import ProtenixOutput

logger = logging.getLogger(__name__)

#: ``LAYERNORM_TYPE`` the Protenix worker runs with, per optimization mode. The worker environment sets it over any
#: inherited value (an image ``ENV``, a caller's shell), so the mode alone decides it. Vanilla is ``openfold``, what the
#: stock image has always run: upstream's unset default, ``fast_layernorm``, JIT-compiles a CUDA extension at first use
#: that the stock image does not carry. The kit modes need ``fast_layernorm``, the fused LayerNorm their lever patches
#: and the kit image ships compiled.
PROTENIX_LAYERNORM: Final[Mapping[str, str]] = {
    "vanilla": "openfold",
    "exact": "fast_layernorm",
    "fast": "fast_layernorm",
}
#: Checkpoints with a template embedder (``template_embedder.n_blocks > 0`` in upstream's model configs). Upstream's
#: ``runner/batch_inference.py`` asserts the same list before template inference; on any other checkpoint a template
#: would be featurized and then ignored by the network.
PROTENIX_TEMPLATE_MODELS: Final[frozenset[str]] = frozenset(
    {"protenix-v2", "protenix_base_default_v1.0.0", "protenix_base_20250630_v1.0.0"}
)
#: Characters a caller A3M row may hold. Upstream's featurizer (``MSACore.sequences_to_array``) aligns uppercase
#: letters and ``-`` and counts every other character as an inserted residue, and its reader (``parse_fasta``) skips
#: ``#`` lines, so any other character would be read differently from the validated row. ``.`` is removed on writing.
_A3M_ROW_ALPHABET: Final[frozenset[str]] = frozenset(string.ascii_letters + "-.")
#: Upstream's ``FeatureAssemblyLine.assemble`` drops the MSA of a chain shorter than this (``len(seq) <= 4``).
_MIN_MSA_CHAIN_LENGTH: Final[int] = 5


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
    #: ``LAYERNORM_TYPE`` per optimization mode, forced into the worker environment.
    LAYERNORM_BY_MODE: ClassVar[Mapping[str, str]] = PROTENIX_LAYERNORM
    #: Checkpoints that can take templates (searched with ``use_template`` or supplied as ``templates``).
    TEMPLATE_MODELS: ClassVar[frozenset[str]] = PROTENIX_TEMPLATE_MODELS
    #: Distributions of the core's own interpreter recorded in ``metadata.runtime`` (the worker reports its own).
    RUNTIME_PACKAGES: ClassVar[tuple[str, ...]] = ("protenix",)

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Create a Protenix core with one reusable model worker."""
        if config and "protenix_command" in config:
            raise ValueError("protenix_command is no longer supported; Protenix uses its Python runner")
        super().__init__(config or {})
        self.optimization: OptimizationResolution | None = None
        self._worker: ModelWorker | None = None
        self._gpu: GpuInfo | None = None
        # A runtime refusal stands for the life of this core: the worker is closed and never silently restarted.
        self._refusal: OptimizationUnavailableError | None = None
        self._metadata_template = self._initialize_metadata(
            model_name=self.DISPLAY_NAME,
            model_version=str(self.config["model_name"]),
        )

    def _initialize(self) -> None:
        """Load the Protenix model into its persistent worker."""
        self._load()

    def _load(self) -> None:
        """Validate configuration and load model weights once per core.

        Raises
        ------
        OptimizationUnavailableError
            If the mode cannot run on this GPU, or the worker refused it now or earlier in this core's life.
        """
        self._raise_standing_refusal()
        _validate_config(self.config, self.DTYPES)
        self._validate_templates(self.config)
        mode = str(self.config["optimization"])
        # Vanilla records the card for provenance when there is one; a kit mode needs it and refuses without.
        device = self.config.get("device")
        self._gpu = describe_gpu(device) if mode == "vanilla" else detect_gpu(device)
        self.optimization = resolve_optimization(self.FAMILY, mode, self._gpu)
        if self._worker is None:
            self._worker = ModelWorker(
                self.config,
                self._worker_env(self.config, self.optimization),
                runtime_path=self.RUNTIME_PATH,
                runtime_class=self.RUNTIME_CLASS,
                label=self.DISPLAY_NAME,
                python_executable=self._worker_python(),
            )
        try:
            self._worker.start()
        except OptimizationUnavailableError as error:
            self._refuse(error)
        self.ready = True

    def _refuse(self, error: OptimizationUnavailableError) -> NoReturn:
        """Close the worker, keep the refusal for every later call, and raise it."""
        if self._worker is not None:
            self._worker.close()
        self.ready = False
        self._refusal = error
        raise error

    def _raise_standing_refusal(self) -> None:
        """Raise the refusal recorded earlier, instead of restarting a worker that would refuse again."""
        if self._refusal is not None:
            raise OptimizationUnavailableError(
                f"{self._refusal} (refused earlier in this runtime; the worker is not restarted)"
            ) from self._refusal

    def _worker_python(self) -> str:
        """Return the interpreter that runs the model runtime."""
        return sys.executable

    def _worker_env(self, config: dict[str, Any], optimization: OptimizationResolution | None) -> dict[str, str]:
        """Return the worker environment for ``config``.

        Parameters
        ----------
        config : dict[str, Any]
            The core's configuration; ``optimization`` picks the LayerNorm and ``msa_server_url`` the MSA server.
        optimization : OptimizationResolution | None
            The resolved mode; a kit mode also names its kit config in ``MODEL_OPT_TARGET_GPU``.

        Returns
        -------
        dict[str, str]
            The inherited environment with the weights root (:attr:`ROOT_ENV`), ``LAYERNORM_TYPE`` for the mode
            (:attr:`LAYERNORM_BY_MODE`, over any inherited value), the MSA server, and the family's additions
            (:meth:`_extend_worker_env`).
        """
        env = command_env(config, {self.ROOT_ENV: str(get_model_cache_dir(self.FAMILY))})
        if optimization is not None and optimization.kit:
            env["MODEL_OPT_TARGET_GPU"] = str(optimization.kit_config).upper()
        # Forced, not defaulted: an image ENV or a caller's shell must not pick the LayerNorm a mode runs.
        env["LAYERNORM_TYPE"] = self.LAYERNORM_BY_MODE[str(config["optimization"])]
        # Protenix's MSA client speaks the ColabFold MMseqs2 API but defaults to its own
        # server, which can queue jobs for a long time; use the configured server instead.
        env["MMSEQS_SERVICE_HOST_URL"] = str(config["msa_server_url"])
        self._extend_worker_env(env, config)
        return env

    def _extend_worker_env(self, env: dict[str, str], config: dict[str, Any]) -> None:
        """Add a family's own variables to the worker environment ``env`` in place (none for Protenix)."""

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
        self._validate_templates(effective_config)
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
                self._raise_standing_refusal()
                if not self.ready:
                    self._load()
                assert self._worker is not None
                try:
                    payload = self._worker.predict(
                        str(input_json),
                        str(output_dir),
                        # The runtime gets staged paths, never the structures themselves.
                        {
                            **effective_config,
                            "templates": None,
                            "template_staging": staged.to_dict() if staged is not None else None,
                        },
                    )
                except OptimizationUnavailableError as error:
                    # A kit that refused mid-run would refuse again: no silent restart, the refusal stands.
                    self._refuse(error)

            # Resolved by the first load, so it is only known once inference has started the worker.
            metadata.optimization = self.optimization.to_dict() if self.optimization else None
            metadata.runtime = self._runtime_record(payload)
            with Timer(f"{self.DISPLAY_NAME} postprocessing") as postprocess_timer:
                output = self._collect_outputs(output_dir, metadata, effective_config)

        output.metadata.preprocessing_time = preprocess_timer.duration
        output.metadata.inference_time = inference_timer.duration
        output.metadata.postprocessing_time = postprocess_timer.duration
        return output

    def _runtime_record(self, payload: Mapping[str, Any] | None) -> dict[str, str]:
        """Return ``metadata.runtime``: this interpreter's provenance, the worker's ``describe()`` and the request.

        Parameters
        ----------
        payload : Mapping[str, Any] | None
            What the worker's ``predict()`` reported for this request.

        Returns
        -------
        dict[str, str]
            One flat record: :func:`~boileroom.provenance.runtime_provenance` keys for the core's interpreter and the
            GPU it resolved, then ``worker.<key>`` for each entry of the worker's ``describe()`` (its interpreter,
            LayerNorm, kernels, kit report) and ``predict.<key>`` for each entry this request reported (resolved
            kernels, template counts, late kit facts). The prefixes keep the three sources from overwriting each other.
            In kit modes also the keys every kit family shares (:func:`~boileroom.provenance.kit_provenance`), from
            this request's settled kit report where it has one, else from the worker's activation report.
        """
        info = dict(self._worker.info) if self._worker is not None else {}
        predicted = dict(payload or {})
        extra = {f"worker.{key}": value for key, value in info.items()}
        extra.update({f"predict.{key}": value for key, value in predicted.items()})
        if self.optimization is not None and self.optimization.kit:

            def latest(key: str) -> str | None:
                value = predicted.get(f"kit.{key}", info.get(f"kit.{key}"))
                return None if value is None else str(value)

            extra.update(
                kit_provenance(
                    commit=info.get("kit.commit"),
                    levers_applied=_split_levers(latest("levers_applied")),
                    levers_fallback=_split_levers(latest("levers_fallback")),
                    partial=latest("partial") == "true",
                )
            )
        return runtime_provenance(self.RUNTIME_PACKAGES, gpu=self._gpu, extra=extra)

    def _validate_templates(self, config: dict[str, Any]) -> None:
        """Refuse template requests this checkpoint or this combination of options would not honour.

        Raises
        ------
        ValueError
            If ``templates`` are combined with ``use_template=True`` (caller templates replace the search), or if
            templates of either kind are asked of a checkpoint without a template embedder.
        """
        supplied = bool(config.get("templates"))
        if supplied and config["use_template"]:
            raise ValueError(
                "caller templates replace the template search for the request; they cannot be combined with "
                "use_template=True"
            )
        if (supplied or config["use_template"]) and config["model_name"] not in self.TEMPLATE_MODELS:
            raise ValueError(
                f"{self.DISPLAY_NAME} checkpoint {config['model_name']!r} has no template embedder, so a template "
                f"would be ignored; template-capable checkpoints: {sorted(self.TEMPLATE_MODELS)}"
            )

    def _resolve_msa(self, config: dict[str, Any]) -> list[str | None] | None:
        """Return the caller's per-chain MSA (``msa``), if any.

        Each entry is written as that chain's *unpaired* MSA (``unpairedMsaPath``); there is no paired input, so
        for a heteromer the cross-chain pairing a server search would provide is lost when ``msa`` is given. An MSA
        the run would not read (``use_msa=False``) is refused instead of silently ignored.
        """
        msa = config.get("msa")
        if msa and not config["use_msa"]:
            raise ValueError("A caller-supplied MSA needs use_msa=True; it would be ignored")
        return msa

    def _stage_templates(
        self, sequence_entry: str, buffer_path: Path, config: dict[str, Any]
    ) -> StagedTemplates | None:
        """Write caller-supplied template structures where Protenix reads them.

        ``templates`` maps a name to mmCIF text and applies to chain
        ``templates_chain`` only. Returns the staged paths, or ``None`` when no
        templates were supplied, in which case nothing about the request changes.
        """
        templates = config.get("templates")
        if not templates:
            return None
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
        staged_templates: StagedTemplates | None = None,
        templates_chain: int = 0,
    ) -> Path:
        """Write the upstream input JSON for ``sequence_entry`` and return its path.

        Parameters
        ----------
        sequence_entry : str
            Protein chains joined by ``:``.
        buffer_path : Path
            Request scratch directory.
        msa : list[str | None] | None
            One A3M per chain, written as that chain's unpaired MSA; ``None`` entries get a query-only file so
            upstream does not search for that chain. Pairing across chains is not represented. Each entry is
            checked and rewritten by :func:`_model_a3m`, so upstream parses exactly the validated rows.
        staged_templates : StagedTemplates | None
            Caller templates for chain ``templates_chain``. Every other protein chain then gets a query-only hit
            file as its ``templatesPath``, so upstream featurizes it without templates instead of searching for some.
        templates_chain : int
            Index of the chain the staged templates belong to.

        Returns
        -------
        Path
            The written ``input.json``.
        """
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
                msa_path = buffer_path / f"chain_{index}.a3m"
                msa_path.write_text(
                    _model_a3m(msa_text, sequence, f"{self.DISPLAY_NAME} msa entry {index}"), encoding="utf-8"
                )
                sequence_records[-1]["proteinChain"]["unpairedMsaPath"] = str(msa_path)
            if staged_templates is not None:
                templates_path = (
                    staged_templates.templates_path
                    if index == templates_chain
                    else str(write_query_only_hits(sequence, buffer_path / f"chain_{index}_no_templates.a3m"))
                )
                sequence_records[-1]["proteinChain"]["templatesPath"] = templates_path

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


def _model_a3m(text: str, sequence: str, label: str) -> str:
    """Validate one chain's caller A3M and return the A3M upstream parses as exactly those rows.

    Upstream's reader and featurizer differ from :func:`boileroom.inputs.a3m_rows` on some characters: they skip
    ``#`` lines, count any character other than an uppercase letter or ``-`` (whitespace, ``*``, digits, ``.``)
    as an inserted residue, and fail on non-ASCII text. Rows holding such a character, other than ``.``, are
    refused. ``.``, an insert-state gap that the validator drops, is removed, so it does not count as a deletion.
    Lowercase insertions are kept: upstream derives ``deletion_matrix`` from them. Wrapped rows are joined, blank
    lines dropped, and headers kept as read. Upstream ignores the MSA of a chain shorter than
    ``_MIN_MSA_CHAIN_LENGTH``, so homolog rows for one are refused rather than silently unused.

    Parameters
    ----------
    text : str
        The caller's A3M for one chain.
    sequence : str
        That chain's sequence.
    label : str
        Names the entry in error messages.

    Returns
    -------
    str
        One ``>header`` line and one row line per record, ``.`` removed.

    Raises
    ------
    ValueError
        If ``a3m_rows`` rejects the text, a row holds a character outside ASCII letters, ``-`` and ``.``, or a
        chain shorter than ``_MIN_MSA_CHAIN_LENGTH`` gets more than its query row.
    """
    a3m_rows(text, sequence)
    records = parse_a3m(text)
    for row_index, (_, row) in enumerate(records):
        invalid = sorted(set(row) - _A3M_ROW_ALPHABET)
        if invalid:
            raise ValueError(
                f"{label}: A3M row {row_index} contains {', '.join(map(repr, invalid))}, which the model's A3M "
                "reader would drop, count as an insertion or fail on; rows may hold only ASCII residue letters, "
                "'-' gaps, lowercase insertions and '.' (no whitespace, comment lines or terminators)"
            )
    if len(records) > 1 and len(sequence) < _MIN_MSA_CHAIN_LENGTH:
        raise ValueError(
            f"{label}: the chain has {len(sequence)} residues and the model ignores the MSA of a chain shorter "
            f"than {_MIN_MSA_CHAIN_LENGTH}; pass None (or the query row alone) for it"
        )
    return "".join(f">{header}\n{row.replace('.', '')}\n" for header, row in records)


def _prepend_path(env: dict[str, str], variable: str, directory: str) -> None:
    """Put ``directory`` first on the search-path ``variable`` of ``env``."""
    env[variable] = os.pathsep.join(filter(None, [directory, env.get(variable, "")]))


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


def _split_levers(flat: str | None) -> list[str]:
    """The lever names of a worker's comma-joined report entry.

    ``"none"`` (the worker's empty list) and no entry at all (the worker's gate reads a missing key as no levers) are
    both none.
    """
    return [] if flat is None or flat == "none" else flat.split(",")


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
