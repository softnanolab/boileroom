"""Resident, single-chain MiniFold inference using the pinned upstream model."""

import os
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from io import StringIO
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import numpy as np
import torch
from biotite.structure import AtomArray

from ...base import FoldingAlgorithm
from ...optimization import describe_gpu
from ...provenance import runtime_provenance
from ...utils import get_model_dir
from .loading import (
    CHECKPOINT_REVISION,
    ESM_ARTIFACTS,
    ESM_MODEL_NAME,
    UPSTREAM_COMMIT,
    checkpoint_path,
    load_state_dict,
)
from .types import MiniFoldOutput


class MiniFoldCore(FoldingAlgorithm):
    """Predict monomers; list inputs run individually to avoid upstream padding artifacts."""

    DEFAULT_CONFIG = {
        "device": "cuda:0",
        "model_size": "48L",
        "num_recycling": 3,
        "compile": False,
        "cache_dir": None,
        "max_sequence_length": 2048,
        "include_fields": None,
    }
    STATIC_CONFIG_KEYS = frozenset({"device", "model_size", "cache_dir", "compile", "max_sequence_length"})

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        self.model: Any = None
        self.alphabet: Any = None
        self._of_config: Any = None
        self._device: torch.device | None = None

    def _validate_config(self, config: Mapping[str, Any]) -> None:
        """Reject unsupported variants and unsafe resource settings before loading weights."""
        super()._validate_config(config)
        if config["model_size"] not in ("12L", "48L"):
            raise ValueError("MiniFold model_size must be '12L' or '48L'")
        for key, minimum in (("num_recycling", 0), ("max_sequence_length", 1)):
            value = config[key]
            if type(value) is not int or value < minimum:
                raise ValueError(f"MiniFold {key} must be an integer >= {minimum}")
        if type(config["compile"]) is not bool:
            raise ValueError("MiniFold compile must be a boolean")

    def _initialize(self) -> None:
        """Load once per live runtime."""
        self._load()

    def _load(self) -> None:
        """Load verified weights without unsafe pickle fallback or duplicate ESM construction."""
        if self.ready:
            return
        from minifold.data.config import model_config
        from minifold.model.model import MiniFoldModel

        cache = Path(self.config["cache_dir"] or get_model_dir() / "minifold").expanduser()
        cache.mkdir(parents=True, exist_ok=True)
        path = checkpoint_path(cache, self.config["model_size"])
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        hparams = checkpoint["hyper_parameters"]
        if hparams["esm_model_name"] != ESM_MODEL_NAME:
            raise ValueError("MiniFold checkpoint requests an unrecognized ESM backbone")
        config = model_config("initial_training", train=False, low_prec=False, long_sequence_inference=False)
        # fair-esm uses the torch hub cache for its official ESM2 weights. Restore
        # the caller's global cache after construction; only the model owns this path.
        previous_cache = torch.hub.get_dir()
        try:
            torch.hub.set_dir(str(cache))
            model = MiniFoldModel(
                esm_model_name=hparams["esm_model_name"],
                num_blocks=hparams["num_blocks"],
                no_bins=hparams["no_bins"],
                config_of=config,
                use_structure_module=True,
                kernels=False,
            )
        finally:
            torch.hub.set_dir(previous_cache)
        load_state_dict(model, checkpoint["state_dict"])
        del checkpoint
        if self.config["compile"]:
            model.fold.miniformer = torch.compile(model.fold.miniformer, dynamic=True, fullgraph=True)
        device = self._resolve_device()
        model = model.to(device).eval()
        self.model, self.alphabet = model, model.lm.alphabet
        self._of_config, self._device = config.data, device
        self.ready = True

    def close(self) -> None:
        """Release model references so the runtime can shut down or reload cleanly."""
        self.model = self.alphabet = self._of_config = self._device = None
        self.ready = False

    def _validated_monomers(self, sequences: str | Sequence[str], config: dict) -> list[str]:
        """Validate the entire request before any weight download or prediction."""
        items = [sequences] if isinstance(sequences, str) else list(sequences)
        if not items:
            raise ValueError("MiniFold requires at least one non-empty sequence")
        for seq in items:
            if not isinstance(seq, str) or not seq:
                raise ValueError("MiniFold requires non-empty amino-acid strings")
            if ":" in seq:
                raise ValueError("MiniFold supports single-chain proteins only; ':' multimers are unsupported")
            if len(seq) > config["max_sequence_length"]:
                raise ValueError(f"MiniFold sequence exceeds max_sequence_length={config['max_sequence_length']}")
        return self._validate_sequences(items)

    def _autocast(self) -> Any:
        """Use BF16 only on a CUDA device that supports it; CPU/T4 use FP32."""
        assert self._device is not None
        if self._device.type == "cuda":
            with torch.cuda.device(self._device):
                if torch.cuda.is_bf16_supported():
                    return torch.autocast("cuda", dtype=torch.bfloat16)
        return nullcontext()

    def _prepare_input(self, sequence: str) -> dict:
        """Prepare one unpadded upstream input, preserving its monomer numerics."""
        from minifold.data.of_data import of_inference
        from minifold.utils.residue_constants import restype_order_with_x_inverse

        features = of_inference(sequence, "predict", self._of_config)
        of_seq = "".join(restype_order_with_x_inverse[x.item()] for x in features["aatype"])
        of_seq = of_seq[: int(features["seq_length"].flatten()[0])]
        encoded = torch.tensor(self.alphabet.encode(of_seq), dtype=torch.long, device=self._device)
        relevant = {"aatype", "seq_mask", "residx_atom37_to_atom14", "atom37_atom_exists"}
        return {
            "seq": encoded.unsqueeze(0),
            "mask": features["seq_mask"][:, 0].bool().unsqueeze(0).to(self._device),
            "batch_of": {
                key: value.unsqueeze(0).to(self._device) for key, value in features.items() if key in relevant
            },
        }

    def fold(self, sequences: str | Sequence[str], options: dict | None = None) -> MiniFoldOutput:
        """Fold monomers in input order, reusing weights and returning CPU-only outputs.

        Lists are evaluated one sequence at a time: upstream padded attention does
        not preserve single-sequence predictions. ``include_fields=None`` returns
        metadata and atom arrays; ``['*']`` includes all available outputs. PAE
        is unavailable. Multimers, user MSA and templates are unsupported.
        """
        config = self._merge_options(options)
        sequences = self._validated_monomers(sequences, config)
        self._load()
        metadata = self._initialize_metadata("MiniFold", config["model_size"])
        metadata.sequence_lengths = list(map(len, sequences))
        metadata.runtime = runtime_provenance(
            ("minifold", "fair-esm"),
            gpu=describe_gpu(str(self._device)),
            extra={
                "minifold.commit": os.environ.get("MINIFOLD_COMMIT", "unknown"),
                "minifold.expected_commit": UPSTREAM_COMMIT,
                "minifold.esm_model": ESM_MODEL_NAME,
                "minifold.esm_sha256": ESM_ARTIFACTS[
                    f"https://dl.fbaipublicfiles.com/fair-esm/models/{ESM_MODEL_NAME}.pt"
                ],
                "minifold.num_recycling": config["num_recycling"],
                "minifold.compile": config["compile"],
                "minifold.checkpoint_revision": CHECKPOINT_REVISION,
                "minifold.batch_mode": "sequential-unpadded",
                "minifold.device": str(self._device),
            },
        )
        arrays, plddt = [], []
        preprocessing = inference = postprocessing = 0.0
        for sequence in sequences:
            start = perf_counter()
            batch = self._prepare_input(sequence)
            preprocessing += perf_counter() - start
            start = perf_counter()
            with torch.inference_mode(), self._autocast():
                raw = self.model(batch, num_recycling=config["num_recycling"])
                # Materialize before stopping the timer (synchronizes accelerator work).
                positions = raw["final_atom_positions"][0].float().cpu().numpy()
                mask = raw["final_atom_mask"][0].float().cpu().numpy()
                confidence = raw["plddt"][0].float().cpu().numpy()
            inference += perf_counter() - start
            del raw, batch
            start = perf_counter()
            arrays.append(self._atom_array(positions, mask, confidence, sequence))
            plddt.append(confidence[: len(sequence)].copy() / 100.0)
            postprocessing += perf_counter() - start
        start = perf_counter()
        include = config.get("include_fields") or []
        full = "*" in include
        lengths = metadata.sequence_lengths
        max_length = max(lengths)
        result = MiniFoldOutput(
            metadata=metadata,
            atom_array=arrays,
            plddt=plddt,
            residue_index=np.stack([np.pad(np.arange(n), (0, max_length - n), constant_values=-1) for n in lengths]),
            chain_index=np.stack(
                [np.pad(np.zeros(n, dtype=int), (0, max_length - n), constant_values=-1) for n in lengths]
            ),
            pdb=self._serialize(arrays, "pdb") if full or "pdb" in include else None,
            cif=self._serialize(arrays, "cif") if full or "cif" in include else None,
        )
        metadata.preprocessing_time = preprocessing
        metadata.inference_time = inference
        metadata.postprocessing_time = postprocessing + perf_counter() - start
        return cast(MiniFoldOutput, self._filter_include_fields(result, config.get("include_fields")))

    @staticmethod
    def _atom_array(positions: np.ndarray, mask: np.ndarray, plddt: np.ndarray, sequence: str) -> AtomArray:
        """Convert atom37 coordinates to one-based structures with percent B-factors."""
        from minifold.utils.residue_constants import atom_types, restype_1to3

        length = len(sequence)
        if positions.shape != (length, 37, 3) or mask.shape != (length, 37) or plddt.shape != (length,):
            raise RuntimeError("MiniFold returned unexpected output shapes")
        if (
            not np.isfinite(mask).all()
            or not np.isfinite(positions[mask.astype(bool)]).all()
            or not np.isfinite(plddt).all()
        ):
            raise RuntimeError("MiniFold returned non-finite coordinates or confidence")
        if ((plddt < 0) | (plddt > 100)).any():
            raise RuntimeError("MiniFold returned confidence outside the expected 0–100 scale")
        residue, atom = np.where(mask.astype(bool))
        if not len(atom):
            raise RuntimeError("MiniFold returned no atoms")
        result = AtomArray(len(atom))
        result.coord = positions[residue, atom]
        result.chain_id[:] = "A"
        result.res_id = residue + 1
        result.res_name = np.array([restype_1to3.get(sequence[index], "UNK") for index in residue])
        result.atom_name = np.array([atom_types[index] for index in atom])
        result.element = np.array([name[0] for name in result.atom_name])
        result.set_annotation("b_factor", plddt[residue])
        return result

    @staticmethod
    def _serialize(arrays: list[AtomArray], kind: str) -> list[str]:
        """Serialize exactly the same one-based structure to PDB or mmCIF."""
        from biotite.structure.io import pdb, pdbx

        strings = []
        for array in arrays:
            file: Any = pdb.PDBFile() if kind == "pdb" else pdbx.CIFFile()
            (pdb.set_structure if kind == "pdb" else pdbx.set_structure)(file, array)
            stream = StringIO()
            file.write(stream)
            strings.append(stream.getvalue())
        return strings
