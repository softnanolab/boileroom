"""Core ESMFold2 implementation without Modal dependencies."""

from __future__ import annotations

import dataclasses
import logging
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any, ClassVar, cast

import numpy as np

from ...base import FoldingAlgorithm, PredictionMetadata
from ...utils import MODAL_MODEL_DIR, Timer, safe_mkdir, validate_sequence
from .payloads import decode_structure_input
from .types import (
    CovalentBond,
    DistogramConditioning,
    DNAInput,
    ESMFold2Output,
    LigandInput,
    Modification,
    MSAInput,
    PocketConditioning,
    ProteinInput,
    RNAInput,
    StructurePredictionInput,
)

logger = logging.getLogger(__name__)

SequenceInput = ProteinInput | RNAInput | DNAInput | LigandInput
ESMFold2FoldInput = (
    str
    | Sequence[str]
    | StructurePredictionInput
    | Sequence[StructurePredictionInput]
    | Sequence[SequenceInput]
    | Mapping[str, Any]
    | Sequence[Mapping[str, Any]]
)


@dataclass(frozen=True)
class _FoldRequest:
    """One normalized fold request plus its sequence-length metadata."""

    input: StructurePredictionInput
    sequence_length: int


class ESMFold2Core(FoldingAlgorithm):
    """Biohub ESMFold2 all-atom structure prediction model."""

    DEFAULT_CONFIG: ClassVar[dict[str, Any]] = {
        "device": "cuda:0",
        "model_name": "biohub/ESMFold2",
        "cache_dir": None,
        "ccd_cache_dir": None,
        "dtype": None,
        "num_loops": 3,
        "num_sampling_steps": 50,
        "num_diffusion_samples": 1,
        "seed": None,
        "noise_scale": None,
        "step_scale": None,
        "max_inference_sigma": None,
        "early_exit": False,
        "complex_id": "pred",
        "include_fields": None,
    }
    STATIC_CONFIG_KEYS: ClassVar[frozenset[str]] = frozenset(
        {"device", "model_name", "cache_dir", "ccd_cache_dir", "dtype"}
    )

    def __init__(self, config: dict | None = None) -> None:
        """Create an ESMFold2 core instance."""
        super().__init__(config)
        self._metadata_template = self._initialize_metadata(
            model_name="ESMFold2",
            model_version=str(self.config["model_name"]),
        )
        self.model_dir: str | None = os.environ.get("MODEL_DIR", MODAL_MODEL_DIR)
        self._device: Any | None = None
        self.model: Any | None = None
        self.input_builder: Any | None = None

    def _initialize(self) -> None:
        """Load the ESMFold2 model and input builder."""
        self._load()

    def _resolve_cache_dir(self, config_key: str, default_subdir: str) -> Path:
        """Resolve and create a configured or model-volume cache directory."""
        configured = self.config.get(config_key)
        if configured is not None:
            cache_dir = Path(str(configured))
        else:
            if self.model_dir is None:
                raise ValueError(f"model_dir must be set when {config_key} is not provided")
            cache_dir = Path(self.model_dir) / default_subdir
        safe_mkdir(cache_dir, parents=True)
        return cache_dir

    def _load(self) -> None:
        """Load Biohub ESMFold2 from Hugging Face and prepare the input builder."""
        from esm.models.esmfold2 import ESMFold2InputBuilder
        from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

        cache_dir = self._resolve_cache_dir("cache_dir", "esmfold2")
        ccd_cache_dir = self._resolve_cache_dir("ccd_cache_dir", "esmfold2")

        kwargs: dict[str, Any] = {"cache_dir": str(cache_dir)}
        if self.config.get("dtype") is not None:
            kwargs["dtype"] = self.config["dtype"]

        if self.model is None:
            self.model = ESMFold2Model.from_pretrained(str(self.config["model_name"]), **kwargs)

        self._device = self._resolve_device()
        self.model = self.model.to(self._device)
        self.model.eval()
        self._ensure_ccd_cache(ccd_cache_dir)
        self.input_builder = ESMFold2InputBuilder(ccd_cache=ccd_cache_dir)
        self.ready = True

    @staticmethod
    def _ensure_ccd_cache(ccd_cache_dir: Path) -> None:
        """Download the CCD pickle to the direct path expected by Biohub ESMFold2."""
        ccd_path = ccd_cache_dir / "ccd.pkl"
        if ccd_path.exists():
            return

        from huggingface_hub import hf_hub_download

        hf_hub_download(repo_id="biohub/ESMFold2", filename="ccd.pkl", local_dir=str(ccd_cache_dir))

    def fold(self, sequences: ESMFold2FoldInput, options: dict | None = None) -> ESMFold2Output:
        """Predict one or more structures with ESMFold2."""
        effective_config = self._merge_options(options)
        self._validate_effective_config(effective_config)
        requests = self._coerce_requests(sequences)

        if self.model is None or self.input_builder is None:
            logger.warning("Model not loaded. Forcing the model to load... Next time call _load() first.")
            self._load()
        assert self.model is not None and self.input_builder is not None, "Model not loaded"

        results: list[Any] = []
        sequence_lengths: list[int] = []
        preprocessing_time = 0.0
        inference_time = 0.0
        postprocessing_time = 0.0

        for request_index, request in enumerate(requests):
            folded, timing = self._fold_one(request.input, effective_config, request_index)
            results.extend(folded)
            sequence_lengths.extend([request.sequence_length] * len(folded))
            preprocessing_time += timing["preprocessing"]
            inference_time += timing["inference"]
            postprocessing_time += timing["postprocessing"]

        metadata = dataclasses.replace(
            self._metadata_template,
            sequence_lengths=sequence_lengths,
            preprocessing_time=preprocessing_time,
            inference_time=inference_time,
            postprocessing_time=postprocessing_time,
        )
        return self._convert_results(results, metadata, effective_config)

    def _fold_one(
        self,
        prediction_input: StructurePredictionInput,
        config: dict[str, Any],
        request_index: int,
    ) -> tuple[list[Any], dict[str, float]]:
        """Run one ESMFold2 structure input through preprocess, model, and decode."""
        import torch
        from esm.models.esmfold2.processor import _seed_context

        assert self.model is not None and self.input_builder is not None, "Model not loaded"

        esm_input = self._to_esm_structure_prediction_input(prediction_input)
        seed = cast(int | None, config.get("seed"))
        num_diffusion_samples = int(config["num_diffusion_samples"])
        complex_id_base = str(config.get("complex_id") or "pred")
        complex_id = complex_id_base if request_index == 0 else f"{complex_id_base}_{request_index}"

        with Timer("ESMFold2 preprocessing") as preprocess_timer:
            features, chain_infos = self.input_builder.prepare_input(esm_input, seed=seed, device=self.model.device)

        sampler_kwargs = self._sampler_kwargs(config)
        with Timer("ESMFold2 inference") as inference_timer, torch.no_grad(), _seed_context(seed):
            output = self.model(
                **features,
                num_loops=int(config["num_loops"]),
                num_sampling_steps=int(config["num_sampling_steps"]),
                num_diffusion_samples=num_diffusion_samples,
                early_exit=bool(config["early_exit"]),
                **sampler_kwargs,
            )

        with Timer("ESMFold2 postprocessing") as postprocess_timer:
            decoded = self.input_builder.decode(
                output,
                features,
                chain_infos,
                num_diffusion_samples=num_diffusion_samples,
                complex_id=complex_id,
            )

        results = decoded if isinstance(decoded, list) else [decoded]
        timing = {
            "preprocessing": preprocess_timer.duration,
            "inference": inference_timer.duration,
            "postprocessing": postprocess_timer.duration,
        }
        return results, timing

    @staticmethod
    def _validate_effective_config(config: dict[str, Any]) -> None:
        """Validate dynamic inference options before launching expensive model work."""
        for key in ("num_loops", "num_sampling_steps", "num_diffusion_samples"):
            value = config[key]
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"ESMFold2 option {key!r} must be a positive integer.")

        seed = config.get("seed")
        if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool) or seed < 0):
            raise ValueError("ESMFold2 option 'seed' must be a non-negative integer or None.")

        for key in ("noise_scale", "step_scale", "max_inference_sigma"):
            value = config.get(key)
            if value is None:
                continue
            if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(float(value)):
                raise ValueError(f"ESMFold2 option {key!r} must be a finite number or None.")

    @staticmethod
    def _sampler_kwargs(config: dict[str, Any]) -> dict[str, Any]:
        """Return optional sampler controls accepted by the Biohub model call."""
        kwargs = {}
        for key in ("noise_scale", "step_scale", "max_inference_sigma"):
            if config.get(key) is not None:
                kwargs[key] = config[key]
        return kwargs

    def _coerce_requests(self, sequences: ESMFold2FoldInput) -> list[_FoldRequest]:
        """Coerce public BoilerRoom inputs into one or more ESMFold2 structure inputs."""
        if isinstance(sequences, str):
            return [self._request_from_protein_string(sequences)]

        if isinstance(sequences, StructurePredictionInput):
            return [self._request_from_structure_input(sequences)]

        if isinstance(sequences, Mapping):
            return [self._request_from_structure_input(decode_structure_input(cast(Mapping[str, Any], sequences)))]

        if not isinstance(sequences, Sequence):
            raise TypeError("ESMFold2.fold expects a sequence string, structure input, or a sequence of those inputs.")

        items = list(sequences)
        if not items:
            raise ValueError("ESMFold2.fold received an empty input sequence.")

        if all(isinstance(item, str) for item in items):
            return [self._request_from_protein_string(cast(str, item)) for item in items]

        if all(isinstance(item, StructurePredictionInput) for item in items):
            return [self._request_from_structure_input(cast(StructurePredictionInput, item)) for item in items]

        if all(isinstance(item, Mapping) for item in items):
            return [
                self._request_from_structure_input(decode_structure_input(cast(Mapping[str, Any], item)))
                for item in items
            ]

        if all(self._is_sequence_input(item) for item in items):
            structure_input = StructurePredictionInput(sequences=cast(list[SequenceInput], items))
            return [self._request_from_structure_input(structure_input)]

        raise TypeError(
            "ESMFold2.fold input lists must contain only strings, only StructurePredictionInput objects, "
            "or only molecule input dataclasses."
        )

    @staticmethod
    def _is_sequence_input(value: object) -> bool:
        """Return whether a value is one lightweight molecule input dataclass."""
        return isinstance(value, ProteinInput | RNAInput | DNAInput | LigandInput)

    def _request_from_protein_string(self, sequence: str) -> _FoldRequest:
        """Convert a colon-delimited protein string into one structure request."""
        chains = sequence.replace("|", ":").split(":")
        if any(chain == "" for chain in chains):
            raise ValueError(f"Invalid ESMFold2 protein sequence {sequence!r}: empty chain near ':'.")
        for chain in chains:
            validate_sequence(chain)
        structure_input = StructurePredictionInput(
            sequences=[ProteinInput(id=self._chain_id(index), sequence=chain) for index, chain in enumerate(chains)]
        )
        return _FoldRequest(input=structure_input, sequence_length=sum(len(chain) for chain in chains))

    def _request_from_structure_input(self, prediction_input: StructurePredictionInput) -> _FoldRequest:
        """Validate a rich structure input and attach sequence-length metadata."""
        if not prediction_input.sequences:
            raise ValueError("StructurePredictionInput.sequences must not be empty.")
        return _FoldRequest(input=prediction_input, sequence_length=self._structure_input_length(prediction_input))

    @staticmethod
    def _chain_id(index: int) -> str:
        """Return spreadsheet-style chain IDs: A, B, ..., Z, AA, AB, ..."""
        index += 1
        parts = []
        while index:
            index, remainder = divmod(index - 1, 26)
            parts.append(chr(65 + remainder))
        return "".join(reversed(parts))

    def _structure_input_length(self, prediction_input: StructurePredictionInput) -> int:
        """Return the metadata length for all sequence-like entities in an input."""
        return sum(self._sequence_input_length(item) for item in prediction_input.sequences)

    def _sequence_input_length(self, item: SequenceInput) -> int:
        """Return the metadata length contribution for one chain or ligand input."""
        multiplier = self._id_count(item.id)
        if isinstance(item, ProteinInput | RNAInput | DNAInput):
            sequence = item.sequence.replace("|", ":")
            chains = sequence.split(":")
            if any(chain == "" for chain in chains):
                raise ValueError(f"Invalid ESMFold2 sequence {sequence!r}: empty chain near ':'.")
            if isinstance(item, ProteinInput):
                for chain in chains:
                    validate_sequence(chain)
            return sum(len(chain) for chain in chains) * multiplier
        ccd_count = len(item.ccd or [])
        ligand_count = ccd_count or (1 if item.smiles else 0)
        return max(multiplier, ligand_count)

    @staticmethod
    def _id_count(value: str | list[str]) -> int:
        """Return how many entity copies an ESMFold2 id represents."""
        return len(value) if isinstance(value, list) else 1

    def _to_esm_structure_prediction_input(self, prediction_input: StructurePredictionInput) -> Any:
        """Convert a lightweight structure input to Biohub's native dataclass."""
        from esm.models.esmfold2 import StructurePredictionInput as ESMStructurePredictionInput

        return ESMStructurePredictionInput(
            sequences=[self._to_esm_sequence_input(item) for item in prediction_input.sequences],
            pocket=self._to_esm_pocket_conditioning(prediction_input.pocket),
            distogram_conditioning=self._to_esm_distogram_conditioning(prediction_input.distogram_conditioning),
            covalent_bonds=self._to_esm_covalent_bonds(prediction_input.covalent_bonds),
        )

    def _to_esm_sequence_input(self, item: SequenceInput) -> Any:
        """Convert one lightweight chain or ligand input to Biohub's native type."""
        from esm.models.esmfold2 import DNAInput as ESMDNAInput
        from esm.models.esmfold2 import LigandInput as ESMLigandInput
        from esm.models.esmfold2 import ProteinInput as ESMProteinInput
        from esm.models.esmfold2 import RNAInput as ESMRNAInput

        if isinstance(item, ProteinInput):
            return ESMProteinInput(
                id=item.id,
                sequence=item.sequence,
                modifications=self._to_esm_modifications(item.modifications),
                msa=self._to_esm_msa(item.msa),
            )
        if isinstance(item, RNAInput):
            return ESMRNAInput(
                id=item.id,
                sequence=item.sequence,
                modifications=self._to_esm_modifications(item.modifications),
            )
        if isinstance(item, DNAInput):
            return ESMDNAInput(
                id=item.id,
                sequence=item.sequence,
                modifications=self._to_esm_modifications(item.modifications),
            )
        return ESMLigandInput(id=item.id, smiles=item.smiles, ccd=item.ccd)

    @staticmethod
    def _to_esm_modifications(modifications: list[Modification] | None) -> list[Any] | None:
        """Convert optional lightweight residue modifications to Biohub objects."""
        if modifications is None:
            return None
        from esm.models.esmfold2 import Modification as ESMModification

        return [
            ESMModification(position=modification.position, ccd=modification.ccd, smiles=modification.smiles)
            for modification in modifications
        ]

    @staticmethod
    def _to_esm_msa(msa: MSAInput | Any | None) -> Any | None:
        """Convert lightweight or native MSA inputs to Biohub's MSA object."""
        if msa is None:
            return None
        if isinstance(msa, MSAInput):
            if msa.sequences is None:
                raise ValueError(
                    "ESMFold2 currently requires in-memory MSA sequences; "
                    "file-backed MSA paths are reserved for other model adapters."
                )
            from esm.models.esmfold2 import MSA

            return MSA.from_sequences(msa.sequences, remove_insertions=msa.remove_insertions)
        from esm.models.esmfold2 import MSA

        if isinstance(msa, MSA):
            return msa
        if isinstance(msa, list):
            return MSA.from_sequences(msa)
        return msa

    @staticmethod
    def _to_esm_pocket_conditioning(pocket: PocketConditioning | None) -> Any | None:
        """Convert optional pocket conditioning to Biohub's native dataclass."""
        if pocket is None:
            return None
        from esm.utils.structure.input_builder import PocketConditioning as ESMPocketConditioning

        return ESMPocketConditioning(binder_chain_id=pocket.binder_chain_id, contacts=pocket.contacts)

    @staticmethod
    def _to_esm_distogram_conditioning(
        distograms: list[DistogramConditioning] | None,
    ) -> list[Any] | None:
        """Convert optional distogram conditioning entries to Biohub objects."""
        if distograms is None:
            return None
        from esm.models.esmfold2 import DistogramConditioning as ESMDistogramConditioning

        return [
            ESMDistogramConditioning(chain_id=distogram.chain_id, distogram=distogram.distogram)
            for distogram in distograms
        ]

    @staticmethod
    def _to_esm_covalent_bonds(bonds: list[CovalentBond] | None) -> list[Any] | None:
        """Convert optional covalent-bond constraints to Biohub objects."""
        if bonds is None:
            return None
        from esm.models.esmfold2 import CovalentBond as ESMCovalentBond

        return [
            ESMCovalentBond(
                chain_id1=bond.chain_id1,
                res_idx1=bond.res_idx1,
                atom_idx1=bond.atom_idx1,
                chain_id2=bond.chain_id2,
                res_idx2=bond.res_idx2,
                atom_idx2=bond.atom_idx2,
            )
            for bond in bonds
        ]

    def _convert_results(
        self,
        results: list[Any],
        metadata: PredictionMetadata,
        config: dict[str, Any],
    ) -> ESMFold2Output:
        """Convert Biohub decoded outputs to BoilerRoom's normalized output type."""
        include_fields = cast(list[str] | None, config.get("include_fields"))
        atom_arrays: list[Any] = []
        cif_values: list[str] | None = [] if self._wants_field(include_fields, "cif") else None
        pdb_values: list[str] | None = [] if self._wants_field(include_fields, "pdb") else None

        plddt: list[np.ndarray | None] | None = [] if self._wants_field(include_fields, "plddt") else None
        ptm: list[Any] | None = [] if self._wants_field(include_fields, "ptm") else None
        iptm: list[Any] | None = [] if self._wants_field(include_fields, "iptm") else None
        pae: list[np.ndarray | None] | None = [] if self._wants_field(include_fields, "pae") else None
        distogram: list[np.ndarray | None] | None = [] if self._wants_field(include_fields, "distogram") else None
        pair_chains_iptm: list[np.ndarray | None] | None = (
            [] if self._wants_field(include_fields, "pair_chains_iptm") else None
        )
        residue_index: list[np.ndarray | None] | None = (
            [] if self._wants_field(include_fields, "residue_index") else None
        )
        entity_id: list[np.ndarray | None] | None = [] if self._wants_field(include_fields, "entity_id") else None

        for result in results:
            cif_string = result.complex.to_mmcif()
            atom_array = self._cif_to_atom_array(cif_string)
            atom_arrays.append(atom_array)

            if cif_values is not None:
                cif_values.append(cif_string)
            if pdb_values is not None:
                pdb_values.append(self._atom_array_to_pdb(atom_array))
            if plddt is not None:
                plddt.append(self._tensor_to_numpy(result.plddt))
            if ptm is not None:
                ptm.append(result.ptm)
            if iptm is not None:
                iptm.append(result.iptm)
            if pae is not None:
                pae.append(self._tensor_to_numpy(result.pae))
            if distogram is not None:
                distogram.append(self._tensor_to_numpy(result.distogram))
            if pair_chains_iptm is not None:
                pair_chains_iptm.append(self._tensor_to_numpy(result.pair_chains_iptm))
            if residue_index is not None:
                residue_index.append(self._tensor_to_numpy(result.residue_index))
            if entity_id is not None:
                entity_id.append(self._tensor_to_numpy(result.entity_id))

        full_output = ESMFold2Output(
            metadata=metadata,
            atom_array=atom_arrays,
            plddt=plddt,
            ptm=ptm,
            iptm=iptm,
            pae=pae,
            distogram=distogram,
            pair_chains_iptm=pair_chains_iptm,
            residue_index=residue_index,
            entity_id=entity_id,
            pdb=pdb_values,
            cif=cif_values,
        )
        filtered = self._filter_include_fields(full_output, include_fields)
        return cast(ESMFold2Output, filtered)

    @staticmethod
    def _wants_field(include_fields: list[str] | None, field: str) -> bool:
        """Return whether a named optional output field was requested."""
        return include_fields is not None and ("*" in include_fields or field in include_fields)

    @staticmethod
    def _tensor_to_numpy(value: Any) -> np.ndarray | None:
        """Detach a tensor-like value and return it as a NumPy array."""
        if value is None:
            return None
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        return np.asarray(value)

    @staticmethod
    def _cif_to_atom_array(cif_string: str) -> Any:
        """Parse an mmCIF string into a Biotite atom array."""
        from biotite.structure.io.pdbx import CIFFile, get_structure

        return get_structure(CIFFile.read(StringIO(cif_string)), model=1)

    @staticmethod
    def _atom_array_to_pdb(atom_array: Any) -> str:
        """Serialize a Biotite atom array to PDB text."""
        from biotite.structure.io.pdb import PDBFile, set_structure

        pdb_file = PDBFile()
        set_structure(pdb_file, atom_array)
        buffer = StringIO()
        pdb_file.write(buffer)
        return buffer.getvalue()
