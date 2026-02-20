"""MiniFold core algorithm."""

import logging
import urllib.request
from pathlib import Path
from typing import Any, List, Optional, Sequence, Union, cast

import numpy as np
import torch
import torch.nn.functional as F
from biotite.structure import AtomArray

from ...base import FoldingAlgorithm
from ...utils import Timer, get_model_dir

from .types import MiniFoldOutput
from ..esm.linker import store_multimer_properties

logger = logging.getLogger(__name__)

MODEL_URL_48L = "https://huggingface.co/jwohlwend/minifold/resolve/main/minifold_48L_final.ckpt"
MODEL_URL_12L = "https://huggingface.co/jwohlwend/minifold/resolve/main/minifold_12L_final.ckpt"


class MiniFoldCore(FoldingAlgorithm):
    """MiniFold protein structure prediction model."""

    DEFAULT_CONFIG = {
        "device": "cuda:0",
        "model_size": "48L",  # "48L" or "12L"
        "num_recycling": 3,
        "compile": False,
        "kernels": False,
        "cache_dir": None,  # defaults to MODEL_DIR/minifold
        "glycine_linker": "",  # empty = direct concat (monomer default)
        "include_fields": None,  # Optional[List[str]] - controls which fields to include in output
    }
    STATIC_CONFIG_KEYS = {"device", "model_size", "cache_dir", "compile", "kernels"}

    def __init__(self, config: dict = {}) -> None:
        super().__init__(config)
        self.metadata = self._initialize_metadata(
            model_name="MiniFold",
            model_version=self.config["model_size"],
        )
        self._device: torch.device | None = None
        self.model = None
        self.alphabet = None
        self._of_config = None

    def _initialize(self) -> None:
        """Initialize the model during container startup."""
        self._load()

    def _load(self) -> None:
        """Load the model, alphabet, and checkpoint, then prepare for inference."""
        from minifold.model.model import MiniFoldModel
        from minifold.data.config import model_config
        from esm.pretrained import load_model_and_alphabet

        model_size = self.config["model_size"]
        cache_dir = self.config.get("cache_dir")
        if cache_dir is None:
            cache_dir = str(get_model_dir() / "minifold")

        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)

        # ESM2 weights should be cached alongside the minifold checkpoint
        torch.hub.set_dir(str(cache_path))

        checkpoint_path = cache_path / f"minifold_{model_size}.ckpt"
        if not checkpoint_path.exists():
            url = MODEL_URL_48L if model_size == "48L" else MODEL_URL_12L
            logger.info(f"Downloading MiniFold {model_size} checkpoint to {checkpoint_path}")
            urllib.request.urlretrieve(url, str(checkpoint_path))  # noqa: S310

        ckpt = torch.load(str(checkpoint_path), map_location="cpu")
        hparams = ckpt["hyper_parameters"]

        config_of = model_config(
            "initial_training",
            train=False,
            low_prec=False,
            long_sequence_inference=False,
        )
        model = MiniFoldModel(
            esm_model_name=hparams["esm_model_name"],
            num_blocks=hparams["num_blocks"],
            no_bins=hparams["no_bins"],
            config_of=config_of,
            use_structure_module=True,
            kernels=self.config["kernels"],
        )

        _, alphabet = load_model_and_alphabet(hparams["esm_model_name"])

        state_dict = ckpt["state_dict"]
        state_dict = {k: v for k, v in state_dict.items() if "boundaries" not in k}
        state_dict = {k: v for k, v in state_dict.items() if "mid_points" not in k}
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)

        if self.config["compile"]:
            model.fold.miniformer = torch.compile(
                model.fold.miniformer,
                dynamic=True,
                fullgraph=True,
            )

        self._device = self._resolve_device()
        model = model.to(self._device)
        model.eval()

        self.model = model
        self.alphabet = alphabet

        # Second config call to get the .data sub-config for input preparation
        self._of_config = model_config(
            "initial_training",
            train=False,
            low_prec=False,
            long_sequence_inference=False,
        ).data

        self.ready = True

    def fold(self, sequences: Union[str, Sequence[str]], options: Optional[dict] = None) -> MiniFoldOutput:
        """Predict protein structure(s) from one or more amino acid sequence(s)."""
        effective_config = self._merge_options(options)

        if self.model is None or self.alphabet is None:
            logger.warning("Model not loaded. Forcing the model to load... Next time call _load() first.")
            self._load()
        assert self.model is not None and self.alphabet is not None, "Model not loaded"

        validated_sequences = self._validate_sequences(sequences)
        self.metadata.sequence_lengths = self._compute_sequence_lengths(validated_sequences)

        is_multimer = any(":" in seq for seq in validated_sequences)
        if is_multimer:
            model_batch, multimer_properties = self._tokenize_multimer(validated_sequences, effective_config)
        else:
            model_batch = self._prepare_batch(validated_sequences)
            multimer_properties = None

        with Timer("Model Inference") as timer:
            with torch.inference_mode():
                autocast_device = "cuda" if self._device.type == "cuda" else "cpu"
                with torch.autocast(autocast_device, dtype=torch.bfloat16):
                    raw_outputs = self.model(model_batch, num_recycling=effective_config["num_recycling"])

        result = self._convert_outputs(
            raw_outputs,
            validated_sequences,
            multimer_properties,
            prediction_time=timer.duration,
            config=effective_config,
        )

        return result

    def _prepare_input(self, sequence: str):
        """Prepare input tensors for a single sequence.

        Adapted from minifold predict.py:prepare_input.
        Returns (encoded_seq, mask, open_fold_batch).
        """
        from minifold.data.of_data import of_inference
        from minifold.utils.residue_constants import restype_order_with_x_inverse

        open_fold_batch = of_inference(sequence, "predict", self._of_config)

        of_seq = "".join([restype_order_with_x_inverse[x.item()] for x in open_fold_batch["aatype"]])[
            : open_fold_batch["seq_length"]
        ]

        assert self.alphabet is not None
        encoded_seq = self.alphabet.encode(of_seq)
        encoded_seq = torch.tensor(encoded_seq, dtype=torch.long)
        mask = open_fold_batch["seq_mask"][:, 0].bool()

        relevant = {"aatype", "seq_mask", "residx_atom37_to_atom14", "atom37_atom_exists"}
        open_fold_batch = {k: v for k, v in open_fold_batch.items() if k in relevant}

        return encoded_seq, mask, open_fold_batch

    def _prepare_batch(self, sequences: List[str]) -> dict:
        """Prepare a batched input dict from a list of sequences.

        Adapted from minifold predict.py:predict (batching logic).
        """
        feats = [self._prepare_input(seq) for seq in sequences]

        max_len = max(len(seq) for seq, _, _ in feats)

        if self.config["kernels"]:
            max_len = (max_len + 127) // 128 * 128

        seq = torch.stack([F.pad(s, (0, max_len - len(s)), value=20) for s, _, _ in feats])
        mask = torch.stack([F.pad(m, (0, max_len - len(m)), value=0) for _, m, _ in feats])

        # Collate OpenFold batch
        batch_of: dict[str, Any] = {}
        for _, _, feats_of in feats:
            for k, v in feats_of.items():
                batch_of.setdefault(k, []).append(v)

        for k, v in batch_of.items():
            batch_of[k] = torch.stack(
                [
                    F.pad(
                        item,
                        [0] * 2 * (len(item.shape) - 1) + [0, max_len - item.shape[0]],
                        value=0,
                    )
                    for item in v
                ]
            )

        model_batch = {
            "seq": seq.to(self._device),
            "mask": mask.to(self._device),
            "batch_of": {k: v.to(self._device) for k, v in batch_of.items()},
        }

        return model_batch

    def _tokenize_multimer(self, sequences: List[str], config: dict) -> tuple[dict, dict[str, torch.Tensor]]:
        """Prepare input for multimer sequences containing ':' chain separators."""
        glycine_linker = config["glycine_linker"]
        linker_map, residue_index, chain_index = store_multimer_properties(sequences, glycine_linker)

        joined_sequences = [seq.replace(":", glycine_linker) for seq in sequences]
        model_batch = self._prepare_batch(joined_sequences)

        multimer_properties = {
            "linker_map": linker_map,
            "residue_index": residue_index,
            "chain_index": chain_index,
        }

        return model_batch, multimer_properties

    def _mask_linker_region(
        self,
        atom_arrays: List[AtomArray],
        plddt_list: List[np.ndarray],
        pae_list: Optional[List[np.ndarray]],
        linker_map: torch.Tensor,
        residue_index: torch.Tensor,
        chain_index: torch.Tensor,
    ) -> tuple[List[AtomArray], List[np.ndarray], Optional[List[np.ndarray]], np.ndarray, np.ndarray]:
        """Remove linker positions from outputs and extract residue/chain indices."""
        masked_atom_arrays = []
        masked_plddt = []
        masked_pae = []
        residue_index_list = []
        chain_index_list = []

        for batch_idx in range(len(atom_arrays)):
            multimer = linker_map[batch_idx].clone()
            multimer = multimer.masked_fill(multimer == -1, 0).cpu().numpy()
            keep_positions = np.where(multimer == 1)[0]

            # Mask atom array: keep only atoms belonging to kept residues
            aa = atom_arrays[batch_idx]
            kept_res_ids = set(keep_positions.tolist())
            mask = np.array([aa.res_id[i] in kept_res_ids for i in range(len(aa))])
            masked_aa = aa[mask]

            # Update res_id and chain_id from multimer properties
            res_idx = residue_index[batch_idx, keep_positions].cpu().numpy().astype(int)
            ch_idx = chain_index[batch_idx, keep_positions].cpu().numpy().astype(int)

            # Map residue index and chain index onto the masked atom array
            for i, pos in enumerate(keep_positions):
                atom_mask = masked_aa.res_id == pos
                masked_aa.res_id[atom_mask] = res_idx[i]
                masked_aa.chain_id[atom_mask] = chr(65 + ch_idx[i])

            masked_atom_arrays.append(masked_aa)
            masked_plddt.append(plddt_list[batch_idx][keep_positions])

            if pae_list is not None:
                masked_pae.append(pae_list[batch_idx][keep_positions][:, keep_positions])

            residue_index_list.append(res_idx)
            chain_index_list.append(ch_idx)

        max_len = max(arr.shape[0] for arr in residue_index_list)
        padded_res = np.stack([np.pad(arr, (0, max_len - len(arr)), constant_values=-1) for arr in residue_index_list])
        padded_chain = np.stack([np.pad(arr, (0, max_len - len(arr)), constant_values=-1) for arr in chain_index_list])

        return (
            masked_atom_arrays,
            masked_plddt,
            masked_pae if pae_list is not None else None,
            padded_res,
            padded_chain,
        )

    def _convert_outputs(
        self,
        raw_outputs: dict,
        sequences: List[str],
        multimer_properties: Optional[dict[str, torch.Tensor]],
        prediction_time: float,
        config: dict,
    ) -> MiniFoldOutput:
        """Convert raw model outputs into a MiniFoldOutput."""
        atom_positions = raw_outputs["final_atom_positions"].float().cpu().numpy()  # (B, L, 37, 3)
        atom_mask = raw_outputs["final_atom_mask"].float().cpu().numpy()  # (B, L, 37)
        plddt_raw = raw_outputs["plddt"].float().cpu().numpy()  # (B, L)

        batch_size = atom_positions.shape[0]
        seq_lengths = [len(seq.replace(":", config.get("glycine_linker", ""))) for seq in sequences]

        atom_array_list = self._convert_outputs_to_atomarray(atom_positions, atom_mask, plddt_raw, sequences, config)

        plddt_list = [plddt_raw[i, : seq_lengths[i]] for i in range(batch_size)]
        pae_list = None  # not available from MiniFold

        # Multimer: mask linker regions
        if multimer_properties is not None:
            (
                atom_array_list,
                plddt_list,
                pae_list,
                residue_index_arr,
                chain_index_arr,
            ) = self._mask_linker_region(
                atom_array_list,
                plddt_list,
                pae_list,
                **multimer_properties,
            )
        else:
            # Monomer: generate chain_index and residue_index
            max_len = max(seq_lengths)
            residue_index_arr = np.stack(
                [np.pad(np.arange(sl), (0, max_len - sl), constant_values=-1) for sl in seq_lengths]
            )
            chain_index_arr = np.stack(
                [np.pad(np.zeros(sl, dtype=np.int32), (0, max_len - sl), constant_values=-1) for sl in seq_lengths]
            )

        self.metadata.prediction_time = prediction_time

        include_fields = config.get("include_fields")
        pdb = None
        cif = None
        if include_fields and ("*" in include_fields or "pdb" in include_fields):
            pdb = self._convert_outputs_to_pdb(atom_array_list)
        if include_fields and ("*" in include_fields or "cif" in include_fields):
            cif = self._convert_outputs_to_cif(atom_array_list)

        full_output = MiniFoldOutput(
            metadata=self.metadata,
            atom_array=atom_array_list,
            plddt=plddt_list,
            pae=pae_list,
            residue_index=residue_index_arr,
            chain_index=chain_index_arr,
            pdb=pdb,
            cif=cif,
        )

        filtered = self._filter_include_fields(full_output, include_fields)
        return cast(MiniFoldOutput, filtered)

    def _convert_outputs_to_atomarray(
        self,
        atom_positions: np.ndarray,
        atom_mask: np.ndarray,
        plddt: np.ndarray,
        sequences: List[str],
        config: dict,
    ) -> List[AtomArray]:
        """Convert atom37 tensors to biotite AtomArray list."""
        from biotite.structure import Atom, array
        from minifold.utils.residue_constants import atom_types, restype_1to3

        batch_size, n_residues, n_atoms = atom_mask.shape

        arrays = []
        for b in range(batch_size):
            atoms = []
            seq = sequences[b].replace(":", config.get("glycine_linker", ""))
            seq_len = len(seq)

            for res_idx in range(min(n_residues, seq_len)):
                res_letter = seq[res_idx]
                res_name = restype_1to3.get(res_letter, "UNK")

                # Per-residue plddt (broadcast to all atoms)
                res_plddt = plddt[b, res_idx]

                for atom_idx in range(n_atoms):
                    if not atom_mask[b, res_idx, atom_idx]:
                        continue

                    coord = atom_positions[b, res_idx, atom_idx]
                    atom = Atom(
                        coord=coord,
                        chain_id="A",
                        atom_name=atom_types[atom_idx],
                        res_name=res_name,
                        res_id=res_idx,  # 0-indexed
                        element=atom_types[atom_idx][0],
                        b_factor=res_plddt,
                    )
                    atoms.append(atom)
            arrays.append(array(atoms))
        return arrays

    def _convert_outputs_to_pdb(self, atom_array: List[AtomArray]) -> list[str]:
        from biotite.structure.io.pdb import PDBFile, set_structure
        from io import StringIO

        pdbs = []
        for a in atom_array:
            structure_file = PDBFile()
            set_structure(structure_file, a)
            string = StringIO()
            structure_file.write(string)
            pdbs.append(string.getvalue())
        return pdbs

    def _convert_outputs_to_cif(self, atom_array: List[AtomArray]) -> list[str]:
        from biotite.structure.io.pdbx import CIFFile, set_structure
        from io import StringIO

        cifs = []
        for a in atom_array:
            structure_file = CIFFile()
            set_structure(structure_file, a)
            string = StringIO()
            structure_file.write(string)
            cifs.append(string.getvalue())
        return cifs
