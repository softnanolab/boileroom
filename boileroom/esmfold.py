"""ESMFold implementation for protein structure prediction using Meta AI's ESM-2 model."""

import os
import logging
from dataclasses import dataclass
from typing import Optional, List, Union

import modal
import numpy as np
from biotite.structure import AtomArray
from . import app
from .base import FoldingAlgorithm, StructurePrediction, PredictionMetadata
from .images import esm_image
from .images.volumes import model_weights
from .utils import MINUTES, MODEL_DIR, GPUS_AVAIL_ON_MODAL
from .utils import Timer
from .linker import compute_position_ids, store_multimer_properties


# Set up basic logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# TODO: turn this into a Pydantic model instead
@dataclass
class ESMFoldOutput(StructurePrediction):
    """Output from ESMFold prediction including all model outputs."""

    # TODO: we should figure out what should be the verbosity of the output,
    # as a usual user does not need all of this information

    # Required by StructurePrediction protocol
    positions: np.ndarray  # (model_layer, batch_size, residue, atom=14, xyz=3)
    metadata: PredictionMetadata

    # Additional ESMFold-specific outputs
    frames: np.ndarray  # (model_layer, batch_size, residue, qxyz=7)
    sidechain_frames: np.ndarray  # (model_layer, batch_size, residue, 8, 4, 4) [rot matrix per sidechain]
    unnormalized_angles: np.ndarray  # (model_layer, batch_size, residue, 7, 2) [torsion angles]
    angles: np.ndarray  # (model_layer, batch_size, residue, 7, 2) [torsion angles]
    states: np.ndarray  # (model_layer, batch_size, residue, ???)
    s_s: np.ndarray  # (batch_size, residue, 1024)
    s_z: np.ndarray  # (batch_size, residue, residue, 128)
    distogram_logits: np.ndarray  # (batch_size, residue, residue, 64) ???
    lm_logits: np.ndarray  # (batch_size, residue, 23) ???
    aatype: np.ndarray  # (batch_size, residue) amino acid identity
    atom14_atom_exists: np.ndarray  # (batch_size, residue, atom=14)
    residx_atom14_to_atom37: np.ndarray  # (batch_size, residue, atom=14)
    residx_atom37_to_atom14: np.ndarray  # (batch_size, residue, atom=37)
    atom37_atom_exists: np.ndarray  # (batch_size, residue, atom=37)
    residue_index: np.ndarray  # (batch_size, residue)
    lddt_head: np.ndarray  # (model_layer, batch_size, residue, atom=37, 50) ??
    plddt: np.ndarray  # (batch_size, residue, atom=37)
    ptm_logits: np.ndarray  # (batch_size, residue, residue, 64) ???
    ptm: np.ndarray  # float # TODO: make it into a float when sending to the client
    aligned_confidence_probs: np.ndarray  # (batch_size, residue, residue, 64)
    predicted_aligned_error: np.ndarray  # (batch_size, residue, residue)
    max_predicted_aligned_error: np.ndarray  # float # TODO: make it into a float when sending to the client
    chain_index: np.ndarray  # (batch_size, residue)
    # TODO: maybe add this to the output to clearly indicate padded residues
    atom_array: Optional[AtomArray] = None  # 0-indexed
    pdb: Optional[list[str]] = None  # 1-indexed
    cif: Optional[list[str]] = None

    # TODO: can add a save method here (to a pickle and a pdb file) that can be run locally
    # TODO: add verification of the outputs, and primarily the shape of all the arrays
    # (see test_esmfold_batch_multimer_linkers for the exact batched shapes)


with esm_image.imports():
    import torch
    from transformers import EsmForProteinFolding, AutoTokenizer

GPU_TO_USE = os.environ.get("BOILEROOM_GPU", "T4")

if GPU_TO_USE not in GPUS_AVAIL_ON_MODAL:
    raise ValueError(
        f"GPU specified in BOILEROOM_GPU environment variable ('{GPU_TO_USE}') not available on "
        f"Modal. Please choose from: {GPUS_AVAIL_ON_MODAL}"
    )


@app.cls(
    image=esm_image,
    gpu=GPU_TO_USE,
    timeout=20 * MINUTES,
    container_idle_timeout=10 * MINUTES,
    volumes={MODEL_DIR: model_weights},
)
class ESMFold(FoldingAlgorithm):
    """ESMFold protein structure prediction model."""

    # TODO: maybe this config should be input to the fold function, so that it can
    # changed programmatically on a single ephermal app, rather than re-creating the app?
    DEFAULT_CONFIG = {
        # ESMFold model config
        "output_pdb": False,
        "output_cif": False,
        "output_atomarray": False,
        # Chain linking and positioning config
        "glycine_linker": "",
        "position_ids_skip": 512,
    }

    # We need to properly asses whether using this or the original ESMFold is better
    # based on speed, accuracy, bugs, etc.; as well as customizability
    # For instance, if we want to also allow differently sized structure modules, than this would be good
    # TODO: we should add a settings dictionary or something, that would make it easier to add new options
    # TODO: maybe use OmegaConf instead to make it easier instead of config
    def __init__(self, config: dict = {}) -> None:
        """Initialize ESMFold."""
        super().__init__(config)
        self.metadata = self._initialize_metadata(
            model_name="ESMFold",
            model_version="v4.49.0",  # HuggingFace transformers version
        )
        self.model_dir: Optional[str] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[EsmForProteinFolding] = None

    @modal.enter()
    def _initialize(self) -> None:
        """Initialize the model during container startup. This helps us determine whether we run locally or remotely."""
        self.model_dir = os.environ.get("HF_MODEL_DIR", MODEL_DIR)
        self._load()

    def _load(self) -> None:
        """Load the ESMFold model and tokenizer."""
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1", cache_dir=self.model_dir)
        if self.model is None:
            self.model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", cache_dir=self.model_dir)
        self.device = "cuda"
        self.model = self.model.cuda()
        self.model.eval()
        self.model.trunk.set_chunk_size(64)
        self.ready = True

    @modal.method()
    def fold(self, sequences: Union[str, List[str]]) -> ESMFoldOutput:
        """Predict protein structure(s) using ESMFold."""
        if self.tokenizer is None or self.model is None:
            logger.warning("Model not loaded. Forcing the model to load... Next time call _load() first.")
            self._load()
        assert self.tokenizer is not None and self.model is not None, "Model not loaded"

        if isinstance(sequences, str):
            sequences = [sequences]

        sequences = self._validate_sequences(sequences)
        self.metadata.sequence_lengths = self._compute_sequence_lengths(sequences)

        tokenized_input, multimer_properties = self._tokenize_sequences(sequences)

        with Timer("Model Inference") as timer:
            with torch.inference_mode():
                outputs = self.model(**tokenized_input)

        outputs = self._convert_outputs(outputs, multimer_properties, timer.duration)
        return outputs

    def _tokenize_sequences(self, sequences: List[str]) -> tuple[dict, dict[str, torch.Tensor] | None]:
        assert self.tokenizer is not None, "Tokenizer not loaded"
        if ":" in "".join(sequences):  # MULTIMER setting
            tokenized, multimer_properties = self._tokenize_multimer(sequences)
        else:  # MONOMER setting
            tokenized = self.tokenizer(
                sequences, return_tensors="pt", add_special_tokens=False, padding=True, truncation=True, max_length=1024
            )
            multimer_properties = None
        tokenized = {k: v.to(self.device) for k, v in tokenized.items()}

        return tokenized, multimer_properties

    def _tokenize_multimer(self, sequences: List[str]) -> torch.Tensor:
        assert self.tokenizer is not None, "Tokenizer not loaded"
        # Store multimer properties first
        linker_map, residue_index, chain_index = store_multimer_properties(sequences, self.config["glycine_linker"])

        # Create tokenized input using list comprehension directly
        glycine_linker = self.config["glycine_linker"]
        tokenized = self.tokenizer(
            [seq.replace(":", glycine_linker) for seq in sequences],
            padding=True,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )

        # Add position IDs
        tokenized["position_ids"] = compute_position_ids(sequences, glycine_linker, self.config["position_ids_skip"])

        # Create attention mask (1 means keep, 0 means mask)
        # This also masks padding tokens, which are -1
        tokenized["attention_mask"] = (linker_map == 1).to(torch.int32)

        return tokenized, {"linker_map": linker_map, "residue_index": residue_index, "chain_index": chain_index}

    def _mask_linker_region(
        self,
        outputs: dict,
        linker_map: torch.Tensor,
        residue_index: torch.Tensor,
        chain_index: torch.Tensor,
    ) -> dict:
        """Mask the linker region in the outputs and track padding information.
        This includes all the metrics.

        Args:
            outputs: Dictionary containing model outputs

        Returns:
            dict: Updated outputs with linker regions masked and padding information
        """
        assert isinstance(linker_map, torch.Tensor), "linker_map must be a tensor"

        positions = []
        frames = []
        sidechain_frames = []
        unnormalized_angles = []
        angles = []
        states = []
        lddt_head = []

        s_s = []
        lm_logits = []
        aatype = []
        atom14_atom_exists = []
        residx_atom14_to_atom37 = []
        residx_atom37_to_atom14 = []
        atom37_atom_exists = []
        plddt = []

        s_z = []
        distogram_logits = []
        ptm_logits = []
        aligned_confidence_probs = []
        predicted_aligned_error = []

        _residue_index = []
        _chain_index = []

        for batch_idx, multimer in enumerate(linker_map):
            # Drop the -1 values, meaning 1s refer to residues we want to keep
            multimer = multimer.masked_fill(multimer == -1, 0).cpu().numpy()
            # Chain indices are the ones that were not masked, hence they were kept and are thus 1
            chain_indices = np.where(multimer == 1)[0]

            # 3rd dim is residue index
            positions.append(outputs["positions"][:, batch_idx, chain_indices])
            frames.append(outputs["frames"][:, batch_idx, chain_indices])
            sidechain_frames.append(outputs["sidechain_frames"][:, batch_idx, chain_indices])
            unnormalized_angles.append(outputs["unnormalized_angles"][:, batch_idx, chain_indices])
            angles.append(outputs["angles"][:, batch_idx, chain_indices])
            states.append(outputs["states"][:, batch_idx, chain_indices])
            lddt_head.append(outputs["lddt_head"][:, batch_idx, chain_indices])

            # 2nd dim is residue index
            s_s.append(outputs["s_s"][batch_idx, chain_indices])
            lm_logits.append(outputs["lm_logits"][batch_idx, chain_indices])
            aatype.append(outputs["aatype"][batch_idx, chain_indices])
            atom14_atom_exists.append(outputs["atom14_atom_exists"][batch_idx, chain_indices])
            residx_atom14_to_atom37.append(outputs["residx_atom14_to_atom37"][batch_idx, chain_indices])
            residx_atom37_to_atom14.append(outputs["residx_atom37_to_atom14"][batch_idx, chain_indices])
            atom37_atom_exists.append(outputs["atom37_atom_exists"][batch_idx, chain_indices])
            plddt.append(outputs["plddt"][batch_idx, chain_indices])

            # 2D properties that are per residue pair; thus residues is both the 2nd and 3rd dim
            s_z.append(outputs["s_z"][batch_idx, chain_indices][:, chain_indices])
            distogram_logits.append(outputs["distogram_logits"][batch_idx, chain_indices][:, chain_indices])
            ptm_logits.append(outputs["ptm_logits"][batch_idx, chain_indices][:, chain_indices])
            aligned_confidence_probs.append(
                outputs["aligned_confidence_probs"][batch_idx, chain_indices][:, chain_indices]
            )
            predicted_aligned_error.append(
                outputs["predicted_aligned_error"][batch_idx, chain_indices][:, chain_indices]
            )

            # Custom outputs, that also have 2nd dimension as residue index
            _residue_index.append(residue_index[batch_idx, chain_indices].cpu().numpy())
            _chain_index.append(chain_index[batch_idx, chain_indices].cpu().numpy())

        def pad_and_stack(
            arrays: list[np.ndarray], residue_dim: Union[int, List[int]], batch_dim: int, intermediate_dim: bool = False
        ) -> np.ndarray:
            """Pad arrays to match the largest size in the residue dimension and stack them in the batch dimension.

            Args:
                arrays: List of NumPy arrays to pad and stack
                residue_dim: Dimension(s) to pad to match sizes
                batch_dim: Dimension to stack the arrays along
                intermediate_dim: Whether the array has an intermediate dimension to preserve

            Returns:
                Stacked and padded NumPy array
            """
            if isinstance(residue_dim, int):
                max_size = max(arr.shape[residue_dim] for arr in arrays)
                padded_arrays = []
                for arr in arrays:
                    padding = [(0, 0)] * arr.ndim
                    padding[residue_dim] = (0, max_size - arr.shape[residue_dim])
                    padded_arrays.append(np.pad(arr, padding, mode="constant", constant_values=-1))
            elif isinstance(residue_dim, list):
                # Multi-dimension padding (e.g., for 2D matrices)
                max_sizes = []
                for dim in residue_dim:
                    max_sizes.append(max(arr.shape[dim] for arr in arrays))

                padded_arrays = []
                for arr in arrays:
                    padding = [(0, 0)] * arr.ndim
                    for dim, max_size in zip(residue_dim, max_sizes):
                        padding[dim] = (0, max_size - arr.shape[dim])
                    padded_arrays.append(np.pad(arr, padding, mode="constant", constant_values=-1))

            # Handle intermediate dimensions differently
            if intermediate_dim:
                # Stack along axis=1 to preserve intermediate dim as first dimension
                return np.stack(padded_arrays, axis=1)
            else:
                return np.stack(padded_arrays, axis=batch_dim)

        # 2nd dimension is the batch size, 3rd dimension was the residue index (without batch it's the 2nd dim)
        # These are not done same as below is because of getting the 8 intermediate outputs from StructureModule
        outputs["positions"] = pad_and_stack(positions, residue_dim=1, batch_dim=0, intermediate_dim=True)
        outputs["frames"] = pad_and_stack(frames, residue_dim=1, batch_dim=0, intermediate_dim=True)
        outputs["sidechain_frames"] = pad_and_stack(sidechain_frames, residue_dim=1, batch_dim=0, intermediate_dim=True)
        outputs["unnormalized_angles"] = pad_and_stack(
            unnormalized_angles, residue_dim=1, batch_dim=0, intermediate_dim=True
        )
        outputs["angles"] = pad_and_stack(angles, residue_dim=1, batch_dim=0, intermediate_dim=True)
        outputs["states"] = pad_and_stack(states, residue_dim=1, batch_dim=0, intermediate_dim=True)
        outputs["lddt_head"] = pad_and_stack(lddt_head, residue_dim=1, batch_dim=0, intermediate_dim=True)

        # 1st dimension is the batch size, 2nd dimension was the residue index (without batch it's the 1st dim)
        outputs["s_s"] = pad_and_stack(s_s, residue_dim=0, batch_dim=0)
        outputs["lm_logits"] = pad_and_stack(lm_logits, residue_dim=0, batch_dim=0)
        outputs["aatype"] = pad_and_stack(aatype, residue_dim=0, batch_dim=0)
        outputs["atom14_atom_exists"] = pad_and_stack(atom14_atom_exists, residue_dim=0, batch_dim=0)
        outputs["residx_atom14_to_atom37"] = pad_and_stack(residx_atom14_to_atom37, residue_dim=0, batch_dim=0)
        outputs["residx_atom37_to_atom14"] = pad_and_stack(residx_atom37_to_atom14, residue_dim=0, batch_dim=0)
        outputs["atom37_atom_exists"] = pad_and_stack(atom37_atom_exists, residue_dim=0, batch_dim=0)
        outputs["plddt"] = pad_and_stack(plddt, residue_dim=0, batch_dim=0)

        # 2D properties, otherwise same as above
        outputs["s_z"] = pad_and_stack(s_z, residue_dim=[0, 1], batch_dim=0)
        outputs["distogram_logits"] = pad_and_stack(distogram_logits, residue_dim=[0, 1], batch_dim=0)
        outputs["ptm_logits"] = pad_and_stack(ptm_logits, residue_dim=[0, 1], batch_dim=0)
        outputs["aligned_confidence_probs"] = pad_and_stack(aligned_confidence_probs, residue_dim=[0, 1], batch_dim=0)
        outputs["predicted_aligned_error"] = pad_and_stack(predicted_aligned_error, residue_dim=[0, 1], batch_dim=0)

        # Custom
        outputs["chain_index"] = pad_and_stack(_chain_index, residue_dim=0, batch_dim=0)
        outputs["residue_index"] = pad_and_stack(_residue_index, residue_dim=0, batch_dim=0)

        return outputs

    def _convert_outputs(
        self,
        outputs: dict,
        multimer_properties: dict[str, torch.Tensor] | None,
        prediction_time: float,
    ) -> ESMFoldOutput:
        """Convert model outputs to ESMFoldOutput format."""

        outputs = {k: v.cpu().numpy() for k, v in outputs.items()}
        if multimer_properties is not None:
            # TODO: maybe add a proper MULTIMER flag?
            outputs = self._mask_linker_region(outputs, **multimer_properties)
        else:  # only MONOMERs
            outputs["chain_index"] = np.zeros(outputs["residue_index"].shape, dtype=np.int32)

        self.metadata.prediction_time = prediction_time

        atom_array = self._convert_outputs_to_atomarray(outputs)
        if self.config["output_pdb"]:
            outputs["pdb"] = self._convert_outputs_to_pdb(atom_array)
        if self.config["output_cif"]:
            outputs["cif"] = self._convert_outputs_to_cif(atom_array)
        if self.config["output_atomarray"]:
            outputs["atom_array"] = atom_array

        return ESMFoldOutput(metadata=self.metadata, **outputs)

    def _convert_outputs_to_atomarray(self, outputs: dict) -> AtomArray:
        """Convert ESMFold outputs to a Biotite AtomArray.

        Args:
            outputs: Dictionary containing ESMFold model outputs

        Returns:
            AtomArray: Biotite structure representation
        """
        from biotite.structure import Atom, array
        from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
        from transformers.models.esm.openfold_utils.residue_constants import atom_types, restypes, restype_1to3

        # Convert atom14 to atom37 format
        atom_positions = atom14_to_atom37(
            outputs["positions"][-1], outputs
        )  # (model_layer, batch, residue, atom37, xyz)
        atom_mask = outputs["atom37_atom_exists"]  # (batch, residue, atom37)

        assert len(atom_types) == atom_positions.shape[2] == 37, "Atom types must be 37"

        # Get batch and residue dimensions
        batch_size, n_residues, n_atoms = atom_mask.shape

        # Create list to store atoms
        arrays = []

        # Process each protein in the batch
        for b in range(batch_size):
            atoms = []  # clear out the atoms list
            # Process each residue
            for res_idx in range(n_residues):
                # Get chain ID (convert numeric index to letter A-Z)
                chain_id = chr(65 + outputs["chain_index"][b, res_idx])  # A=65 in ASCII

                # Get residue name (3-letter code)
                aa_type = outputs["aatype"][b, res_idx]  # id representing residue identity
                res_name = restypes[aa_type]  # 1-letter residue identity
                res_name = restype_1to3[res_name]  # 3-letter residue identity

                # Process each atom in the residue
                for atom_idx in range(n_atoms):  # loops through all 37 atom types
                    # Skip if atom doesn't exist
                    if not atom_mask[b, res_idx, atom_idx]:
                        continue

                    # Get atom coordinates
                    coord = atom_positions[b, res_idx, atom_idx]

                    # Create Atom object
                    atom = Atom(
                        coord=coord,
                        chain_id=chain_id,
                        atom_name=atom_types[atom_idx],
                        res_name=res_name,
                        res_id=outputs["residue_index"][b, res_idx],  # 0-indexed
                        element=atom_types[atom_idx][0],
                        # we only support C, N, O, S, [according to OpenFold Protein class]
                        # element is thus the first character of any atom name (according to PDB nomenclature)
                        b_factor=outputs["plddt"][b, res_idx, atom_idx],
                    )
                    atoms.append(atom)
            arrays.append(array(atoms))
        return arrays

    def _convert_outputs_to_pdb(self, atom_array: AtomArray) -> list[str]:
        # TODO: this might make more sense to do locally, instead of doing it on the Modal instance
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

    def _convert_outputs_to_cif(self, atom_array: AtomArray) -> list[str]:
        # TODO: this might make more sense to do locally, instead of doing it on the Modal instance
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


def get_esmfold(gpu_type="T4", config: dict = {}):
    """
    Note that the app will still show that's using T4, but the actual method / function call will use the correct GPU,
    and display accordingly in the Modal dashboard.
    """
    Model = ESMFold.with_options(gpu=gpu_type)  # type: ignore
    return Model(config=config)
