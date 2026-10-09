"""Model registry and shared contract metadata."""

from dataclasses import dataclass, replace
from importlib import import_module
from typing import Any, Literal

from ..images.metadata import get_model_image_spec

TaskMethod = Literal["fold", "embed"]
TaskKind = Literal["structure", "embedding"]

ESM_IMAGE_NAME = get_model_image_spec("esm").image_name
ESMFOLD2_IMAGE_NAME = get_model_image_spec("esmfold2").image_name
CHAI_IMAGE_NAME = get_model_image_spec("chai").image_name
BOLTZ_IMAGE_NAME = get_model_image_spec("boltz").image_name
PROTENIX_IMAGE_NAME = get_model_image_spec("protenix").image_name
OPENDDE_IMAGE_NAME = get_model_image_spec("opendde").image_name
RF3_IMAGE_NAME = get_model_image_spec("rf3").image_name
ALPHAFOLD2_MULTIMER_IMAGE_NAME = get_model_image_spec("alphafold").image_name


@dataclass(frozen=True)
class ModelContract:
    """Backend-agnostic behavioral contract for a public model wrapper."""

    task_method: TaskMethod
    task_kind: TaskKind
    static_config_keys: frozenset[str]
    minimal_output_fields: tuple[str, ...]
    optional_output_fields: tuple[str, ...] = ()
    supports_batch: bool = True
    supports_multimer: bool = False
    supports_include_fields: bool = True


@dataclass(frozen=True)
class ModelSpec:
    """Runtime metadata for a public model wrapper."""

    key: str
    public_name: str
    family: str
    wrapper_class_path: str
    modal_class_path: str | None
    apptainer_core_class_path: str | None
    apptainer_image_name: str | None
    contract: ModelContract
    supported_backends: tuple[str, ...] = ("modal",)
    default_backend: str = "modal"
    # Families whose kit modes (optimization="exact" / "fast") need a different image than the default one run those
    # modes there: the Modal class on the kit image, and the kit image key (see KIT_IMAGE_SPECS in images/metadata.py)
    # for Apptainer. ``optimization="vanilla"`` never touches them.
    kit_modal_class_path: str | None = None
    kit_image_key: str | None = None


def resolve_object(dotted_path: str) -> Any:
    """Import and return the object addressed by a dotted path."""

    module_path, separator, attr_name = dotted_path.rpartition(".")
    if not separator:
        raise ValueError(f"Invalid dotted path: {dotted_path}")

    module = import_module(module_path)
    return getattr(module, attr_name)


ESMFOLD_SPEC = ModelSpec(
    key="esmfold",
    public_name="ESMFold",
    family="esm",
    wrapper_class_path="boileroom.models.esm.esmfold.ESMFold",
    modal_class_path="boileroom.models.esm.esmfold.ModalESMFold",
    apptainer_core_class_path="boileroom.models.esm.core.ESMFoldCore",
    apptainer_image_name=ESM_IMAGE_NAME,
    supported_backends=("modal", "apptainer"),
    contract=ModelContract(
        task_method="fold",
        task_kind="structure",
        static_config_keys=frozenset({"device"}),
        minimal_output_fields=("metadata", "atom_array"),
        optional_output_fields=(
            "frames",
            "sidechain_frames",
            "unnormalized_angles",
            "angles",
            "states",
            "s_s",
            "s_z",
            "distogram_logits",
            "lm_logits",
            "aatype",
            "atom14_atom_exists",
            "residx_atom14_to_atom37",
            "residx_atom37_to_atom14",
            "atom37_atom_exists",
            "residue_index",
            "lddt_head",
            "plddt",
            "ptm_logits",
            "ptm",
            "aligned_confidence_probs",
            "pae",
            "max_pae",
            "chain_index",
            "pdb",
            "cif",
        ),
        supports_multimer=True,
    ),
)

ESM2_SPEC = ModelSpec(
    key="esm2",
    public_name="ESM2",
    family="esm",
    wrapper_class_path="boileroom.models.esm.esm2.ESM2",
    modal_class_path="boileroom.models.esm.esm2.ModalESM2",
    apptainer_core_class_path="boileroom.models.esm.core.ESM2Core",
    apptainer_image_name=ESM_IMAGE_NAME,
    supported_backends=("modal", "apptainer"),
    contract=ModelContract(
        task_method="embed",
        task_kind="embedding",
        static_config_keys=frozenset({"device", "model_name"}),
        minimal_output_fields=("metadata", "embeddings", "chain_index", "residue_index"),
        optional_output_fields=("hidden_states", "lm_logits"),
        supports_multimer=True,
    ),
)

ESMC_SPEC = ModelSpec(
    key="esmc",
    public_name="ESMC",
    family="esm3",
    wrapper_class_path="boileroom.models.esm3.esmc.ESMC",
    modal_class_path="boileroom.models.esm3.esmc.ModalESMC",
    apptainer_core_class_path="boileroom.models.esm3.core.ESMCCore",
    apptainer_image_name=ESMFOLD2_IMAGE_NAME,
    supported_backends=("modal", "apptainer"),
    contract=ModelContract(
        task_method="embed",
        task_kind="embedding",
        static_config_keys=frozenset({"device", "model_name"}),
        minimal_output_fields=("metadata", "embeddings", "chain_index", "residue_index"),
        optional_output_fields=("hidden_states", "lm_logits"),
        supports_multimer=True,
    ),
)

ESM3_SPEC = ModelSpec(
    key="esm3",
    public_name="ESM3",
    family="esm3",
    wrapper_class_path="boileroom.models.esm3.esm3.ESM3",
    modal_class_path="boileroom.models.esm3.esm3.ModalESM3",
    apptainer_core_class_path="boileroom.models.esm3.core.ESM3Core",
    apptainer_image_name=ESMFOLD2_IMAGE_NAME,
    supported_backends=("modal", "apptainer"),
    contract=ModelContract(
        task_method="embed",
        task_kind="embedding",
        static_config_keys=frozenset({"device", "model_name"}),
        minimal_output_fields=("metadata", "embeddings", "chain_index", "residue_index"),
        optional_output_fields=(
            "lm_logits",
            "sasa_logits",
            "secondary_structure_logits",
            "function_logits",
            "residue_annotation_logits",
        ),
        supports_multimer=True,
    ),
)

ESMFOLD2_SPEC = ModelSpec(
    key="esmfold2",
    public_name="ESMFold2",
    family="esmfold2",
    wrapper_class_path="boileroom.models.esmfold2.esmfold2.ESMFold2",
    modal_class_path="boileroom.models.esmfold2.esmfold2.ModalESMFold2",
    apptainer_core_class_path="boileroom.models.esmfold2.core.ESMFold2Core",
    apptainer_image_name=ESMFOLD2_IMAGE_NAME,
    supported_backends=("modal", "apptainer"),
    kit_modal_class_path="boileroom.models.esmfold2.modal_kit.ModalESMFold2Kit",
    kit_image_key="esmfold2",
    contract=ModelContract(
        task_method="fold",
        task_kind="structure",
        static_config_keys=frozenset(
            {"device", "model_name", "revision", "cache_dir", "ccd_cache_dir", "dtype", "optimization", "kit_msa"}
        ),
        minimal_output_fields=("metadata", "atom_array"),
        optional_output_fields=(
            "plddt",
            "ptm",
            "iptm",
            "pae",
            "distogram",
            "pair_chains_iptm",
            "residue_index",
            "entity_id",
            "pdb",
            "cif",
        ),
        supports_multimer=True,
    ),
)

CHAI1_SPEC = ModelSpec(
    key="chai1",
    public_name="Chai1",
    family="chai",
    wrapper_class_path="boileroom.models.chai.chai1.Chai1",
    modal_class_path="boileroom.models.chai.chai1.ModalChai1",
    apptainer_core_class_path="boileroom.models.chai.core.Chai1Core",
    apptainer_image_name=CHAI_IMAGE_NAME,
    supported_backends=("modal", "apptainer"),
    contract=ModelContract(
        task_method="fold",
        task_kind="structure",
        static_config_keys=frozenset({"device"}),
        minimal_output_fields=("metadata", "atom_array"),
        optional_output_fields=("pae", "pde", "plddt", "ptm", "iptm", "per_chain_iptm", "cif"),
        supports_batch=False,
    ),
)

BOLTZ2_SPEC = ModelSpec(
    key="boltz2",
    public_name="Boltz2",
    family="boltz",
    wrapper_class_path="boileroom.models.boltz.boltz2.Boltz2",
    modal_class_path="boileroom.models.boltz.boltz2.ModalBoltz2",
    apptainer_core_class_path="boileroom.models.boltz.core.Boltz2Core",
    apptainer_image_name=BOLTZ_IMAGE_NAME,
    supported_backends=("modal", "apptainer"),
    contract=ModelContract(
        task_method="fold",
        task_kind="structure",
        static_config_keys=frozenset(
            {
                "device",
                "cache_dir",
                "no_kernels",
                "recycling_steps",
                "sampling_steps",
                "diffusion_samples",
                "max_parallel_samples",
                "step_scale",
                "subsample_msa",
                "num_subsampled_msa",
                "write_full_pae",
                "write_full_pde",
            }
        ),
        minimal_output_fields=("metadata", "atom_array"),
        optional_output_fields=("confidence", "plddt", "ptm", "iptm", "pae", "pde", "pdb", "cif"),
        supports_multimer=True,
    ),
)

PROTENIX_SPEC = ModelSpec(
    key="protenix",
    public_name="Protenix",
    family="protenix",
    wrapper_class_path="boileroom.models.protenix.protenix.Protenix",
    modal_class_path="boileroom.models.protenix.protenix.ModalProtenix",
    apptainer_core_class_path="boileroom.models.protenix.core.ProtenixCore",
    apptainer_image_name=PROTENIX_IMAGE_NAME,
    supported_backends=("modal", "apptainer"),
    kit_modal_class_path="boileroom.models.protenix.modal_kit.ModalProtenixKit",
    kit_image_key="protenix",
    contract=ModelContract(
        task_method="fold",
        task_kind="structure",
        static_config_keys=frozenset(
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
        ),
        minimal_output_fields=("metadata", "atom_array"),
        optional_output_fields=(
            "confidence",
            "plddt",
            "ptm",
            "iptm",
            "pae",
            "token_chain_ids",
            "token_res_ids",
            "atom_plddt",
            "seeds",
            "sample_ranks",
            "pdb",
            "cif",
        ),
        supports_batch=False,
        supports_multimer=True,
    ),
)

OPENDDE_SPEC = ModelSpec(
    key="opendde",
    public_name="OpenDDE",
    family="opendde",
    wrapper_class_path="boileroom.models.opendde.opendde.OpenDDE",
    modal_class_path="boileroom.models.opendde.opendde.ModalOpenDDE",
    apptainer_core_class_path="boileroom.models.opendde.core.OpenDDECore",
    apptainer_image_name=OPENDDE_IMAGE_NAME,
    supported_backends=PROTENIX_SPEC.supported_backends,
    contract=replace(
        PROTENIX_SPEC.contract,
        static_config_keys=PROTENIX_SPEC.contract.static_config_keys | {"opendde_python"},
    ),
)

RF3_SPEC = ModelSpec(
    key="rf3",
    public_name="RF3",
    family="rf3",
    wrapper_class_path="boileroom.models.rf3.rf3.RF3",
    modal_class_path="boileroom.models.rf3.rf3.ModalRF3",
    apptainer_core_class_path="boileroom.models.rf3.core.RF3Core",
    apptainer_image_name=RF3_IMAGE_NAME,
    supported_backends=("modal", "apptainer"),
    kit_modal_class_path="boileroom.models.rf3.modal_kit.ModalRF3Kit",
    kit_image_key="rf3",
    contract=ModelContract(
        task_method="fold",
        task_kind="structure",
        static_config_keys=frozenset(
            {
                "device",
                "n_recycles",
                "diffusion_batch_size",
                "num_steps",
                "checkpoint_path",
                "rf3_python",
                "optimization",
            }
        ),
        minimal_output_fields=("metadata", "atom_array"),
        optional_output_fields=(
            "confidence",
            "plddt",
            "ptm",
            "iptm",
            "pae",
            "token_chain_ids",
            "token_res_ids",
            "atom_plddt",
            "seeds",
            "sample_ranks",
            "sample_indices",
            "pdb",
            "cif",
        ),
        supports_batch=False,
        supports_multimer=True,
    ),
)

SAE_SPEC = ModelSpec(
    key="sae",
    public_name="SAE",
    family="sae",
    wrapper_class_path="boileroom.models.sae.sae.SAE",
    modal_class_path="boileroom.models.sae.sae.ModalSAE",
    apptainer_core_class_path="boileroom.models.sae.core.SAECore",
    # Reuses the shared Biohub ESM runtime image (same as ESM-C / ESM3 / ESMFold2).
    apptainer_image_name=ESMFOLD2_IMAGE_NAME,
    supported_backends=("modal", "apptainer"),
    contract=ModelContract(
        task_method="embed",
        task_kind="embedding",
        static_config_keys=frozenset(
            {
                "device",
                "feature_source",
                "normalize_features",
                "num_features",
                "k",
                "sae_layer",
                "activation",
                "esmc_model_name",
                "sae_repo_id",
                "forge_model",
                "forge_sae_model",
                "forge_url",
                "forge_token",
            }
        ),
        minimal_output_fields=("metadata", "pooled_features", "chain_index", "residue_index"),
        optional_output_fields=("features",),
        supports_multimer=True,
    ),
)

ALPHAFOLD2_MULTIMER_SPEC = ModelSpec(
    key="alphafold2_multimer",
    public_name="AlphaFold2Multimer",
    family="alphafold",
    wrapper_class_path="boileroom.models.alphafold.alphafold2_multimer.AlphaFold2Multimer",
    modal_class_path="boileroom.models.alphafold.alphafold2_multimer.ModalAlphaFold2Multimer",
    apptainer_core_class_path="boileroom.models.alphafold.core.AlphaFold2MultimerCore",
    apptainer_image_name=ALPHAFOLD2_MULTIMER_IMAGE_NAME,
    supported_backends=("modal", "apptainer"),
    contract=ModelContract(
        task_method="fold",
        task_kind="structure",
        static_config_keys=frozenset(
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
        ),
        minimal_output_fields=("metadata", "atom_array"),
        optional_output_fields=("ranking", "plddt", "ptm", "iptm", "pae", "pdb", "cif"),
        supports_batch=False,
        supports_multimer=True,
    ),
)

MODEL_SPECS = (
    ESMFOLD_SPEC,
    ESM2_SPEC,
    ESMFOLD2_SPEC,
    ESMC_SPEC,
    ESM3_SPEC,
    CHAI1_SPEC,
    BOLTZ2_SPEC,
    SAE_SPEC,
    PROTENIX_SPEC,
    OPENDDE_SPEC,
    RF3_SPEC,
    ALPHAFOLD2_MULTIMER_SPEC,
)
MODEL_SPECS_BY_KEY = {spec.key: spec for spec in MODEL_SPECS}
MODEL_SPECS_BY_PUBLIC_NAME = {spec.public_name: spec for spec in MODEL_SPECS}


def get_model_spec(identifier: str) -> ModelSpec:
    """Return a registered model specification by key or public class name."""

    if identifier in MODEL_SPECS_BY_KEY:
        return MODEL_SPECS_BY_KEY[identifier]
    if identifier in MODEL_SPECS_BY_PUBLIC_NAME:
        return MODEL_SPECS_BY_PUBLIC_NAME[identifier]
    raise KeyError(f"Unknown model spec: {identifier}")


__all__ = [
    "ALPHAFOLD2_MULTIMER_SPEC",
    "BOLTZ2_SPEC",
    "CHAI1_SPEC",
    "ESM2_SPEC",
    "ESM3_SPEC",
    "ESMC_SPEC",
    "ESMFOLD2_SPEC",
    "ESMFOLD_SPEC",
    "MODEL_SPECS",
    "MODEL_SPECS_BY_KEY",
    "MODEL_SPECS_BY_PUBLIC_NAME",
    "SAE_SPEC",
    "ModelContract",
    "ModelSpec",
    "PROTENIX_SPEC",
    "OPENDDE_SPEC",
    "RF3_SPEC",
    "get_model_spec",
    "resolve_object",
]
