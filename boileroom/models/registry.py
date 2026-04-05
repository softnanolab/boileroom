"""Model registry and shared contract metadata."""

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Literal

from ..images.metadata import get_model_image_spec

TaskMethod = Literal["fold", "embed"]
TaskKind = Literal["structure", "embedding"]

ESM_IMAGE_NAME = get_model_image_spec("esm").image_name
CHAI_IMAGE_NAME = get_model_image_spec("chai").image_name
BOLTZ_IMAGE_NAME = get_model_image_spec("boltz").image_name


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
        optional_output_fields=("hidden_states",),
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
        static_config_keys=frozenset({"device", "cache_dir", "no_kernels"}),
        minimal_output_fields=("metadata", "atom_array"),
        optional_output_fields=("confidence", "plddt", "pae", "pde", "pdb", "cif"),
        supports_multimer=True,
    ),
)

MODEL_SPECS = (ESMFOLD_SPEC, ESM2_SPEC, CHAI1_SPEC, BOLTZ2_SPEC)
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
    "BOLTZ2_SPEC",
    "CHAI1_SPEC",
    "ESM2_SPEC",
    "ESMFOLD_SPEC",
    "MODEL_SPECS",
    "MODEL_SPECS_BY_KEY",
    "MODEL_SPECS_BY_PUBLIC_NAME",
    "ModelContract",
    "ModelSpec",
    "get_model_spec",
    "resolve_object",
]
