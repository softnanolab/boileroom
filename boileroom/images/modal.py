"""Helpers for Modal images backed by published Docker images, and for the optimization-kit images."""

from __future__ import annotations

from modal import Image

from ..utils import HOURS, MODAL_MODEL_DIR
from .metadata import (
    RuntimeImageSpec,
    format_image_reference,
    get_image_tag,
    get_kit_image_source,
    get_kit_image_spec,
    get_model_image_spec,
    render_modal_runtime_env,
)

# The ESMFold2 kit image compiles flash-attn, TransformerEngine and xformers from source: 1893 s on 64 cores. Here that is
# one build step with its own CPU and memory request, which a Dockerfile RUN step on Modal cannot make. kit_wheels.sh
# caps its parallel jobs at one per 9 GB of available memory, so the memory request matters as much as the cores.
ESMFOLD2_KIT_COMPILE_CPU = 48
ESMFOLD2_KIT_COMPILE_MEMORY_MIB = 196608
ESMFOLD2_KIT_COMPILE_TIMEOUT = 4 * HOURS
# A100, H100 and H200 (compute capability 8.0 and 9.0); ``img_ef2_fa`` would be 9.0 only.
ESMFOLD2_KIT_STACK = "img_esmfold2_a100"
_KIT_SCRIPTS_DIR = "/opt/boileroom-kit"
_ESMFOLD2_KIT_WORKDIR = "/kit/esmfold2"


def get_modal_image(identifier: str) -> Image:
    """Return a Modal image sourced from the published Docker image for a model."""
    spec = get_model_image_spec(identifier)
    image = Image.from_registry(format_image_reference(spec.image_name, get_image_tag()))
    return image.env(render_modal_runtime_env(spec, MODAL_MODEL_DIR))


def compile_esmfold2_kit() -> None:
    """Build step of the ESMFold2 kit image: compile the CUDA extensions, then install the kit.

    Runs inside the image that ``boileroom/models/esmfold2/kit/Dockerfile`` produces with ``WHEELS_FROM=defer``, on the
    builder that :func:`get_modal_kit_image` requests. It only calls the two scripts the Dockerfile itself runs when
    ``WHEELS_FROM=build``, so the Modal build and a local ``docker build`` end in the same image. Standard library only:
    the image has no boileroom dependencies at this point.
    """
    import subprocess

    # The first step of the build the benchmarked images came from: /tmp world-writable with the sticky bit.
    subprocess.run(["chmod", "1777", "/tmp"], check=True)
    for script in ("kit_wheels.sh", "kit_finish.sh"):
        subprocess.run(["sh", f"{_KIT_SCRIPTS_DIR}/{script}"], check=True, cwd=_ESMFOLD2_KIT_WORKDIR)


def _build_kit_image(spec: RuntimeImageSpec) -> Image:
    """Build a kit image on Modal from its Dockerfile in this repository."""
    if not spec.dockerfile_path.is_file():
        raise FileNotFoundError(
            f"The Dockerfile of the {spec.key} kit image is not in this installation ({spec.dockerfile_path}). "
            f"Use an image from a registry instead: set BOILEROOM_KIT_IMAGE_SOURCE=registry."
        )
    if spec.key == "esmfold2":
        image = Image.from_dockerfile(
            spec.dockerfile_path,
            context_dir=spec.context_path,
            build_args={"STACK": ESMFOLD2_KIT_STACK, "WHEELS_FROM": "defer"},
        )
        return image.run_function(
            compile_esmfold2_kit,
            cpu=ESMFOLD2_KIT_COMPILE_CPU,
            memory=ESMFOLD2_KIT_COMPILE_MEMORY_MIB,
            timeout=ESMFOLD2_KIT_COMPILE_TIMEOUT,
            env={"STACK": ESMFOLD2_KIT_STACK, "BUILD_JOBS": str(ESMFOLD2_KIT_COMPILE_CPU)},
        )
    return Image.from_dockerfile(spec.dockerfile_path, context_dir=spec.context_path)


def get_modal_kit_image(identifier: str) -> Image:
    """Return the Modal image that runs ``optimization="exact"`` and ``"fast"`` for a model family.

    By default Modal builds it from the kit Dockerfile in this repository (the first use takes tens of minutes, then
    Modal caches it). With ``BOILEROOM_KIT_IMAGE_SOURCE=registry`` it pulls ``<repository>/boileroom-<family>-kit:<tag>``
    instead, for an image you built and pushed yourself: nothing publishes the kit images.
    """
    spec = get_kit_image_spec(identifier)
    if get_kit_image_source() == "registry":
        image = Image.from_registry(format_image_reference(spec.image_name, get_image_tag()))
    else:
        image = _build_kit_image(spec)
    return image.env(render_modal_runtime_env(spec, MODAL_MODEL_DIR))
