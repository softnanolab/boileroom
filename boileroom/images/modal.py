"""Helpers for Modal images backed by published Docker images, and for the optimization-kit images."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Final

from modal import Image

from ..provenance import IMAGE_REF_ENV
from ..utils import HOURS, MODAL_MODEL_DIR
from .metadata import (
    RuntimeImageSpec,
    format_image_reference,
    get_image_tag,
    get_kit_image_source,
    get_kit_image_spec,
    get_model_image_spec,
    kit_build_reference,
    kit_image_reference,
    render_modal_runtime_env,
)

# The ESMFold2 kit image compiles flash-attn, TransformerEngine and xformers from source: 1893 s on 64 cores. Here that is
# one build step with its own CPU and memory request, which a Dockerfile RUN step on Modal cannot make. kit_wheels.sh
# caps its parallel jobs at one per 9 GB of available memory, so the memory request matters as much as the cores.
ESMFOLD2_KIT_COMPILE_CPU = 48
ESMFOLD2_KIT_COMPILE_MEMORY_MIB = 196608
ESMFOLD2_KIT_COMPILE_TIMEOUT = 4 * HOURS
# Modal's own default timeout of a build step; enough for the ESMFold2 kit's finish step (pip installs and imports).
KIT_BUILD_STEP_TIMEOUT = 1 * HOURS
# A100, H100 and H200 (compute capability 8.0 and 9.0); ``img_ef2_fa`` would be 9.0 only and fail on the A100 class.
ESMFOLD2_KIT_STACK = "img_esmfold2_a100"
# The single definition of the ESMFold2 kit build, read by the build and by its BOILEROOM_IMAGE_REF alike.
ESMFOLD2_KIT_BUILD_ARGS: Final[Mapping[str, str]] = MappingProxyType(
    {"STACK": ESMFOLD2_KIT_STACK, "WHEELS_FROM": "defer"}
)
ESMFOLD2_KIT_COMPILE_ENV: Final[Mapping[str, str]] = MappingProxyType(
    {"STACK": ESMFOLD2_KIT_STACK, "BUILD_JOBS": str(ESMFOLD2_KIT_COMPILE_CPU)}
)
_KIT_SCRIPTS_DIR = "/opt/boileroom-kit"
_ESMFOLD2_KIT_WORKDIR = "/kit/esmfold2"


def _with_image_ref(image: Image, reference: str) -> Image:
    """Return ``image`` with BOILEROOM_IMAGE_REF set to ``reference`` as its last layer.

    The runtime reports this value as ``image_ref`` in each prediction's provenance (boileroom.provenance). It is the
    last layer so that changing it never invalidates a cached build step underneath.
    """
    return image.env({IMAGE_REF_ENV: reference})


def get_modal_image(identifier: str) -> Image:
    """Return a Modal image sourced from the published Docker image for a model."""
    spec = get_model_image_spec(identifier)
    reference = format_image_reference(spec.image_name, get_image_tag())
    image = Image.from_registry(reference).env(render_modal_runtime_env(spec, MODAL_MODEL_DIR))
    return _with_image_ref(image, reference)


def compile_esmfold2_kit() -> None:
    """First build step of the ESMFold2 kit image on Modal: compile and install the CUDA extensions (``kit_wheels.sh``).

    Runs inside the image that ``boileroom/models/esmfold2/kit/Dockerfile`` produces with ``WHEELS_FROM=defer``, on the
    sized builder that :func:`get_modal_kit_image` requests. It runs the script the Dockerfile's first ``WHEELS_FROM``
    step runs when ``WHEELS_FROM=build``; :func:`finish_esmfold2_kit` runs the second as its own build step, so that a
    failed finish (a transient pip error, a failing smoke check) is retried without recompiling. Standard library only:
    the image has no boileroom dependencies at this point.
    """
    import subprocess

    # The first step of the build the benchmarked images came from: /tmp world-writable with the sticky bit.
    subprocess.run(["chmod", "1777", "/tmp"], check=True)
    subprocess.run(["sh", f"{_KIT_SCRIPTS_DIR}/kit_wheels.sh"], check=True, cwd=_ESMFOLD2_KIT_WORKDIR)


def finish_esmfold2_kit() -> None:
    """Second build step of the ESMFold2 kit image on Modal: install the kit and smoke-check it (``kit_finish.sh``).

    Runs on top of :func:`compile_esmfold2_kit`, as the Dockerfile's second ``WHEELS_FROM`` step does when
    ``WHEELS_FROM=build``, so the Modal build and a local ``docker build`` end in the same image. ``kit_finish.sh`` ends
    with ``kit_smoke.py``, which fails this step (and so the image build) when a compiled kernel does not import or the
    fork would select a slow path. Standard library only.
    """
    import subprocess

    subprocess.run(["sh", f"{_KIT_SCRIPTS_DIR}/kit_finish.sh"], check=True, cwd=_ESMFOLD2_KIT_WORKDIR)


@dataclass(frozen=True)
class KitBuildStep:
    """One function Modal runs on top of a kit image's Dockerfile, as its own (separately cached) build step.

    Attributes
    ----------
    function : Callable[[], None]
        The step. Its source is part of the image reference.
    env : Mapping[str, str]
        Environment of the step.
    cpu : int | None
        CPU request of the step, or None for Modal's default.
    memory_mib : int | None
        Memory request of the step in MiB, or None for Modal's default.
    timeout : int
        Timeout of the step, in seconds. It cannot change the image, so the reference leaves it out.
    """

    function: Callable[[], None]
    env: Mapping[str, str] = field(default_factory=dict)
    cpu: int | None = None
    memory_mib: int | None = None
    timeout: int = KIT_BUILD_STEP_TIMEOUT


@dataclass(frozen=True)
class KitBuild:
    """How Modal builds one kit image on top of its Dockerfile: the single source of the build and of its reference.

    :func:`_build_kit_image` builds from these fields and :meth:`reference_inputs` hashes the same fields, so a build
    input added here changes BOILEROOM_IMAGE_REF along with the image.

    Attributes
    ----------
    build_args : Mapping[str, str]
        Docker build args passed to the Dockerfile.
    steps : tuple[KitBuildStep, ...]
        The build steps Modal runs on top of the Dockerfile image, in order.
    """

    build_args: Mapping[str, str] = field(default_factory=dict)
    steps: tuple[KitBuildStep, ...] = ()

    def reference_inputs(self) -> dict[str, str]:
        """Return the build inputs besides the Dockerfile and its context, by name, for :func:`kit_build_reference`.

        A step's CPU request is covered through its environment where it matters (``BUILD_JOBS``); its memory is
        listed because ``kit_wheels.sh`` sizes its parallel jobs by the memory it is given.
        """
        inputs = {f"build-arg {name}": value for name, value in self.build_args.items()}
        for index, step in enumerate(self.steps):
            inputs.update({f"step {index} env {name}": value for name, value in step.env.items()})
            if step.memory_mib is not None:
                inputs[f"step {index} memory-mib"] = str(step.memory_mib)
            inputs[f"step {index} function"] = inspect.getsource(step.function)
        return inputs


def _kit_build(spec: RuntimeImageSpec) -> KitBuild:
    """Return the Modal build of a kit image: the ESMFold2 kit compiles its CUDA extensions, the others need nothing."""
    if spec.key == "esmfold2":
        compile_step = KitBuildStep(
            function=compile_esmfold2_kit,
            env=ESMFOLD2_KIT_COMPILE_ENV,
            cpu=ESMFOLD2_KIT_COMPILE_CPU,
            memory_mib=ESMFOLD2_KIT_COMPILE_MEMORY_MIB,
            timeout=ESMFOLD2_KIT_COMPILE_TIMEOUT,
        )
        # Default builder resources: the finish step installs and imports, it compiles nothing.
        finish_step = KitBuildStep(function=finish_esmfold2_kit)
        return KitBuild(build_args=ESMFOLD2_KIT_BUILD_ARGS, steps=(compile_step, finish_step))
    return KitBuild()


def _build_kit_image(spec: RuntimeImageSpec, build: KitBuild) -> Image:
    """Build a kit image on Modal from its Dockerfile in this repository, as ``build`` describes."""
    if not spec.dockerfile_path.is_file():
        raise FileNotFoundError(
            f"The Dockerfile of the {spec.key} kit image is not in this installation ({spec.dockerfile_path}). "
            f"Use an image from a registry instead: set BOILEROOM_KIT_IMAGE_SOURCE=registry."
        )
    dockerfile_kwargs: dict[str, Any] = {"build_args": dict(build.build_args)} if build.build_args else {}
    image = Image.from_dockerfile(spec.dockerfile_path, context_dir=spec.context_path, **dockerfile_kwargs)
    for step in build.steps:
        image = image.run_function(
            step.function,
            cpu=step.cpu,
            memory=step.memory_mib,
            timeout=step.timeout,
            env=dict(step.env),
        )
    return image


def get_modal_kit_image(identifier: str) -> Image:
    """Return the Modal image that runs ``optimization="exact"`` and ``"fast"`` for a model family.

    By default Modal builds it from the kit Dockerfile in this repository (the first use takes tens of minutes, then
    Modal caches it). With ``BOILEROOM_KIT_IMAGE_SOURCE=registry`` it pulls the published kit image instead:
    ``<repository>/boileroom-<family>-kit@<digest>`` at the digest pinned in ``KIT_IMAGE_DIGESTS``, or
    ``<repository>/boileroom-<family>-kit:<tag>`` when ``BOILEROOM_KIT_IMAGE_TAG`` names a tag. ``BOILEROOM_IMAGE_TAG``
    does not apply to kit images.

    Parameters
    ----------
    identifier : str
        Model family key, such as ``"esmfold2"`` or ``"protenix"``.

    Returns
    -------
    Image
        The kit image, with BOILEROOM_IMAGE_REF naming the registry reference or the Dockerfile build as its last layer.
    """
    spec = get_kit_image_spec(identifier)
    if get_kit_image_source() == "registry":
        reference = kit_image_reference(spec)
        image = Image.from_registry(reference)
    else:
        build = _kit_build(spec)
        image = _build_kit_image(spec, build)
        reference = kit_build_reference(spec, build.reference_inputs())
    return _with_image_ref(image.env(render_modal_runtime_env(spec, MODAL_MODEL_DIR)), reference)
