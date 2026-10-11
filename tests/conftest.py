"""Pytest configuration for the boileroom package."""

import contextlib
import os
import pathlib
import re

import pytest

from boileroom.images.metadata import (
    DOCKER_REPOSITORY_ENV,
    IMAGE_TAG_ENV,
    KIT_IMAGE_SOURCE_ENV,
    KIT_IMAGE_TAG_ENV,
    get_kit_image_source,
    normalize_docker_repository,
    normalize_kit_image_tag,
)
from boileroom.utils import GPUS_AVAIL_ON_MODAL

_APPTAINER_DEVICE_RE = re.compile(r"^(cpu|cuda(:\d+)?)$")
#: Environment opt-in for the ``kit`` tests, equivalent to ``--run-kit``.
RUN_KIT_ENV = "BOILEROOM_RUN_KIT"
_RUN_KIT_VALUES = {"1": True, "true": True, "yes": True, "": False, "0": False, "false": False, "no": False}


def pytest_report_header(config: pytest.Config) -> list[str]:
    """Report the runtime image tag and Docker repository at the top of every pytest run.

    Both Modal and Apptainer backends pull the boileroom package from a prebuilt
    Docker image. Surfacing the selected tag in the pytest header avoids ambiguity
    about which image the integration tests are actually exercising.

    Parameters
    ----------
    config : pytest.Config
        The active pytest configuration.

    Returns
    -------
    list[str]
        One-line-per-entry header additions.
    """
    try:
        from boileroom.images.metadata import (
            KIT_IMAGE_DIGESTS,
            get_docker_repository,
            get_image_tag,
            resolve_registry_tag,
        )
    except ImportError as exc:  # pragma: no cover - defensive; keep pytest running if the module is missing
        return [f"boileroom image: <unresolved: {exc!s}>"]

    image_tag = config.getoption("--image-tag")
    raw_backend = config.getoption("--backend")
    raw_family, raw_sep, raw_backend_tag = raw_backend.partition(":")
    if raw_family.strip() == "apptainer" and raw_sep and raw_backend_tag.strip():
        tag = resolve_registry_tag(raw_backend_tag)
        source = "from --backend tag"
    elif image_tag:
        tag = resolve_registry_tag(image_tag)
        source = "from --image-tag"
    else:
        tag = get_image_tag()
        override = os.environ.get(IMAGE_TAG_ENV)
        source = f"override via {IMAGE_TAG_ENV}" if override else "from pyproject.toml"
    repository = get_docker_repository()
    lines = [f"boileroom image: {repository}/boileroom-<family>:{tag} ({source})"]
    lines.append(f"boileroom kit image: {_kit_image_header(raw_backend, repository, len(KIT_IMAGE_DIGESTS))}")
    return lines


def _kit_image_header(raw_backend: str, repository: str, pinned: int) -> str:
    """Describe where the selected backend gets the kit images (``optimization="exact"`` / ``"fast"``).

    Parameters
    ----------
    raw_backend : str
        The ``--backend`` value.
    repository : str
        The Docker repository namespace in effect.
    pinned : int
        How many kit images have a pinned digest.

    Returns
    -------
    str
        The header text after ``boileroom kit image:``.
    """
    family, _, inline_tag = raw_backend.partition(":")
    kit_tag = os.environ.get(KIT_IMAGE_TAG_ENV, "").strip()
    if family.strip() == "apptainer" and inline_tag.strip():
        return f"{repository}/boileroom-<family>-kit:{inline_tag.strip()} (from --backend tag)"
    if family.strip() == "modal":
        try:
            source = get_kit_image_source()
        except ValueError as exc:
            return f"<invalid {KIT_IMAGE_SOURCE_ENV}: {exc}>"
        if source == "build":
            ignored = f"; {KIT_IMAGE_TAG_ENV}={kit_tag} is ignored" if kit_tag else ""
            return (
                f"built on Modal from boileroom/models/<family>/kit/Dockerfile ({KIT_IMAGE_SOURCE_ENV}=build{ignored})"
            )
    if kit_tag:
        return f"{repository}/boileroom-<family>-kit:{kit_tag} ({KIT_IMAGE_TAG_ENV})"
    return f"{repository}/boileroom-<family>-kit@<pinned digest> ({pinned} pinned)"


def pytest_addoption(parser):
    """Register pytest CLI options used by the test suite.

    Adds the following command-line options:
    - --backend: selects the execution backend ("modal" or "apptainer[:<tag>]"); defaults to "modal".
    - --gpu: Modal-only GPU class (e.g. "T4", "A100-80GB").
    - --device: Apptainer-only CUDA device (e.g. "cuda:0", "cpu"). Defaults to cuda:0.
    - --docker-user: overrides the Docker Hub user or namespace for image lookup.
    - --image-tag: overrides the stock runtime image tag for both Modal and Apptainer tests (not the kit images).
    - --kit-image-tag: names the kit images (optimization="exact" / "fast") by tag instead of their pinned digests.
    - --run-kit: runs the opt-in GPU tests marked ``kit`` (also BOILEROOM_RUN_KIT=1); they are skipped otherwise.

    Parameters
    ----------
    parser : Any
        The pytest parser object used to register custom command-line options.
    """
    parser.addoption(
        "--backend",
        action="store",
        default="modal",
        help=(
            "Execution backend for models in tests: 'modal' (default) or "
            "'apptainer' (runs locally in containers). Apptainer accepts an "
            "optional image tag via 'apptainer:<tag>' (e.g. 'apptainer:sha-abc1234')."
        ),
    )
    parser.addoption(
        "--gpu",
        action="store",
        default=None,
        help="Modal-only: GPU class to reserve (e.g. T4, A100-40GB, A100-80GB). Ignored for apptainer.",
    )
    parser.addoption(
        "--device",
        action="store",
        default=None,
        help="Apptainer-only: CUDA device string (e.g. cuda:0, cuda:1, cpu). Ignored for modal. Defaults to cuda:0.",
    )
    parser.addoption(
        "--docker-user",
        action="store",
        default=None,
        help="Docker Hub user or namespace for image lookup (sets BOILEROOM_DOCKER_REPOSITORY).",
    )
    parser.addoption(
        "--image-tag",
        action="store",
        default=None,
        help="Stock runtime image tag for Modal and Apptainer image lookup (sets BOILEROOM_IMAGE_TAG; kit images ignore it).",
    )
    parser.addoption(
        "--kit-image-tag",
        action="store",
        default=None,
        help=(
            "Kit image tag for optimization='exact'/'fast' (sets BOILEROOM_KIT_IMAGE_TAG); without it the kit images "
            "are pulled by their pinned digests. On Modal it needs BOILEROOM_KIT_IMAGE_SOURCE=registry, since Modal "
            "builds the kit images from their Dockerfiles by default."
        ),
    )
    parser.addoption(
        "--run-kit",
        action="store_true",
        default=False,
        help=(
            f"Run the opt-in GPU tests of the optimization kit modes (marker 'kit'; same as {RUN_KIT_ENV}=1). They "
            "fold on paid GPUs: see each tests/*/test_*_kit_integration.py docstring for its cost."
        ),
    )


def pytest_configure(config: pytest.Config) -> None:
    """Validate CLI options and apply Docker repository overrides before test modules import wrappers."""
    backend = config.getoption("--backend")
    family, _, _ = backend.partition(":")
    family = family.strip()
    if family not in ("modal", "apptainer"):
        raise pytest.UsageError(f"--backend must be 'modal' or 'apptainer[:<tag>]'; got {backend!r}")
    if family == "modal" and ":" in backend:
        raise pytest.UsageError("--backend 'modal' does not accept a ':<tag>' suffix; use --image-tag instead.")

    gpu = config.getoption("--gpu")
    device = config.getoption("--device")
    if family == "modal":
        if device is not None:
            raise pytest.UsageError("--device is apptainer-only; for the modal backend pick a GPU class with --gpu.")
        if gpu is not None and gpu not in GPUS_AVAIL_ON_MODAL:
            raise pytest.UsageError(f"--gpu={gpu!r} is not a Modal GPU class; expected one of {GPUS_AVAIL_ON_MODAL}.")
    else:  # apptainer
        if gpu is not None:
            raise pytest.UsageError(
                "--gpu is modal-only; for the apptainer backend pick a CUDA device with --device (e.g. cuda:0)."
            )
        if device is not None and not _APPTAINER_DEVICE_RE.match(device):
            raise pytest.UsageError(
                f"--device={device!r} is not a valid Apptainer device; expected 'cpu' or 'cuda[:<N>]'."
            )

    if docker_user := config.getoption("--docker-user"):
        os.environ[DOCKER_REPOSITORY_ENV] = normalize_docker_repository(docker_user)
    if image_tag := config.getoption("--image-tag"):
        os.environ[IMAGE_TAG_ENV] = image_tag
    if family == "modal":
        try:
            kit_source = get_kit_image_source()
        except ValueError as exc:
            raise pytest.UsageError(str(exc)) from None
    if kit_image_tag := config.getoption("--kit-image-tag"):
        try:
            normalized_kit_tag = normalize_kit_image_tag(kit_image_tag)
        except ValueError as exc:
            raise pytest.UsageError(f"--kit-image-tag: {exc}") from None
        if normalized_kit_tag is not None and family == "modal" and kit_source == "build":
            # Modal builds the kit images from their Dockerfiles by default and never reads a kit tag.
            raise pytest.UsageError(
                f"--kit-image-tag has no effect on Modal while {KIT_IMAGE_SOURCE_ENV}=build (the default): Modal builds "
                f"the kit images from their Dockerfiles. Set {KIT_IMAGE_SOURCE_ENV}=registry to run a published tag."
            )
        if normalized_kit_tag is not None:
            os.environ[KIT_IMAGE_TAG_ENV] = normalized_kit_tag


def _run_kit_requested(config: pytest.Config) -> bool:
    """Return whether the ``kit`` tests were opted into, by ``--run-kit`` or :data:`RUN_KIT_ENV`.

    Raises
    ------
    pytest.UsageError
        If :data:`RUN_KIT_ENV` holds something other than ``1/true/yes`` or ``0/false/no`` (or nothing).
    """
    raw = os.environ.get(RUN_KIT_ENV, "").strip().lower()
    if raw not in _RUN_KIT_VALUES:
        raise pytest.UsageError(f"{RUN_KIT_ENV}={raw!r}: expected 1/true/yes to run the kit tests, or 0/false/no")
    return bool(config.getoption("--run-kit")) or _RUN_KIT_VALUES[raw]


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip the ``kit`` tests unless they were opted into (``--run-kit`` or ``BOILEROOM_RUN_KIT=1``).

    They fold on paid GPUs, so selecting them by marker or path alone (``-m integration``) is not enough.
    """
    if _run_kit_requested(config):
        return
    skip = pytest.mark.skip(
        reason=f"kit GPU test: opt in with --run-kit or {RUN_KIT_ENV}=1 (cost in the module docstring)"
    )
    for item in items:
        if item.get_closest_marker("kit") is not None:
            item.add_marker(skip)


@pytest.fixture(autouse=True, scope="session")
def model_dir():
    """
    Set the MODEL_DIR environment variable to the repository's .model_cache directory for the test session.

    Computes the path as two levels up from this file joined with ".model_cache" and assigns that string to the MODEL_DIR environment variable.
    """
    os.environ["MODEL_DIR"] = str(pathlib.Path(__file__).parent.parent / ".model_cache")


@pytest.fixture(scope="session")
def backend_option(request):
    """Provide the selected backend from the `--backend` command-line option.

    Parameters
    ----------
    request : Any
        Pytest request object.

    Returns
    -------
    str
        The backend string as provided via `--backend` (defaults to "modal"). ``--image-tag`` reaches the stock images
        through BOILEROOM_IMAGE_TAG, not through this string, so it never names a kit image.
    """
    return request.config.getoption("--backend")


@pytest.fixture(scope="session")
def gpu_device(request):
    """Modal GPU class from ``--gpu`` (e.g. ``T4``, ``A100-80GB``).

    Modal-only. For the apptainer backend, prefer ``device_option`` instead.

    Returns
    -------
    Optional[str]
        ``--gpu`` value, or ``None`` if not set.
    """
    return request.config.getoption("--gpu")


def _backend_family(backend: str) -> str:
    """Return the backend family ("modal" or "apptainer") from a backend string."""
    return backend.split(":", 1)[0].strip()


@pytest.fixture(scope="session")
def device_option(backend_option: str, request) -> str | None:
    """Resolve the device kwarg for model wrappers based on the active backend.

    For the modal backend this returns ``--gpu`` (a Modal GPU class such as
    ``T4`` or ``A100-80GB``, or ``None`` to let the wrapper apply its per-class
    default). For the apptainer backend this returns ``--device`` (a CUDA
    device string such as ``cuda:0`` or ``cpu``), defaulting to ``cuda:0``.

    Validity of each flag against the active backend is enforced by
    ``pytest_configure``.
    """
    if _backend_family(backend_option) == "apptainer":
        return request.config.getoption("--device") or "cuda:0"
    return request.config.getoption("--gpu")


@pytest.fixture(scope="session")
def output_ctx(backend_option: str):
    """Provide a factory that builds a fresh per-call context manager for the active backend.

    Usage::

        with output_ctx(), Model(backend=backend_option) as model:
            ...

    For the Modal backend, each call returns ``modal.enable_output()``. For Apptainer
    each call returns ``contextlib.nullcontext()``. This preserves the original
    "fresh context per use" semantics of pre-refactor tests while keeping the
    Modal import gated behind backend selection.
    """
    is_modal = _backend_family(backend_option) == "modal"

    def _make():
        if is_modal:
            from modal import enable_output

            return enable_output()
        return contextlib.nullcontext()

    return _make


@pytest.fixture
def test_sequences() -> dict[str, str]:
    """Provide a set of canonical protein sequence test cases used by tests.

    Returns
    -------
    dict[str, str]
        Mapping of test case names to sequence strings:
        - "short": a short valid peptide sequence.
        - "medium": a longer valid protein sequence.
        - "invalid": a sequence containing invalid characters.
        - "multimer": two identical valid sequences joined by ':' to represent a multimer.
    """
    return {
        "short": "MLKNVHVLVLGAGDVGSVVVRLLEK",
        "medium": "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKT",
        "invalid": "MALWMRLLPX123LLALWGPD",
        "multimer": (
            "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKT:"
            "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKT"
        ),
    }


@pytest.fixture
def data_dir() -> pathlib.Path:
    return pathlib.Path(__file__).parent / "data"


@pytest.fixture(params=[10, 25, 50])
def glycine_linker(request) -> str:
    return "G" * request.param
