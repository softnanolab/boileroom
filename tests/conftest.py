"""Pytest configuration for the boileroom package."""

import contextlib
import os
import pathlib
import re

import pytest

from boileroom.images.metadata import DOCKER_REPOSITORY_ENV, MODAL_IMAGE_TAG_ENV, normalize_docker_repository
from boileroom.utils import GPUS_AVAIL_ON_MODAL

_APPTAINER_DEVICE_RE = re.compile(r"^(cpu|cuda(:\d+)?)$")


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
        from boileroom.images.metadata import get_docker_repository, get_modal_image_tag, resolve_registry_tag
    except ImportError as exc:  # pragma: no cover - defensive; keep pytest running if the module is missing
        return [f"boileroom image: <unresolved: {exc!s}>"]

    image_tag = config.getoption("--image-tag")
    raw_backend = config.getoption("--backend")
    backend = _backend_with_image_tag(raw_backend, image_tag)
    family, _, backend_tag = backend.partition(":")
    raw_family, raw_sep, raw_backend_tag = raw_backend.partition(":")
    has_inline_backend_tag = raw_family.strip() == "apptainer" and bool(raw_sep) and bool(raw_backend_tag.strip())
    if family.strip() == "apptainer" and backend_tag.strip() and has_inline_backend_tag:
        tag = resolve_registry_tag(backend_tag)
        source = "from --backend tag"
    elif image_tag:
        tag = resolve_registry_tag(image_tag)
        source = "from --image-tag"
    else:
        tag = get_modal_image_tag()
        override = os.environ.get(MODAL_IMAGE_TAG_ENV)
        source = f"override via {MODAL_IMAGE_TAG_ENV}" if override else "from pyproject.toml"
    repository = get_docker_repository()
    return [f"boileroom image: {repository}/boileroom-<family>:{tag} ({source})"]


def pytest_addoption(parser):
    """Register pytest CLI options used by the test suite.

    Adds the following command-line options:
    - --backend: selects the execution backend ("modal" or "apptainer[:<tag>]"); defaults to "modal".
    - --gpu: Modal-only GPU class (e.g. "T4", "A100-80GB").
    - --device: Apptainer-only CUDA device (e.g. "cuda:0", "cpu"). Defaults to cuda:0.
    - --docker-user: overrides the Docker Hub user or namespace for image lookup.
    - --image-tag: overrides the runtime image tag for both Modal and Apptainer tests.

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
        help="Runtime image tag for Modal and Apptainer image lookup.",
    )


def _backend_with_image_tag(backend: str, image_tag: str | None) -> str:
    """Return the backend selector after applying the pytest image-tag option."""
    family, sep, tag = backend.partition(":")
    if family.strip() == "apptainer":
        inline_tag = tag.strip() if sep else ""
        if inline_tag:
            return f"apptainer:{inline_tag}"
        if image_tag:
            return f"apptainer:{image_tag}"
        return "apptainer"
    return backend


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
        os.environ[MODAL_IMAGE_TAG_ENV] = image_tag


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
        The backend string as provided via `--backend` (defaults to "modal").
    """
    return _backend_with_image_tag(request.config.getoption("--backend"), request.config.getoption("--image-tag"))


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
