"""Pytest configuration for the boileroom package."""

import os
import pathlib

import pytest

from boileroom.images.metadata import DOCKER_REPOSITORY_ENV, MODAL_IMAGE_TAG_ENV, normalize_docker_repository


def pytest_report_header(config: pytest.Config) -> list[str]:
    """Report the runtime image tag and Docker repository at the top of every pytest run.

    The Modal backend pulls the boileroom package from a prebuilt Docker image whose
    tag is resolved by ``boileroom.images.metadata.get_modal_image_tag``. Surfacing
    that tag in the pytest header avoids ambiguity about which image the integration
    tests are actually exercising.

    Parameters
    ----------
    config : pytest.Config
        The active pytest configuration (unused, required by the hook signature).

    Returns
    -------
    list[str]
        One-line-per-entry header additions.
    """
    try:
        from boileroom.images.metadata import MODAL_IMAGE_TAG_ENV, get_docker_repository, get_modal_image_tag
    except ImportError as exc:  # pragma: no cover - defensive; keep pytest running if the module is missing
        return [f"boileroom image: <unresolved: {exc!s}>"]

    tag = get_modal_image_tag()
    repository = get_docker_repository()
    override = os.environ.get(MODAL_IMAGE_TAG_ENV)
    source = f"override via {MODAL_IMAGE_TAG_ENV}" if override else "from pyproject.toml"
    return [f"boileroom image: {repository}/boileroom-<family>:{tag} ({source})"]


def pytest_addoption(parser):
    """Register pytest CLI options used by the test suite.

    Adds two command-line options:
    - --backend: selects the execution backend for tests; allowed values are "modal" and "apptainer", defaults to "modal".
    - --gpu: specifies the GPU type for Modal backend tests (examples: "A100-40GB", "A100-80GB", "T4"); defaults to None which uses the test-suite default.
    - --docker-user: overrides the default Docker Hub user or namespace for Modal image lookup; defaults to None, which uses the package default.
    - --image-tag: selects the runtime image tag for Modal image lookup; defaults to None, which uses the current package version.

    Parameters
    ----------
    parser : Any
        The pytest parser object used to register custom command-line options.
    """
    parser.addoption(
        "--backend",
        action="store",
        default="modal",
        choices=("modal", "apptainer"),
        help="Execution backend for models in tests: modal (default) or apptainer (runs locally in containers)",
    )
    parser.addoption(
        "--gpu",
        action="store",
        default=None,
        help="GPU type for Modal backend tests (e.g., A100-40GB, A100-80GB, T4). Defaults to None (uses default T4).",
    )
    parser.addoption(
        "--docker-user",
        action="store",
        default=None,
        help="Docker Hub user or namespace for Modal image lookup.",
    )
    parser.addoption(
        "--image-tag",
        action="store",
        default=None,
        help="Runtime image tag for Modal image lookup.",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Apply image lookup options before test modules import Modal wrappers."""
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
    return request.config.getoption("--backend")


@pytest.fixture(scope="session")
def gpu_device(request):
    """Provide the selected GPU device type from the `--gpu` command-line option.

    Parameters
    ----------
    request : Any
        Pytest request object.

    Returns
    -------
    Optional[str]
        The GPU device string as provided via `--gpu`, or `None` if the option was not set.
    """
    return request.config.getoption("--gpu")


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
