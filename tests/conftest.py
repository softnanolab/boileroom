"""Pytest configuration for the boileroom package."""

import os
import pathlib
import pytest


def pytest_addoption(parser):
    """
    Register pytest CLI options used by the test suite.
    
    Adds two command-line options:
    - --backend: selects the execution backend for tests; allowed value is "modal" and defaults to "modal".
    - --gpu: specifies the GPU type for Modal backend tests (examples: "A100-40GB", "A100-80GB", "T4"); defaults to None which uses the test-suite default.
    
    Parameters:
        parser: The pytest parser object used to register custom command-line options.
    """
    parser.addoption(
        "--backend",
        action="store",
        default="modal",
        choices=("modal",),
        help="Execution backend for models in tests: modal (default)",
    )
    parser.addoption(
        "--gpu",
        action="store",
        default=None,
        help="GPU type for Modal backend tests (e.g., A100-40GB, A100-80GB, T4). Defaults to None (uses default T4).",
    )


@pytest.fixture(autouse=True, scope="session")
def model_dir():
    """
    Set the MODEL_DIR environment variable to the repository's .model_cache directory for the test session.
    
    Computes the path as two levels up from this file joined with ".model_cache" and assigns that string to the MODEL_DIR environment variable.
    """
    os.environ["MODEL_DIR"] = str(pathlib.Path(__file__).parent.parent / ".model_cache")


@pytest.fixture(scope="session")
def gpu_device(request):
    """
    Provide the selected GPU device type from the `--gpu` command-line option.
    
    Returns:
        The GPU device string as provided via `--gpu`, or `None` if the option was not set.
    """
    return request.config.getoption("--gpu")


@pytest.fixture
def test_sequences() -> dict[str, str]:
    """
    Provide a set of canonical protein sequence test cases used by tests.
    
    Returns:
        dict[str, str]: Mapping of test case names to sequence strings:
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