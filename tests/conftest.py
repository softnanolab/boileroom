"""Pytest configuration for the boileroom package."""

import os
import pathlib
import pytest


def pytest_addoption(parser):
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
    os.environ["MODEL_DIR"] = str(pathlib.Path(__file__).parent.parent / ".model_cache")


@pytest.fixture(scope="session")
def gpu_device(request):
    """Fixture that provides the GPU device type from the --gpu command-line option."""
    return request.config.getoption("--gpu")


@pytest.fixture
def test_sequences() -> dict[str, str]:
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
