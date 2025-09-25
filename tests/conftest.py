"""Pytest configuration for the boileroom package."""

import os
import pathlib
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--backend",
        action="store",
        default="modal",
        choices=("modal"),
        help="Execution backend for models in tests: modal (default)",
    )


@pytest.fixture(autouse=True, scope="session")
def model_dir():
    os.environ["MODEL_DIR"] = str(pathlib.Path(__file__).parent.parent / ".model_cache")


# @pytest.fixture
# def run_backend(request):
#     mode = request.config.getoption("--backend")

#     def select(method):
#         # method is e.g. model.fold or model.embed
#         return getattr(method, "local" if mode == "local" else "remote")

#     return select


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
