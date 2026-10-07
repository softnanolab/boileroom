"""Contract tests for the Apptainer model server's load and error handling, run without FastAPI, uvicorn or a model.

``server.py`` rewires imports and LD_LIBRARY_PATH when imported, so each test runs it in its own interpreter, with small
stand-ins for ``fastapi``, ``pydantic`` and ``uvicorn`` on the path and a fake core module.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from boileroom.optimization import REFUSAL_PROCESS_EXIT_CODE

REPO_ROOT = Path(__file__).resolve().parents[2]
SERVER_PATH = REPO_ROOT / "boileroom" / "backend" / "server.py"

_STUBS = {
    "fastapi/__init__.py": """
class HTTPException(Exception):
    def __init__(self, status_code, detail=None):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class FastAPI:
    def _route(self, *args, **kwargs):
        return lambda function: function

    get = post = on_event = _route
""",
    "fastapi/responses.py": """
class JSONResponse:
    def __init__(self, content=None, status_code=200):
        self.content = content
        self.status_code = status_code
""",
    "pydantic/__init__.py": """
class BaseModel:
    def __init__(self, **fields):
        for name, value in type(self).__dict__.get('__annotations__', {}).items():
            setattr(self, name, getattr(type(self), name, None))
        self.__dict__.update(fields)
""",
    "uvicorn/__init__.py": """
import os
import sys


def run(app, host, port):
    with open(os.environ['FAKE_UVICORN_LOG'], 'w') as handle:
        handle.write(f'{host}:{port}')
    if os.environ.get('FAKE_UVICORN_EXIT'):
        sys.exit(int(os.environ['FAKE_UVICORN_EXIT']))
""",
    "fake_cores.py": """
import sys


class OptimizationUnavailableError(RuntimeError):
    '''A kit runtime's own refusal class: it cannot import boileroom.'''


class _Core:
    def __init__(self, config):
        self.config = config

    def _initialize(self):
        pass

    def fold(self, sequences, options=None):
        raise OptimizationUnavailableError('fast needs an sm90 GPU')

    def embed(self, sequences, options=None):
        raise MemoryError('CUDA out of memory')


class LoadsCore(_Core):
    pass


class KitExitsCore(_Core):
    def _initialize(self):
        sys.exit(3)


class RuntimeRefusesCore(_Core):
    def _initialize(self):
        raise OptimizationUnavailableError('no kit in this image')


class BrokenCore(_Core):
    def _initialize(self):
        raise OSError('weights missing')


class ExitsOneCore(_Core):
    def _initialize(self):
        sys.exit(1)
""",
}


@pytest.fixture
def stubs(tmp_path: Path) -> Path:
    root = tmp_path / "stubs"
    for relative, source in _STUBS.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    return root


def _env(stubs: Path, tmp_path: Path, core: str, **extra: str) -> dict[str, str]:
    return {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(stubs), str(REPO_ROOT)]),
        "MODEL_CLASS": f"fake_cores.{core}",
        "MODEL_CONFIG": json.dumps({"device": "cpu"}),
        "DEVICE": "cpu",
        "FAKE_UVICORN_LOG": str(tmp_path / "uvicorn.log"),
        **extra,
    }


def _serve(stubs: Path, tmp_path: Path, core: str, **extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SERVER_PATH), "--port", "8765"],
        env=_env(stubs, tmp_path, core, **extra),
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_a_loaded_model_is_served(stubs: Path, tmp_path: Path) -> None:
    result = _serve(stubs, tmp_path, "LoadsCore")
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "uvicorn.log").read_text() == "127.0.0.1:8765"


@pytest.mark.parametrize("core", ["KitExitsCore", "RuntimeRefusesCore"])
def test_a_refused_load_exits_with_the_refusal_code_before_serving(stubs: Path, tmp_path: Path, core: str) -> None:
    """The load ran in uvicorn's startup event, whose failures all exit 3, so a refusal was indistinguishable."""
    result = _serve(stubs, tmp_path, core)
    assert result.returncode == REFUSAL_PROCESS_EXIT_CODE, result.stderr
    assert "Model load refused" in result.stderr
    assert not (tmp_path / "uvicorn.log").exists()


@pytest.mark.parametrize("core", ["BrokenCore", "ExitsOneCore", "NoSuchCore"])
def test_any_other_load_failure_exits_one(stubs: Path, tmp_path: Path, core: str) -> None:
    result = _serve(stubs, tmp_path, core)
    assert result.returncode == 1, result.stderr
    assert "Model load failed" in result.stderr
    assert not (tmp_path / "uvicorn.log").exists()


def test_uvicorn_startup_failure_is_not_reported_as_a_refusal(stubs: Path, tmp_path: Path) -> None:
    """uvicorn exits with its STARTUP_FAILURE code 3 when it cannot start; that must not read as a refusal."""
    result = _serve(stubs, tmp_path, "LoadsCore", FAKE_UVICORN_EXIT=str(REFUSAL_PROCESS_EXIT_CODE))
    assert result.returncode == 1, result.stderr


def test_uvicorn_exit_codes_other_than_three_pass_through(stubs: Path, tmp_path: Path) -> None:
    result = _serve(stubs, tmp_path, "LoadsCore", FAKE_UVICORN_EXIT="2")
    assert result.returncode == 2, result.stderr


_CALL_SCRIPT = """
import asyncio
import json
import sys

sys.argv = ['server.py']
sys.path.insert(0, {server_dir!r})
import server

server._load_model()
for name, request in (('fold', server.FoldRequest(sequences='MKV')), ('embed', server.EmbedRequest(sequences='MKV'))):
    response = asyncio.run(getattr(server, name)(request))
    print(json.dumps({{'call': name, 'status': response.status_code, 'content': response.content}}))
"""


def test_a_failed_call_names_the_exception_type(stubs: Path, tmp_path: Path) -> None:
    """The client re-raises a refusal from a 500 only if the response names its type."""
    script = _CALL_SCRIPT.format(server_dir=str(SERVER_PATH.parent))
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=_env(stubs, tmp_path, "LoadsCore"),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    fold, embed = (json.loads(line) for line in result.stdout.splitlines())
    assert fold == {
        "call": "fold",
        "status": 500,
        "content": {"detail": "Folding failed: fast needs an sm90 GPU", "error_type": "OptimizationUnavailableError"},
    }
    assert embed["status"] == 500
    assert embed["content"]["error_type"] == "MemoryError"
    assert embed["content"]["detail"] == "Embedding failed: CUDA out of memory"
