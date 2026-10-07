"""Shared resident model worker with isolated dependencies and hard deadlines.

Executed directly by the chosen interpreter, this module stays Python 3.10
compatible and never imports the parent boileroom package in the child.
A runtime implements ``__init__(config, work_dir)`` and ``predict(input, output, options)``, and optionally
``describe()``, which returns a flat dict describing the loaded stack (it becomes :attr:`ModelWorker.info`).
``work_dir`` belongs to the parent, which removes it only after the child exited, so a kit's interpreter-exit hooks
(Protenix's lever report) still find it.

Child-to-parent messages are ``("ok", payload)`` and ``("error", kind, text)``. ``kind`` is ``"refused"`` for a
kit's refusal ``SystemExit`` (code 3 or 5) or an exception whose class is named ``OptimizationUnavailableError`` (a
runtime cannot import boileroom, so it defines a local class of that name), and ``"failed"`` for anything else (other
exit codes included).
"""

from __future__ import annotations

import os
import sys

if __name__ == "__main__":
    # Python put this script's directory (boileroom/models) first on sys.path. Its sibling packages (alphafold, esm,
    # protenix, opendde, ...) would shadow the upstream packages the runtime imports, so drop it before importing more.
    _script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path[:] = [entry for entry in sys.path if os.path.realpath(entry or os.curdir) != _script_dir]

import atexit  # noqa: E402
import contextlib  # noqa: E402
import multiprocessing  # noqa: E402
import runpy  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import threading  # noqa: E402
import traceback  # noqa: E402
from multiprocessing.connection import Connection  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

# Variables that change where the child interpreter finds modules. They are dropped from the child's environment so
# the parent's paths (e.g. the Apptainer service's PYTHONPATH) never reach a runtime with its own interpreter; every
# other PYTHON* setting (PYTHONHASHSEED, PYTHONUNBUFFERED, ...) is kept.
_PATH_ENV_VARS = frozenset(
    {
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONSTARTUP",
        "PYTHONUSERBASE",
        "PYTHONSAFEPATH",
        "PYTHONINSPECT",
        "PYTHONEXECUTABLE",
        "PYTHONPLATLIBDIR",
    }
)
# Exit status of a child that refused: a kit that refuses at interpreter start ends it with os._exit(3). A literal
# copy of boileroom.optimization.REFUSAL_PROCESS_EXIT_CODE (the process contract), held equal by a contract test.
_REFUSAL_PROCESS_EXIT_CODE = 3
# SystemExit codes a kit raises in-process for a refusal: 3 (not active) and 5 (OpenDDE's KernelsRefused). A literal
# copy of boileroom.optimization.KIT_REFUSAL_EXIT_CODES, held equal by a contract test.
_KIT_REFUSAL_EXIT_CODES = frozenset({3, 5})
# What _recv returns once the parent closed its end of the pipe.
_CLOSED = object()


def _is_refusal(error: BaseException) -> bool:
    """Whether ``error`` is a typed refusal (a kit's refusal ``SystemExit`` or a refusal by class name), not a failure.

    The child's copy of :func:`boileroom.optimization.is_refusal`, which it cannot import.
    """
    if isinstance(error, SystemExit):
        return isinstance(error.code, int) and error.code in _KIT_REFUSAL_EXIT_CODES
    return type(error).__name__ == "OptimizationUnavailableError"


def _refusal_text(error: BaseException) -> str:
    """Describe a refusal in one line."""
    if isinstance(error, SystemExit):
        return f"the runtime exited with code {error.code}"
    return str(error) or type(error).__name__


def _recv(connection: Connection) -> Any:
    """Receive the next message from the parent, or :data:`_CLOSED` once it closed the pipe."""
    try:
        return connection.recv()
    except EOFError:
        return _CLOSED


def _serve(connection: Connection, runtime_path: str, runtime_class: str, work_dir: str) -> None:
    """Own the loaded runtime until its parent closes the pipe or a request fails."""
    try:
        namespace = runpy.run_path(runtime_path)
        config = _recv(connection)
        if config is _CLOSED:
            return
        runtime = namespace[runtime_class](config, work_dir)
        describe = getattr(runtime, "describe", None)
        connection.send(("ok", dict(describe()) if callable(describe) else {}))
        while True:
            request = _recv(connection)
            if request is None or request is _CLOSED:
                break
            connection.send(("ok", runtime.predict(*request)))
    except KeyboardInterrupt:
        raise
    except BaseException as error:
        # Only the pipe's EOF (in _recv) is a quiet shutdown; an EOFError of the runtime (a truncated checkpoint) is a
        # failure with its traceback like any other.
        kind = "refused" if _is_refusal(error) else "failed"
        text = _refusal_text(error) if kind == "refused" else traceback.format_exc()
        with contextlib.suppress(EOFError, OSError):
            connection.send(("error", kind, text))
    finally:
        connection.close()


def _child_env(env: dict[str, str]) -> dict[str, str]:
    """Return ``env`` without the variables that change the child interpreter's module search path."""
    return {key: value for key, value in env.items() if key not in _PATH_ENV_VARS}


class ModelWorker:
    """Serialize requests to one model, retaining its weights and runtime caches.

    Attributes
    ----------
    info : dict[str, str]
        What the loaded runtime's ``describe()`` reported (empty before a load and after :meth:`close`).
    """

    def __init__(
        self,
        config: dict[str, Any],
        env: dict[str, str],
        *,
        runtime_path: Path,
        runtime_class: str,
        label: str,
        python_executable: str = sys.executable,
    ) -> None:
        """Describe a runtime without importing its model-specific dependencies."""
        self._runtime_path = str(runtime_path.resolve())
        self._runtime_class = runtime_class
        self._label = label
        self._python = python_executable
        self._config = config.copy()
        self._env = _child_env(env)
        self._process: subprocess.Popen | None = None
        self._connection: Connection | None = None
        self._work_dir: str | None = None
        self._lock = threading.RLock()
        self.info: dict[str, str] = {}
        atexit.register(self.close)

    def start(self) -> dict[str, str]:
        """Load weights once and return the runtime's ``describe()`` record; failed startup leaves it restartable.

        Raises
        ------
        OptimizationUnavailableError
            If the runtime refused to load (a typed refusal, ``SystemExit(3)``/``SystemExit(5)``, or exit code 3).
        RuntimeError
            If the runtime failed to load, timed out or died.
        """
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                return self.info
            self.close()
            self._work_dir = tempfile.mkdtemp(prefix="boileroom-worker-")
            parent, child = multiprocessing.Pipe()
            self._connection = parent
            try:
                self._process = subprocess.Popen(
                    [
                        self._python,
                        # No user site-packages. Not -I: that would also drop PYTHONHASHSEED and the other PYTHON*
                        # settings; the path-affecting ones are removed from the environment instead.
                        "-s",
                        str(Path(__file__).resolve()),
                        str(child.fileno()),
                        self._runtime_path,
                        self._runtime_class,
                        self._work_dir,
                    ],
                    pass_fds=(child.fileno(),),
                    env=self._env,
                )
                child.close()
                self._send(self._config)
                payload = self._receive(self._config["timeout_seconds"])
                self.info = {str(key): str(value) for key, value in (payload or {}).items()}
            except BaseException:
                child.close()
                self.close()
                raise
            return self.info

    def predict(self, input_path: str, output_dir: str, config: dict[str, Any]) -> Any:
        """Reuse the loaded model and return what the runtime's ``predict`` returned.

        The worker is discarded on timeout or failure, so the next call starts a fresh one.
        """
        with self._lock:
            self.start()
            assert self._connection is not None
            try:
                self._send((input_path, output_dir, config))
                return self._receive(config["timeout_seconds"])
            except BaseException:
                self.close()
                raise

    def _send(self, message: Any) -> None:
        """Send one message; a worker that already died (e.g. its kit refused at interpreter start) maps by exit code."""
        assert self._connection is not None
        try:
            self._connection.send(message)
        except (EOFError, OSError) as exc:
            raise self._exit_error() from exc

    def _receive(self, timeout: float | None) -> Any:
        """Receive one reply without depending on upstream stdout or stderr, and return its payload."""
        assert self._connection is not None
        try:
            if not self._connection.poll(timeout):
                raise RuntimeError(f"{self._label} worker timed out after {timeout} seconds")
            message = self._connection.recv()
        except (EOFError, OSError) as exc:
            raise self._exit_error() from exc
        if message[0] == "ok":
            return message[1]
        _, kind, text = message
        if kind == "refused":
            from boileroom.optimization import OptimizationUnavailableError

            raise OptimizationUnavailableError(f"{self._label} refused: {text}")
        raise RuntimeError(f"{self._label} worker failed:\n{text}")

    def _exit_error(self) -> Exception:
        """Map a worker that closed its pipe without replying to a refusal (exit code 3) or a failure.

        Only the process contract's code 3 is a refusal here: the child reports a kit's in-process refusal
        ``SystemExit`` (3 or 5) over the pipe, so no other exit status of the child means one.
        """
        code: int | None = None
        if self._process is not None:
            with contextlib.suppress(subprocess.TimeoutExpired):
                code = self._process.wait(timeout=5)
        if code == _REFUSAL_PROCESS_EXIT_CODE:
            from boileroom.optimization import OptimizationUnavailableError

            return OptimizationUnavailableError(
                f"{self._label} refused: the worker exited with code {code}, the optimization kit's refusal exit"
            )
        return RuntimeError(f"{self._label} worker exited before completing the request (exit code {code})")

    def close(self) -> None:
        """Release the model process, terminating it if it cannot exit cleanly, then remove its work directory."""
        with self._lock:
            self.info = {}
            if self._connection is not None:
                with contextlib.suppress(EOFError, OSError):
                    self._connection.send(None)
                self._connection.close()
                self._connection = None
            if self._process is not None:
                try:
                    self._process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    self._process.terminate()
                    try:
                        self._process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self._process.kill()
                        self._process.wait()
                self._process = None
            if self._work_dir is not None:
                shutil.rmtree(self._work_dir, ignore_errors=True)
                self._work_dir = None


if __name__ == "__main__":
    # The parent removes the work dir after the child exits; this covers a parent that died without closing. Registered
    # first, so it runs after the runtime's own exit hooks (the kit's lever report writes into the dir at exit).
    atexit.register(shutil.rmtree, sys.argv[4], True)
    _serve(Connection(int(sys.argv[1])), sys.argv[2], sys.argv[3], sys.argv[4])
