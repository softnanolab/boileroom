"""Shared resident model worker with isolated dependencies and hard deadlines.

Executed directly by the chosen interpreter, this module stays Python 3.10
compatible and never imports the parent boileroom package in the child.
A runtime implements __init__(config, work_dir) and predict(input, output, options).
"""

from __future__ import annotations

import atexit
import contextlib
import multiprocessing
import runpy
import subprocess
import sys
import threading
import traceback
from multiprocessing.connection import Connection
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any


def _serve(connection: Connection, runtime_path: str, runtime_class: str) -> None:
    """Own the loaded runtime until its parent closes the pipe or a request fails."""
    try:
        namespace = runpy.run_path(runtime_path)
        with TemporaryDirectory(prefix="boileroom-worker-") as work_dir:
            runtime = namespace[runtime_class](connection.recv(), work_dir)
            connection.send(None)
            while True:
                request = connection.recv()
                if request is None:
                    break
                runtime.predict(*request)
                connection.send(None)
    except EOFError:
        pass
    except Exception:
        with contextlib.suppress(EOFError, OSError):
            connection.send(traceback.format_exc())
    finally:
        connection.close()


class ModelWorker:
    """Serialize requests to one model, retaining its weights and runtime caches."""

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
        self._env = env.copy()
        self._process: subprocess.Popen | None = None
        self._connection: Connection | None = None
        self._lock = threading.RLock()
        atexit.register(self.close)

    def start(self) -> None:
        """Load weights once; failed startup leaves the worker restartable."""
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                return
            self.close()
            parent, child = multiprocessing.Pipe()
            self._connection = parent
            try:
                self._process = subprocess.Popen(
                    [
                        self._python,
                        # Do not let our sibling alphafold/protenix packages
                        # shadow the upstream packages in the child interpreter.
                        "-I",
                        str(Path(__file__).resolve()),
                        str(child.fileno()),
                        self._runtime_path,
                        self._runtime_class,
                    ],
                    pass_fds=(child.fileno(),),
                    env=self._env,
                )
                child.close()
                parent.send(self._config)
                self._receive(self._config["timeout_seconds"])
            except BaseException:
                child.close()
                self.close()
                raise

    def predict(self, input_path: str, output_dir: str, config: dict[str, Any]) -> None:
        """Reuse the loaded model and discard the worker on timeout or failure."""
        with self._lock:
            self.start()
            assert self._connection is not None
            try:
                self._connection.send((input_path, output_dir, config))
                self._receive(config["timeout_seconds"])
            except BaseException:
                self.close()
                raise

    def _receive(self, timeout: float | None) -> None:
        """Receive completion without depending on upstream stdout or stderr."""
        assert self._connection is not None
        try:
            if not self._connection.poll(timeout):
                raise RuntimeError(f"{self._label} worker timed out after {timeout} seconds")
            error = self._connection.recv()
        except (EOFError, OSError) as exc:
            raise RuntimeError(f"{self._label} worker exited before completing the request") from exc
        if error is not None:
            raise RuntimeError(f"{self._label} worker failed:\n{error}")

    def close(self) -> None:
        """Release the model process, terminating it if it cannot exit cleanly."""
        with self._lock:
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


if __name__ == "__main__":
    _serve(Connection(int(sys.argv[1])), sys.argv[2], sys.argv[3])
