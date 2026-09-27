"""A persistent, timeout-bounded Protenix process with isolated CUDA state."""

from __future__ import annotations

import atexit
import contextlib
import multiprocessing
import os
import threading
import traceback
from multiprocessing.connection import Connection
from tempfile import TemporaryDirectory
from typing import Any


def _serve(connection: Connection, config: dict[str, Any], env: dict[str, str]) -> None:
    """Own one loaded model until the parent closes the pipe or stops the worker."""
    os.environ.update(env)
    try:
        from .runtime import ProtenixRuntime

        with TemporaryDirectory(prefix="boileroom-protenix-") as work_dir:
            runtime = ProtenixRuntime(config, work_dir)
            connection.send(None)
            while True:
                request = connection.recv()
                if request is None:
                    break
                input_json, output_dir, options = request
                runtime.predict(input_json, output_dir, options)
                connection.send(None)
    except EOFError:
        pass
    except Exception:
        connection.send(traceback.format_exc())
    finally:
        connection.close()


class ProtenixWorker:
    """Serialize requests to a reusable model; discard it on timeout or failure."""

    def __init__(self, config: dict[str, Any], env: dict[str, str]) -> None:
        """Store startup configuration without importing the model dependency."""
        self._config = config.copy()
        self._env = env.copy()
        self._context = multiprocessing.get_context("spawn")
        self._process: Any = None
        self._connection: Connection | None = None
        self._lock = threading.RLock()
        atexit.register(self.close)

    def start(self) -> None:
        """Load the model once; a failed start leaves the worker restartable."""
        with self._lock:
            if self._process is not None and self._process.is_alive():
                return
            self.close()
            parent, child = self._context.Pipe()
            self._connection = parent
            self._process = self._context.Process(target=_serve, args=(child, self._config, self._env), daemon=True)
            try:
                self._process.start()
                child.close()
                self._receive(self._config["timeout_seconds"])
            except BaseException:
                child.close()
                self.close()
                raise

    def predict(self, input_json: str, output_dir: str, config: dict[str, Any]) -> None:
        """Reuse the loaded runner and enforce the request's timeout."""
        with self._lock:
            self.start()
            assert self._connection is not None
            try:
                self._connection.send((input_json, output_dir, config))
                self._receive(config["timeout_seconds"])
            except BaseException:
                self.close()
                raise

    def _receive(self, timeout: float | None) -> None:
        """Wait for completion and surface upstream failures or worker death."""
        assert self._connection is not None
        try:
            if not self._connection.poll(timeout):
                raise RuntimeError(f"Protenix worker timed out after {timeout} seconds")
            error = self._connection.recv()
        except (EOFError, OSError) as exc:
            raise RuntimeError("Protenix worker exited before completing the request") from exc
        if error is not None:
            raise RuntimeError(f"Protenix worker failed:\n{error}")

    def close(self) -> None:
        """Stop the worker and release its model and IPC resources."""
        with self._lock:
            if self._connection is not None:
                with contextlib.suppress(EOFError, OSError):
                    self._connection.send(None)
                self._connection.close()
                self._connection = None
            if self._process is not None:
                if self._process.pid is not None:
                    self._process.join(timeout=1)
                if self._process.is_alive():
                    self._process.terminate()
                    self._process.join(timeout=5)
                    if self._process.is_alive():
                        self._process.kill()
                        self._process.join()
                self._process.close()
                self._process = None
