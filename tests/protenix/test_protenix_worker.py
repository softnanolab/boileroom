"""Exercise persistent worker reuse and recovery with real lightweight processes."""

import os
import time
from pathlib import Path

import pytest


def _fake_worker(connection, config, env):
    """Stand in for GPU inference while keeping the real process/pipe lifecycle."""
    connection.send(None)
    while True:
        try:
            action, output, options = connection.recv()
        except EOFError:
            return
        if action == "hang":
            time.sleep(30)
        if action == "crash":
            os._exit(1)
        if action == "error":
            connection.send("checkpoint failure")
            return
        Path(output).write_text(str(os.getpid()))
        connection.send(None)


@pytest.fixture
def worker(monkeypatch):
    """Keep startup and deadline handling real while avoiding model dependencies."""
    from boileroom.models.protenix import worker as module

    monkeypatch.setattr(module, "_serve", _fake_worker)
    instance = module.ProtenixWorker({"timeout_seconds": 5}, {})
    yield instance
    instance.close()


def test_worker_reuses_process_across_requests(worker, tmp_path) -> None:
    """Two requests use the same resident model process; close releases it."""
    first, second = tmp_path / "first", tmp_path / "second"
    worker.predict("ok", str(first), {"timeout_seconds": 2})
    process = worker._process
    worker.predict("ok", str(second), {"timeout_seconds": 2})
    assert first.read_text() == second.read_text()
    assert process is worker._process and process.is_alive()
    worker.close()
    assert worker._process is None and worker._connection is None


@pytest.mark.parametrize("action, message", [("hang", "timed out"), ("crash", "exited"), ("error", "checkpoint failure")])
def test_worker_failure_releases_process_and_next_request_recovers(worker, tmp_path, action, message) -> None:
    """Timeouts, process death and upstream errors do not poison later calls."""
    with pytest.raises(RuntimeError, match=message):
        worker.predict(action, str(tmp_path / "failed"), {"timeout_seconds": 0.1})
    assert worker._process is None and worker._connection is None
    worker.predict("ok", str(tmp_path / "recovered"), {"timeout_seconds": 2})
    assert (tmp_path / "recovered").read_text()
