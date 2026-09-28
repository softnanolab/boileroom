"""Exercise the shared resident model process and failure recovery without a GPU."""

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest


@pytest.fixture
def worker(monkeypatch, tmp_path):
    """Run the production pipe loop with a lightweight runtime in a real child."""
    from boileroom.models import _worker as module

    entrypoint = tmp_path / "worker.py"
    entrypoint.write_text(Path(module.__file__).read_text())
    monkeypatch.setattr(module, "__file__", str(entrypoint))
    # A sibling package must not shadow a dependency of the loaded runtime.
    (tmp_path / "json").mkdir()
    (tmp_path / "json/__init__.py").write_text("raise RuntimeError('shadowed upstream dependency')\n")

    (tmp_path / "runtime.py").write_text(
        "import os, time, json\n"
        "from pathlib import Path\n"
        "class Runtime:\n"
        "    def __init__(self, config, work_dir):\n"
        "        with open(config['load_log'], 'a') as log: log.write(str(os.getpid()) + '\\n')\n"
        "        if config.get('fail_start'): raise RuntimeError('startup failed')\n"
        "    def predict(self, action, output, options):\n"
        "        print('ordinary upstream stdout', flush=True)\n"
        "        if action == 'hang': time.sleep(30)\n"
        "        if action == 'crash': os._exit(1)\n"
        "        if action == 'error': raise RuntimeError('inference failed')\n"
        "        Path(output).write_text(str(os.getpid()))\n"
    )
    config = {
        "timeout_seconds": 5,
        "load_log": str(tmp_path / "loads"),
    }
    instance = module.ModelWorker(
        config,
        dict(os.environ),
        runtime_path=tmp_path / "runtime.py",
        runtime_class="Runtime",
        label="Test",
        python_executable=sys.executable,
    )
    yield instance, config
    instance.close()


def test_concurrent_jobs_use_one_loaded_process(worker, tmp_path) -> None:
    """Queued requests must not race startup or mix their response messages."""
    instance, config = worker

    def predict(index):
        path = tmp_path / str(index)
        instance.predict("ok", str(path), config)
        return path.read_text()

    with ThreadPoolExecutor(max_workers=4) as pool:
        pids = list(pool.map(predict, range(4)))
    assert len(set(pids)) == 1
    assert Path(config["load_log"]).read_text().splitlines() == [pids[0]]
    process = instance._process
    instance.close()
    assert process.poll() is not None
    assert instance._process is None and instance._connection is None


@pytest.mark.parametrize("action,message", [("hang", "timed out"), ("crash", "exited"), ("error", "inference failed")])
def test_failed_request_releases_model_and_next_request_recovers(worker, tmp_path, action, message) -> None:
    """Time limits, crashes and model exceptions all leave a restartable worker."""
    instance, config = worker
    with pytest.raises(RuntimeError, match=message):
        instance.predict(action, str(tmp_path / "failed"), {**config, "timeout_seconds": 0.1})
    assert instance._process is None and instance._connection is None
    instance.predict("ok", str(tmp_path / "recovered"), config)
    assert len(Path(config["load_log"]).read_text().splitlines()) == 2


def test_failed_start_releases_process_and_pipe(worker) -> None:
    """A failure before ready must not retain a child or block later attempts."""
    instance, config = worker
    instance._config["fail_start"] = True
    with pytest.raises(RuntimeError, match="startup failed"):
        instance.start()
    assert instance._process is None and instance._connection is None
    instance._config["fail_start"] = False
    instance.start()
    assert len(Path(config["load_log"]).read_text().splitlines()) == 2
