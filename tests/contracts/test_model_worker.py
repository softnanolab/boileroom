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
        "import os, sys, time, json\n"
        "from pathlib import Path\n"
        "class OptimizationUnavailableError(RuntimeError):\n"
        "    pass\n"
        "class KernelsRefused(SystemExit):\n"
        "    def __init__(self, problems): super().__init__(5); self.problems = problems\n"
        "def _exit_report(work_dir, note):\n"
        "    with open(os.path.join(work_dir, 'report.jsonl'), 'a') as report: report.write('{}\\n')\n"
        "    Path(note).write_text('report written')\n"
        "class Runtime:\n"
        "    def __init__(self, config, work_dir):\n"
        "        with open(config['load_log'], 'a') as log: log.write(str(os.getpid()) + '\\n')\n"
        "        if config.get('exit_note'): __import__('atexit').register(_exit_report, work_dir, config['exit_note'])\n"
        "        if config.get('fail_start'): raise RuntimeError('startup failed')\n"
        "        if config.get('refuse_start') == 'raise': raise OptimizationUnavailableError('no kit on this card')\n"
        "        if config.get('refuse_start') == 'exit': sys.exit(3)\n"
        "        if config.get('refuse_start') == 'os_exit': os._exit(3)\n"
        "        if config.get('eof_start'): raise EOFError('Ran out of input (truncated checkpoint)')\n"
        "    def describe(self):\n"
        "        return {'stack': 'test', 'levers': 7}\n"
        "    def predict(self, action, output, options):\n"
        "        print('ordinary upstream stdout', flush=True)\n"
        "        if action == 'hang': time.sleep(30)\n"
        "        if action == 'crash': os._exit(1)\n"
        "        if action == 'error': raise RuntimeError('inference failed')\n"
        "        if action == 'value_error': raise ValueError('bad input')\n"
        "        if action == 'refuse': raise OptimizationUnavailableError('lever t10 needs more shared memory')\n"
        "        if action == 'exit3': raise SystemExit(3)\n"
        "        if action == 'os_exit3': os._exit(3)\n"
        "        if action == 'kernels_refused': raise KernelsRefused(['trimul: reference path served'])\n"
        "        if action == 'os_exit5': os._exit(5)\n"
        "        if action == 'exit1': sys.exit(1)\n"
        "        if action == 'eof': raise EOFError('Ran out of input')\n"
        "        Path(output).write_text(str(os.getpid()))\n"
        "        return {'pid': os.getpid(), 'output': output}\n"
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
    assert instance._process is None and instance._connection is None and instance._work_dir is None
    instance._config["fail_start"] = False
    instance.start()
    assert len(Path(config["load_log"]).read_text().splitlines()) == 2


def test_work_dir_outlives_the_runtime_exit_hooks_and_is_removed_on_close(worker, tmp_path) -> None:
    """A kit's interpreter-exit hook (Protenix's lever report) still writes in ``work_dir``; close() then removes it."""
    instance, _ = worker
    instance._config["exit_note"] = str(tmp_path / "exit_note")
    instance.start()
    assert instance._work_dir is not None
    work_dir = Path(instance._work_dir)
    assert work_dir.is_dir()
    instance.close()
    assert (tmp_path / "exit_note").read_text() == "report written"
    assert instance._work_dir is None and not work_dir.exists()


def test_work_dir_is_removed_by_the_child_when_the_parent_dies_without_closing(worker, tmp_path) -> None:
    """A parent killed before close() (SIGKILL, OOM) must not leave the work dir behind, nor cut the exit hooks short."""
    instance, _ = worker
    instance._config["exit_note"] = str(tmp_path / "exit_note")
    instance.start()
    assert instance._work_dir is not None
    work_dir = Path(instance._work_dir)
    # What the child sees when its parent dies: EOF on the pipe, and no close().
    instance._connection.close()
    instance._process.wait(timeout=10)
    # The note is written only after the hook's write into work_dir succeeded.
    assert (tmp_path / "exit_note").read_text() == "report written"
    assert not work_dir.exists()


def test_start_returns_and_keeps_the_runtime_description(worker) -> None:
    instance, _ = worker
    assert instance.info == {}
    assert instance.start() == {"stack": "test", "levers": "7"}
    assert instance.info == {"stack": "test", "levers": "7"}
    instance.close()
    assert instance.info == {}


def test_predict_returns_the_runtime_payload(worker, tmp_path) -> None:
    instance, config = worker
    payload = instance.predict("ok", str(tmp_path / "out"), config)
    assert payload == {"pid": instance._process.pid, "output": str(tmp_path / "out")}


@pytest.mark.parametrize(
    ("action", "message"),
    [
        ("refuse", "Test refused: lever t10 needs more shared memory"),
        ("exit3", "Test refused: the runtime exited with code 3"),
        ("os_exit3", "Test refused: the worker exited with code 3"),
        ("kernels_refused", "Test refused: the runtime exited with code 5"),
    ],
)
def test_runtime_refusal_is_a_typed_optimization_error(worker, tmp_path, action, message) -> None:
    """A kit refusal (typed error, SystemExit or os._exit(3)) must not look like a transient failure to retry."""
    from boileroom.optimization import OptimizationUnavailableError

    instance, config = worker
    with pytest.raises(OptimizationUnavailableError, match=message):
        instance.predict(action, str(tmp_path / "refused"), config)
    assert instance._process is None and instance._connection is None


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("raise", "Test refused: no kit on this card"),
        ("exit", "Test refused: the runtime exited with code 3"),
        ("os_exit", "Test refused: the worker exited with code 3"),
    ],
)
def test_refused_start_is_a_typed_optimization_error(worker, mode, message) -> None:
    from boileroom.optimization import OptimizationUnavailableError

    instance, _ = worker
    instance._config["refuse_start"] = mode
    with pytest.raises(OptimizationUnavailableError, match=message):
        instance.start()
    assert instance._process is None and instance._connection is None and instance.info == {}


def test_runtime_failure_is_a_runtime_error_with_the_traceback(worker, tmp_path) -> None:
    from boileroom.optimization import OptimizationUnavailableError

    instance, config = worker
    with pytest.raises(RuntimeError, match=r"(?s)Test worker failed:\n.*ValueError: bad input") as raised:
        instance.predict("value_error", str(tmp_path / "failed"), config)
    assert not isinstance(raised.value, OptimizationUnavailableError)


@pytest.mark.parametrize(("action", "code"), [("crash", 1), ("os_exit5", 5)])
def test_other_exit_codes_are_failures_naming_the_code(worker, tmp_path, action, code) -> None:
    """Only exit status 3 is a refusal for a dead child: a kit's in-process ``SystemExit(5)`` travels over the pipe."""
    from boileroom.optimization import OptimizationUnavailableError

    instance, config = worker
    with pytest.raises(RuntimeError, match=rf"exit code {code}\)") as raised:
        instance.predict(action, str(tmp_path / "failed"), config)
    assert not isinstance(raised.value, OptimizationUnavailableError)


@pytest.mark.parametrize(("action", "message"), [("exit1", r"SystemExit: 1"), ("eof", r"EOFError: Ran out of input")])
def test_runtime_exits_and_eof_errors_are_failures_with_the_traceback(worker, tmp_path, action, message) -> None:
    """Only the kits' SystemExit(3) is a refusal, and only the pipe's EOF is a quiet shutdown."""
    from boileroom.optimization import OptimizationUnavailableError

    instance, config = worker
    with pytest.raises(RuntimeError, match=rf"(?s)Test worker failed:\n.*{message}") as raised:
        instance.predict(action, str(tmp_path / "failed"), config)
    assert not isinstance(raised.value, OptimizationUnavailableError)


def test_runtime_eof_error_at_startup_reports_its_traceback(worker) -> None:
    instance, _ = worker
    instance._config["eof_start"] = True
    with pytest.raises(RuntimeError, match=r"(?s)Test worker failed:\n.*truncated checkpoint"):
        instance.start()
    assert instance._process is None and instance._connection is None


@pytest.mark.parametrize("config_bytes", [0, 5_000_000])
def test_worker_dead_at_interpreter_start_is_mapped_by_exit_code(tmp_path, config_bytes) -> None:
    """A kit .pth that refuses at interpreter start kills the child before it reads the config.

    A config larger than the pipe buffer then fails the send itself (a broken pipe), which must map like a closed pipe.
    """
    from boileroom.models._worker import ModelWorker
    from boileroom.optimization import OptimizationUnavailableError

    for name, code in (("refuses", 3), ("crashes", 1)):
        interpreter = tmp_path / name
        interpreter.write_text(f"#!/bin/sh\nexit {code}\n")
        interpreter.chmod(0o755)
        instance = ModelWorker(
            {"timeout_seconds": 10, "msa": "A" * config_bytes},
            {},
            runtime_path=tmp_path / "unused.py",
            runtime_class="Runtime",
            label="Kit",
            python_executable=str(interpreter),
        )
        expected = OptimizationUnavailableError if code == 3 else RuntimeError
        with pytest.raises(expected, match=f"exit(ed with)? code {code}") as raised:
            instance.start()
        assert code == 3 or not isinstance(raised.value, OptimizationUnavailableError)
        assert instance._process is None and instance._connection is None


def test_real_worker_keeps_hash_seed_and_drops_paths_and_its_own_directory(tmp_path) -> None:
    """The production entry point, launched from boileroom/models, with the parent's PYTHONPATH pointing there too.

    ``-I`` used to drop ``PYTHONHASHSEED`` (it implies ``-E``); and the script's own directory on ``sys.path`` let
    the sibling ``boileroom/models/protenix`` package shadow the upstream ``protenix`` a runtime imports.
    """
    from boileroom.models import _worker as module

    models_dir = Path(module.__file__).resolve().parent
    assert (models_dir / "protenix" / "__init__.py").is_file()
    (tmp_path / "probe.py").write_text(
        "import importlib.util, os, sys\n"
        "class Probe:\n"
        "    def __init__(self, config, work_dir):\n"
        "        pass\n"
        "    def predict(self, input_path, output_dir, options):\n"
        "        spec = importlib.util.find_spec('protenix')\n"
        "        return {\n"
        "            'hashseed': os.environ.get('PYTHONHASHSEED'),\n"
        "            'hash_randomization': sys.flags.hash_randomization,\n"
        "            'no_user_site': sys.flags.no_user_site,\n"
        "            'pythonpath': os.environ.get('PYTHONPATH'),\n"
        "            'unbuffered': os.environ.get('PYTHONUNBUFFERED'),\n"
        "            'sys_path': [os.path.realpath(entry or os.curdir) for entry in sys.path],\n"
        "            'protenix_origin': spec.origin if spec is not None else None,\n"
        "        }\n"
    )
    env = {**os.environ, "PYTHONHASHSEED": "0", "PYTHONPATH": str(models_dir), "PYTHONUNBUFFERED": "1"}
    instance = module.ModelWorker(
        {"timeout_seconds": 30},
        env,
        runtime_path=tmp_path / "probe.py",
        runtime_class="Probe",
        label="Probe",
        python_executable=sys.executable,
    )
    try:
        seen = instance.predict("", "", {"timeout_seconds": 30})
    finally:
        instance.close()
    assert seen["hashseed"] == "0" and seen["hash_randomization"] == 0
    assert seen["unbuffered"] == "1"
    assert seen["no_user_site"] == 1
    assert seen["pythonpath"] is None
    assert str(models_dir) not in seen["sys_path"]
    origin = seen["protenix_origin"]
    assert origin is None or not Path(origin).resolve().is_relative_to(models_dir)


def test_child_env_strips_only_path_affecting_variables() -> None:
    from boileroom.models._worker import _child_env

    env = {"PYTHONPATH": "/x", "PYTHONHOME": "/y", "PYTHONSTARTUP": "s.py", "PYTHONHASHSEED": "1", "PATH": "/bin"}
    assert _child_env(env) == {"PYTHONHASHSEED": "1", "PATH": "/bin"}
