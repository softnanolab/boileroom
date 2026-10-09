"""RF3 runtime contract tests: checkpoint handling and engine calls, without RF3, the kit or the network."""

import hashlib
import io
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

CONFIG: dict[str, Any] = {
    "n_recycles": 4,
    "diffusion_batch_size": 3,
    "num_steps": 25,
    "seed": 5,
    "early_stopping_plddt_threshold": None,
    "checkpoint_path": None,
    "optimization": "vanilla",
}


@pytest.fixture
def runtime() -> ModuleType:
    """Import the runtime inside tests; it imports nothing heavy until a runtime is built."""
    from boileroom.models.rf3 import runtime

    return runtime


@pytest.fixture
def checkpoint(tmp_path: Path) -> Path:
    path = tmp_path / "weights.ckpt"
    path.write_bytes(b"weights")
    return path


@pytest.fixture
def events() -> list[str]:
    """Order in which the fakes were touched."""
    return []


@pytest.fixture
def fake_rf3(monkeypatch: pytest.MonkeyPatch, events: list[str]) -> SimpleNamespace:
    """Install fake ``rf3`` and ``lightning`` modules; importing the engine class is recorded as an event."""
    engine = Mock(name="engine")
    engine_class = Mock(name="RF3InferenceEngine", return_value=engine)

    class EngineModule(ModuleType):
        def __getattr__(self, name: str) -> Any:
            if name == "RF3InferenceEngine":
                events.append("import rf3 engine")
                return engine_class
            raise AttributeError(name)

    seed_everything = Mock(name="seed_everything")
    fabric = ModuleType("lightning.fabric")
    fabric.seed_everything = seed_everything  # type: ignore[attr-defined]
    lightning = ModuleType("lightning")
    lightning.fabric = fabric  # type: ignore[attr-defined]
    for name, module in {
        "rf3": ModuleType("rf3"),
        "rf3.inference_engines": ModuleType("rf3.inference_engines"),
        "rf3.inference_engines.rf3": EngineModule("rf3.inference_engines.rf3"),
        "lightning": lightning,
        "lightning.fabric": fabric,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    return SimpleNamespace(engine=engine, engine_class=engine_class, seed_everything=seed_everything)


def fake_kit(monkeypatch: pytest.MonkeyPatch, events: list[str], result: Any = None, exit_code: int | None = None):
    """Install a fake ``rosettafold3_opt`` whose ``enable`` returns ``result`` or exits with ``exit_code``."""
    module = ModuleType("rosettafold3_opt")

    def enable(mode: str, strict: bool) -> Any:
        events.append(f"enable {mode} strict={strict}")
        if exit_code is not None:
            raise SystemExit(exit_code)
        return result if result is not None else {"active": True}

    module.enable = enable  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "rosettafold3_opt", module)


# -- Engine wiring ---------------------------------------------------------------------------------------------------


def test_engine_is_built_once_with_the_static_configuration(
    runtime: ModuleType, fake_rf3: SimpleNamespace, checkpoint: Path, tmp_path: Path
) -> None:
    runtime.RF3Runtime({**CONFIG, "checkpoint_path": str(checkpoint)}, str(tmp_path))

    fake_rf3.engine_class.assert_called_once_with(
        ckpt_path=checkpoint, n_recycles=4, diffusion_batch_size=3, num_steps=25, seed=5
    )
    # Weights load at start-up so that a bad checkpoint fails there, not in the first request.
    fake_rf3.engine.initialize.assert_called_once_with()


def test_predict_applies_the_request_seed_and_threshold_and_runs_the_engine(
    runtime: ModuleType, fake_rf3: SimpleNamespace, checkpoint: Path, tmp_path: Path
) -> None:
    built = runtime.RF3Runtime({**CONFIG, "checkpoint_path": str(checkpoint)}, str(tmp_path))

    built.predict("in.json", "out", {**CONFIG, "seed": 9, "early_stopping_plddt_threshold": 0.4})
    built.predict("in2.json", "out2", {**CONFIG, "seed": 11})

    assert [call.args for call in fake_rf3.seed_everything.call_args_list] == [(9,), (11,)]
    assert fake_rf3.seed_everything.call_args.kwargs == {"workers": True}
    assert [call.kwargs for call in fake_rf3.engine.run.call_args_list] == [
        {"inputs": "in.json", "out_dir": "out"},
        {"inputs": "in2.json", "out_dir": "out2"},
    ]
    # The threshold is reset on every request, so one request's early stop never leaks into the next.
    assert fake_rf3.engine.seed == 11 and fake_rf3.engine.early_stopping_plddt_threshold is None
    fake_rf3.engine.initialize.assert_called_once_with()


def test_vanilla_never_touches_the_kit(
    runtime: ModuleType,
    fake_rf3: SimpleNamespace,
    checkpoint: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "rosettafold3_opt", None)  # importing it would raise
    monkeypatch.setenv("ROSETTAFOLD3_OPT_CKPT", "")
    monkeypatch.delenv("ROSETTAFOLD3_OPT_CKPT")

    runtime.RF3Runtime({**CONFIG, "checkpoint_path": str(checkpoint)}, str(tmp_path))

    import os

    assert "ROSETTAFOLD3_OPT_CKPT" not in os.environ


# -- Kit activation --------------------------------------------------------------------------------------------------


def test_kit_is_enabled_before_rf3_is_imported(
    runtime: ModuleType,
    fake_rf3: SimpleNamespace,
    checkpoint: Path,
    tmp_path: Path,
    events: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The kit refuses late activation: it has to run, strictly, before the engine module is imported."""
    monkeypatch.setenv("ROSETTAFOLD3_OPT_CKPT", "")
    fake_kit(monkeypatch, events)

    runtime.RF3Runtime({**CONFIG, "optimization": "exact", "checkpoint_path": str(checkpoint)}, str(tmp_path))

    import os

    assert events == ["enable exact strict=True", "import rf3 engine"]
    assert os.environ["ROSETTAFOLD3_OPT_CKPT"] == str(checkpoint)


def test_kit_exit_becomes_a_runtime_error(
    runtime: ModuleType,
    fake_rf3: SimpleNamespace,
    checkpoint: Path,
    tmp_path: Path,
    events: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The kit exits (code 3) when it cannot activate; that must not take the worker down with it."""
    monkeypatch.setenv("ROSETTAFOLD3_OPT_CKPT", "")
    fake_kit(monkeypatch, events, exit_code=3)

    with pytest.raises(RuntimeError, match="optimization='exact' did not activate: the kit exited with 3"):
        runtime.RF3Runtime({**CONFIG, "optimization": "exact", "checkpoint_path": str(checkpoint)}, str(tmp_path))

    fake_rf3.engine_class.assert_not_called()


def test_inactive_kit_is_an_error(
    runtime: ModuleType,
    fake_rf3: SimpleNamespace,
    checkpoint: Path,
    tmp_path: Path,
    events: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ROSETTAFOLD3_OPT_CKPT", "")
    fake_kit(monkeypatch, events, result={"active": False, "reason": "driver too old"})

    with pytest.raises(RuntimeError, match="did not activate: driver too old"):
        runtime.RF3Runtime({**CONFIG, "optimization": "exact", "checkpoint_path": str(checkpoint)}, str(tmp_path))


def test_missing_kit_names_the_image_needed(
    runtime: ModuleType,
    fake_rf3: SimpleNamespace,
    checkpoint: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ROSETTAFOLD3_OPT_CKPT", "")
    monkeypatch.setitem(sys.modules, "rosettafold3_opt", None)

    with pytest.raises(RuntimeError, match="needs the rosettafold3 kit image"):
        runtime.RF3Runtime({**CONFIG, "optimization": "exact", "checkpoint_path": str(checkpoint)}, str(tmp_path))


# -- Checkpoint resolution -------------------------------------------------------------------------------------------


def serve(monkeypatch: pytest.MonkeyPatch, runtime: ModuleType, payload: bytes, *, sha: bool = True) -> Mock:
    """Make the pinned download return ``payload`` and describe it as the pinned release."""
    monkeypatch.setattr(runtime, "CHECKPOINT_BYTES", len(payload))
    monkeypatch.setattr(runtime, "CHECKPOINT_SHA256", hashlib.sha256(payload if sha else b"other").hexdigest())
    urlopen = Mock(side_effect=lambda url, timeout: io.BytesIO(payload))
    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    return urlopen


def test_explicit_checkpoint_is_used_as_is(runtime: ModuleType, checkpoint: Path) -> None:
    assert runtime._resolve_checkpoint({"checkpoint_path": str(checkpoint)}) == checkpoint


def test_missing_explicit_checkpoint_is_an_error(runtime: ModuleType, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="checkpoint_path does not exist"):
        runtime._resolve_checkpoint({"checkpoint_path": str(tmp_path / "absent.ckpt")})


def test_checkpoint_is_downloaded_verified_and_cached(
    runtime: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("RF3_ROOT_DIR", str(tmp_path / "root"))
    urlopen = serve(monkeypatch, runtime, b"released weights")

    first = runtime._resolve_checkpoint({})
    second = runtime._resolve_checkpoint({})

    assert first == second == tmp_path / "root" / runtime.CHECKPOINT_NAME
    assert first.read_bytes() == b"released weights"
    urlopen.assert_called_once()
    assert urlopen.call_args.args[0] == runtime.CHECKPOINT_URL
    assert list(first.parent.glob("*.part")) == []


def test_cached_checkpoint_with_the_right_size_but_other_bytes_is_downloaded_again(
    runtime: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A file of the pinned size is not trusted on size alone: its digest must match before it is loaded."""
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setenv("RF3_ROOT_DIR", str(root))
    urlopen = serve(monkeypatch, runtime, b"released weights")
    (root / runtime.CHECKPOINT_NAME).write_bytes(b"corrupted weight")  # the same 16 bytes

    resolved = runtime._resolve_checkpoint({})

    assert resolved.read_bytes() == b"released weights"
    urlopen.assert_called_once()


def test_corrupted_download_is_rejected_and_leaves_nothing_behind(
    runtime: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("RF3_ROOT_DIR", str(tmp_path / "root"))
    serve(monkeypatch, runtime, b"tampered weights", sha=False)

    with pytest.raises(RuntimeError, match="has sha256 .* expected"):
        runtime._resolve_checkpoint({})

    assert list((tmp_path / "root").iterdir()) == []


def test_interrupted_download_leaves_no_partial_file(
    runtime: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A connection that drops after some bytes were written must not leave a file that looks like the weights."""
    root = tmp_path / "root"
    monkeypatch.setenv("RF3_ROOT_DIR", str(root))
    seen: dict[str, list[str]] = {}

    class DroppedConnection(io.BytesIO):
        calls = 0

        def read(self, size: int | None = -1) -> bytes:
            DroppedConnection.calls += 1
            if DroppedConnection.calls == 1:
                return b"first chunk"
            seen["during"] = sorted(path.name for path in root.iterdir())
            raise OSError("connection reset")

    monkeypatch.setattr("urllib.request.urlopen", Mock(side_effect=lambda url, timeout: DroppedConnection()))

    with pytest.raises(OSError, match="connection reset"):
        runtime._resolve_checkpoint({})

    assert len(seen["during"]) == 1 and seen["during"][0].endswith(".part")
    assert list(root.iterdir()) == []


def test_unreachable_download_is_reported(runtime: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RF3_ROOT_DIR", str(tmp_path / "root"))
    monkeypatch.setattr("urllib.request.urlopen", Mock(side_effect=OSError("network down")))

    with pytest.raises(OSError, match="network down"):
        runtime._resolve_checkpoint({})

    assert list((tmp_path / "root").iterdir()) == []


def test_truncated_cached_checkpoint_is_fetched_again(
    runtime: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / runtime.CHECKPOINT_NAME).write_bytes(b"trunc")
    monkeypatch.setenv("RF3_ROOT_DIR", str(root))
    urlopen = serve(monkeypatch, runtime, b"released weights")

    path = runtime._resolve_checkpoint({})

    urlopen.assert_called_once()
    assert path.read_bytes() == b"released weights"


def test_pinned_checkpoint_matches_the_documented_release(runtime: ModuleType) -> None:
    """The constants identify the foundry release the image pins; changing them is a deliberate act."""
    assert runtime.CHECKPOINT_NAME == "rf3_foundry_01_24_latest_remapped.ckpt"
    assert runtime.CHECKPOINT_URL.endswith(runtime.CHECKPOINT_NAME)
    assert runtime.CHECKPOINT_BYTES == 3_038_876_446
    assert len(runtime.CHECKPOINT_SHA256) == 64
