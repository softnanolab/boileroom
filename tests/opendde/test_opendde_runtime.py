"""Runner contract tests without importing OpenDDE or loading GPU weights."""

import copy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


def _install_fake_runner(monkeypatch, tmp_path):
    configs = SimpleNamespace(
        model=SimpleNamespace(N_cycle=10),
        sample_diffusion=SimpleNamespace(N_step=200, N_sample=5, guidance={"enable": False}),
        sorted_by_ranking_score=True,
        mutated=False,
    )
    runner = SimpleNamespace(configs=configs, model=SimpleNamespace(N_cycle=10), error_dir=str(tmp_path / "ERR"))
    runner.update_model_configs = Mock()
    runner.init_basics = Mock()
    runner.init_dumper = Mock()
    batch = SimpleNamespace(
        get_default_runner=Mock(return_value=runner),
        init_logging=Mock(),
        preprocess_input=Mock(side_effect=lambda path, **kwargs: path),
    )
    calls = []

    def predict(current_runner, config):
        calls.append((copy.deepcopy(config), current_runner.model.N_cycle))
        config.mutated = True

    monkeypatch.setitem(sys.modules, "runner", ModuleType("runner"))
    monkeypatch.setitem(sys.modules, "runner.batch_inference", batch)
    monkeypatch.setitem(sys.modules, "runner.inference", SimpleNamespace(infer_predict=predict))
    return runner, batch, calls


def test_runtime_loads_once_and_resets_request_state(monkeypatch, tmp_path) -> None:
    """Changed seeds, sampling and guidance never leak between requests; weights load once."""
    from boileroom.models.opendde.core import OpenDDECore
    from boileroom.models.opendde.runtime import OpenDDERuntime

    runner, batch, calls = _install_fake_runner(monkeypatch, tmp_path)
    config = dict(OpenDDECore.DEFAULT_CONFIG)
    runtime = OpenDDERuntime(config, str(tmp_path))
    kwargs = batch.get_default_runner.call_args.kwargs
    assert kwargs["model_name"] == "opendde_v1" and kwargs["dump_dir"] == str(tmp_path)
    runtime.predict(
        "a.json", "a", {**config, "seeds": "2,10", "cycle": 4, "sample": 2, "step": 50, "use_tfg_guidance": True}
    )
    runtime.predict("b.json", "b", config)
    batch.get_default_runner.assert_called_once()
    (first, first_cycle), (second, second_cycle) = calls
    assert first.seeds == [2, 10] and second.seeds == [101]
    assert first_cycle == 4 and second_cycle == 10
    assert first.sample_diffusion.guidance["enable"] is True and second.sample_diffusion.guidance["enable"] is False
    assert first.dump_dir == "a" and second.dump_dir == "b" and not second.mutated
    assert all(call.kwargs["msa_server_mode"] == "colabfold" for call in batch.preprocess_input.call_args_list)


def test_runtime_reports_err_files(monkeypatch, tmp_path) -> None:
    """Upstream swallows some failures into ERR files; they must surface as errors."""
    from boileroom.models.opendde.core import OpenDDECore
    from boileroom.models.opendde.runtime import OpenDDERuntime

    runner, _, _ = _install_fake_runner(monkeypatch, tmp_path)
    config = dict(OpenDDECore.DEFAULT_CONFIG)
    runtime = OpenDDERuntime(config, str(tmp_path))
    Path(runner.error_dir).mkdir()
    (Path(runner.error_dir) / "target.txt").write_text("CUDA out of memory")
    with pytest.raises(RuntimeError, match="(?s)OpenDDE inference failed.*CUDA out of memory"):
        runtime.predict("c.json", "c", config)


@pytest.mark.parametrize("mode", ["exact", "fast"])
def test_kit_is_enabled_before_runner_import(monkeypatch, tmp_path, mode) -> None:
    """The kit refuses late activation, so it must run strictly before `runner` is imported."""
    from boileroom.models.opendde.core import OpenDDECore
    from boileroom.models.opendde.runtime import OpenDDERuntime

    _install_fake_runner(monkeypatch, tmp_path)
    order = []
    kit = ModuleType("opendde_opt")
    kit.enable = lambda m, strict: order.append(("enable", m, strict)) or {"active": True}
    monkeypatch.setitem(sys.modules, "opendde_opt", kit)
    batch = sys.modules["runner.batch_inference"]
    real = batch.get_default_runner
    batch.get_default_runner = lambda **kw: order.append("runner") or real(**kw)
    OpenDDERuntime({**OpenDDECore.DEFAULT_CONFIG, "optimization": mode}, str(tmp_path))
    assert order == [("enable", mode, True), "runner"]


def test_kit_mode_without_kit_image_fails_by_name(monkeypatch, tmp_path) -> None:
    """A vanilla-only image must say so instead of silently running unoptimized."""
    from boileroom.models.opendde.core import OpenDDECore
    from boileroom.models.opendde.runtime import OpenDDERuntime

    _install_fake_runner(monkeypatch, tmp_path)
    monkeypatch.setitem(sys.modules, "opendde_opt", None)
    with pytest.raises(RuntimeError, match="OpenDDE kit image"):
        OpenDDERuntime({**OpenDDECore.DEFAULT_CONFIG, "optimization": "fast"}, str(tmp_path))


def test_kit_inactive_report_is_an_error(monkeypatch, tmp_path) -> None:
    """A kit that reports inactive must abort loading."""
    from boileroom.models.opendde.core import OpenDDECore
    from boileroom.models.opendde.runtime import OpenDDERuntime

    _install_fake_runner(monkeypatch, tmp_path)
    kit = ModuleType("opendde_opt")
    kit.enable = lambda m, strict: {"active": False, "reason": "no gpu"}
    monkeypatch.setitem(sys.modules, "opendde_opt", kit)
    with pytest.raises(RuntimeError, match="did not activate: no gpu"):
        OpenDDERuntime({**OpenDDECore.DEFAULT_CONFIG, "optimization": "exact"}, str(tmp_path))
