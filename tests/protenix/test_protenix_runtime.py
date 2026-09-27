"""Runner contract tests without importing Protenix or loading GPU weights."""

import copy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


def test_runtime_reuses_weights_and_resets_request_state(monkeypatch, tmp_path) -> None:
    """Changed seeds, lengths and sampling settings must not reuse stale state."""
    from boileroom.models.protenix.core import ProtenixCore
    from boileroom.models.protenix.runtime import ProtenixRuntime

    configs = SimpleNamespace(
        model=SimpleNamespace(N_cycle=10),
        sample_diffusion=SimpleNamespace(N_step=200, N_sample=5, guidance=SimpleNamespace(enable=False)),
        sorted_by_ranking_score=True,
        mutated_by_previous_prediction=False,
    )
    runner = SimpleNamespace(configs=configs, model=SimpleNamespace(N_cycle=10), error_dir=str(tmp_path / "ERR"))
    runner.update_model_configs = Mock()
    runner.init_basics = Mock()
    runner.init_dumper = Mock()
    batch = ModuleType("runner.batch_inference")
    batch.get_default_runner = Mock(return_value=runner)
    batch.inference_configs = {}
    batch.init_logging = Mock()
    batch.preprocess_input = Mock(side_effect=lambda path, **kwargs: path)
    inference = ModuleType("runner.inference")
    calls = []

    def predict(current_runner, config):
        calls.append((id(current_runner), copy.deepcopy(config), current_runner.model.N_cycle))
        config.mutated_by_previous_prediction = True

    inference.infer_predict = predict
    monkeypatch.setitem(sys.modules, "runner", ModuleType("runner"))
    monkeypatch.setitem(sys.modules, "runner.batch_inference", batch)
    monkeypatch.setitem(sys.modules, "runner.inference", inference)
    config = dict(ProtenixCore.DEFAULT_CONFIG)
    runtime = ProtenixRuntime(config, str(tmp_path))
    runtime.predict("first.json", "first", {**config, "seeds": "2,10", "cycle": 4, "sample": 2, "step": 50})
    runtime.predict("second.json", "second", config)
    batch.get_default_runner.assert_called_once()
    first, second = calls
    assert first[0] == second[0]
    assert first[1].seeds == [2, 10] and second[1].seeds == [101]
    assert first[2] == 4 and second[2] == 10
    assert first[1].sample_diffusion.N_sample == 2 and second[1].sample_diffusion.N_sample == 5
    assert first[1].sample_diffusion.N_step == 50 and second[1].sample_diffusion.N_step == 200
    assert first[1].input_json_path == "first.json" and second[1].input_json_path == "second.json"
    assert first[1].dump_dir == "first" and second[1].dump_dir == "second"
    assert not second[1].mutated_by_previous_prediction
    assert runner.init_basics.call_count == runner.init_dumper.call_count == 2
    # Upstream sometimes swallows exceptions and writes ERR files instead.
    Path(runner.error_dir).mkdir()
    (Path(runner.error_dir) / "target.txt").write_text("CUDA out of memory")
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        runtime.predict("third.json", "third", config)
