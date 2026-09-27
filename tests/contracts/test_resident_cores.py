"""The two folding adapters must share the package's persistent core lifecycle."""

import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize(
    "family,class_name",
    [("protenix", "ProtenixCore"), ("alphafold", "AlphaFold2MultimerCore")],
)
def test_core_loads_once_for_multiple_fold_jobs(monkeypatch, tmp_path, family, class_name) -> None:
    """Repeated folds use one worker and fresh files, and closing releases it."""
    module = importlib.import_module(f"boileroom.models.{family}.core")
    worker = Mock()
    factory = Mock(return_value=worker)
    monkeypatch.setattr(module, "ModelWorker", factory)
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    core = getattr(module, class_name)({"use_msa_server": False} if family == "alphafold" else {})
    monkeypatch.setattr(core, "_collect_outputs", lambda output, metadata, config: SimpleNamespace(metadata=metadata))
    observed = []

    def predict(input_path, output_dir, config):
        assert Path(input_path).is_file() and Path(output_dir).is_dir()
        observed.append((input_path, output_dir, config.copy()))

    worker.predict.side_effect = predict
    core._initialize()
    core.fold("AAAA:CCCC")
    core.fold("AAAA")
    factory.assert_called_once()
    worker.start.assert_called_once()
    assert len(observed) == 2
    assert observed[0][0] != observed[1][0] and observed[0][1] != observed[1][1]
    assert not any(Path(item[1]).exists() for item in observed)
    assert core.ready
    core.close()
    worker.close.assert_called_once()
    assert not core.ready


def test_af2_materializes_msa_before_crossing_interpreter_boundary(monkeypatch, tmp_path) -> None:
    """Python 3.12 MSAInput objects must never need unpickling in ColabFold's 3.10."""
    from boileroom.inputs import MSAInput
    from boileroom.models.alphafold import core as module

    worker = Mock()
    monkeypatch.setattr(module, "ModelWorker", Mock(return_value=worker))
    core = module.AlphaFold2MultimerCore({"data_dir": str(tmp_path)})
    monkeypatch.setattr(core, "_collect_outputs", lambda output, metadata, config: SimpleNamespace(metadata=metadata))

    def predict(input_path, output_dir, config):
        assert Path(input_path).suffix == ".a3m"
        assert Path(input_path).read_text().startswith("#")
        assert "msa" not in config
        assert config["random_seed"] == 7

    worker.predict.side_effect = predict
    core.fold("AAAA:CCCC", {"msa": MSAInput(sequences=["AAAA:CCCC"]), "random_seed": 7})
    worker.predict.assert_called_once()
