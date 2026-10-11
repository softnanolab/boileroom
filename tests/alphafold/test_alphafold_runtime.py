"""ColabFold API contracts without loading its JAX dependencies."""

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    """Install a small upstream stand-in that records model and request state."""
    from boileroom.models.alphafold.core import AlphaFold2MultimerCore
    from boileroom.models.alphafold.runtime import AlphaFold2MultimerRuntime

    runners = [object()]
    models = SimpleNamespace(load_models_and_params=Mock(return_value=runners))
    download = SimpleNamespace(download_alphafold_params=Mock())
    msa: tuple[list[str], None, list[str], list[int], list[dict[str, object]]] = (
        [">query\nAAAA"],
        None,
        ["AAAA"],
        [2],
        [{}],
    )
    batch = SimpleNamespace(
        get_msa_and_templates=Mock(return_value=msa),
        unserialize_msa=Mock(return_value=msa),
        msa_to_str=Mock(return_value="alignment\n"),
        generate_input_feature=Mock(side_effect=lambda *args: ({"msa": np.zeros((8, 8))}, {})),
        predict_structure=Mock(),
    )
    utils = SimpleNamespace(get_queries=Mock(return_value=([("name", ["AAAA", "AAAA"], None, None)], True)))
    for name, value in {
        "colabfold": ModuleType("colabfold"),
        "colabfold.alphafold": ModuleType("colabfold.alphafold"),
        "colabfold.alphafold.models": models,
        "colabfold.download": download,
        "colabfold.batch": batch,
        "colabfold.input": utils,
        "colabfold.utils": SimpleNamespace(setup_logging=Mock()),
    }.items():
        monkeypatch.setitem(sys.modules, name, value)
    config = {**AlphaFold2MultimerCore.DEFAULT_CONFIG, "data_dir": str(tmp_path)}
    return AlphaFold2MultimerRuntime(config, str(tmp_path)), models, batch, utils, config


def test_runner_and_parameters_survive_changed_requests(runtime, tmp_path) -> None:
    """Each request gets fresh features while weights and runners retain identity."""
    resident, models, batch, utils, config = runtime
    first_dir, second_dir = tmp_path / "first", tmp_path / "second"
    resident.predict("input.fasta", str(first_dir), config)
    first = batch.predict_structure.call_args.kwargs

    utils.get_queries.return_value = ([("name", "AAAA", ["supplied alignment"], None)], False)
    batch.unserialize_msa.return_value = ([">query\nAAAA"], None, ["AAAA"], [1], [{}])
    batch.generate_input_feature.side_effect = lambda *args: ({"msa": np.zeros((4, 4))}, {})
    resident.predict("input.a3m", str(second_dir), {**config, "random_seed": 2, "num_seeds": 3, "use_amber": True})
    second = batch.predict_structure.call_args.kwargs

    models.load_models_and_params.assert_called_once()
    assert first["model_runner_and_params"] is second["model_runner_and_params"] is resident.runners
    assert first["feature_dict"] is not second["feature_dict"]
    assert first["sequences_lengths"] == [4, 4] and second["sequences_lengths"] == [4]
    assert first["random_seed"] == 0 and second["random_seed"] == 2
    assert second["num_seeds"] == 3 and second["num_relax"] == 15
    assert first["msa_pad_depth"] == second["msa_pad_depth"] == 8
    batch.get_msa_and_templates.assert_called_once()
    batch.unserialize_msa.assert_called_once_with(["supplied alignment"], "AAAA")
    assert (first_dir / "query.done.txt").exists() and (second_dir / "query.done.txt").exists()


@pytest.mark.parametrize("stage", ["get_msa_and_templates", "predict_structure"])
def test_upstream_failure_propagates_without_completion_marker(runtime, tmp_path, stage) -> None:
    """Do not silently report completion after MSA or GPU errors."""
    resident, _, batch, _, config = runtime
    getattr(batch, stage).side_effect = RuntimeError("upstream failed")
    with pytest.raises(RuntimeError, match="upstream failed"):
        resident.predict("input.fasta", str(tmp_path / "failed"), config)
    assert not (tmp_path / "failed" / "query.done.txt").exists()


def test_supplied_alignment_templates_use_reference_search_path(runtime, tmp_path) -> None:
    """Template searches preserve the supplied alignment for feature construction."""
    resident, _, batch, utils, config = runtime
    resident.config["use_templates"] = True
    alignment = tmp_path / "given.a3m"
    alignment.write_text("provided\n")
    utils.get_queries.return_value = ([("name", ["AAAA", "AAAA"], alignment, None)], True)
    batch.get_msa_and_templates.return_value = ([], None, ["AAAA"], [2], [{"template": "new"}])
    resident.predict(str(alignment), str(tmp_path / "out"), config)
    batch.unserialize_msa.assert_called_once_with(["provided\n"], ["AAAA", "AAAA"])
    assert batch.get_msa_and_templates.call_args.kwargs["msa_mode"] == "single_sequence"
    assert batch.generate_input_feature.call_args.args[4] == [{"template": "new"}]


def test_isolated_runtime_sources_support_python_310() -> None:
    """The isolated interpreter cannot parse Python 3.12-only syntax."""
    import ast

    root = Path(__file__).resolve().parents[2] / "boileroom/models/alphafold"
    for path in (root / "runtime.py", root.parent / "_worker.py"):
        ast.parse(path.read_text(), feature_version=(3, 10))
