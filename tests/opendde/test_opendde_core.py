"""OpenDDE core tests: the Protenix runner contract in its own interpreter, without heavy imports."""

import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from boileroom.base import PredictionMetadata
from boileroom.optimization import OptimizationUnavailableError


@pytest.fixture
def module():
    """Import the core inside tests, never at module scope."""
    from boileroom.models.opendde import core

    return core


@pytest.fixture
def sample_outputs(tmp_path: Path) -> Path:
    """Write one OpenDDE-shaped sample (identical layout to Protenix 2.0)."""
    from biotite.structure import AtomArray
    from biotite.structure.io.pdbx import CIFFile, set_structure

    atoms = AtomArray(4)
    atoms.chain_id = np.array(["A"] * 2 + ["B"] * 2)
    atoms.res_id = np.array([1, 1, 1, 1])
    atoms.res_name[:] = "ALA"
    atoms.atom_name = np.tile(["N", "CA"], 2)
    atoms.element = np.tile(["N", "C"], 2)
    atoms.coord = np.arange(12).reshape(4, 3)
    cif = CIFFile()
    set_structure(cif, atoms)
    directory = tmp_path / "boileroom_target" / "seed_101" / "predictions"
    directory.mkdir(parents=True)
    cif.write(directory / "boileroom_target_sample_0.cif")
    (directory / "boileroom_target_summary_confidence_sample_0.json").write_text(
        json.dumps({"plddt": 80.0, "ptm": 0.7, "iptm": 0.6, "ranking_score": 0.65})
    )
    (directory / "boileroom_target_full_data_sample_0.json").write_text(
        json.dumps(
            {
                "token_pair_pae": [[1.0, 2.0], [2.0, 1.0]],
                "atom_to_token_idx": [0, 0, 1, 1],
                "token_asym_id": [0, 1],
                "atom_plddt": [0.8, 0.6, 0.9, 0.7],
            }
        )
    )
    return tmp_path


def test_defaults_and_static_keys(module) -> None:
    """OpenDDE defaults to its single released model and adds the interpreter as a static key."""
    core = module.OpenDDECore()
    assert core.config["model_name"] == "opendde_v1"
    assert core.config["optimization"] == "vanilla"
    assert core.config["opendde_python"] == "/opt/opendde/bin/python"
    assert {"optimization", "opendde_python", "model_name"} <= module.OpenDDECore.STATIC_CONFIG_KEYS


def test_writes_protein_chain_json_like_protenix(module, tmp_path: Path) -> None:
    """Chains are colon-joined into A..Z proteinChain records with per-chain A3M passthrough."""
    target = ">query\nAAAA\n>hit\nAAcAA\n"
    path = module.OpenDDECore()._write_input_json("AAAA:CCCC", tmp_path, [target, None])
    records = json.loads(path.read_text())[0]["sequences"]
    assert [r["proteinChain"]["id"] for r in records] == [["A"], ["B"]]
    assert Path(records[0]["proteinChain"]["unpairedMsaPath"]).read_text() == target
    assert Path(records[1]["proteinChain"]["unpairedMsaPath"]).read_text() == ">query\nCCCC\n"


def test_invalid_input_names_opendde(module, tmp_path: Path) -> None:
    """Errors carry the OpenDDE label, not Protenix's."""
    with pytest.raises(ValueError, match="OpenDDE input requires"):
        module.OpenDDECore()._write_input_json("AAAA::CCCC", tmp_path)
    with pytest.raises(ValueError, match="exactly one top-level"):
        module.OpenDDECore().fold(["AAAA", "CCCC"])


@pytest.mark.parametrize("dtype", ["fp16", "bogus"])
def test_dtype_fp16_is_rejected(module, dtype: str) -> None:
    """Upstream OpenDDE only offers bf16 and fp32."""
    with pytest.raises(ValueError, match="dtype must be bf16, fp32"):
        module.OpenDDECore().fold("AAAA", options={"dtype": dtype})


def test_collect_outputs_returns_opendde_output(module, sample_outputs: Path) -> None:
    """Shared Protenix parsing yields an OpenDDEOutput with PAE, token ids, and scalar scores."""
    from boileroom.models.opendde.types import OpenDDEOutput

    core = module.OpenDDECore()
    output = core._collect_outputs(
        sample_outputs,
        PredictionMetadata("OpenDDE", "opendde_v1", [2]),
        {**core.config, "sample": 1, "include_fields": ["*"]},
    )
    assert type(output) is OpenDDEOutput
    assert output.seeds == [101] and output.sample_ranks == [0]
    assert output.pae is not None and output.pae[0].shape == (2, 2)
    assert output.token_chain_ids is not None and output.token_chain_ids[0].tolist() == ["A", "B"]
    assert output.ptm is not None and output.ptm[0][0] == pytest.approx(0.7)


def test_missing_confidence_fails_with_opendde_label(module, sample_outputs: Path) -> None:
    """Incomplete output is reported under the OpenDDE name."""
    next(sample_outputs.glob("**/*_full_data_*.json")).unlink()
    core = module.OpenDDECore()
    with pytest.raises(RuntimeError, match="OpenDDE produced incomplete output"):
        core._collect_outputs(sample_outputs, PredictionMetadata("OpenDDE", "x", [2]), {**core.config, "sample": 1})


def test_command_env(module, monkeypatch, tmp_path: Path) -> None:
    """Weights root, MSA server, JIT caches and fast LayerNorm are set for the worker."""
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    for key in ("OPENDDE_ROOT_DIR", "MODEL_OPT_JIT_ROOT", "TRITON_CACHE_DIR", "TORCH_EXTENSIONS_DIR", "LAYERNORM_TYPE"):
        monkeypatch.delenv(key, raising=False)
    config = {**module.OpenDDECore().config, "msa_server_url": "https://msa.example"}
    env = module._command_env(config)
    assert env["OPENDDE_ROOT_DIR"] == str(tmp_path / "opendde")
    assert env["MMSEQS_SERVICE_HOST_URL"] == "https://msa.example"
    assert env["LAYERNORM_TYPE"] == "fast_layernorm"
    assert env["TRITON_CACHE_DIR"].startswith(str(tmp_path / "opendde" / "jit"))
    assert "MODEL_OPT_TARGET_GPU" not in env


def test_kit_env_names_the_resolved_gpu(module, monkeypatch, tmp_path: Path) -> None:
    """Kit modes pass the resolved kit config to the worker."""
    from boileroom.optimization import GpuInfo, resolve_optimization

    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    monkeypatch.delenv("MODEL_OPT_TARGET_GPU", raising=False)
    resolution = resolve_optimization("opendde", "fast", GpuInfo("NVIDIA H200", (9, 0)))
    assert module._command_env(module.OpenDDECore().config, resolution)["MODEL_OPT_TARGET_GPU"] == "H100"


def test_worker_uses_isolated_interpreter_and_opendde_runtime(module, monkeypatch, tmp_path: Path) -> None:
    """The worker is built once with the venv interpreter and the OpenDDE runtime class."""
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    factory = Mock()
    import boileroom.models.protenix.core as protenix_core

    monkeypatch.setattr(protenix_core, "ModelWorker", factory)
    core = module.OpenDDECore({"opendde_python": "/venv/bin/python"})
    core._initialize()
    core._initialize()
    factory.assert_called_once()
    kwargs = factory.call_args.kwargs
    assert kwargs["python_executable"] == "/venv/bin/python"
    assert kwargs["runtime_class"] == "OpenDDERuntime" and kwargs["label"] == "OpenDDE"
    assert kwargs["runtime_path"].name == "runtime.py" and kwargs["runtime_path"].parent.name == "opendde"


def test_kit_mode_refused_on_unserved_gpu_before_worker(module, monkeypatch) -> None:
    """An L40S cannot run the kit; the refusal names the card and no worker starts."""
    import boileroom.models.protenix.core as protenix_core
    from boileroom.optimization import GpuInfo

    factory = Mock()
    monkeypatch.setattr(protenix_core, "ModelWorker", factory)
    monkeypatch.setattr(protenix_core, "detect_gpu", lambda device=None: GpuInfo("NVIDIA L40S", (8, 9)))
    with pytest.raises(OptimizationUnavailableError, match="cannot run opendde on NVIDIA L40S"):
        module.OpenDDECore({"optimization": "fast"})._initialize()
    factory.assert_not_called()


def test_metadata_records_optimization(module, monkeypatch, tmp_path: Path) -> None:
    """The requested and resolved mode land in the output metadata."""
    import boileroom.models.protenix.core as protenix_core
    from boileroom.optimization import GpuInfo

    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(protenix_core, "ModelWorker", Mock())
    monkeypatch.setattr(protenix_core, "detect_gpu", lambda device=None: GpuInfo("NVIDIA A100-SXM4-40GB", (8, 0)))
    core = module.OpenDDECore({"optimization": "exact"})
    seen = {}
    core._collect_outputs = lambda output, metadata, config: seen.setdefault("m", metadata) and Mock(metadata=metadata)  # type: ignore[method-assign]
    core._initialize()
    core.fold("AAAA")
    assert seen["m"].optimization["active"] == "exact" and seen["m"].optimization["kit_config"] == "a100"
