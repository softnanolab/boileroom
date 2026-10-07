"""OpenDDE core tests: the Protenix runner contract in its own interpreter, without heavy imports."""

import json
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest

from boileroom.base import PredictionMetadata
from boileroom.optimization import OptimizationUnavailableError


class FakeWorker:
    """CPU stand-in for ``ModelWorker``: records calls, reports a ``describe()`` and a predict payload."""

    info_on_start: dict[str, Any] = {"python": "3.11.5", "layernorm": "torch", "kit.active": "False"}
    payload: dict[str, Any] = {"kernel.resolved.trimul": "cuequivariance", "templates.staged": 0}

    def __init__(self, config: dict[str, Any], env: dict[str, str], **kwargs: Any) -> None:
        self.env = env
        self.kwargs = kwargs
        self.info: dict[str, Any] = {}
        self.calls: list[str] = []
        self.start_error: Exception | None = None
        self.predict_error: Exception | None = None

    def start(self) -> dict[str, Any]:
        self.calls.append("start")
        if self.start_error is not None:
            raise self.start_error
        self.info = dict(self.info_on_start)
        return self.info

    def predict(self, input_json: str, output_dir: str, config: dict[str, Any]) -> dict[str, Any]:
        self.calls.append("predict")
        if self.predict_error is not None:
            raise self.predict_error
        return dict(self.payload)

    def close(self) -> None:
        self.calls.append("close")
        self.info = {}


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
    """Chains are colon-joined into A..Z proteinChain records, each with its canonical per-chain A3M."""
    target = ">query\nAAAAA\n>hit\nAAcAAA\n"
    path = module.OpenDDECore()._write_input_json("AAAAA:CCCC", tmp_path, [target, None])
    records = json.loads(path.read_text())[0]["sequences"]
    assert [r["proteinChain"]["id"] for r in records] == [["A"], ["B"]]
    assert Path(records[0]["proteinChain"]["unpairedMsaPath"]).read_text() == target
    assert Path(records[1]["proteinChain"]["unpairedMsaPath"]).read_text() == ">query\nCCCC\n"


def test_caller_a3m_reaches_opendde_without_dots_or_wrapping(module, tmp_path: Path) -> None:
    """OpenDDE's A3M reader is Protenix's: ``.`` would count as a deleted residue, so the written file drops it."""
    caller = ">query\nMKTAY\n\n>hit\nMK.tT.\nAY.\n"
    path = module.OpenDDECore()._write_input_json("MKTAY", tmp_path, [caller])
    written = Path(json.loads(path.read_text())[0]["sequences"][0]["proteinChain"]["unpairedMsaPath"]).read_text()
    assert written == ">query\nMKTAY\n>hit\nMKtTAY\n"


@pytest.mark.parametrize(
    ("chains", "msa", "match"),
    [
        pytest.param("MKTAY", [">query\nMKTAY\n>hit\nMKT A\n"], "A3M row 1 contains ' '", id="space"),
        pytest.param("MKTAY", [">query\nMKTAY\n>hit\nMKTA*\n"], "A3M row 1 contains '\\*'", id="terminator"),
        pytest.param("MKTAY", [">query\nMKTAY\n>hit\nMKTA\n#\n"], "A3M row 1 contains '#'", id="comment-line"),
        pytest.param("MKTAY:CCCC", [None, ">query\nCCCC\n>hit\nCCCC\n"], "the chain has 4 residues", id="short"),
    ],
)
def test_caller_a3m_opendde_would_misread_is_refused(module, tmp_path: Path, chains, msa, match) -> None:
    """Rows the shared ``a3m_rows`` check accepts but OpenDDE would read differently, or ignore, are refused."""
    with pytest.raises(ValueError, match=f"OpenDDE msa entry {len(msa) - 1}: .*{match}"):
        module.OpenDDECore()._write_input_json(chains, tmp_path, msa)


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
    assert output.token_chain_ids is not None
    assert output.token_chain_ids[0] is not None and output.token_chain_ids[0].tolist() == ["A", "B"]
    assert output.ptm is not None
    assert output.ptm[0] is not None and output.ptm[0][0] == pytest.approx(0.7)


def test_missing_confidence_fails_with_opendde_label(module, sample_outputs: Path) -> None:
    """Incomplete output is reported under the OpenDDE name."""
    next(sample_outputs.glob("**/*_full_data_*.json")).unlink()
    core = module.OpenDDECore()
    with pytest.raises(RuntimeError, match="OpenDDE produced incomplete output"):
        core._collect_outputs(sample_outputs, PredictionMetadata("OpenDDE", "x", [2]), {**core.config, "sample": 1})


def _worker_env(module: ModuleType, config: dict[str, Any], resolution: Any = None) -> dict[str, str]:
    return module.OpenDDECore()._worker_env(config, resolution)


def test_command_env(module, monkeypatch, tmp_path: Path) -> None:
    """Weights root, MSA server, the JIT root and stock OpenDDE's torch LayerNorm are set for a vanilla worker."""
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    for key in (
        "OPENDDE_ROOT_DIR",
        "MODEL_OPT_JIT_ROOT",
        "MODEL_OPT_WEIGHTS_DIGEST_DIR",
        "TRITON_CACHE_DIR",
        "TORCH_EXTENSIONS_DIR",
        "LAYERNORM_TYPE",
    ):
        monkeypatch.delenv(key, raising=False)
    config = {**module.OpenDDECore().config, "msa_server_url": "https://msa.example"}
    env = _worker_env(module, config)
    assert env["OPENDDE_ROOT_DIR"] == str(tmp_path / "opendde")
    assert env["MMSEQS_SERVICE_HOST_URL"] == "https://msa.example"
    assert env["LAYERNORM_TYPE"] == "torch"
    assert env["MODEL_OPT_JIT_ROOT"] == str(tmp_path / "opendde" / "jit")
    assert env["MODEL_OPT_WEIGHTS_DIGEST_DIR"] == str(tmp_path / "opendde" / "jit" / "weights")
    # Un-keyed compile caches would let an A100 and an H100 worker on one volume load each other's builds: the
    # runtime keys them under the root by stack (test_opendde_runtime.py), so the core presets none.
    assert not {"TRITON_CACHE_DIR", "TORCH_EXTENSIONS_DIR"} & env.keys()
    assert "MODEL_OPT_TARGET_GPU" not in env


def test_worker_env_keeps_a_caller_jit_root(module, monkeypatch, tmp_path: Path) -> None:
    """A JIT root (or compile cache) the caller set is kept, not overwritten with the model directory's."""
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("MODEL_OPT_JIT_ROOT", "/scratch/jit")
    monkeypatch.setenv("TRITON_CACHE_DIR", "/scratch/triton")
    env = _worker_env(module, module.OpenDDECore().config)
    assert env["MODEL_OPT_JIT_ROOT"] == "/scratch/jit"
    assert env["TRITON_CACHE_DIR"] == "/scratch/triton"


def test_default_kernels_are_protenix_defaults(module) -> None:
    """OpenDDE inherits the triangle-kernel defaults (cuEquivariance) instead of restating them."""
    from boileroom.models.protenix.core import ProtenixCore

    for key in ("trimul_kernel", "triatt_kernel"):
        assert module.OpenDDECore.DEFAULT_CONFIG[key] == ProtenixCore.DEFAULT_CONFIG[key] == "cuequivariance"


@pytest.mark.parametrize(
    ("mode", "layernorm"), [("vanilla", "torch"), ("exact", "fast_layernorm"), ("fast", "fast_layernorm")]
)
def test_layernorm_is_set_per_mode_over_the_inherited_value(
    module: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: str, layernorm: str
) -> None:
    """The image or caller environment cannot move a mode off its LayerNorm (the old image ENV set fast_layernorm)."""
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    for inherited in ("fast_layernorm", "torch", "openfold"):
        monkeypatch.setenv("LAYERNORM_TYPE", inherited)
        env = _worker_env(module, {**module.OpenDDECore().config, "optimization": mode})
        assert env["LAYERNORM_TYPE"] == layernorm


def test_worker_env_leads_path_with_the_venv_bin_and_sets_the_kit_process_env(
    module: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The venv's ninja builds the fused LayerNorm, so its ``bin`` leads PATH; the kit's process env always holds."""
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("PATH", "/usr/local/cuda-12.6/bin:/usr/bin")
    for name, inherited in (("PYTHONHASHSEED", "random"), ("CFLAGS", "-g -O0"), ("PYTHONDONTWRITEBYTECODE", "")):
        monkeypatch.setenv(name, inherited)
    env = _worker_env(module, {**module.OpenDDECore().config, "opendde_python": "/venv/bin/python"})
    assert env["PATH"].split(":") == ["/venv/bin", "/usr/local/cuda-12.6/bin", "/usr/bin"]
    assert {name: env[name] for name in module.OPENDDE_PROCESS_ENV} == {
        "PYTHONHASHSEED": "0",
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "CFLAGS": "-g0",
    }
    monkeypatch.delenv("PATH")
    assert _worker_env(module, {**module.OpenDDECore().config, "opendde_python": "/venv/bin/python"})["PATH"] == (
        "/venv/bin"
    )


def test_core_hands_the_mode_layernorm_to_the_worker(module, monkeypatch, tmp_path: Path) -> None:
    """The worker the core builds carries its family's LayerNorm for the mode, not Protenix's."""
    import boileroom.models.protenix.core as protenix_core
    from boileroom.optimization import GpuInfo

    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("LAYERNORM_TYPE", "fast_layernorm")
    monkeypatch.setattr(protenix_core, "ModelWorker", FakeWorker)
    monkeypatch.setattr(protenix_core, "describe_gpu", lambda device=None: GpuInfo("NVIDIA A100-SXM4-40GB", (8, 0)))
    core = module.OpenDDECore()
    core._initialize()
    assert core._worker is not None and core._worker.env["LAYERNORM_TYPE"] == "torch"


def test_worker_env_scopes_gcc13_libstdcxx_to_the_venv(
    module: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The venv's libstdc++ directory leads the worker's library path, ahead of any inherited entries."""
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("LD_LIBRARY_PATH", "/usr/local/nvidia/lib64")
    config = {**module.OpenDDECore().config, "opendde_python": "/opt/opendde/bin/python"}
    assert _worker_env(module, config)["LD_LIBRARY_PATH"] == "/opt/opendde/lib:/usr/local/nvidia/lib64"
    monkeypatch.delenv("LD_LIBRARY_PATH")
    assert _worker_env(module, config)["LD_LIBRARY_PATH"] == "/opt/opendde/lib"


def test_dockerfile_keeps_libstdcxx_off_the_global_library_path() -> None:
    """Ubuntu's libstdc++ on the image-wide path broke the system Python (glibc 2.38 vs 2.36)."""
    dockerfile = (Path(__file__).parents[2] / "boileroom/models/opendde/Dockerfile").read_text()
    env_lines = [line for line in dockerfile.splitlines() if "LD_LIBRARY_PATH=" in line and "RUN" not in line]
    assert env_lines
    assert all("/opt/opendde/lib" not in line for line in env_lines)
    assert "ubuntu:24.04" not in dockerfile


def test_kit_env_names_the_resolved_gpu(module, monkeypatch, tmp_path: Path) -> None:
    """Kit modes pass the resolved kit config to the worker."""
    from boileroom.optimization import GpuInfo, resolve_optimization

    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    monkeypatch.delenv("MODEL_OPT_TARGET_GPU", raising=False)
    resolution = resolve_optimization("opendde", "fast", GpuInfo("NVIDIA H200", (9, 0)))
    assert _worker_env(module, module.OpenDDECore().config, resolution)["MODEL_OPT_TARGET_GPU"] == "H100"


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
    # One runtime module serves both families; the runtime class selects OpenDDE.
    assert kwargs["runtime_path"].name == "runtime.py" and kwargs["runtime_path"].parent.name == "protenix"


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
    """The mode, kit config and card land in the output metadata."""
    import boileroom.models.protenix.core as protenix_core
    from boileroom.optimization import GpuInfo

    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(protenix_core, "ModelWorker", FakeWorker)
    monkeypatch.setattr(protenix_core, "detect_gpu", lambda device=None: GpuInfo("NVIDIA A100-SXM4-40GB", (8, 0)))
    core = module.OpenDDECore({"optimization": "exact"})
    seen: dict[str, Any] = {}
    core._collect_outputs = lambda output, metadata, config: seen.setdefault("m", metadata) and Mock(metadata=metadata)
    core.fold("AAAA")  # first fold: the optimization is resolved by this call's own load
    assert seen["m"].optimization == {
        "mode": "exact",
        "kit_config": "a100",
        "gpu_name": "NVIDIA A100-SXM4-40GB",
        "capability": "sm80",
    }


def test_triangle_kernels_default_to_cuequivariance(module) -> None:
    """Upstream's ``auto`` could pick another kernel by itself; the default asks for cuEquivariance explicitly."""
    core = module.OpenDDECore()
    assert core.config["trimul_kernel"] == "cuequivariance"
    assert core.config["triatt_kernel"] == "cuequivariance"


def test_metadata_runtime_merges_provenance_worker_and_request(module, monkeypatch, tmp_path: Path) -> None:
    """``metadata.runtime`` is one flat record: core provenance, the worker's describe() and the predict payload."""
    import platform

    import boileroom.models.protenix.core as protenix_core
    from boileroom.optimization import GpuInfo

    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(protenix_core, "ModelWorker", FakeWorker)
    monkeypatch.setattr(protenix_core, "describe_gpu", lambda device=None: GpuInfo("NVIDIA A100-SXM4-40GB", (8, 0)))
    core = module.OpenDDECore()
    core._collect_outputs = lambda output, metadata, config: Mock(metadata=metadata)
    runtime = core.fold("AAAA").metadata.runtime
    assert runtime["python"] == platform.python_version()
    assert runtime["gpu"] == "NVIDIA A100-SXM4-40GB" and runtime["gpu_capability"] == "sm80"
    assert "boileroom" in runtime and "protenix" not in runtime
    assert runtime["worker.python"] == "3.11.5" and runtime["worker.layernorm"] == "torch"
    assert runtime["predict.kernel.resolved.trimul"] == "cuequivariance"
    assert runtime["predict.templates.staged"] == "0"


def test_refused_start_closes_the_worker_and_stands(module, monkeypatch, tmp_path: Path) -> None:
    """A kit that refuses at load is not restarted by the next call: the refusal stands for this core."""
    import boileroom.models.protenix.core as protenix_core
    from boileroom.optimization import GpuInfo

    workers: list[FakeWorker] = []

    def refusing_worker(*args: Any, **kwargs: Any) -> FakeWorker:
        worker = FakeWorker(*args, **kwargs)
        worker.start_error = OptimizationUnavailableError("OpenDDE refused: the fused LayerNorm did not load")
        workers.append(worker)
        return worker

    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(protenix_core, "ModelWorker", refusing_worker)
    monkeypatch.setattr(protenix_core, "detect_gpu", lambda device=None: GpuInfo("NVIDIA H100 80GB HBM3", (9, 0)))
    core = module.OpenDDECore({"optimization": "exact"})
    with pytest.raises(OptimizationUnavailableError, match="fused LayerNorm did not load"):
        core._initialize()
    assert workers[0].calls == ["start", "close"] and not core.ready
    with pytest.raises(OptimizationUnavailableError, match="refused earlier in this runtime"):
        core.fold("AAAA")
    assert len(workers) == 1 and workers[0].calls == ["start", "close"]


def test_atom_plddt_error_names_the_family_without_a_version(module, sample_outputs: Path) -> None:
    """The out-of-range pLDDT error carries the family label only (it read "OpenDDE 2.0")."""
    path = next(sample_outputs.glob("**/*_full_data_*.json"))
    full = json.loads(path.read_text())
    full["atom_plddt"] = [80.0, 60.0, 90.0, 70.0]
    path.write_text(json.dumps(full))
    core = module.OpenDDECore()
    with pytest.raises(RuntimeError, match=r"^OpenDDE atom pLDDT must be in \[0, 1\]$"):
        core._collect_outputs(sample_outputs, PredictionMetadata("OpenDDE", "x", [2]), {**core.config, "sample": 1})


def test_templates_go_to_one_chain_and_the_others_get_query_only_hits(module, tmp_path: Path) -> None:
    """With caller templates, every other chain reads a query-only hit file instead of being searched."""
    from boileroom.models.protenix.templates import StagedTemplates

    staged = StagedTemplates(
        templates_path=str(tmp_path / "templates" / "hits.a3m"),
        mmcif_dir=str(tmp_path / "templates" / "mmcif"),
        release_dates_path=str(tmp_path / "templates" / "release_dates.json"),
        obsolete_pdbs_path=str(tmp_path / "templates" / "obsolete.dat"),
        cache_dir=str(tmp_path / "templates" / "cache"),
        query="CCCC",
        count=1,
        names=("t",),
    )
    path = module.OpenDDECore()._write_input_json("AAAA:CCCC:DDDDD", tmp_path, None, staged, 1)
    records = [r["proteinChain"] for r in json.loads(path.read_text())[0]["sequences"]]
    assert records[1]["templatesPath"] == staged.templates_path
    for record in (records[0], records[2]):
        other = Path(record["templatesPath"])
        assert other != Path(staged.templates_path)
        assert other.read_text() == f">query\n{record['sequence']}\n"
    assert all("unpairedMsaPath" not in record for record in records)


def test_without_templates_no_chain_gets_a_templates_path(module, tmp_path: Path) -> None:
    """No caller templates: upstream's own use_template behaviour is untouched."""
    path = module.OpenDDECore()._write_input_json("AAAA:CCCC", tmp_path)
    assert all("templatesPath" not in r["proteinChain"] for r in json.loads(path.read_text())[0]["sequences"])


def test_unpaired_msa_is_an_unknown_key(module) -> None:
    """The unreleased ``unpaired_msa`` alias is gone; it is refused, not silently ignored."""
    with pytest.raises(ValueError, match="does not accept config keys \\['unpaired_msa'\\]"):
        module.OpenDDECore().fold("AAAA", options={"unpaired_msa": [">q\nAAAA\n"]})


def test_static_key_per_call_is_refused_by_the_wrapper_and_the_core(monkeypatch: pytest.MonkeyPatch) -> None:
    """The wrapper refuses before dispatch (no remote load); the core refuses the same key for direct callers."""
    from boileroom.models.opendde.core import OpenDDECore
    from boileroom.models.opendde.opendde import OpenDDE

    backend = Mock()
    wrapper = OpenDDE.__new__(OpenDDE)
    wrapper._backend = backend
    with pytest.raises(ValueError, match="can only be set at initialization.*opendde_python"):
        wrapper.fold("AAAA", options={"opendde_python": "/elsewhere/bin/python"})
    assert backend.mock_calls == []

    core = OpenDDECore()
    started = Mock()
    monkeypatch.setattr(core, "_load", started)
    with pytest.raises(ValueError, match="can only be set at initialization.*opendde_python"):
        core.fold("AAAA", options={"opendde_python": "/elsewhere/bin/python"})
    started.assert_not_called()
