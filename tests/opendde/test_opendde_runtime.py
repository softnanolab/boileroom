"""OpenDDE worker-runtime contract tests, CPU-only, with fake runner and kit modules.

OpenDDE's runtime is :class:`OpenDDERuntime` in the shared Protenix-family runtime file, which the worker runs by
path in the model image's interpreter; these tests load it the same way.
"""

import copy
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

RUNTIME_PATH = Path(__file__).resolve().parents[2] / "boileroom" / "models" / "protenix" / "runtime.py"
TEMPLATE_UTILS = "opendde.data.template.template_utils"
HEALTHY_REPORT = {
    "active": True,
    "mode": "exact",
    "partial": False,
    "levers_planned": ["lnstream", "cueq_triatt"],
    "levers_applied": ["lnstream", "cueq_triatt"],
    "levers_fallback": [],
    "levers_unavailable": [],
    "levers_not_in_arm": ["fp8_trimul"],
    "fallback_reasons": {},
    "line": "a100",
}


def _config(**overrides: Any) -> dict[str, Any]:
    from boileroom.models.opendde.core import OpenDDECore

    return {**OpenDDECore.DEFAULT_CONFIG, **overrides}


@pytest.fixture
def rt() -> Any:
    """The runtime module, loaded from its file as the worker child does."""
    spec = importlib.util.spec_from_file_location("boileroom_opendde_runtime_test", RUNTIME_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch) -> None:
    # setenv then delenv registers an undo, so a value the runtime or a fake kit sets is removed afterwards.
    for name in (
        "LAYERNORM_TYPE",
        "BOILEROOM_KIT_COMMIT",
        "MODEL_OPT_JIT_ROOT",
        "MODEL_OPT_STACK_KEY",
        "TRITON_CACHE_DIR",
        "TORCH_EXTENSIONS_DIR",
    ):
        monkeypatch.setenv(name, "x")
        monkeypatch.delenv(name)
    # The core starts every worker with its mode's LayerNorm (vanilla: torch); FakeKit sets the kit modes' value.
    monkeypatch.setenv("LAYERNORM_TYPE", "torch")


class Fake:
    """The fake upstream runner; ``auto`` kernels resolve to ``auto_resolves_to``, as upstream does on the device."""

    def __init__(self, monkeypatch, tmp_path: Path) -> None:
        self.auto_resolves_to = "cuequivariance"
        template = SimpleNamespace(
            prot_template_mmcif_dir="/data/mmcif",
            release_dates_path="/data/dates.json",
            obsolete_pdbs_path="/data/obsolete.json",
            prot_template_cache_dir="/data/shared_cache",
            fetch_remote=True,
            kalign_binary_path="",
        )
        self.configs = SimpleNamespace(
            model=SimpleNamespace(N_cycle=10),
            sample_diffusion=SimpleNamespace(N_step=200, N_sample=5, guidance={"enable": False}),
            sorted_by_ranking_score=True,
            data=SimpleNamespace(template=template),
            use_template=False,
            num_workers=16,
            dtype="bf16",
            mutated=False,
        )
        # A CPU device by default, so the runtime's compute-capability probe never reads a real GPU of the test host.
        self.runner = SimpleNamespace(
            configs=self.configs,
            model=SimpleNamespace(N_cycle=10),
            error_dir=str(tmp_path / "ERR"),
            device=SimpleNamespace(type="cpu"),
        )
        self.runner.update_model_configs = Mock()
        self.runner.init_basics = Mock()
        self.runner.init_dumper = Mock()
        self.order: list[str] = []
        self.seen: list[tuple[Any, int]] = []
        self.featurizer_calls: list[tuple[str, int]] = []
        self.batch = SimpleNamespace(
            get_default_runner=Mock(side_effect=self._get_default_runner),
            init_logging=Mock(),
            preprocess_input=Mock(side_effect=lambda path, **kwargs: path),
        )
        fake = self

        class TemplateHitFeaturizer:
            def get_templates(self, sequence_uid, query_sequence, hits, max_template_date=None):
                count = dict(fake.featurizer_calls).get(query_sequence, 0)
                return SimpleNamespace(features=[{}] * count, hits=[], errors=[], warnings=[]), None

        self.featurizer = TemplateHitFeaturizer
        utils = ModuleType(TEMPLATE_UTILS)
        utils.TemplateHitFeaturizer = TemplateHitFeaturizer  # type: ignore[attr-defined]
        for name in ("runner", "opendde", "opendde.data", "opendde.data.template"):
            monkeypatch.setitem(sys.modules, name, ModuleType(name))
        monkeypatch.setitem(sys.modules, "runner.batch_inference", self.batch)
        monkeypatch.setitem(sys.modules, "runner.inference", SimpleNamespace(infer_predict=self._infer_predict))
        monkeypatch.setitem(sys.modules, TEMPLATE_UTILS, utils)
        for name in ("cuequivariance_torch", "cuequivariance_ops_torch"):
            monkeypatch.setitem(sys.modules, name, ModuleType(name))

    def _get_default_runner(self, **kwargs: Any) -> Any:
        self.order.append("runner")
        self.kwargs = kwargs
        for attribute, key in (("triangle_attention", "triatt_kernel"), ("triangle_multiplicative", "trimul_kernel")):
            requested = kwargs[key]
            setattr(self.configs, attribute, self.auto_resolves_to if requested == "auto" else requested)
        return self.runner

    def _infer_predict(self, runner: Any, configs: Any) -> None:
        self.seen.append((copy.deepcopy(configs), runner.model.N_cycle))
        featurizer = self.featurizer()
        for query, _ in self.featurizer_calls:
            featurizer.get_templates("uid", query, [])
        configs.mutated = True


class KernelsRefused(SystemExit):
    """The shape of ``opendde_opt.lncensus.KernelsRefused``."""

    def __init__(self, problems: list[str]) -> None:
        super().__init__(5)
        self.problems = problems


class FakeKit:
    """A fake ``opendde_opt`` and the submodules the runtime reads."""

    def __init__(self, monkeypatch, fake: Fake) -> None:
        self.fake = fake
        self.report = dict(HEALTHY_REPORT)
        self.refreshed: dict[str, Any] = dict(HEALTHY_REPORT)
        self.module = ModuleType("opendde_opt")
        self.module.enable = self.enable  # type: ignore[attr-defined]
        self.module.status = lambda: dict(self.report)  # type: ignore[attr-defined]
        self.enable_calls: list[tuple[str, bool]] = []
        self.lncensus = ModuleType("opendde_opt.lncensus")
        self.lncensus.census = Mock(side_effect=self._census)  # type: ignore[attr-defined]
        self.lncensus.arm = Mock(side_effect=self._arm)  # type: ignore[attr-defined]
        self.lncensus.enforce = Mock()  # type: ignore[attr-defined]
        self.lncensus.record = Mock(return_value={"line": "kernels: cueq triatt+trimul served"})  # type: ignore[attr-defined]
        self.modes = ModuleType("opendde_opt.modes")
        self.modes.resolve = Mock(return_value=SimpleNamespace(line="a100", mode="exact"))  # type: ignore[attr-defined]
        self.modes.kernel_expectations = Mock(return_value={"triatt": "cuequivariance"})  # type: ignore[attr-defined]
        self.modes.route_word = Mock(return_value="kit")  # type: ignore[attr-defined]
        self.stack = ModuleType("opendde_opt.stack")
        self.stack.tree_root = Mock(return_value="/kit")  # type: ignore[attr-defined]
        self.stack.refresh = Mock(side_effect=lambda predicted, facts: dict(self.refreshed))  # type: ignore[attr-defined]
        self.settings = ModuleType("opendde_opt.settings")
        self.settings.ln_requested = Mock(return_value=True)  # type: ignore[attr-defined]
        self.alloc = ModuleType("opendde_opt.alloc")
        self.alloc.token_floor = Mock(return_value=256)  # type: ignore[attr-defined]
        self.inputs = ModuleType("opendde_opt.inputs")
        self.inputs.load_query = Mock(side_effect=lambda path: {"path": path})  # type: ignore[attr-defined]
        self.lnstream = ModuleType("opendde_opt.lnstream")
        self.lnstream.STATS = {"state": "serving", "reason": ""}  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "opendde_opt", self.module)
        for name in ("lncensus", "modes", "stack", "settings", "alloc", "inputs", "lnstream"):
            monkeypatch.setitem(sys.modules, f"opendde_opt.{name}", getattr(self, name))
        monkeypatch.setenv("LAYERNORM_TYPE", "fast_layernorm")

    def enable(self, mode: str, strict: bool) -> dict[str, Any]:
        self.fake.order.append("enable")
        self.enable_calls.append((mode, strict))
        # As the kit does on activation, unless the caller already chose a LayerNorm.
        import os

        os.environ.setdefault("LAYERNORM_TYPE", "fast_layernorm")
        return dict(self.report)

    def _census(self, strict: bool, stream: Any) -> dict[str, Any]:
        self.fake.order.append("census")
        return {"backend": "fast_layernorm_v2", "loaded": True}

    def _arm(self, route: str, expected: Any, **kwargs: Any) -> None:
        self.fake.order.append("arm")


@pytest.fixture
def fake(monkeypatch, tmp_path) -> Fake:
    return Fake(monkeypatch, tmp_path)


def _staging(tmp_path: Path, query: str = "QUERYSEQ") -> dict[str, Any]:
    return {
        "templates_path": str(tmp_path / "hits.a3m"),
        "mmcif_dir": "/stage/mmcif",
        "release_dates_path": "/stage/d.json",
        "obsolete_pdbs_path": "/stage/o.json",
        "cache_dir": str(tmp_path / "template_cache"),
        "query": query,
        "count": 1,
        "names": ["mine"],
    }


def test_runtime_loads_once_and_resets_request_state(rt, fake, tmp_path) -> None:
    """Changed seeds, sampling and guidance never leak between requests; weights load once."""
    config = _config()
    runtime = rt.OpenDDERuntime(config, str(tmp_path))
    kwargs = fake.batch.get_default_runner.call_args.kwargs
    assert kwargs["model_name"] == "opendde_v1" and kwargs["dump_dir"] == str(tmp_path)
    payload = runtime.predict(
        "a.json", "a", {**config, "seeds": "2,10", "cycle": 4, "sample": 2, "step": 50, "use_tfg_guidance": True}
    )
    runtime.predict("b.json", "b", config)
    fake.batch.get_default_runner.assert_called_once()
    (first, first_cycle), (second, second_cycle) = fake.seen
    assert first.seeds == [2, 10] and second.seeds == [101]
    assert first_cycle == 4 and second_cycle == 10
    assert first.sample_diffusion.guidance["enable"] is True and second.sample_diffusion.guidance["enable"] is False
    assert first.dump_dir == "a" and second.dump_dir == "b" and not second.mutated
    assert first.num_workers == 0
    assert all(call.kwargs["msa_server_mode"] == "colabfold" for call in fake.batch.preprocess_input.call_args_list)
    assert payload["kernel.resolved.triangle_attention"] == "cuequivariance"
    info = runtime.describe()
    assert info["runtime"] == "OpenDDE" and info["layernorm_type"] == "torch"
    assert info["kernel.requested.triangle_attention"] == "cuequivariance"  # the core's default
    assert info["kernel.resolved.triangle_multiplicative"] == "cuequivariance"
    assert "opendde" in info and all(isinstance(value, str) for value in info.values())


def test_runtime_reports_err_files(rt, fake, tmp_path) -> None:
    """Upstream swallows some failures into ERR files; they must surface as errors."""
    runtime = rt.OpenDDERuntime(_config(), str(tmp_path))
    Path(fake.runner.error_dir).mkdir()
    (Path(fake.runner.error_dir) / "target.txt").write_text("CUDA out of memory")
    with pytest.raises(RuntimeError, match="(?s)OpenDDE inference failed.*CUDA out of memory"):
        runtime.predict("c.json", "c", _config())


def test_staged_templates_get_a_per_request_cache(rt, fake, monkeypatch, tmp_path) -> None:
    """OpenDDE's parse cache is keyed by the synthetic template id, so a shared one would serve stale templates."""
    monkeypatch.setattr(rt, "_kalign_path", lambda: "/usr/bin/kalign")
    runtime = rt.OpenDDERuntime(_config(), str(tmp_path))
    fake.featurizer_calls = [("QUERYSEQ", 1)]
    payload = runtime.predict("a.json", "a", {**_config(), "template_staging": _staging(tmp_path / "one")})
    runtime.predict("b.json", "b", {**_config(), "template_staging": _staging(tmp_path / "two")})
    fake.featurizer_calls = []
    runtime.predict("c.json", "c", _config())
    (one, _), (two, _), (plain, _) = fake.seen
    assert one.data.template.prot_template_cache_dir == str(tmp_path / "one" / "template_cache")
    assert two.data.template.prot_template_cache_dir == str(tmp_path / "two" / "template_cache")
    assert one.data.template.fetch_remote is False and one.data.template.prot_template_mmcif_dir == "/stage/mmcif"
    assert plain.data.template.prot_template_cache_dir == "/data/shared_cache" and not plain.use_template
    assert payload["templates.featurized"] == "1"


def test_dropped_staged_template_fails_the_request(rt, fake, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(rt, "_kalign_path", lambda: "/usr/bin/kalign")
    runtime = rt.OpenDDERuntime(_config(), str(tmp_path))
    fake.featurizer_calls = [("QUERYSEQ", 0)]
    with pytest.raises(RuntimeError, match="OpenDDE featurized 0 of 1 staged template"):
        runtime.predict("a.json", "a", {**_config(), "template_staging": _staging(tmp_path)})


def _fake_cuda(monkeypatch, fake: "Fake", major: int) -> None:
    """Put the fake runner on a CUDA device of compute capability ``major``.0 (a fake ``torch`` reports it)."""
    cuda = SimpleNamespace(is_available=lambda: True, get_device_capability=lambda device=None: (major, 0))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    fake.runner.device = SimpleNamespace(type="cuda", index=0)


@pytest.mark.parametrize("major", [None, 8, 9])
def test_auto_kernel_resolving_to_torch_refuses(rt, fake, monkeypatch, tmp_path, major) -> None:
    """``auto`` silently picks the torch kernels on a card without cuEquivariance; that is refused, not served."""
    if major is not None:
        _fake_cuda(monkeypatch, fake, major)
    fake.auto_resolves_to = "torch"
    with pytest.raises(
        rt.OptimizationUnavailableError, match="triangle_attention resolved to 'torch' where 'cuequivariance'"
    ):
        rt.OpenDDERuntime(_config(triatt_kernel="auto", trimul_kernel="auto"), str(tmp_path))


@pytest.mark.parametrize("requested", ["cuequivariance", "auto"])
def test_vanilla_serves_upstreams_cc7_fallback_and_records_it(
    rt, fake, monkeypatch, tmp_path, capsys, requested
) -> None:
    """On a V100/T4 upstream runs the torch kernels in fp32 whatever was asked; vanilla serves that, visibly."""
    _fake_cuda(monkeypatch, fake, 7)
    fake.auto_resolves_to = "torch"
    # apply_runtime_compatibility overrides even an explicit cuEquivariance request on compute capability 7.x.
    fake.batch.get_default_runner.side_effect = lambda **kwargs: _cc7_runner(fake, **kwargs)
    runtime = rt.OpenDDERuntime(_config(triatt_kernel=requested, trimul_kernel=requested), str(tmp_path))
    info = runtime.describe()
    assert info["kernel.cc7_fallback"] == "true"
    assert info["kernel.resolved.triangle_attention"] == "torch"
    assert "compute-capability-7.x fallback" in capsys.readouterr().err
    # A bf16 request must not undo upstream's fp32 on these cards.
    runtime.predict("a.json", "a", {**_config(), "dtype": "bf16"})
    assert fake.seen[-1][0].dtype == "fp32"


def _cc7_runner(fake: "Fake", **kwargs: Any) -> Any:
    runner = fake._get_default_runner(**kwargs)
    fake.configs.triangle_attention = fake.configs.triangle_multiplicative = "torch"
    fake.configs.dtype = "fp32"
    return runner


def test_healthy_ampere_vanilla_records_no_fallback_and_keeps_the_dtype(rt, fake, monkeypatch, tmp_path) -> None:
    _fake_cuda(monkeypatch, fake, 8)
    runtime = rt.OpenDDERuntime(_config(), str(tmp_path))
    assert runtime.describe()["kernel.cc7_fallback"] == "false"
    runtime.predict("a.json", "a", {**_config(), "dtype": "bf16"})
    assert fake.seen[-1][0].dtype == "bf16"


def test_kit_mode_on_cc7_refuses_the_torch_kernels(rt, fake, monkeypatch, tmp_path) -> None:
    """The fallback is stock OpenDDE's policy only: a kit mode that lands on the torch kernels refuses."""
    FakeKit(monkeypatch, fake)
    _fake_cuda(monkeypatch, fake, 7)
    fake.batch.get_default_runner.side_effect = lambda **kwargs: _cc7_runner(fake, **kwargs)
    with pytest.raises(rt.OptimizationUnavailableError, match="resolved to 'torch' where 'cuequivariance'"):
        rt.OpenDDERuntime(_config(optimization="exact"), str(tmp_path))


def test_torch_kernels_by_name_run_in_vanilla(rt, fake, tmp_path) -> None:
    runtime = rt.OpenDDERuntime(_config(triatt_kernel="torch", trimul_kernel="torch"), str(tmp_path))
    assert runtime.describe()["kernel.resolved.triangle_attention"] == "torch"
    assert runtime.describe()["kernel.cc7_fallback"] == "false"


def test_runtime_leaves_path_to_the_core(rt, fake, monkeypatch, tmp_path) -> None:
    """The core's worker environment alone puts the venv's ``bin`` (its ninja) on PATH; the runtime does not edit it."""
    monkeypatch.setattr(rt.sys, "executable", str(tmp_path / "opendde" / "bin" / "python"))
    monkeypatch.setenv("PATH", "/venv/bin:/usr/bin")
    rt.OpenDDERuntime(_config(), str(tmp_path))
    assert os.environ["PATH"] == "/venv/bin:/usr/bin"


def test_vanilla_with_the_fused_layernorm_refuses(rt, fake, monkeypatch, tmp_path) -> None:
    """Vanilla is stock OpenDDE (torch LayerNorm); a worker started with fast_layernorm was not started by the core."""
    monkeypatch.setenv("LAYERNORM_TYPE", "fast_layernorm")
    with pytest.raises(rt.OptimizationUnavailableError, match="optimization='vanilla' runs LAYERNORM_TYPE='torch'"):
        rt.OpenDDERuntime(_config(), str(tmp_path))
    assert "runner" not in fake.order


def _fake_jit_cache(monkeypatch, key: str = "torch2.7.1-cu126-sm90") -> list[str]:
    calls: list[str] = []
    package = ModuleType("opt_core")
    module = ModuleType("opt_core.jit_cache")

    def key_facts() -> dict[str, Any]:
        calls.append("key_facts")
        return {"key": key, "parts": {}, "unknown": [], "word": key}

    module.key_facts = key_facts  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "opt_core", package)
    monkeypatch.setitem(sys.modules, "opt_core.jit_cache", module)
    return calls


def test_jit_caches_are_keyed_by_the_stack(rt, fake, monkeypatch, tmp_path) -> None:
    """Triton and torch-extension builds land under ``<root>/<stack key>/``, as the kit's configs/<gpu>.env has them."""
    _fake_jit_cache(monkeypatch)
    root = tmp_path / "jit"
    monkeypatch.setenv("MODEL_OPT_JIT_ROOT", str(root))
    runtime = rt.OpenDDERuntime(_config(), str(tmp_path))
    assert os.environ["MODEL_OPT_STACK_KEY"] == "torch2.7.1-cu126-sm90"
    assert os.environ["TRITON_CACHE_DIR"] == str(root / "torch2.7.1-cu126-sm90" / "triton")
    assert os.environ["TORCH_EXTENSIONS_DIR"] == str(root / "torch2.7.1-cu126-sm90" / "torch_ext")
    assert runtime.describe()["jit.stack_key"] == "torch2.7.1-cu126-sm90"


def test_jit_caches_the_caller_set_are_kept(rt, fake, monkeypatch, tmp_path) -> None:
    calls = _fake_jit_cache(monkeypatch)
    monkeypatch.setenv("MODEL_OPT_JIT_ROOT", str(tmp_path / "jit"))
    monkeypatch.setenv("MODEL_OPT_STACK_KEY", "pinned")
    monkeypatch.setenv("TRITON_CACHE_DIR", "/scratch/triton")
    rt.OpenDDERuntime(_config(), str(tmp_path))
    assert calls == []
    assert os.environ["TRITON_CACHE_DIR"] == "/scratch/triton"
    assert os.environ["TORCH_EXTENSIONS_DIR"] == str(tmp_path / "jit" / "pinned" / "torch_ext")


def test_jit_caches_without_the_kit_core_stay_unset(rt, fake, monkeypatch, tmp_path) -> None:
    """No opt_core to key them: torch's own defaults, never an un-keyed shared directory."""
    monkeypatch.setitem(sys.modules, "opt_core.jit_cache", None)
    monkeypatch.setenv("MODEL_OPT_JIT_ROOT", str(tmp_path / "jit"))
    runtime = rt.OpenDDERuntime(_config(), str(tmp_path))
    assert not {"TRITON_CACHE_DIR", "TORCH_EXTENSIONS_DIR", "MODEL_OPT_STACK_KEY"} & os.environ.keys()
    assert runtime.describe()["jit.stack_key"] == "absent"


def test_jit_caches_with_an_unreadable_stack_stay_unset(rt, fake, monkeypatch, tmp_path) -> None:
    _fake_jit_cache(monkeypatch)
    sys.modules["opt_core.jit_cache"].key_facts = Mock(side_effect=OSError("nvidia-smi hung"))  # type: ignore[attr-defined]
    monkeypatch.setenv("MODEL_OPT_JIT_ROOT", str(tmp_path / "jit"))
    runtime = rt.OpenDDERuntime(_config(), str(tmp_path))
    assert not {"TRITON_CACHE_DIR", "TORCH_EXTENSIONS_DIR"} & os.environ.keys()
    assert runtime.describe()["jit.stack_key"] == "unknown"


def test_jit_caches_without_a_root_are_untouched(rt, fake, monkeypatch, tmp_path) -> None:
    calls = _fake_jit_cache(monkeypatch)
    runtime = rt.OpenDDERuntime(_config(), str(tmp_path))
    assert calls == [] and "TRITON_CACHE_DIR" not in os.environ
    assert runtime.describe()["jit.stack_key"] == "none"


def test_layernorm_type_opendde_does_not_read_refuses(rt, fake, monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("LAYERNORM_TYPE", "openfold")
    with pytest.raises(
        rt.OptimizationUnavailableError, match="'vanilla' runs LAYERNORM_TYPE='torch'.* with 'openfold'"
    ):
        rt.OpenDDERuntime(_config(), str(tmp_path))


def test_unset_layernorm_type_refuses_even_where_upstream_defaults_to_the_table_value(
    rt, fake, monkeypatch, tmp_path
) -> None:
    """The core always sets it; unset means a worker started outside the core, so it is refused, not assumed."""
    monkeypatch.delenv("LAYERNORM_TYPE", raising=False)
    with pytest.raises(rt.OptimizationUnavailableError, match="'vanilla' runs LAYERNORM_TYPE='torch'.* with it unset"):
        rt.OpenDDERuntime(_config(), str(tmp_path))
    assert "runner" not in fake.order


@pytest.mark.parametrize("mode", ["exact", "fast"])
def test_kit_activates_and_arms_its_census_before_the_runner_loads(rt, fake, monkeypatch, tmp_path, mode) -> None:
    """The kit refuses late activation, and its kernel census wraps the runner's init, so both come first."""
    kit = FakeKit(monkeypatch, fake)
    kit.report = {**HEALTHY_REPORT, "mode": mode}
    runtime = rt.OpenDDERuntime(_config(optimization=mode), str(tmp_path))
    assert fake.order == ["enable", "census", "arm", "runner"]
    assert kit.enable_calls == [(mode, False)]
    kit.modes.resolve.assert_called_once()
    assert kit.modes.resolve.call_args.args[:2] == (mode, "/kit")
    kit.modes.kernel_expectations.assert_called_once_with("a100", ln_requested=True, n_gpu=1, knobs={})
    arm = kit.lncensus.arm.call_args
    assert arm.args == ("kit", {"triatt": "cuequivariance"})
    assert arm.kwargs["strict"] is True and arm.kwargs["layernorm"]["backend"] == "fast_layernorm_v2"
    # The kit's LayerNorm export, set during activation, is what the check reads.
    info = runtime.describe()
    assert info["layernorm_type"] == "fast_layernorm" and info["layernorm.backend"] == "fast_layernorm_v2"
    assert info["kit.active"] == "true" and info["kit.mode"] == mode and info["kit.line"] == "a100"
    assert info["kit.levers_not_in_arm"] == "fp8_trimul" and info["kit.commit"] == "unknown"


def test_kit_late_checks_run_after_each_prediction(rt, fake, monkeypatch, tmp_path) -> None:
    kit = FakeKit(monkeypatch, fake)
    runtime = rt.OpenDDERuntime(_config(optimization="exact"), str(tmp_path))
    payload = runtime.predict("a.json", "a", _config(optimization="exact"))
    kit.lncensus.enforce.assert_called_once_with("predict")
    refresh = kit.stack.refresh.call_args.kwargs
    assert refresh["predicted"] is True
    assert refresh["facts"] == {"det": False, "n_gpu": 1, "rank": 0, "token_floors": 256, "stock_knobs": {}}
    kit.inputs.load_query.assert_called_once_with("a.json")
    assert payload["kit.lnstream"] == "serving" and payload["kit.kernels"] == "kernels: cueq triatt+trimul served"
    assert payload["kit.partial"] == "false"


def test_kit_named_gaps_land_in_the_payload(rt, fake, monkeypatch, tmp_path) -> None:
    """refresh(predicted=True) counts a lever with no readable counter as applied and names the gap; keep it."""
    kit = FakeKit(monkeypatch, fake)
    kit.refreshed = {
        **HEALTHY_REPORT,
        "levers_uncounted": {"cueq_triatt": "no install record and no readable counter: a named gap"},
        "levers_asides": {"fpf_trimul": "planes exceed free memory"},
        "levers_inert_reasons": {"tmpl_dedup": "no templates"},
        "lever_buckets": {"lnstream": "applied", "cueq_triatt": "uncounted"},
    }
    runtime = rt.OpenDDERuntime(_config(optimization="exact"), str(tmp_path))
    payload = runtime.predict("a.json", "a", _config(optimization="exact"))
    assert payload["kit.levers_uncounted"] == '{"cueq_triatt":"no install record and no readable counter: a named gap"}'
    assert payload["kit.levers_asides"] == '{"fpf_trimul":"planes exceed free memory"}'
    assert payload["kit.levers_inert_reasons"] == '{"tmpl_dedup":"no templates"}'
    assert payload["kit.lever_buckets"] == '{"cueq_triatt":"uncounted","lnstream":"applied"}'


@pytest.mark.parametrize(
    ("refreshed", "stats", "message"),
    [
        ({**HEALTHY_REPORT, "partial": True}, {"state": "serving"}, "is partial after the prediction"),
        (
            {**HEALTHY_REPORT, "levers_fallback": ["cueq_triatt"], "fallback_reasons": {"cueq_triatt": "raised"}},
            {"state": "serving"},
            r"levers_fallback=\['cueq_triatt'\]",
        ),
        (dict(HEALTHY_REPORT), {"state": "fallback", "reason": "stream sync"}, "LayerNorm is 'fallback' .stream sync"),
        (dict(HEALTHY_REPORT), {"state": "armed"}, "LayerNorm is 'armed'"),
    ],
    ids=["partial", "fallback", "lnstream-fallback", "lnstream-never-served"],
)
def test_kit_degradation_during_the_prediction_refuses(
    rt, fake, monkeypatch, tmp_path, refreshed, stats, message
) -> None:
    kit = FakeKit(monkeypatch, fake)
    runtime = rt.OpenDDERuntime(_config(optimization="exact"), str(tmp_path))
    kit.refreshed = refreshed
    kit.lnstream.STATS = stats  # type: ignore[attr-defined]
    with pytest.raises(rt.OptimizationUnavailableError, match=message):
        runtime.predict("a.json", "a", _config(optimization="exact"))


def test_kernel_census_exit_5_becomes_a_refusal(rt, fake, monkeypatch, tmp_path) -> None:
    kit = FakeKit(monkeypatch, fake)

    def refuse(route: str, expected: Any, **kwargs: Any) -> None:
        raise KernelsRefused(["triangle_attention: cuequivariance absent"])

    kit.lncensus.arm.side_effect = refuse
    with pytest.raises(rt.OptimizationUnavailableError, match=r"code 5 .*cuequivariance absent"):
        rt.OpenDDERuntime(_config(optimization="exact"), str(tmp_path))
    assert "runner" not in fake.order


def test_census_at_predict_refuses(rt, fake, monkeypatch, tmp_path) -> None:
    kit = FakeKit(monkeypatch, fake)
    runtime = rt.OpenDDERuntime(_config(optimization="exact"), str(tmp_path))
    kit.lncensus.enforce.side_effect = KernelsRefused(["trimul: reference path served 12 calls"])
    with pytest.raises(rt.OptimizationUnavailableError, match="reference path served"):
        runtime.predict("a.json", "a", _config(optimization="exact"))


def test_layernorm_census_not_loaded_refuses(rt, fake, monkeypatch, tmp_path) -> None:
    kit = FakeKit(monkeypatch, fake)

    class NotLoaded(Exception):
        pass

    kit.lncensus.census.side_effect = NotLoaded("fused LayerNorm module did not load")
    with pytest.raises(rt.OptimizationUnavailableError, match="LayerNorm census refused: fused LayerNorm"):
        rt.OpenDDERuntime(_config(optimization="fast"), str(tmp_path))


def test_kit_mode_with_a_stock_kernel_refuses(rt, fake, monkeypatch, tmp_path) -> None:
    """A stated torch kernel would make the kit step its levers aside without telling anyone."""
    FakeKit(monkeypatch, fake)
    with pytest.raises(rt.OptimizationUnavailableError, match="trimul_kernel='torch' needs optimization='vanilla'"):
        rt.OpenDDERuntime(_config(optimization="exact", trimul_kernel="torch"), str(tmp_path))
    assert "runner" not in fake.order


def test_kit_mode_with_torch_layernorm_refuses(rt, fake, monkeypatch, tmp_path) -> None:
    FakeKit(monkeypatch, fake)
    monkeypatch.setenv("LAYERNORM_TYPE", "torch")
    with pytest.raises(rt.OptimizationUnavailableError, match="'exact' runs LAYERNORM_TYPE='fast_layernorm'"):
        rt.OpenDDERuntime(_config(optimization="exact"), str(tmp_path))


def test_kit_enable_exit_3_becomes_a_refusal(rt, fake, monkeypatch, tmp_path) -> None:
    kit = FakeKit(monkeypatch, fake)

    def refuse(mode: str, strict: bool) -> Any:
        raise SystemExit(3)

    kit.module.enable = refuse  # type: ignore[attr-defined]
    with pytest.raises(rt.OptimizationUnavailableError, match="opendde_opt.enable.'exact'. exited with code 3"):
        rt.OpenDDERuntime(_config(optimization="exact"), str(tmp_path))


@pytest.mark.parametrize("code", [1, 2, 128 + 15])
def test_kit_enable_failure_exit_is_a_failure_not_a_refusal(rt, fake, monkeypatch, tmp_path, code) -> None:
    """Only exit 3 and 5 are refusals; a crash or usage error must not become a standing refusal of the mode."""
    kit = FakeKit(monkeypatch, fake)

    def crash(mode: str, strict: bool) -> Any:
        raise SystemExit(code)

    kit.module.enable = crash  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError, match=f"exited with code {code} .a failure, not a refusal") as error:
        rt.OpenDDERuntime(_config(optimization="exact"), str(tmp_path))
    assert not isinstance(error.value, rt.OptimizationUnavailableError)


def test_kit_refusal_subclass_is_a_refusal(rt, fake, monkeypatch, tmp_path) -> None:
    """OpenDDE's BigRefusal and SmallFloorRefusal subclass OpenModeError; they refuse like their base."""
    kit = FakeKit(monkeypatch, fake)
    open_mode_error = type("OpenModeError", (RuntimeError,), {})
    big_refusal = type("BigRefusal", (open_mode_error,), {})

    def refuse(mode: str, strict: bool) -> Any:
        raise big_refusal("2400 tokens exceed the fast arm's floor")

    kit.module.enable = refuse  # type: ignore[attr-defined]
    with pytest.raises(rt.OptimizationUnavailableError, match="refused: 2400 tokens exceed"):
        rt.OpenDDERuntime(_config(optimization="fast"), str(tmp_path))


def test_kit_mode_without_kit_image_refuses_by_name(rt, fake, monkeypatch, tmp_path) -> None:
    """A vanilla-only image must say so instead of silently running unoptimized."""
    monkeypatch.setitem(sys.modules, "opendde_opt", None)
    with pytest.raises(rt.OptimizationUnavailableError, match="needs the OpenDDE kit image"):
        rt.OpenDDERuntime(_config(optimization="fast"), str(tmp_path))


def test_kit_inactive_report_refuses(rt, fake, monkeypatch, tmp_path) -> None:
    kit = FakeKit(monkeypatch, fake)
    kit.report = {"active": False, "reason": "no gpu", "exit_code": 3}
    with pytest.raises(rt.OptimizationUnavailableError, match="not active at activation: no gpu") as error:
        rt.OpenDDERuntime(_config(optimization="exact"), str(tmp_path))
    assert type(error.value).__name__ == "OptimizationUnavailableError"
    assert fake.order == ["enable"]
