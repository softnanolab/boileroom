"""Protenix worker-runtime contract tests, CPU-only, with fake runner and kit modules.

The runtime runs in the model image's own interpreter (``runpy.run_path``), so these tests load it by path, as
the worker does, instead of importing it through ``boileroom``.
"""

import ast
import copy
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

RUNTIME_PATH = Path(__file__).resolve().parents[2] / "boileroom" / "models" / "protenix" / "runtime.py"
TEMPLATE_UTILS = "protenix.data.template.template_utils"
#: The kit's report on a healthy A100 exact run: levers outside this card's arm are listed, nothing fell back.
A100_EXACT_REPORT = {
    "active": True,
    "mode": "exact",
    "partial": False,
    "levers_applied": ["lnstream", "cueq_tri"],
    "levers_fallback": [],
    "levers_unavailable": [],
    "levers_not_in_arm": ["pad8", "dit_attn_exact", "atom_attn_exact", "triatt_prologue_cuda"],
    "fallback_reasons": {},
    "kit_version": "2.0.0+kit",
}


def _config(**overrides: Any) -> dict[str, Any]:
    from boileroom.models.protenix.core import ProtenixCore

    return {**ProtenixCore.DEFAULT_CONFIG, **overrides}


@pytest.fixture
def rt() -> Any:
    """The runtime module, loaded from its file as the worker child does."""
    spec = importlib.util.spec_from_file_location("boileroom_protenix_runtime_test", RUNTIME_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch) -> None:
    # setenv then delenv registers an undo, so a value the runtime sets is removed after the test.
    for name in ("LAYERNORM_TYPE", "PTX_LEVER_REPORT", "BOILEROOM_KIT_COMMIT"):
        monkeypatch.setenv(name, "x")
        monkeypatch.delenv(name)
    # The core starts every worker with its mode's LayerNorm (vanilla: openfold); FakeKit sets the kit modes' value.
    monkeypatch.setenv("LAYERNORM_TYPE", "openfold")


class Fake:
    """The fake upstream: runner modules, template featurizer, and what each prediction saw."""

    def __init__(self, monkeypatch, tmp_path: Path) -> None:
        template = SimpleNamespace(
            prot_template_mmcif_dir="/data/mmcif",
            release_dates_path="/data/dates.json",
            obsolete_pdbs_path="/data/obsolete.json",
            prot_template_cache_dir="/data/cache",
            fetch_remote=True,
            kalign_binary_path="",
        )
        self.configs = SimpleNamespace(
            model=SimpleNamespace(N_cycle=10),
            sample_diffusion=SimpleNamespace(N_step=200, N_sample=5, guidance=SimpleNamespace(enable=False)),
            sorted_by_ranking_score=True,
            data=SimpleNamespace(template=template),
            use_template=False,
            num_workers=16,
            triangle_attention="cuequivariance",
            triangle_multiplicative="cuequivariance",
            dtype="bf16",
            mutated=False,
        )
        self.runner = SimpleNamespace(
            configs=self.configs, model=SimpleNamespace(N_cycle=10), error_dir=str(tmp_path / "ERR")
        )
        self.runner.update_model_configs = Mock()
        self.runner.init_basics = Mock()
        self.runner.init_dumper = Mock()
        self.order: list[str] = []
        self.seen: list[tuple[Any, int]] = []
        #: (query sequence, features returned) per featurizer call each prediction makes.
        self.featurizer_calls: list[tuple[str, int]] = []
        self.batch = SimpleNamespace(
            get_default_runner=Mock(side_effect=self._get_default_runner),
            inference_configs={},
            init_logging=Mock(),
            preprocess_input=Mock(side_effect=lambda path, **kwargs: path),
        )
        fake = self

        class TemplateHitFeaturizer:
            def get_templates(self, sequence_uid, query_sequence, hits, max_template_date=None):
                count = dict(fake.featurizer_calls).get(query_sequence, 0)
                errors = [] if count else [f"{sequence_uid}: dropped"]
                return SimpleNamespace(features=[{}] * count, hits=[], errors=errors, warnings=[]), None

        self.featurizer = TemplateHitFeaturizer
        utils = ModuleType(TEMPLATE_UTILS)
        utils.TemplateHitFeaturizer = TemplateHitFeaturizer  # type: ignore[attr-defined]
        for name in ("runner", "protenix", "protenix.data", "protenix.data.template"):
            monkeypatch.setitem(sys.modules, name, ModuleType(name))
        monkeypatch.setitem(sys.modules, "runner.batch_inference", self.batch)
        monkeypatch.setitem(sys.modules, "runner.inference", SimpleNamespace(infer_predict=self._infer_predict))
        monkeypatch.setitem(sys.modules, TEMPLATE_UTILS, utils)
        for name in ("cuequivariance_torch", "cuequivariance_ops_torch"):
            module = ModuleType(name)
            monkeypatch.setitem(sys.modules, name, module)

    def _get_default_runner(self, **kwargs: Any) -> Any:
        self.order.append("runner")
        return self.runner

    def _infer_predict(self, runner: Any, configs: Any) -> None:
        self.seen.append((copy.deepcopy(configs), runner.model.N_cycle))
        featurizer = self.featurizer()
        for query, _ in self.featurizer_calls:
            featurizer.get_templates("uid", query, [], max_template_date="2021-09-30")
        configs.mutated = True


class FakeKit:
    """A fake ``protenix_opt`` with its ``stack`` and the forward kit's BLK2 counters."""

    def __init__(self, monkeypatch, fake: Fake, report: dict[str, Any] | None = None) -> None:
        self.fake = fake
        self.report = dict(A100_EXACT_REPORT if report is None else report)
        self.status_report: dict[str, Any] | None = None
        self.late: dict[str, Any] = {}
        self.enable_calls: list[tuple[str, bool]] = []
        self.module = ModuleType("protenix_opt")
        self.module.enable = self.enable  # type: ignore[attr-defined]
        self.module.status = lambda: dict(self.status_report or self.report)  # type: ignore[attr-defined]
        self.stack = ModuleType("protenix_opt.stack")
        self.stack.late_records = Mock(return_value=[{"lever": "cueq_tri"}])  # type: ignore[attr-defined]
        self.stack.reconcile = Mock(side_effect=lambda rep, records: {**rep, **self.late})  # type: ignore[attr-defined]
        # The kit's lever registry (protenix_opt/report.py IMPL): lever -> (implementing source, origin).
        self.registry = ModuleType("protenix_opt.report")
        self.registry.IMPL = {  # type: ignore[attr-defined]
            "nomask": ("forward/flashpairformer/src/ptx_trunk2_levers.py", "kit"),
            "cueq_tri": ("opt_core/kernels/trimul", "core"),
        }
        monkeypatch.setitem(sys.modules, "protenix_opt.report", self.registry)
        self.blk2 = ModuleType("ptx_trunk2_levers")
        self.blk2._STATS = {"blk2_tri_stock_fallback": 0, "portability_lines": []}  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "protenix_opt", self.module)
        monkeypatch.setitem(sys.modules, "protenix_opt.stack", self.stack)
        monkeypatch.setitem(sys.modules, "ptx_trunk2_levers", self.blk2)
        monkeypatch.setenv("LAYERNORM_TYPE", "fast_layernorm")

    def enable(self, mode: str, strict: bool) -> dict[str, Any]:
        self.fake.order.append("enable")
        self.enable_calls.append((mode, strict))
        return dict(self.report)


@pytest.fixture
def fake(monkeypatch, tmp_path) -> Fake:
    return Fake(monkeypatch, tmp_path)


def _staging(tmp_path: Path, query: str = "QUERYSEQ", count: int = 1) -> dict[str, Any]:
    return {
        "templates_path": str(tmp_path / "hits.a3m"),
        "mmcif_dir": "/stage/mmcif",
        "release_dates_path": "/stage/d.json",
        "obsolete_pdbs_path": "/stage/o.json",
        "cache_dir": "/stage/template_cache",
        "query": query,
        "count": count,
        "names": ["mine"],
    }


def test_runtime_reuses_weights_and_resets_request_state(rt, fake, tmp_path) -> None:
    """Changed seeds, lengths and sampling settings must not reuse stale state."""
    config = _config()
    runtime = rt.ProtenixRuntime(config, str(tmp_path))
    first_payload = runtime.predict(
        "first.json", "first", {**config, "seeds": "2,10", "cycle": 4, "sample": 2, "step": 50}
    )
    runtime.predict("second.json", "second", config)
    fake.batch.get_default_runner.assert_called_once()
    assert fake.batch.inference_configs["dump_dir"] == str(tmp_path)
    assert all(call.kwargs["msa_server_mode"] == "colabfold" for call in fake.batch.preprocess_input.call_args_list)
    (first, first_cycle), (second, second_cycle) = fake.seen
    assert first.seeds == [2, 10] and second.seeds == [101]
    assert first_cycle == 4 and second_cycle == 10
    assert first.sample_diffusion.N_sample == 2 and second.sample_diffusion.N_sample == 5
    assert first.sample_diffusion.N_step == 50 and second.sample_diffusion.N_step == 200
    assert first.input_json_path == "first.json" and second.input_json_path == "second.json"
    assert first.dump_dir == "first" and second.dump_dir == "second"
    assert not second.mutated
    # Featurization stays in this process so the template count sees every call.
    assert first.num_workers == 0
    assert fake.runner.init_basics.call_count == fake.runner.init_dumper.call_count == 2
    assert first_payload == {
        "kernel.resolved.triangle_attention": "cuequivariance",
        "kernel.resolved.triangle_multiplicative": "cuequivariance",
        "kernel.resolved.dtype": "bf16",
    }


def test_runtime_reports_err_files(rt, fake, tmp_path) -> None:
    """Upstream swallows some failures into ERR files; they must surface as errors."""
    runtime = rt.ProtenixRuntime(_config(), str(tmp_path))
    Path(fake.runner.error_dir).mkdir()
    (Path(fake.runner.error_dir) / "target.txt").write_text("CUDA out of memory")
    with pytest.raises(RuntimeError, match="(?s)Protenix inference failed.*CUDA out of memory"):
        runtime.predict("third.json", "third", _config())


def test_runtime_points_featurizer_at_staged_templates(rt, fake, monkeypatch, tmp_path) -> None:
    """Staged templates switch the featurizer to the caller's files for that request only."""
    monkeypatch.setattr(rt, "_kalign_path", lambda: "/usr/bin/kalign")
    config = _config()
    runtime = rt.ProtenixRuntime(config, str(tmp_path))
    fake.featurizer_calls = [("QUERYSEQ", 1), ("OTHERCHAIN", 0)]
    payload = runtime.predict("a.json", "a", {**config, "template_staging": _staging(tmp_path)})
    fake.featurizer_calls = []
    runtime.predict("b.json", "b", config)

    (staged, _), (plain, _) = fake.seen
    template = staged.data.template
    assert staged.use_template and template.prot_template_mmcif_dir == "/stage/mmcif"
    assert template.fetch_remote is False and template.kalign_binary_path == "/usr/bin/kalign"
    assert template.release_dates_path == "/stage/d.json" and template.obsolete_pdbs_path == "/stage/o.json"
    # Protenix reads a set cache directory without the pickle as "no template", silently: it is switched off.
    assert template.prot_template_cache_dir == ""
    assert fake.batch.preprocess_input.call_args_list[0].kwargs["use_template"] is True
    assert payload["templates.staged"] == payload["templates.featurized"] == "1"
    assert payload["templates.other_chains"] == "0"
    # The next request starts from the pristine copy, and the featurizer is restored.
    assert not plain.use_template and plain.data.template.prot_template_mmcif_dir == "/data/mmcif"
    assert plain.data.template.fetch_remote is True and plain.data.template.prot_template_cache_dir == "/data/cache"
    assert fake.featurizer.get_templates.__qualname__.endswith("TemplateHitFeaturizer.get_templates")


def test_staged_templates_cannot_be_combined_with_database_templates(rt, fake, tmp_path) -> None:
    runtime = rt.ProtenixRuntime(_config(), str(tmp_path))
    with pytest.raises(ValueError, match="use_template=True"):
        runtime.predict("a.json", "a", {**_config(use_template=True), "template_staging": _staging(tmp_path)})


@pytest.mark.parametrize(
    ("calls", "message"),
    [
        ([("QUERYSEQ", 0)], "featurized 0 of 1 staged template"),
        ([("QUERYSEQ", 1), ("OTHERCHAIN", 2)], "and 2 for other chains"),
        ([], r"\(0 featurizer call"),
    ],
    ids=["dropped", "other-chain-searched", "never-featurized"],
)
def test_featurized_template_count_must_match_the_staged_count(rt, fake, monkeypatch, tmp_path, calls, message) -> None:
    """A template the featurizer dropped (or one found for another chain) fails the request loudly."""
    monkeypatch.setattr(rt, "_kalign_path", lambda: "/usr/bin/kalign")
    runtime = rt.ProtenixRuntime(_config(), str(tmp_path))
    fake.featurizer_calls = calls
    with pytest.raises(RuntimeError, match=message) as error:
        runtime.predict("a.json", "a", {**_config(), "template_staging": _staging(tmp_path)})
    assert not isinstance(error.value, rt.OptimizationUnavailableError)
    if calls and calls[0][1] == 0:
        assert "uid: dropped" in str(error.value)


def test_describe_records_the_vanilla_stack(rt, fake, tmp_path) -> None:
    info = rt.ProtenixRuntime(_config(trimul_kernel="torch"), str(tmp_path)).describe()
    assert info["runtime"] == "Protenix" and info["optimization"] == "vanilla"
    assert info["layernorm_type"] == "openfold"
    assert info["kernel.requested.triangle_multiplicative"] == "torch"
    assert info["kernel.resolved.triangle_attention"] == "cuequivariance"
    assert info["kernel.resolved.dtype"] == "bf16"
    for key in ("python", "protenix", "torch", "cuda", "gpu", "cuequivariance_torch", "cuequivariance_ops_torch"):
        assert key in info
    assert not any(key.startswith("kit.") for key in info)
    assert all(isinstance(value, str) for value in info.values())


@pytest.mark.parametrize("mode", ["exact", "fast"])
def test_healthy_kit_mode_loads_and_records_its_report(rt, fake, monkeypatch, tmp_path, mode) -> None:
    """The kit activates before any upstream import; a report with levers outside the card's arm still runs."""
    kit = FakeKit(monkeypatch, fake, {**A100_EXACT_REPORT, "mode": mode})
    imported: list[str] = []
    real_import = rt.importlib.import_module

    def tracking_import(name: str, *args: Any) -> Any:
        if name.startswith("cuequivariance"):
            fake.order.append(name)
        imported.append(name)
        return real_import(name, *args)

    monkeypatch.setattr(rt.importlib, "import_module", tracking_import)
    runtime = rt.ProtenixRuntime(_config(optimization=mode), str(tmp_path))
    assert kit.enable_calls == [(mode, False)]
    assert fake.order == ["enable", "cuequivariance_torch", "cuequivariance_ops_torch", "runner"]
    # The lever report lives with the runtime's own files, as the kit's CLI places it.
    assert rt.os.environ["PTX_LEVER_REPORT"] == str(tmp_path / "opt_lever_report.jsonl")
    info = runtime.describe()
    assert info["kit.active"] == "true" and info["kit.partial"] == "false" and info["kit.mode"] == mode
    assert info["kit.levers_not_in_arm"] == "pad8,dit_attn_exact,atom_attn_exact,triatt_prologue_cuda"
    assert info["kit.levers_fallback"] == "none" and info["kit.fallback_reasons"] == "{}"
    assert info["kit.commit"] == "unknown"


@pytest.mark.parametrize(
    ("report", "message"),
    [
        ({"active": False, "reason": "no gpu"}, "is not active at activation: no gpu"),
        ({**A100_EXACT_REPORT, "partial": True}, "is partial at activation"),
        (
            {**A100_EXACT_REPORT, "levers_fallback": ["cueq_tri"], "fallback_reasons": {"cueq_tri": "sm70"}},
            r"levers_fallback=\['cueq_tri'\].*sm70",
        ),
        ({**A100_EXACT_REPORT, "levers_unavailable": ["lnstream"]}, r"levers_unavailable=\['lnstream'\]"),
    ],
    ids=["inactive", "partial", "fallback", "unavailable"],
)
def test_degraded_kit_report_refuses(rt, fake, monkeypatch, tmp_path, report, message) -> None:
    """A kit that does not run the whole mode is a typed refusal, before the model loads."""
    FakeKit(monkeypatch, fake, report)
    with pytest.raises(rt.OptimizationUnavailableError, match=message) as error:
        rt.ProtenixRuntime(_config(optimization="exact"), str(tmp_path))
    # The worker matches refusals by class name.
    assert type(error.value).__name__ == "OptimizationUnavailableError"
    assert "runner" not in fake.order


def test_kit_status_degraded_after_load_refuses(rt, fake, monkeypatch, tmp_path) -> None:
    kit = FakeKit(monkeypatch, fake)
    kit.status_report = {**A100_EXACT_REPORT, "levers_fallback": ["dit_attn"]}
    with pytest.raises(rt.OptimizationUnavailableError, match="partial after loading the model"):
        rt.ProtenixRuntime(_config(optimization="exact"), str(tmp_path))


def test_kit_system_exit_becomes_a_refusal(rt, fake, monkeypatch, tmp_path) -> None:
    """The kit refuses a mode with ``SystemExit(3)``; it must not kill the worker as an unexplained exit."""
    kit = FakeKit(monkeypatch, fake)

    def refuse(mode: str, strict: bool) -> Any:
        raise SystemExit(3)

    kit.module.enable = refuse  # type: ignore[attr-defined]
    with pytest.raises(rt.OptimizationUnavailableError, match="exited with code 3 .the optimization kit refused"):
        rt.ProtenixRuntime(_config(optimization="exact"), str(tmp_path))


def test_kit_mode_without_kit_image_refuses_by_name(rt, fake, monkeypatch, tmp_path) -> None:
    monkeypatch.setitem(sys.modules, "protenix_opt", None)
    monkeypatch.setenv("LAYERNORM_TYPE", "fast_layernorm")
    with pytest.raises(rt.OptimizationUnavailableError, match="needs the Protenix kit image"):
        rt.ProtenixRuntime(_config(optimization="fast"), str(tmp_path))


def test_kit_mode_without_cuequivariance_ops_refuses(rt, fake, monkeypatch, tmp_path) -> None:
    FakeKit(monkeypatch, fake)
    monkeypatch.setitem(sys.modules, "cuequivariance_ops_torch", None)
    with pytest.raises(rt.OptimizationUnavailableError, match="cuequivariance_ops_torch failed to import"):
        rt.ProtenixRuntime(_config(optimization="exact"), str(tmp_path))
    assert "runner" not in fake.order


@pytest.mark.parametrize(
    ("mode", "layernorm", "message"),
    [
        ("exact", "openfold", "'exact' runs LAYERNORM_TYPE='fast_layernorm'"),
        ("exact", "torch", "'exact' runs LAYERNORM_TYPE='fast_layernorm'"),
        ("vanilla", "fastlayernorm", "'vanilla' runs LAYERNORM_TYPE='openfold'.* with 'fastlayernorm'"),
        # Unset is stock Protenix's own default: a worker started outside the core would run vanilla on the fused
        # LayerNorm. Empty is a non-fused LayerNorm upstream, but not the table's value either.
        ("vanilla", None, "'vanilla' runs LAYERNORM_TYPE='openfold'.* with it unset"),
        ("exact", None, "'exact' runs LAYERNORM_TYPE='fast_layernorm'.* with it unset"),
        ("vanilla", "", "'vanilla' runs LAYERNORM_TYPE='openfold'.* with ''"),
        ("vanilla", "fast_layernorm", "'vanilla' runs LAYERNORM_TYPE='openfold'"),
        ("vanilla", "torch", "'vanilla' runs LAYERNORM_TYPE='openfold'"),
    ],
)
def test_layernorm_type_is_checked_against_the_mode(rt, fake, monkeypatch, tmp_path, mode, layernorm, message) -> None:
    if mode != "vanilla":
        FakeKit(monkeypatch, fake)
    if layernorm is None:
        monkeypatch.delenv("LAYERNORM_TYPE", raising=False)
    else:
        monkeypatch.setenv("LAYERNORM_TYPE", layernorm)
    with pytest.raises(rt.OptimizationUnavailableError, match=message):
        rt.ProtenixRuntime(_config(optimization=mode), str(tmp_path))
    assert "runner" not in fake.order


def test_vanilla_records_a_stock_layernorm(rt, fake, monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("LAYERNORM_TYPE", "openfold")
    assert rt.ProtenixRuntime(_config(), str(tmp_path)).describe()["layernorm_type"] == "openfold"


def test_late_kit_facts_land_in_the_payload(rt, fake, monkeypatch, tmp_path) -> None:
    kit = FakeKit(monkeypatch, fake)
    kit.late = {"templates": {"real": 1, "dropped": 0, "dropped_reasons": {}, "all_dummy_items": []}}
    kit.blk2._STATS["portability_lines"] = ["no BLK2 cells for sm80", "tmax exact"]
    runtime = rt.ProtenixRuntime(_config(optimization="exact"), str(tmp_path))
    payload = runtime.predict("a.json", "a", _config(optimization="exact"))
    kit.stack.late_records.assert_called_once_with(str(tmp_path / "opt_lever_report.jsonl"), rt.os.getpid())
    assert payload["kit.active"] == "true" and payload["kit.partial"] == "false"
    assert payload["kit.templates.real"] == "1" and payload["kit.templates.dropped"] == "0"
    assert payload["kit.blk2"] == "loaded"
    assert payload["kit.blk2.portability"] == "no BLK2 cells for sm80 | tmax exact"
    assert payload["kernel.resolved.triangle_attention"] == "cuequivariance"
    assert all(isinstance(value, str) for value in payload.values())


@pytest.mark.parametrize(
    "late",
    [
        {"partial": True},
        {"levers_fallback": ["cueq_tri"], "fallback_reasons": {"cueq_tri": "kernel raised"}},
    ],
)
def test_late_kit_fallback_refuses(rt, fake, monkeypatch, tmp_path, late) -> None:
    """A lever that fell back during the prediction (the kit's late records) refuses the request."""
    kit = FakeKit(monkeypatch, fake)
    runtime = rt.ProtenixRuntime(_config(optimization="exact"), str(tmp_path))
    kit.late = late
    with pytest.raises(rt.OptimizationUnavailableError, match="partial after the prediction"):
        runtime.predict("a.json", "a", _config(optimization="exact"))


def test_blk2_stock_fallback_refuses(rt, fake, monkeypatch, tmp_path) -> None:
    kit = FakeKit(monkeypatch, fake)
    runtime = rt.ProtenixRuntime(_config(optimization="fast"), str(tmp_path))
    kit.blk2._STATS["blk2_tri_stock_fallback"] = 1
    with pytest.raises(rt.OptimizationUnavailableError, match="BLK2 fused triangle attention failed"):
        runtime.predict("a.json", "a", _config(optimization="fast"))


def test_blk2_module_never_loaded_refuses_when_its_levers_were_applied(rt, fake, monkeypatch, tmp_path) -> None:
    """Applied levers whose module never loaded never ran: refused, not silently labelled as the mode."""
    FakeKit(monkeypatch, fake, {**A100_EXACT_REPORT, "levers_applied": ["nomask", "cueq_tri"]})
    runtime = rt.ProtenixRuntime(_config(optimization="exact"), str(tmp_path))
    monkeypatch.delitem(sys.modules, "ptx_trunk2_levers")
    with pytest.raises(rt.OptimizationUnavailableError, match=r"never loaded.*\['nomask'\] never ran"):
        runtime.predict("a.json", "a", _config(optimization="exact"))


def test_blk2_module_never_loaded_is_recorded_when_none_of_its_levers_apply(rt, fake, monkeypatch, tmp_path) -> None:
    FakeKit(monkeypatch, fake)
    runtime = rt.ProtenixRuntime(_config(optimization="exact"), str(tmp_path))
    monkeypatch.delitem(sys.modules, "ptx_trunk2_levers")
    payload = runtime.predict("a.json", "a", _config(optimization="exact"))
    assert payload["kit.blk2"] == "not-loaded" and "kit.blk2.portability" not in payload


def test_kit_named_gaps_land_in_the_payload(rt, fake, monkeypatch, tmp_path) -> None:
    """Levers the kit counts applied without per-call evidence keep their named gap in the provenance."""
    kit = FakeKit(monkeypatch, fake)
    kit.late = {"levers_uncounted": {"cueq_tri": "no readable counter"}, "notes": ["tiles from cache"]}
    runtime = rt.ProtenixRuntime(_config(optimization="exact"), str(tmp_path))
    payload = runtime.predict("a.json", "a", _config(optimization="exact"))
    assert payload["kit.levers_uncounted"] == '{"cueq_tri":"no readable counter"}'
    assert payload["kit.notes"] == "tiles from cache"


def test_kit_commit_comes_from_the_kit_or_the_environment_never_a_git_checkout(rt, fake, monkeypatch, tmp_path) -> None:
    """A ``.git`` above the package may be another repository's (an editable install in a checkout): never read."""
    kit = FakeKit(monkeypatch, fake)
    runtime = rt.ProtenixRuntime(_config(optimization="exact"), str(tmp_path))
    package = tmp_path / "host" / "src" / "protenix_opt"
    package.mkdir(parents=True)
    git = tmp_path / "host" / ".git"
    (git / "refs" / "heads").mkdir(parents=True)
    (git / "HEAD").write_text("ref: refs/heads/main\n")
    (git / "refs" / "heads" / "main").write_text("abc123\n")
    kit.module.__file__ = str(package / "__init__.py")
    assert runtime.describe()["kit.commit"] == "unknown"
    monkeypatch.setenv("BOILEROOM_KIT_COMMIT", "feedface")
    assert runtime.describe()["kit.commit"] == "feedface"
    kit.module.__commit__ = "0badc0de"  # type: ignore[attr-defined]
    assert runtime.describe()["kit.commit"] == "0badc0de"


def test_guard_names_census_problems_and_refusal_classes(rt) -> None:
    class KernelsRefused(SystemExit):
        def __init__(self, problems: list[str]) -> None:
            super().__init__(5)
            self.problems = problems

    def census() -> None:
        raise KernelsRefused(["trimul: torch serves"])

    with pytest.raises(
        rt.OptimizationUnavailableError, match=r"code 5 .the kernel census refused.*trimul: torch serves"
    ):
        rt._guard("OpenDDE", "the kernel census", census)

    class NotLoaded(Exception):
        pass

    def not_loaded() -> None:
        raise NotLoaded("layer norm module not loaded")

    with pytest.raises(rt.OptimizationUnavailableError, match="refused: layer norm module not loaded"):
        rt._guard("OpenDDE", "the LayerNorm census", not_loaded)
    with pytest.raises(KeyError):
        rt._guard("Protenix", "x", {}.__getitem__, "missing")


@pytest.mark.parametrize("code", [3, 5])
def test_guard_refuses_on_the_refusal_exit_codes(rt, code: int) -> None:
    with pytest.raises(rt.OptimizationUnavailableError, match=f"exited with code {code}"):
        rt._guard("Protenix", "protenix_opt.enable('exact')", sys.exit, code)


@pytest.mark.parametrize("code", [0, 1, 2, 130, None])
def test_guard_turns_other_exit_codes_into_failures(rt, code: int | None) -> None:
    """A crash, a usage error or a signal is a failure: as a refusal the core would store it and never retry."""
    with pytest.raises(RuntimeError, match=f"exited with code {code} .a failure, not a refusal") as error:
        rt._guard("Protenix", "protenix_opt.enable('exact')", sys.exit, code)
    assert not isinstance(error.value, rt.OptimizationUnavailableError)


def test_guard_refuses_subclasses_of_the_refusal_classes(rt) -> None:
    """OpenDDE's BigRefusal subclasses OpenModeError; matching by exact class name missed it."""
    open_mode_error = type("OpenModeError", (RuntimeError,), {})
    big_refusal = type("BigRefusal", (open_mode_error,), {})

    def refuse() -> None:
        raise big_refusal("too many tokens for the fast arm")

    with pytest.raises(rt.OptimizationUnavailableError, match="refused: too many tokens"):
        rt._guard("OpenDDE", "opendde_opt.enable('fast')", refuse)
    assert {"ActivationError", "NotLoaded", "OpenModeError"} == rt.KIT_REFUSAL_CLASS_NAMES


def test_flat_values_are_strings(rt) -> None:
    assert rt._flat(None) == "none" and rt._flat([]) == "none" and rt._flat(()) == "none"
    assert rt._flat(True) == "true" and rt._flat(False) == "false"
    assert rt._flat(["a", "b"]) == "a,b" and rt._flat({"b": 1, "a": [2]}) == '{"a":[2],"b":1}'


def test_provenance_sentinels_for_what_is_not_there(rt, monkeypatch) -> None:
    """Missing facts use boileroom's provenance words, never an empty string or prose."""
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    assert rt._torch_info() == {"torch": "not-loaded", "cuda": "not-loaded", "gpu": "not-loaded"}
    cuda = SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(__version__="", version=SimpleNamespace(cuda=None), cuda=cuda)
    )
    assert rt._torch_info() == {"torch": "unknown", "cuda": "none", "gpu": "none"}
    monkeypatch.setitem(sys.modules, "boileroom_no_such_module", None)
    assert rt._package_version(("boileroom-no-such-dist",), "boileroom_no_such_module") == "absent"
    assert {rt.NOT_LOADED, rt.ABSENT, rt.NONE, rt.UNKNOWN} == rt.PROVENANCE_SENTINELS


def test_runtime_template_guard_says_what_the_core_says(rt, fake, tmp_path) -> None:
    """The worker keeps the core's guard for its own protocol, worded the same so either refusal reads alike."""
    from boileroom.models.protenix.core import ProtenixCore

    core = ProtenixCore({"model_name": "protenix_base_default_v1.0.0"})
    with pytest.raises(ValueError) as core_error:
        core._validate_templates({**core.config, "use_template": True, "templates": {"t": "data_t\n"}})
    runtime = rt.ProtenixRuntime(_config(), str(tmp_path))
    with pytest.raises(ValueError) as worker_error:
        runtime.predict("a.json", "a", {**_config(use_template=True), "template_staging": _staging(tmp_path)})
    assert str(worker_error.value) == str(core_error.value)


def test_runtime_source_supports_python_310() -> None:
    """The model images run Python 3.10, and the runtime never imports boileroom."""
    source = RUNTIME_PATH.read_text()
    tree = ast.parse(source, feature_version=(3, 10))
    imported = {
        alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names
    } | {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    assert "boileroom" not in imported
