"""Worker runtimes for the pinned Protenix 2.0 and OpenDDE 1.1.1 Python inference runners.

Both upstreams descend from the same runner (``runner.batch_inference`` / ``runner.inference``) and the same
template pipeline, so one module serves both: :class:`ProtenixRuntime` and :class:`OpenDDERuntime` are thin
subclasses of :class:`FoldRuntime` that name their upstream, kit package and LayerNorm semantics.

This file runs inside the model image's own interpreter (``runpy.run_path`` in the worker child), so it must stay
importable on Python 3.10 and must never import ``boileroom``. It defines its own
:class:`OptimizationUnavailableError`; the worker matches refusals by class name.

Every degradation is loud or visible:

- a kit mode that is inactive, partial, or reports any fallen-back or unavailable lever refuses with
  :class:`OptimizationUnavailableError`, at activation, after the model loads, and again after every prediction
  (the kits' late records);
- the kits' ``SystemExit`` (3 = kit refusal, 5 = OpenDDE's kernel census refusal) becomes the same typed refusal;
  any other exit code is a failure (``RuntimeError``), never a standing refusal;
- ``LAYERNORM_TYPE`` must be the mode's value (the core's table, copied here) and the resolved triangle kernels the
  requested ones; OpenDDE's own compute-capability-7.x fallback (torch kernels, fp32) is served in ``vanilla`` only,
  and recorded (``kernel.cc7_fallback``);
- caller templates are counted after featurization, and a staged template the featurizer dropped fails the request;
- in kit modes, the forward kit's PairformerStack CUDA graphs are dropped once the native TriMul adapter frees a
  workspace they read, and recaptured on demand (``kit.stack_graph_resets`` counts the resets);
- :meth:`FoldRuntime.describe` and every prediction's payload are flat ``str -> str`` provenance records.
"""

from __future__ import annotations

import contextlib
import copy
import importlib
import json
import os
import platform
import shutil
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

#: The kits' exit codes, as their CLIs document them.
KIT_REFUSAL_EXIT_CODE = 3
KERNELS_REFUSED_EXIT_CODE = 5
_EXIT_MEANINGS = {
    KIT_REFUSAL_EXIT_CODE: "the optimization kit refused the mode",
    KERNELS_REFUSED_EXIT_CODE: "the kernel census refused: a required kernel is absent or fell back",
}
#: The ``SystemExit`` codes that are refusals (those with a meaning above); any other code (1 = failed, 2 = usage) is a
#: failure. Equal to ``boileroom.optimization.KIT_REFUSAL_EXIT_CODES`` (this file cannot import boileroom; a contract
#: test holds them equal).
KIT_REFUSAL_EXIT_CODES = frozenset(_EXIT_MEANINGS)
#: Exception classes the kits raise for a refusal, matched by name anywhere in the MRO (the kits are not importable
#: here). A literal copy of ``boileroom.optimization.KIT_REFUSAL_CLASS_NAMES``; ``KernelsRefused`` is a ``SystemExit``
#: (code 5), so its exit code, not its name, makes it a refusal.
KIT_REFUSAL_CLASS_NAMES = frozenset({"ActivationError", "NotLoaded", "OpenModeError"})
#: Provenance values for "not established", as ``boileroom/provenance.py`` spells them (literal copies; a contract
#: test holds this set inside provenance's).
NOT_LOADED = "not-loaded"
ABSENT = "absent"
NONE = "none"
UNKNOWN = "unknown"
PROVENANCE_SENTINELS = frozenset({NOT_LOADED, ABSENT, NONE, UNKNOWN})
#: Report keys copied into describe() and the payload, in this order.
_REPORT_KEYS = (
    "active",
    "mode",
    "partial",
    "levers_applied",
    "levers_fallback",
    "levers_unavailable",
    "levers_not_in_arm",
    "levers_inert",
    "levers_gated_off",
    "fallback_reasons",
    # The kits' named gaps and step-asides: a lever counted applied without its own per-call evidence
    # (``levers_uncounted``), a call stepped aside by rule (``levers_asides``), why a lever is inert.
    "levers_uncounted",
    "levers_asides",
    "levers_inert_reasons",
    "lever_buckets",
    "notes",
    "kit_version",
    "package_version",
    "protenix_version",
    "opendde_version",
    "line",
    "gpu",
)
#: Kit modules whose import pulls in the cuEquivariance kernels every kit mode serves.
CUEQUIVARIANCE_MODULES = ("cuequivariance_torch", "cuequivariance_ops_torch")
_CUEQUIVARIANCE_DISTS = {
    "cuequivariance_torch": ("cuequivariance-torch",),
    "cuequivariance_ops_torch": (
        "cuequivariance-ops-torch-cu12",
        "cuequivariance-ops-torch-cu13",
        "cuequivariance-ops-torch-cu11",
        "cuequivariance-ops-torch",
    ),
}
#: Requested triangle-kernel values that must resolve to the cuEquivariance kernels.
_CUEQ_REQUESTS = frozenset({"auto", "cuequivariance"})
#: Environment variable holding the kit's source commit; the kit images set it to the commit they were built from.
KIT_COMMIT_ENV = "BOILEROOM_KIT_COMMIT"


class OptimizationUnavailableError(RuntimeError):
    """The requested optimization (or the kernel path it needs) is unavailable; the worker reports it as refused."""


class FoldRuntime:
    """Load one runner and serve requests without reloading its weights.

    Parameters
    ----------
    config : dict[str, Any]
        The model config. Keys read: ``optimization``, ``model_name``, ``seeds``, ``cycle``, ``step``,
        ``sample``, ``dtype``, ``use_msa``, ``use_template``, ``trimul_kernel``, ``triatt_kernel``,
        ``enable_cache``, ``enable_fusion``, ``enable_tf32``, ``use_tfg_guidance``.
    work_dir : str
        A private directory for this runtime's own files.

    Raises
    ------
    OptimizationUnavailableError
        If a kit mode cannot run in full on this image and card, or the LayerNorm or triangle-kernel path is
        not the one the mode needs.
    """

    #: Shown in messages and in ``describe()["runtime"]``.
    LABEL = ""
    #: The kit package of this family's kit image.
    KIT_PACKAGE = ""
    #: Distribution names of the upstream package (first found wins), and its import name.
    UPSTREAM_DISTS: tuple[str, ...] = ()
    UPSTREAM_MODULE = ""
    #: Module holding ``TemplateHitFeaturizer``.
    TEMPLATE_UTILS = ""
    #: ``LAYERNORM_TYPE`` per optimization mode: a literal copy of the core's ``LAYERNORM_BY_MODE`` (this file cannot
    #: import boileroom; ``tests/contracts/test_layernorm_contract.py`` holds the two equal). Any other value, or none
    #: (upstream's own default), refuses.
    LAYERNORM_BY_MODE: dict[str, str] = {}
    #: Whether staged requests get a per-request template cache directory (else the cache is switched off).
    TEMPLATE_CACHE_PER_REQUEST = False
    #: The forward kit's PairformerStack CUDA graphs, and the native TriMul adapter whose per-geometry workspaces
    #: those graphs read (see :meth:`_drop_stale_stack_graphs`).
    STACK_GRAPH_MODULE = "fpf_stackgraph.stackgraph"
    TRIMUL_NATIVE_MODULE = "opt_core.kernels.trimul.native"

    def __init__(self, config: dict[str, Any], work_dir: str) -> None:
        self.mode = str(config.get("optimization") or "vanilla")
        self.work_dir = work_dir
        self.kit: Any = None
        self.report: dict[str, Any] | None = None
        #: The TriMul geometry caches live after the previous request, held so none is freed before the next check.
        self._trimul_caches: dict[Any, Any] = {}
        self.stack_graph_resets = 0
        self.requested_kernels = {
            "triangle_attention": str(config["triatt_kernel"]),
            "triangle_multiplicative": str(config["trimul_kernel"]),
        }
        self._prepare_jit_cache()
        if self.mode != "vanilla":
            # The kits refuse late activation, so this precedes any upstream import.
            self._enable_kit()
            self._import_cuequivariance()
        self.layernorm = self._check_layernorm_env()
        if self.mode != "vanilla":
            self._arm_kit_census()
        self.runner = _guard(self.LABEL, "loading the model", self._load_runner, config, work_dir)
        self.resolved = _resolved_kernels(self.runner.configs)
        self._check_loaded()
        if self.kit is not None:
            self.report = self._gate(_guard(self.LABEL, "the kit status", self.kit.status), "after loading the model")
        # Upstream mutates inference settings based on input length. Start each
        # request from a pristine copy while retaining the model's parameters.
        self._base_config = copy.deepcopy(self.runner.configs)

    # ------------------------------------------------------------------ provenance
    def describe(self) -> dict[str, str]:
        """Return a flat provenance record of the loaded stack.

        Returns
        -------
        dict[str, str]
            ``runtime``, ``optimization``, ``python``, ``torch``, ``cuda``, ``gpu``, ``<upstream>`` (its
            package version), ``cuequivariance_torch``, ``cuequivariance_ops_torch``, ``layernorm_type``,
            ``kernel.requested.<site>`` and ``kernel.resolved.<site>`` for both triangle sites plus
            ``kernel.resolved.dtype``; in kit modes also ``kit.commit`` and ``kit.<field>`` for the
            activation report's fields (see ``_REPORT_KEYS``).
        """
        info: dict[str, str] = {
            "runtime": self.LABEL,
            "optimization": self.mode,
            "python": platform.python_version(),
            self.UPSTREAM_MODULE: _package_version(self.UPSTREAM_DISTS, self.UPSTREAM_MODULE),
            "layernorm_type": self.layernorm,
        }
        info.update(_torch_info())
        for module, dists in _CUEQUIVARIANCE_DISTS.items():
            info[module] = _package_version(dists, module)
        for site, value in self.requested_kernels.items():
            info[f"kernel.requested.{site}"] = value
        for site, value in self.resolved.items():
            info[f"kernel.resolved.{site}"] = value
        if self.kit is not None:
            info["kit.commit"] = _kit_commit(self.kit)
            info.update(_report_facts(self.report or {}))
        info.update(self._describe_extra())
        return info

    def _describe_extra(self) -> dict[str, str]:
        return {}

    # ------------------------------------------------------------------ requests
    def predict(self, input_json: str, output_dir: str, config: dict[str, Any]) -> dict[str, str]:
        """Run one request with fresh paths, seeds, sampling settings and dumper.

        Parameters
        ----------
        input_json : str
            The upstream query JSON.
        output_dir : str
            Where upstream writes this request's outputs.
        config : dict[str, Any]
            The request config; ``template_staging`` (a ``StagedTemplates.to_dict()``) points the featurizer at
            caller templates for this request only.

        Returns
        -------
        dict[str, str]
            Flat provenance for this request: ``kernel.resolved.*``, ``templates.*`` when templates were
            staged, and the kit's reconciled ``kit.*`` facts in kit modes.

        Raises
        ------
        OptimizationUnavailableError
            If a kit lever fell back during the run.
        RuntimeError
            If upstream recorded a failure, or fewer or more templates were featurized than were staged.
        ValueError
            If caller templates are combined with ``use_template=True``.
        """
        from runner.batch_inference import preprocess_input
        from runner.inference import infer_predict

        configs = copy.deepcopy(self._base_config)
        configs.seeds = [int(seed) for seed in str(config["seeds"]).split(",")]
        configs.model.N_cycle = config["cycle"]
        configs.sample_diffusion.N_step = config["step"]
        configs.sample_diffusion.N_sample = config["sample"]
        self._set_guidance(configs, config["use_tfg_guidance"])
        configs.dtype = self._request_dtype(config)
        configs.use_msa = config["use_msa"]
        configs.dump_dir = output_dir
        # Featurize in this process so the template count below sees every call.
        configs.num_workers = 0
        staging = config.get("template_staging")
        if staging is not None and config["use_template"]:
            raise ValueError(
                "caller templates replace the template search for the request; they cannot be combined with "
                "use_template=True"
            )
        use_template = bool(config["use_template"]) or staging is not None
        if staging is not None:
            self._point_at_staged_templates(configs, staging)
        configs.use_template = use_template
        configs.input_json_path = preprocess_input(
            input_json,
            out_dir=output_dir,
            use_msa=config["use_msa"],
            use_template=use_template,
            msa_server_mode="colabfold",
        )
        self.runner.configs = configs
        # The runner also caches the cycle count directly on the model.
        self.runner.model.N_cycle = config["cycle"]
        self.runner.update_model_configs(configs)
        self.runner.init_basics()
        self.runner.init_dumper(need_atom_confidence=True, sorted_by_ranking_score=configs.sorted_by_ranking_score)
        with self._count_templates(staging is not None) as featurized:
            _guard(self.LABEL, "the prediction", infer_predict, self.runner, configs)
        if self.kit is not None:
            self._drop_stale_stack_graphs()
        errors = sorted(Path(self.runner.error_dir).glob("*.txt"))
        if errors:
            details = "\n".join(path.read_text(encoding="utf-8") for path in errors)
            raise RuntimeError(f"{self.LABEL} inference failed:\n{details[-12000:]}")
        payload = {f"kernel.resolved.{site}": value for site, value in _resolved_kernels(configs).items()}
        if staging is not None:
            payload.update(_check_template_counts(self.LABEL, staging, featurized))
        if self.kit is not None:
            payload.update(self._late_kit_facts(input_json))
            payload["kit.stack_graph_resets"] = str(self.stack_graph_resets)
        return payload

    def _drop_stale_stack_graphs(self) -> None:
        """Drop the kit's PairformerStack CUDA graphs once a TriMul workspace they read may be freed.

        The forward kit's stack graph (``fpf_stackgraph.stackgraph``) captures the PairformerStack once per token
        count and replays it for later requests of that count, with the addresses of the native TriMul adapter's
        workspaces for that geometry baked in. The adapter keeps only its two most recent geometries (``_SHARED`` in
        ``opt_core.kernels.trimul.native``, kit commit ``f4f62fa``) and frees the rest, so a length that recurs after
        two other lengths would replay its graph over memory other tensors now own (Protenix ``exact``: NaN PAE). This
        runtime holds the caches that were live after the previous request, so none is freed before this check; when
        one of them left ``_SHARED`` or was rebuilt, every stack graph goes (``reset_cache``), and each length
        captures again on its next request. It assumes one request's own geometries fit the adapter's LRU.

        A no-op when the kit's modules are not loaded or lack these names; re-check this when the kit commit changes.
        """
        native = sys.modules.get(self.TRIMUL_NATIVE_MODULE)
        stack_graph = sys.modules.get(self.STACK_GRAPH_MODULE)
        shared = getattr(native, "_SHARED", None)
        if not isinstance(shared, dict) or not callable(getattr(stack_graph, "reset_cache", None)):
            return
        with getattr(native, "_LOCK", None) or contextlib.nullcontext():
            live = dict(shared)
        if any(live.get(key) is not cache for key, cache in self._trimul_caches.items()):
            torch = sys.modules.get("torch")
            if torch is not None and torch.cuda.is_available():
                torch.cuda.synchronize()  # no replay in flight while the graphs are released
            stack_graph.reset_cache()
            self.stack_graph_resets += 1
        self._trimul_caches = live

    def _point_at_staged_templates(self, configs: Any, staging: dict[str, Any]) -> None:
        # Caller templates: point the featurizer at the staged directory and forbid it from fetching
        # anything. The architecture does not depend on use_template, so one runner serves both kinds of
        # request; only the data path differs.
        template = configs.data.template
        template.prot_template_mmcif_dir = staging["mmcif_dir"]
        template.release_dates_path = staging["release_dates_path"]
        template.obsolete_pdbs_path = staging["obsolete_pdbs_path"]
        template.fetch_remote = False
        template.kalign_binary_path = _kalign_path()
        # A shared parse cache keyed by the synthetic ids would serve one request's template to the next.
        template.prot_template_cache_dir = staging["cache_dir"] if self.TEMPLATE_CACHE_PER_REQUEST else ""

    @contextlib.contextmanager
    def _count_templates(self, enabled: bool) -> Iterator[list[tuple[str, int, list[str]]]]:
        """Record ``(query_sequence, featurized count, upstream errors and warnings)`` per featurizer call."""
        calls: list[tuple[str, int, list[str]]] = []
        if not enabled:
            yield calls
            return
        module = importlib.import_module(self.TEMPLATE_UTILS)
        cls = module.TemplateHitFeaturizer
        original = cls.__dict__.get("get_templates", cls.get_templates)

        def get_templates(featurizer: Any, *args: Any, **kwargs: Any) -> Any:
            result = original(featurizer, *args, **kwargs)
            query = kwargs.get("query_sequence", args[1] if len(args) > 1 else "")
            search = result[0] if isinstance(result, tuple) else result
            notes = [str(note) for note in list(getattr(search, "errors", None) or [])]
            notes += [str(note) for note in list(getattr(search, "warnings", None) or [])]
            calls.append((str(query), len(getattr(search, "features", None) or []), notes))
            return result

        cls.get_templates = get_templates
        try:
            yield calls
        finally:
            cls.get_templates = original

    # ------------------------------------------------------------------ family hooks
    def _load_runner(self, config: dict[str, Any], work_dir: str) -> Any:
        raise NotImplementedError

    def _set_guidance(self, configs: Any, enabled: bool) -> None:
        raise NotImplementedError

    def _prepare_kit_env(self) -> None:
        """Set the kit's process environment before activation."""

    def _prepare_jit_cache(self) -> None:
        """Point the compile caches at their directories before anything compiles."""

    def _request_dtype(self, config: dict[str, Any]) -> str:
        """Return the dtype a request runs with (the requested one)."""
        return str(config["dtype"])

    def _arm_kit_census(self) -> None:
        """Arm the kit's own kernel guards before the model loads."""

    def _check_loaded(self) -> None:
        """Check the loaded model's kernel path."""

    def _late_kit_facts(self, input_json: str) -> dict[str, str]:
        raise NotImplementedError

    # ------------------------------------------------------------------ kit
    def _enable_kit(self) -> None:
        try:
            self.kit = importlib.import_module(self.KIT_PACKAGE)
        except ImportError as error:
            raise OptimizationUnavailableError(
                f"optimization={self.mode!r} needs the {self.LABEL} kit image ({self.KIT_PACKAGE} is not importable: "
                f"{error})"
            ) from None
        self._prepare_kit_env()
        report = _guard(
            self.LABEL, f"{self.KIT_PACKAGE}.enable({self.mode!r})", self.kit.enable, self.mode, strict=False
        )
        self.report = self._gate(report, "at activation")

    def _import_cuequivariance(self) -> None:
        # After the kit's activation (it exports the kernels' own environment) and before any model import.
        for module in CUEQUIVARIANCE_MODULES:
            try:
                importlib.import_module(module)
            except Exception as error:  # an ABI mismatch raises OSError/RuntimeError, not only ImportError
                raise OptimizationUnavailableError(
                    f"optimization={self.mode!r} needs the cuEquivariance kernels, but {module} failed to import: "
                    f"{type(error).__name__}: {error}"
                ) from None

    def _gate(self, report: Any, when: str) -> dict[str, Any]:
        """Return ``report`` when the kit runs the whole mode, else refuse naming what fell back."""
        report = dict(report or {})
        if not report.get("active"):
            reason = report.get("reason") or "no reason given"
            raise OptimizationUnavailableError(
                f"{self.LABEL} optimization={self.mode!r} is not active {when}: {reason}"
            )
        fallback = list(report.get("levers_fallback") or [])
        unavailable = list(report.get("levers_unavailable") or [])
        if report.get("partial") or fallback or unavailable:
            reasons = report.get("fallback_reasons") or {}
            raise OptimizationUnavailableError(
                f"{self.LABEL} optimization={self.mode!r} is partial {when}: levers_fallback={fallback} "
                f"levers_unavailable={unavailable} reasons={_flat(reasons)}"
            )
        return report

    # ------------------------------------------------------------------ LayerNorm
    def _check_layernorm_env(self) -> str:
        """Return the effective ``LAYERNORM_TYPE`` after checking it is the mode's value (:attr:`LAYERNORM_BY_MODE`)."""
        value = os.environ.get("LAYERNORM_TYPE")
        expected = self.LAYERNORM_BY_MODE.get(self.mode)
        if value is None or value != expected:
            running = "it unset (upstream's own default)" if value is None else repr(value)
            raise OptimizationUnavailableError(
                f"{self.LABEL} optimization={self.mode!r} runs LAYERNORM_TYPE={expected!r}, but the worker runs with "
                f"{running}; the core sets it per mode, so the worker was started outside the core"
            )
        return value


class ProtenixRuntime(FoldRuntime):
    """Protenix 2.0 runner, optionally under the ``protenix_opt`` kit."""

    LABEL = "Protenix"
    KIT_PACKAGE = "protenix_opt"
    UPSTREAM_DISTS = ("protenix",)
    UPSTREAM_MODULE = "protenix"
    TEMPLATE_UTILS = "protenix.data.template.template_utils"
    # protenix/model/triangular/layers.py:33 reads fast_layernorm when unset; any other value is a non-fused LN.
    LAYERNORM_BY_MODE = {"vanilla": "openfold", "exact": "fast_layernorm", "fast": "fast_layernorm"}
    # Protenix's process() returns an empty hit, silently, for a cache directory without the pickle.
    TEMPLATE_CACHE_PER_REQUEST = False
    #: The lever report the kit's modules dump at exit, and ``late_records`` reads back.
    LEVER_REPORT_ENV = "PTX_LEVER_REPORT"
    LEVER_REPORT_NAME = "opt_lever_report.jsonl"
    #: The forward kit's BLK2 module and its counters.
    BLK2_MODULE = "ptx_trunk2_levers"

    def _load_runner(self, config: dict[str, Any], work_dir: str) -> Any:
        from runner.batch_inference import get_default_runner, inference_configs, init_logging

        init_logging()
        inference_configs["dump_dir"] = work_dir
        return get_default_runner(**_runner_kwargs(config))

    def _set_guidance(self, configs: Any, enabled: bool) -> None:
        configs.sample_diffusion.guidance.enable = enabled

    def _prepare_kit_env(self) -> None:
        # As the kit's CLI does: the lever report lives outside the output directory.
        os.environ.setdefault(self.LEVER_REPORT_ENV, os.path.join(self.work_dir, self.LEVER_REPORT_NAME))

    def _late_kit_facts(self, input_json: str) -> dict[str, str]:
        stack = importlib.import_module(f"{self.KIT_PACKAGE}.stack")
        records = stack.late_records(os.environ.get(self.LEVER_REPORT_ENV), os.getpid())
        report = self._gate(stack.reconcile(dict(self.report or {}), records), "after the prediction")
        facts = _report_facts(report)
        templates = report.get("templates")
        if isinstance(templates, dict):
            for key in ("real", "dropped", "dropped_reasons", "all_dummy_items"):
                if key in templates:
                    facts[f"kit.templates.{key}"] = _flat(templates[key])
        stats = getattr(sys.modules.get(self.BLK2_MODULE), "_STATS", None)
        if not isinstance(stats, dict):
            # The kit's exit tally names this state ("levers were never loaded in this process"): refuse when the
            # report counts a lever that module implements as applied, else record that the counters are absent.
            owned = self._blk2_levers(report)
            if owned:
                raise OptimizationUnavailableError(
                    f"Protenix optimization={self.mode!r} fell back after the prediction: {self.BLK2_MODULE} was "
                    f"never loaded in this process, so the applied levers {owned} never ran"
                )
            facts["kit.blk2"] = NOT_LOADED
        else:
            facts["kit.blk2"] = "loaded"
            # The fused triangle attention failed on first use and the stock op served the rest of the process.
            fallback = stats.get("blk2_tri_stock_fallback")
            if fallback:
                raise OptimizationUnavailableError(
                    f"Protenix optimization={self.mode!r} fell back after the prediction: BLK2 fused triangle "
                    f"attention failed and the stock op served ({fallback})"
                )
            # Expected per-card routes (no BLK2 cells for this arch, tmax exact sub-path, untested sizes): shown, not refused.
            facts["kit.blk2.portability"] = (
                " | ".join(str(line) for line in stats.get("portability_lines") or []) or NONE
            )
        return facts

    def _blk2_levers(self, report: dict[str, Any]) -> list[str]:
        """Return the applied levers the kit's own registry (``report.IMPL``) places in the BLK2 module."""
        impl = getattr(importlib.import_module(f"{self.KIT_PACKAGE}.report"), "IMPL", None) or {}
        source = f"/{self.BLK2_MODULE}.py"
        return sorted(
            lever for lever in report.get("levers_applied") or [] if str((impl.get(lever) or ("",))[0]).endswith(source)
        )


class OpenDDERuntime(FoldRuntime):
    """OpenDDE 1.1.1 runner, optionally under the ``opendde_opt`` kit."""

    LABEL = "OpenDDE"
    KIT_PACKAGE = "opendde_opt"
    UPSTREAM_DISTS = ("opendde",)
    UPSTREAM_MODULE = "opendde"
    TEMPLATE_UTILS = "opendde.data.template.template_utils"
    # opendde/model/triangular/layers.py:25 reads torch when unset.
    LAYERNORM_BY_MODE = {"vanilla": "torch", "exact": "fast_layernorm", "fast": "fast_layernorm"}
    # OpenDDE's process() writes <id>_<chain>.pkl into the cache and reads it back on the next request.
    TEMPLATE_CACHE_PER_REQUEST = True
    #: runner.configs attribute -> the config key that requested it.
    _SITE_KEYS = {"triangle_attention": "triatt_kernel", "triangle_multiplicative": "trimul_kernel"}
    #: The compile caches keyed under ``$MODEL_OPT_JIT_ROOT/<stack key>/``, as the kit's ``configs/<gpu>.env`` lays them out.
    JIT_CACHE_DIRS = {"TRITON_CACHE_DIR": "triton", "TORCH_EXTENSIONS_DIR": "torch_ext"}

    def __init__(self, config: dict[str, Any], work_dir: str) -> None:
        self.kernels_record: dict[str, Any] = {}
        self.layernorm_census: dict[str, Any] = {}
        self.cc7_fallback = False
        self.jit_stack_key = NONE
        super().__init__(config, work_dir)

    def _prepare_jit_cache(self) -> None:
        # The kit's configs/<gpu>.env exports these in its shell launcher; boileroom enables the kit in-process, so the
        # runtime lays them out the same way: keyed by torch, CUDA and compute capability (opt_core.jit_cache.key_facts,
        # import-free of torch), so one MODEL_OPT_JIT_ROOT volume shared by A100 and H100 workers, or kept across a
        # torch or CUDA bump, never serves a build of another stack. A directory the caller already set is kept.
        # key_facts equals the launcher's opendde_opt.modes.jit_cache_key() when torch, CUDA and the capability are
        # known; when one is not, it names a directory of this process alone instead of a shared "...-unknown" bucket.
        root = os.environ.get("MODEL_OPT_JIT_ROOT")
        if not root:
            return
        try:
            jit_cache = importlib.import_module("opt_core.jit_cache")
        except ImportError:
            # No kit core in this interpreter (the kit modes need it and refuse at activation): torch's own defaults.
            self.jit_stack_key = ABSENT
            return
        try:
            key = os.environ.get("MODEL_OPT_STACK_KEY") or str(jit_cache.key_facts()["key"])
        except Exception as error:  # an unreadable stack must not share another stack's builds: torch's defaults
            sys.stderr.write(f"[boileroom] no JIT stack key ({error}); compile caches stay at torch's defaults\n")
            self.jit_stack_key = UNKNOWN
            return
        os.environ["MODEL_OPT_STACK_KEY"] = key
        for variable, subdirectory in self.JIT_CACHE_DIRS.items():
            os.environ.setdefault(variable, os.path.join(root, key, subdirectory))
        self.jit_stack_key = key

    def _request_dtype(self, config: dict[str, Any]) -> str:
        # Upstream forces fp32 with the cc7 torch kernels at runner init; a request's dtype must not undo it.
        return "fp32" if self.cc7_fallback else str(config["dtype"])

    def _load_runner(self, config: dict[str, Any], work_dir: str) -> Any:
        from runner.batch_inference import get_default_runner, init_logging

        init_logging()
        return get_default_runner(**_runner_kwargs(config), dump_dir=work_dir)

    def _set_guidance(self, configs: Any, enabled: bool) -> None:
        configs.sample_diffusion.guidance["enable"] = enabled  # a plain dict in OpenDDE

    def _kit_module(self, name: str) -> Any:
        return importlib.import_module(f"{self.KIT_PACKAGE}.{name}")

    def _arm_kit_census(self) -> None:
        for site, value in self.requested_kernels.items():
            if value not in _CUEQ_REQUESTS:
                # A stated stock kernel makes the kit step its levers of that site aside; the library route cannot tell it.
                raise OptimizationUnavailableError(
                    f"OpenDDE optimization={self.mode!r} runs the kit's triangle kernels; {self._SITE_KEYS[site]}="
                    f"{value!r} needs optimization='vanilla'"
                )
        lncensus = self._kit_module("lncensus")
        modes = self._kit_module("modes")
        stack = self._kit_module("stack")
        settings = self._kit_module("settings")
        # The LayerNorm backend upstream actually bound, before any weights load (NotLoaded refuses).
        self.layernorm_census = dict(
            _guard(self.LABEL, "the LayerNorm census", lncensus.census, strict=True, stream=sys.stderr) or {}
        )
        resolution = _guard(
            self.LABEL, "resolving the kit line", modes.resolve, self.mode, stack.tree_root(), os.environ
        )
        line = resolution.line
        expected = modes.kernel_expectations(line, ln_requested=settings.ln_requested(), n_gpu=1, knobs={})
        route = modes.route_word(line, getattr(resolution, "mode", self.mode), 1)
        # Presence, device probe, and the resolved selection at runner init (KernelsRefused, exit 5).
        _guard(
            self.LABEL,
            "the kernel census",
            lncensus.arm,
            route,
            expected,
            strict=True,
            stream=sys.stderr,
            layernorm=self.layernorm_census,
        )

    def _check_loaded(self) -> None:
        # Upstream's apply_runtime_compatibility (requires_cc7_fallback): on compute capability 7.x it runs both triangle
        # sites on torch kernels in fp32, whatever was requested. Stock OpenDDE's own policy, so vanilla serves it, and
        # records it; the kit modes never do (they run on A100/H100 only, and their kernels are the point).
        self.cc7_fallback = self.mode == "vanilla" and _device_major(getattr(self.runner, "device", None)) == 7
        for site, resolved in self.resolved.items():
            if site not in self._SITE_KEYS:
                continue
            requested = self.requested_kernels[site]
            expected = "cuequivariance" if requested in _CUEQ_REQUESTS else requested
            if resolved != expected and not (self.cc7_fallback and resolved == "torch"):
                raise OptimizationUnavailableError(
                    f"OpenDDE {site} resolved to {resolved!r} where {expected!r} was expected "
                    f"({self._SITE_KEYS[site]}={requested!r}); set it to 'torch' to run the torch kernel by name"
                )
        if self.cc7_fallback:
            sys.stderr.write(
                "[boileroom] OpenDDE runs upstream's compute-capability-7.x fallback: torch triangle kernels in fp32 "
                f"(resolved {_flat(self.resolved)}); recorded as kernel.cc7_fallback=true\n"
            )

    def _describe_extra(self) -> dict[str, str]:
        extra: dict[str, str] = {
            "kernel.cc7_fallback": _flat(self.cc7_fallback),
            "jit.stack_key": self.jit_stack_key,
        }
        if self.layernorm_census:
            extra["layernorm.backend"] = _flat(self.layernorm_census.get("backend"))
        return extra

    def _late_kit_facts(self, input_json: str) -> dict[str, str]:
        lncensus = self._kit_module("lncensus")
        stack = self._kit_module("stack")
        alloc = self._kit_module("alloc")
        inputs = self._kit_module("inputs")
        lnstream = self._kit_module("lnstream")
        # The kernel census with this pass's counters: a reference path where the kit says a kernel serves refuses.
        _guard(self.LABEL, "the kernel census", lncensus.enforce, "predict")
        facts = {
            "det": False,
            "n_gpu": 1,
            "rank": 0,
            "token_floors": alloc.token_floor(inputs.load_query(input_json)),
            "stock_knobs": {},
        }
        report = stack.refresh(predicted=True, facts=facts)
        report = self._gate(report, "after the prediction")
        planned = list(report.get("levers_planned") or report.get("levers_applied") or [])
        state = lnstream.STATS.get("state")
        if "lnstream" in planned and state != "serving":
            raise OptimizationUnavailableError(
                f"OpenDDE optimization={self.mode!r} fell back after the prediction: the current-stream LayerNorm "
                f"is {state!r} ({lnstream.STATS.get('reason')})"
            )
        result = _report_facts(report)
        result["kit.lnstream"] = _flat(state)
        self.kernels_record = dict(lncensus.record() or {})
        result["kit.kernels"] = _flat(self.kernels_record.get("line"))
        return result


# ---------------------------------------------------------------------- helpers
def _device_major(device: Any) -> int | None:
    """The CUDA compute-capability major of ``device`` (the current device for None), or None without one."""
    torch = sys.modules.get("torch")
    if torch is None or (device is not None and getattr(device, "type", "cuda") != "cuda"):
        return None
    try:
        if not torch.cuda.is_available():
            return None
        return int(torch.cuda.get_device_capability(device)[0])
    except Exception:  # a broken driver: no capability, so no fallback is accepted
        return None


def _runner_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_name": config["model_name"],
        "seeds": [int(seed) for seed in str(config["seeds"]).split(",")],
        "n_cycle": config["cycle"],
        "n_step": config["step"],
        "n_sample": config["sample"],
        "dtype": config["dtype"],
        "use_msa": config["use_msa"],
        "use_template": config["use_template"],
        "trimul_kernel": config["trimul_kernel"],
        "triatt_kernel": config["triatt_kernel"],
        "enable_cache": config["enable_cache"],
        "enable_fusion": config["enable_fusion"],
        "enable_tf32": config["enable_tf32"],
        "use_tfg_guidance": config["use_tfg_guidance"],
        "need_atom_confidence": True,
    }


def _guard(label: str, what: str, call: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Run ``call``, turning the kits' refusals (exit 3 or 5, a refusal class) into :class:`OptimizationUnavailableError`.

    Any other ``SystemExit`` code becomes a ``RuntimeError``; any other exception propagates unchanged.
    """
    try:
        return call(*args, **kwargs)
    except SystemExit as error:
        code = error.code
        problems = getattr(error, "problems", None)
        detail = f": {' '.join(str(p) for p in problems)}" if problems else ""
        if not isinstance(code, int) or code not in KIT_REFUSAL_EXIT_CODES:
            # A crash (1) or a usage error (2) is a failure: refusing would store it as a standing refusal.
            raise RuntimeError(f"{label}: {what} exited with code {code} (a failure, not a refusal){detail}") from None
        raise OptimizationUnavailableError(
            f"{label}: {what} exited with code {code} ({_EXIT_MEANINGS[code]}){detail}"
        ) from None
    except Exception as error:
        if _is_kit_refusal(error):
            raise OptimizationUnavailableError(f"{label}: {what} refused: {error}") from error
        raise


def _is_kit_refusal(error: BaseException) -> bool:
    """Whether a class in ``error``'s MRO is one of :data:`KIT_REFUSAL_CLASS_NAMES` (OpenDDE's ``BigRefusal`` included)."""
    return any(cls.__name__ in KIT_REFUSAL_CLASS_NAMES for cls in type(error).__mro__)


def _resolved_kernels(configs: Any) -> dict[str, str]:
    return {
        "triangle_attention": _flat(getattr(configs, "triangle_attention", None)),
        "triangle_multiplicative": _flat(getattr(configs, "triangle_multiplicative", None)),
        "dtype": _flat(getattr(configs, "dtype", None)),
    }


def _check_template_counts(
    label: str, staging: dict[str, Any], calls: list[tuple[str, int, list[str]]]
) -> dict[str, str]:
    """Fail unless exactly the staged templates were featurized for the staged chain, and none elsewhere.

    A dropped template is a property of this input, not of the optimization, so it is a ``RuntimeError``
    (the request fails; nothing retries or falls back to another mode).
    """
    staged = int(staging["count"])
    query = str(staging["query"])
    on_query = [count for seq, count, _ in calls if seq == query]
    elsewhere = sum(count for seq, count, _ in calls if seq != query)
    featurized = max(on_query) if on_query else 0
    total = sum(count for _, count, _ in calls)
    if staged not in on_query or total != staged:
        notes = "; ".join(note for _, _, call_notes in calls for note in call_notes)[-4000:]
        raise RuntimeError(
            f"{label} featurized {featurized} of {staged} staged template(s) for the templated chain and "
            f"{elsewhere} for other chains ({len(calls)} featurizer call(s)); upstream said: {notes or 'nothing'}"
        )
    return {
        "templates.staged": str(staged),
        "templates.featurized": str(featurized),
        "templates.other_chains": str(elsewhere),
    }


def _report_facts(report: dict[str, Any]) -> dict[str, str]:
    return {f"kit.{key}": _flat(report[key]) for key in _REPORT_KEYS if key in report}


def _flat(value: Any) -> str:
    """One string for a report value: lists comma-joined, mappings as sorted JSON, ``"none"`` for None or empty."""
    if value is None or (isinstance(value, list | tuple | set | frozenset) and not value):
        return NONE
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list | tuple | set | frozenset):
        items = sorted(value, key=str) if isinstance(value, set | frozenset) else value
        return ",".join(_flat(item) for item in items)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    return str(value)


def _package_version(dists: tuple[str, ...], module: str) -> str:
    from importlib import metadata

    for dist in dists:
        try:
            return metadata.version(dist)
        except metadata.PackageNotFoundError:
            continue
    loaded = sys.modules.get(module)
    version = getattr(loaded, "__version__", None)
    return str(version) if version else ABSENT


def _torch_info() -> dict[str, str]:
    torch = sys.modules.get("torch")
    if torch is None:
        return {"torch": NOT_LOADED, "cuda": NOT_LOADED, "gpu": NOT_LOADED}
    info = {
        "torch": str(getattr(torch, "__version__", None) or UNKNOWN),
        # A loaded CPU-only torch build has no CUDA: recorded as such, as boileroom/provenance.py does.
        "cuda": str(getattr(getattr(torch, "version", None), "cuda", None) or NONE),
    }
    try:
        info["gpu"] = torch.cuda.get_device_name() if torch.cuda.is_available() else NONE
    except Exception as error:  # a broken driver must not hide the rest of the record
        info["gpu"] = f"unavailable: {error}"
    return info


def _kit_commit(kit: Any) -> str:
    """The kit's source commit: its own attribute, else ``$BOILEROOM_KIT_COMMIT`` (set by the kit images), else unknown.

    No ``.git`` is walked: a checkout above the package need not be the kit's (an editable install inside another
    repository), and its HEAD would be recorded as the kit's commit.
    """
    for attribute in ("__commit__", "KIT_COMMIT", "COMMIT"):
        value = getattr(kit, attribute, None)
        if isinstance(value, str) and value:
            return value
    return os.environ.get(KIT_COMMIT_ENV) or UNKNOWN


def _kalign_path() -> str:
    path = shutil.which("kalign")
    if path is None:
        raise RuntimeError("templates need the kalign binary on PATH (apt-get install kalign)")
    return path
