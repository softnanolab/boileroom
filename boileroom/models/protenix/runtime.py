"""Adapter for the pinned Protenix 2.0 Python inference runner."""

from __future__ import annotations

import copy
import shutil
from pathlib import Path
from typing import Any


class ProtenixRuntime:
    """Load one runner and refresh request state without reloading its weights."""

    def __init__(self, config: dict[str, Any], work_dir: str) -> None:
        """Load weights after the worker has configured its environment."""
        mode = config.get("optimization", "vanilla")
        if mode != "vanilla":
            _enable_kit(mode)  # the kit refuses late activation, so it must precede any protenix import
        from runner.batch_inference import get_default_runner, inference_configs, init_logging

        init_logging()
        inference_configs["dump_dir"] = work_dir
        self.runner = get_default_runner(
            model_name=config["model_name"],
            seeds=[int(seed) for seed in config["seeds"].split(",")],
            n_cycle=config["cycle"],
            n_step=config["step"],
            n_sample=config["sample"],
            dtype=config["dtype"],
            use_msa=config["use_msa"],
            use_template=config["use_template"],
            trimul_kernel=config["trimul_kernel"],
            triatt_kernel=config["triatt_kernel"],
            enable_cache=config["enable_cache"],
            enable_fusion=config["enable_fusion"],
            enable_tf32=config["enable_tf32"],
            use_tfg_guidance=config["use_tfg_guidance"],
            need_atom_confidence=True,
        )
        # Upstream mutates inference settings based on input length. Start each
        # request from a pristine copy while retaining the model's parameters.
        self._base_config = copy.deepcopy(self.runner.configs)

    def predict(self, input_json: str, output_dir: str, config: dict[str, Any]) -> None:
        """Run a request with fresh paths, seeds, sampling settings and dumper."""
        from runner.batch_inference import preprocess_input
        from runner.inference import infer_predict

        configs = copy.deepcopy(self._base_config)
        configs.seeds = [int(seed) for seed in config["seeds"].split(",")]
        configs.model.N_cycle = config["cycle"]
        configs.sample_diffusion.N_step = config["step"]
        configs.sample_diffusion.N_sample = config["sample"]
        configs.sample_diffusion.guidance.enable = config["use_tfg_guidance"]
        configs.dtype = config["dtype"]
        configs.use_msa = config["use_msa"]
        configs.dump_dir = output_dir
        staging = config.get("template_staging")
        use_template = config["use_template"] or staging is not None
        if staging is not None:
            # Caller-supplied templates: point the featurizer at the staged
            # directory and forbid it from fetching anything. The architecture
            # does not depend on use_template, so one runner serves both kinds
            # of request; only the data path differs.
            template = configs.data.template
            template.prot_template_mmcif_dir = staging["mmcif_dir"]
            template.release_dates_path = staging["release_dates_path"]
            template.obsolete_pdbs_path = staging["obsolete_pdbs_path"]
            template.fetch_remote = False
            template.kalign_binary_path = _kalign_path()
        configs.use_template = use_template
        configs.input_json_path = preprocess_input(
            input_json,
            out_dir=output_dir,
            use_msa=config["use_msa"],
            use_template=use_template,
            msa_server_mode="colabfold",
        )
        self.runner.configs = configs
        # Protenix also caches the cycle count directly on the model.
        self.runner.model.N_cycle = config["cycle"]
        self.runner.update_model_configs(configs)
        self.runner.init_basics()
        self.runner.init_dumper(need_atom_confidence=True, sorted_by_ranking_score=configs.sorted_by_ranking_score)
        infer_predict(self.runner, configs)
        errors = sorted(Path(self.runner.error_dir).glob("*.txt"))
        if errors:
            details = "\n".join(path.read_text(encoding="utf-8") for path in errors)
            raise RuntimeError(f"Protenix inference failed:\n{details[-12000:]}")


def _kalign_path() -> str:
    path = shutil.which("kalign")
    if path is None:
        raise RuntimeError("templates need the kalign binary on PATH (apt-get install kalign)")
    return path


def _enable_kit(mode: str) -> None:
    try:
        import protenix_opt
    except ImportError as error:
        raise RuntimeError(
            f"optimization={mode!r} needs the protenix kit image (protenix_opt is not installed)"
        ) from error
    report = protenix_opt.enable(mode, strict=True)
    if not report.get("active"):
        raise RuntimeError(f"optimization={mode!r} did not activate: {report.get('reason')}")
