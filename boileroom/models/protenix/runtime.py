"""Adapter for the pinned Protenix 2.0 Python inference runner."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any


class ProtenixRuntime:
    """Load one runner and refresh request state without reloading its weights."""

    def __init__(self, config: dict[str, Any], work_dir: str) -> None:
        """Load weights after the worker has configured its environment."""
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
        configs.input_json_path = preprocess_input(
            input_json, out_dir=output_dir, use_msa=config["use_msa"], use_template=config["use_template"]
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
