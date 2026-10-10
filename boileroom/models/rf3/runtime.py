"""Adapter for the pinned RoseTTAFold 3 (RosettaCommons foundry) inference engine.

Runs inside the isolated worker interpreter: nothing here may import boileroom.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import urllib.request
from pathlib import Path
from typing import Any

CHECKPOINT_NAME = "rf3_foundry_01_24_latest_remapped.ckpt"
CHECKPOINT_URL = f"https://files.ipd.uw.edu/pub/rf3/{CHECKPOINT_NAME}"
CHECKPOINT_SHA256 = "364ef592fd8042a9cf4176d045015190f8322f961ccca38d891b20ca578d3bb0"
CHECKPOINT_BYTES = 3_038_876_446


class RF3Runtime:
    """Load one inference engine and serve requests without reloading its weights."""

    def __init__(self, config: dict[str, Any], work_dir: str) -> None:
        """Fetch the checkpoint, activate the kit when asked to, and load the model."""
        checkpoint = _resolve_checkpoint(config)
        mode = config.get("optimization", "vanilla")
        self.mode = str(mode)
        self.checkpoint_sha256 = _sha256(checkpoint)
        self.kit_report: dict[str, Any] = {}
        if mode != "vanilla":
            # The kit reads the checkpoint path from its own variable, and refuses late activation:
            # it has to precede any rf3 import.
            os.environ["ROSETTAFOLD3_OPT_CKPT"] = str(checkpoint)
            self.kit_report = _enable_kit(mode)
        from rf3.inference_engines.rf3 import RF3InferenceEngine

        self.engine = RF3InferenceEngine(
            ckpt_path=checkpoint,
            n_recycles=config["n_recycles"],
            diffusion_batch_size=config["diffusion_batch_size"],
            num_steps=config["num_steps"],
            seed=config["seed"],
        )
        # Load the weights now so that a failure surfaces at startup, not in the first request.
        self.engine.initialize()

    def describe(self) -> dict[str, str]:
        """Describe the actual isolated inference interpreter and verified weights."""
        torch = sys.modules.get("torch")
        facts = {
            "python": platform.python_version(),
            "torch": str(getattr(torch, "__version__", "not-loaded")),
            "cuda": str(getattr(getattr(torch, "version", None), "cuda", None) or "none"),
            "optimization": self.mode,
            "weights.sha256": self.checkpoint_sha256,
        }
        if torch is not None and torch.cuda.is_initialized():
            device = torch.cuda.current_device()
            facts["gpu"] = torch.cuda.get_device_name(device)
            facts["gpu_capability"] = "sm" + "".join(map(str, torch.cuda.get_device_capability(device)))
        try:
            facts["rc-foundry"] = importlib.metadata.version("rc-foundry")
        except importlib.metadata.PackageNotFoundError:
            facts["rc-foundry"] = "absent"
        if self.kit_report:
            self.kit_report = sys.modules["rosettafold3_opt"].status()
            if not self.kit_report.get("active"):
                raise RuntimeError(f"RF3 optimization={self.mode!r} became inactive: {self.kit_report.get('reason')}")
            facts["kit.commit"] = os.environ.get("BOILEROOM_KIT_COMMIT", "unknown")
            for key in ("active", "partial", "levers_applied", "levers_fallback"):
                facts[f"kit.{key}"] = json.dumps(self.kit_report.get(key), sort_keys=True)
        return facts

    def predict(self, input_json: str, output_dir: str, config: dict[str, Any]) -> dict[str, str]:
        """Fold ``input_json`` into ``output_dir/<example name>/`` with this request's seed and threshold."""
        from lightning.fabric import seed_everything

        seed = int(config["seed"])
        seed_everything(seed, workers=True)
        # The engine names its output after ``seed`` and applies the threshold on every run.
        self.engine.seed = seed
        self.engine.early_stopping_plddt_threshold = config["early_stopping_plddt_threshold"]
        self.engine.run(inputs=input_json, out_dir=output_dir)
        return self.describe()


def _resolve_checkpoint(config: dict[str, Any]) -> Path:
    """Return the checkpoint path, downloading and verifying the pinned release when it is not on disk."""
    override = config.get("checkpoint_path")
    if override:
        path = Path(override)
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint_path does not exist: {path}")
        return path
    root = Path(os.environ["RF3_ROOT_DIR"])
    path = root / CHECKPOINT_NAME
    if path.is_file() and path.stat().st_size == CHECKPOINT_BYTES and _sha256(path) == CHECKPOINT_SHA256:
        return path
    root.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.{os.getpid()}.part")
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(CHECKPOINT_URL, timeout=300) as response, partial.open("wb") as handle:
            while chunk := response.read(1 << 24):
                handle.write(chunk)
                digest.update(chunk)
        if digest.hexdigest() != CHECKPOINT_SHA256:
            raise RuntimeError(
                f"RF3 checkpoint from {CHECKPOINT_URL} has sha256 {digest.hexdigest()}, expected {CHECKPOINT_SHA256}"
            )
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)
    return path


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
        return digest.hexdigest()


def _enable_kit(mode: str) -> dict[str, Any]:
    try:
        import rosettafold3_opt
    except ImportError as error:
        raise RuntimeError(
            f"optimization={mode!r} needs the rosettafold3 kit image (rosettafold3_opt is not installed)"
        ) from error
    try:
        report = rosettafold3_opt.enable(mode, strict=True)
    except SystemExit as error:  # the kit exits (code 3) after printing why it is not active
        raise RuntimeError(f"optimization={mode!r} did not activate: the kit exited with {error.code}") from error
    if not report.get("active"):
        raise RuntimeError(f"optimization={mode!r} did not activate: {report.get('reason')}")
    return report
