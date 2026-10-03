"""Core OpenDDE implementation: Protenix's runner contract in an isolated interpreter."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar, cast

from ...optimization import OptimizationResolution
from ...utils import get_model_cache_dir
from ..protenix.core import ProtenixCore
from ..protenix.core import _command_env as _protenix_command_env
from .types import OpenDDEOutput

# OpenDDE 1.1.1 needs Python 3.11 and torch 2.7.1 (the kit-pinned stack), so it lives in its own
# virtualenv, like ColabFold for AlphaFold2. The boileroom interpreter never imports it.
DEFAULT_OPENDDE_PYTHON = "/opt/opendde/bin/python"


class OpenDDECore(ProtenixCore):
    """OpenDDE structure prediction model (AF3-style, Protenix 2.0 runner API)."""

    DEFAULT_CONFIG: ClassVar[dict[str, Any]] = {
        **ProtenixCore.DEFAULT_CONFIG,
        "model_name": "opendde_v1",
        "trimul_kernel": "auto",
        "triatt_kernel": "auto",
        "opendde_python": DEFAULT_OPENDDE_PYTHON,
    }
    STATIC_CONFIG_KEYS: ClassVar[frozenset[str]] = ProtenixCore.STATIC_CONFIG_KEYS | {"opendde_python"}
    FAMILY: ClassVar[str] = "opendde"
    DISPLAY_NAME: ClassVar[str] = "OpenDDE"
    ROOT_ENV: ClassVar[str] = "OPENDDE_ROOT_DIR"
    RUNTIME_CLASS: ClassVar[str] = "OpenDDERuntime"
    RUNTIME_PATH: ClassVar[Path] = Path(__file__).with_name("runtime.py")
    OUTPUT_CLASS: ClassVar[type[OpenDDEOutput]] = OpenDDEOutput
    DTYPES: ClassVar[frozenset[str]] = frozenset({"bf16", "fp32"})

    def fold(self, sequences: str | Sequence[str], options: dict | None = None) -> OpenDDEOutput:
        """Run OpenDDE prediction for one sequence entry; ``:`` joins protein chains."""
        return cast(OpenDDEOutput, super().fold(sequences, options))

    def _worker_python(self) -> str:
        return str(self.config["opendde_python"])

    def _worker_env(self, config: dict[str, Any], optimization: OptimizationResolution | None) -> dict[str, str]:
        return _command_env(config, optimization)


def _command_env(config: dict[str, Any], optimization: OptimizationResolution | None = None) -> dict[str, str]:
    """Return the OpenDDE worker environment: weights root, MSA server, kit and JIT caches."""
    env = _protenix_command_env(config, optimization, "opendde", "OPENDDE_ROOT_DIR")
    # Persist compiled kernels (Triton, the fused LayerNorm extension) next to the weights so a
    # restarted container skips the multi-minute JIT warm-up.
    cache = get_model_cache_dir("opendde") / "jit"
    env.setdefault("MODEL_OPT_JIT_ROOT", str(cache))
    env.setdefault("TRITON_CACHE_DIR", str(cache / "triton"))
    env.setdefault("TORCH_EXTENSIONS_DIR", str(cache / "torch_ext"))
    env.setdefault("MODEL_OPT_WEIGHTS_DIGEST_DIR", str(cache / "weights"))
    # Request upstream's fused LayerNorm extension in every mode. The image has no ninja, so the extension cannot
    # build and torch's layer_norm runs (docs/optimization.md); the measured speedups are without it.
    env.setdefault("LAYERNORM_TYPE", "fast_layernorm")
    # The kit's ``exact`` kernels need GLIBCXX_3.4.32 (GCC 13). That libstdc++ lives in the venv's ``lib`` and is
    # visible to this worker only: a global LD_LIBRARY_PATH entry would also load it into the system Python 3.12.
    venv_lib = Path(str(config["opendde_python"])).parent.parent / "lib"
    env["LD_LIBRARY_PATH"] = os.pathsep.join(filter(None, [str(venv_lib), env.get("LD_LIBRARY_PATH", "")]))
    return env
