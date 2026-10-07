"""Core OpenDDE implementation: Protenix's runner contract in an isolated interpreter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar, Final, cast

from ...utils import get_model_cache_dir
from ..protenix.core import ProtenixCore, _prepend_path
from .types import OpenDDEOutput

# OpenDDE 1.1.1 needs Python 3.11 and torch 2.7.1 (the kit-pinned stack), so it lives in its own
# virtualenv, like ColabFold for AlphaFold2. The boileroom interpreter never imports it.
DEFAULT_OPENDDE_PYTHON = "/opt/opendde/bin/python"
#: ``LAYERNORM_TYPE`` the OpenDDE worker runs with, per optimization mode, set over any inherited value. Vanilla is
#: ``torch``, upstream's own default and stock OpenDDE as released. The kit modes need ``fast_layernorm``: the kit's
#: stock base and its lnstream lever run upstream's fused LayerNorm, which JIT-builds at first use with the venv's ninja
#: and the CUDA 12.6 compiler, and the runtime refuses the mode if it does not load.
OPENDDE_LAYERNORM: Final[Mapping[str, str]] = {
    "vanilla": "torch",
    "exact": "fast_layernorm",
    "fast": "fast_layernorm",
}
#: The kit image's process environment (``opendde/environment/Dockerfile`` ENV at the kit commit): a fixed hash seed,
#: unbuffered output, no bytecode written into the kit tree, no debug info in the launchers Triton builds at run time.
#: Set on the worker only, over any inherited value, so boileroom's own Python 3.12 server keeps its defaults.
OPENDDE_PROCESS_ENV: Final[Mapping[str, str]] = {
    "PYTHONHASHSEED": "0",
    "PYTHONUNBUFFERED": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "CFLAGS": "-g0",
}
#: OpenDDE's one released checkpoint has a template embedder (``template_embedder.n_blocks = 2``).
OPENDDE_TEMPLATE_MODELS: Final[frozenset[str]] = frozenset({"opendde_v1"})


class OpenDDECore(ProtenixCore):
    """OpenDDE structure prediction model (AF3-style, Protenix 2.0 runner API).

    The triangle kernels default to ``cuequivariance``, as Protenix's do: upstream's ``auto`` lets it pick another
    kernel by itself, whereas an explicit request runs cuEquivariance or fails. The kernel that ran is recorded in
    ``metadata.runtime`` (``predict.kernel.resolved.*``).
    """

    DEFAULT_CONFIG: ClassVar[dict[str, Any]] = {
        **ProtenixCore.DEFAULT_CONFIG,
        "model_name": "opendde_v1",
        "opendde_python": DEFAULT_OPENDDE_PYTHON,
    }
    STATIC_CONFIG_KEYS: ClassVar[frozenset[str]] = ProtenixCore.STATIC_CONFIG_KEYS | {"opendde_python"}
    FAMILY: ClassVar[str] = "opendde"
    DISPLAY_NAME: ClassVar[str] = "OpenDDE"
    ROOT_ENV: ClassVar[str] = "OPENDDE_ROOT_DIR"
    RUNTIME_CLASS: ClassVar[str] = "OpenDDERuntime"
    # One runtime module serves both families (they share the runner and template pipeline).
    RUNTIME_PATH: ClassVar[Path] = Path(__file__).parent.parent / "protenix" / "runtime.py"
    OUTPUT_CLASS: ClassVar[type[OpenDDEOutput]] = OpenDDEOutput
    DTYPES: ClassVar[frozenset[str]] = frozenset({"bf16", "fp32"})
    LAYERNORM_BY_MODE: ClassVar[Mapping[str, str]] = OPENDDE_LAYERNORM
    TEMPLATE_MODELS: ClassVar[frozenset[str]] = OPENDDE_TEMPLATE_MODELS
    #: The core's interpreter does not hold OpenDDE; the worker's ``describe()`` reports the venv's packages.
    RUNTIME_PACKAGES: ClassVar[tuple[str, ...]] = ()

    def fold(self, sequences: str | Sequence[str], options: dict | None = None) -> OpenDDEOutput:
        """Run OpenDDE prediction for one sequence entry; ``:`` joins protein chains."""
        return cast(OpenDDEOutput, super().fold(sequences, options))

    def _worker_python(self) -> str:
        """Return the venv interpreter that runs OpenDDE."""
        return str(self.config["opendde_python"])

    def _extend_worker_env(self, env: dict[str, str], config: dict[str, Any]) -> None:
        """Add the JIT cache root, the kit's process environment and the venv's search paths to ``env``.

        Parameters
        ----------
        env : dict[str, str]
            The worker environment, updated in place.
        config : dict[str, Any]
            The core's configuration (``opendde_python`` names the venv).
        """
        # Persist compiled kernels (Triton, the fused LayerNorm extension) next to the weights so a restarted container
        # skips the multi-minute JIT warm-up. The runtime keys the Triton and torch-extension caches under this root by
        # the stack (torch, CUDA, compute capability), as the kit's configs/<gpu>.env does, so A100 and H100 workers
        # sharing the volume never load each other's builds.
        cache = get_model_cache_dir(self.FAMILY) / "jit"
        env.setdefault("MODEL_OPT_JIT_ROOT", str(cache))
        env.setdefault("MODEL_OPT_WEIGHTS_DIGEST_DIR", str(cache / "weights"))
        # The kit's process environment (its image's ENV); the worker gets it whatever launched it.
        env.update(OPENDDE_PROCESS_ENV)
        # The fused LayerNorm extension is built by torch's extension builder, which runs ``ninja`` from PATH; the
        # venv's ninja (the kit lock pins it) sits in the interpreter's ``bin``, off the image PATH.
        venv_bin = Path(str(config["opendde_python"])).parent
        _prepend_path(env, "PATH", str(venv_bin))
        # The kit's ``exact`` kernels need GLIBCXX_3.4.32 (GCC 13). That libstdc++ lives in the venv's ``lib`` and is
        # visible to this worker only: a global LD_LIBRARY_PATH entry would also load it into the system Python 3.12.
        _prepend_path(env, "LD_LIBRARY_PATH", str(venv_bin.parent / "lib"))
