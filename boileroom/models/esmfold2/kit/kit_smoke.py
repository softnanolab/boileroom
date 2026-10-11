"""Build-time smoke check of the ESMFold2 kit image's accelerated kernels.

Runs as the last step of ``kit_finish.sh`` (``python -I``, so no boileroom and no working-directory imports) on the
image builder, which has no GPU: every check here is an import or an import-time flag, never a kernel launch. It fails
the build when a compiled kernel wheel does not import or when the fork would select a slow fallback path, so that a
broken image is never published for the runtime refusal to discover later.

The fork's ``_flash_attn_available`` and the rotary flag are deliberately not asserted: they read the GPU at import
time and are false on a CPU builder. The run-time gate in ``boileroom.models.esmfold2.core`` checks them on the GPU.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable
from types import ModuleType

#: Compiled kernel modules the kit's fast paths import.
KERNEL_MODULES = (
    "flash_attn",
    "flash_attn_2_cuda",
    "flash_attn.ops.triton.rotary",
    "transformer_engine.pytorch",
    "xformers.ops",
)
#: Import-time switches of the transformers fork that must be true for the fast paths, by module attribute.
ESMC_FLAGS = ("_te_available", "_xformers_available")

ImportModule = Callable[[str], ModuleType]


def check(import_module: ImportModule = importlib.import_module) -> tuple[list[str], str]:
    """Run every smoke check and collect the failures.

    Parameters
    ----------
    import_module : Callable[[str], ModuleType]
        Module importer; tests pass a fake.

    Returns
    -------
    tuple[list[str], str]
        The failure messages (empty when the image passes) and the kit's metadata words, or ``"unread"`` when
        ``esmfold2_opt.attn`` does not import.
    """
    failures: list[str] = []

    def load(name: str) -> ModuleType | None:
        try:
            return import_module(name)
        except Exception as error:
            failures.append(f"import {name}: {type(error).__name__}: {error}")
            return None

    for name in KERNEL_MODULES:
        load(name)
    attn = load("esmfold2_opt.attn")
    if attn is None:
        return failures, "unread"
    for _word, _want, module in attn.REQUIRED:
        load(module)
    common = load(attn.COMMON_MODULE)
    if common is not None and not getattr(common, attn.FLAG, False):
        failures.append(f"{attn.COMMON_MODULE}.{attn.FLAG} is false: atom attention would fall back to SDPA")
    esmc = load(attn.ESMC_MODULE)
    if esmc is not None:
        for flag in ESMC_FLAGS:
            if not getattr(esmc, flag, False):
                failures.append(f"{attn.ESMC_MODULE}.{flag} is false: the ESMC language model would use a slow path")
    try:
        refusal = attn.require_refusal_metadata(environ={attn.ENV_REQUIRE: "1"})
    except Exception as error:
        failures.append(f"esmfold2_opt.attn.require_refusal_metadata: {type(error).__name__}: {error}")
    else:
        if refusal is not None:
            failures.append(f"the kit refuses this image with {attn.ENV_REQUIRE}=1: {refusal}")
    try:
        words = str(attn.metadata_words())
    except Exception as error:
        failures.append(f"esmfold2_opt.attn.metadata_words: {type(error).__name__}: {error}")
        words = "unread"
    return failures, words


def main(import_module: ImportModule = importlib.import_module) -> int:
    """Print the kit's metadata words and return ``0``, or report every failure on stderr and return ``1``."""
    failures, words = check(import_module)
    print(f"esmfold2 kit smoke: {words}")
    if failures:
        print("esmfold2 kit smoke FAILED:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
