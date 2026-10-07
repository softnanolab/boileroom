"""Modal functions the kit integration tests run inside the runtime images.

The tests import this module inside the test functions that need it, never at collection, and call its functions
under ``app.run()``. Modal mounts this file in the container as ``/root/_kit_remote.py`` and imports it there by name,
so it imports only the standard library, ``modal`` and ``boileroom`` at module scope (no pytest, no test helpers) and
stays Python 3.10-compatible (the Protenix kit image runs Python 3.11). Its functions return plain data; the tests
score it on the client.

Each purpose has its own ``modal.App``, because ``app.run()`` builds every image registered on its app: a guard image
(a kit image minus one package) is never built for a fallback run, nor the other way round.

Importing this module in a container evaluates all four images again, so each image carries the whole image lookup of
the run (the stock tag and the kit source and tag), not only its own: without it the kit container cannot resolve the
stock image. The boileroom source is mounted with its non-Python files (the kit Dockerfiles and scripts), since
:func:`~boileroom.images.modal.get_modal_kit_image` with ``BOILEROOM_KIT_IMAGE_SOURCE=build`` checks that the Dockerfile
is there.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import modal

from boileroom.images.metadata import (
    IMAGE_TAG_ENV,
    KIT_IMAGE_SOURCE_ENV,
    KIT_IMAGE_TAG_ENV,
    get_image_tag,
    get_kit_image_source,
    get_kit_image_tag,
)
from boileroom.images.modal import get_modal_image, get_modal_kit_image
from boileroom.images.volumes import model_weights
from boileroom.utils import MODAL_MODEL_DIR

#: ``KIT_REQUIRE_FAST_ENV`` of ``boileroom.models.esmfold2.core``, repeated so the client never imports the core.
_ESMFOLD2_REQUIRE_FAST_ENV = "ESMFOLD2_OPT_REQUIRE_FAST_ENV"
#: Module-name prefixes (vanilla esm 3.4.1, and the kit's transformers fork) and the flash-attention switches they read.
_FLASH_MODULE_PREFIXES = ("esm.models", "transformers.models.esm")
_FLASH_FLAGS = (
    "FLASH_ATTN_AVAILABLE",
    "FLASH_ATTN_INSTALLED",
    "FLASH_ATTN_ROTARY_INSTALLED",
    "_flash_attn_available",
    "_flash_attn_rotary_available",
)
_KIT_ATTN_WORDS = ("atom_attn", "esmc_mlp", "esmc_attn", "esmc_rope")
_VOLUMES = {MODAL_MODEL_DIR: model_weights}


def _skip_bytecode(path: Path) -> bool:
    """Ignore rule of the boileroom source mount: everything but bytecode caches is mounted."""
    return "__pycache__" in Path(path).parts or Path(path).suffix == ".pyc"


def _image_lookup_env() -> dict[str, str]:
    """Image lookup of this run: the stock tag, the kit source and, when set, the kit tag."""
    env = {IMAGE_TAG_ENV: get_image_tag(), KIT_IMAGE_SOURCE_ENV: get_kit_image_source()}
    if (kit_tag := get_kit_image_tag()) is not None:
        env[KIT_IMAGE_TAG_ENV] = kit_tag
    return env


def _with_source(image: modal.Image) -> modal.Image:
    return image.env(_image_lookup_env()).add_local_python_source("boileroom", ignore=_skip_bytecode)


def _without_module(image: modal.Image, distributions: tuple[str, ...], module: str) -> modal.Image:
    """Return ``image`` with ``distributions`` uninstalled; the build fails if ``module`` can still be found."""
    check = f"import importlib.util, sys; sys.exit(importlib.util.find_spec({module!r}) is not None)"
    return image.run_commands(
        f"python -m pip uninstall -y {' '.join(distributions)}",
        f'python -c "{check}" || (echo "{module} is still importable after the uninstall" >&2; exit 1)',
    )


esmfold2_kit_fallback_app = modal.App("boileroom-test-esmfold2-kit-fallback")
esmfold2_stock_fallback_app = modal.App("boileroom-test-esmfold2-stock-fallback")
esmfold2_kit_guard_app = modal.App("boileroom-test-esmfold2-kit-guard")
protenix_kit_guard_app = modal.App("boileroom-test-protenix-kit-guard")

esmfold2_kit_image = _with_source(get_modal_kit_image("esmfold2"))
esmfold2_stock_image = _with_source(get_modal_image("esmfold2"))
esmfold2_guard_image = _with_source(_without_module(get_modal_kit_image("esmfold2"), ("flash-attn",), "flash_attn"))
protenix_guard_image = _with_source(
    _without_module(
        get_modal_kit_image("protenix"),
        ("cuequivariance-ops-torch-cu13", "cuequivariance-ops-torch-cu12"),
        "cuequivariance_ops_torch",
    )
)


def _set_flash_flags(value: bool) -> list[tuple[str, str, bool]]:
    """Set every loaded flash-attention switch to ``value``; return ``(module, flag, previous value)`` per switch."""
    import sys

    changed: list[tuple[str, str, bool]] = []
    for name, module in list(sys.modules.items()):
        if module is None or not name.startswith(_FLASH_MODULE_PREFIXES):
            continue
        for flag in _FLASH_FLAGS:
            if hasattr(module, flag):
                changed.append((name, flag, bool(getattr(module, flag))))
                setattr(module, flag, value)
    return changed


def _restore_flags(changed: list[tuple[str, str, bool]]) -> None:
    import sys

    for name, flag, previous in changed:
        setattr(sys.modules[name], flag, previous)


def _esmfold2_fallback(mode: str, sequence: str, seeds: list[int], options: dict[str, Any]) -> dict[str, Any]:
    """Fold ``sequence`` per seed as served, then with every flash-attention switch forced off, on one loaded core.

    In a kit mode the post-fold settle (which would refuse the forced-off folds) is bypassed for them, then run once
    with the switches still off to show that it refuses; ``ESMFOLD2_OPT_REQUIRE_FAST_ENV`` is ``0`` while they run.

    Returns
    -------
    dict[str, Any]
        ``served`` / ``forced``: PAE per seed (float32); ``flipped``: ``(module, flag, previous)`` per switch;
        ``served_words`` / ``forced_words``: the kit's attention words (kit modes); ``settle_error``: the refusal of
        the real settle after the forced folds (kit modes); ``runtime``: the served ``metadata.runtime``.
    """
    import importlib
    import os

    import numpy as np

    from boileroom.models.esmfold2.core import ESMFold2Core
    from boileroom.optimization import initialize_core

    core: Any = ESMFold2Core({"optimization": mode})
    failure = initialize_core(core)
    if failure is not None:
        raise failure
    kit = mode != "vanilla"
    attn: Any = importlib.import_module("esmfold2_opt.attn") if kit else None

    def words() -> dict[str, str] | None:
        if attn is None:
            return None
        state = attn.state(core.model)
        return {word: str(state.get(word, "unread")) for word in _KIT_ATTN_WORDS}

    def fold_all() -> tuple[list[Any], dict[str, str]]:
        paes: list[Any] = []
        runtime: dict[str, str] = {}
        for seed in seeds:
            output = core.fold(sequence, options={**options, "seed": seed})
            if output.pae is None:
                raise RuntimeError("the fold returned no PAE; options must include it")
            paes.append(np.asarray(output.pae[0], dtype=np.float32))
            runtime = dict(output.metadata.runtime or {})
        return paes, runtime

    served_words = words()
    served, runtime = fold_all()
    flipped = _set_flash_flags(False)
    result: dict[str, Any] = {
        "served": served,
        "forced": [],
        "flipped": flipped,
        "served_words": served_words,
        "forced_words": None,
        "settle_error": None,
        "runtime": runtime,
    }
    if not any(previous for _, _, previous in flipped):
        _restore_flags(flipped)
        return result
    previous_require = os.environ.get(_ESMFOLD2_REQUIRE_FAST_ENV)
    try:
        if kit:
            # The real settle would refuse each forced-off fold; it runs once afterwards instead.
            core._settle_kit_report = lambda num_diffusion_samples: dict(core._runtime or {})
            os.environ[_ESMFOLD2_REQUIRE_FAST_ENV] = "0"
        result["forced_words"] = words()
        result["forced"], _ = fold_all()
        if kit:
            del core._settle_kit_report
            if previous_require is None:
                os.environ.pop(_ESMFOLD2_REQUIRE_FAST_ENV, None)
            else:
                os.environ[_ESMFOLD2_REQUIRE_FAST_ENV] = previous_require
            try:
                core._settle_kit_report(int(options["num_diffusion_samples"]))
            except Exception as error:
                result["settle_error"] = f"{type(error).__name__}: {error}"
    finally:
        _restore_flags(flipped)
        core.__dict__.pop("_settle_kit_report", None)
        if previous_require is None:
            os.environ.pop(_ESMFOLD2_REQUIRE_FAST_ENV, None)
        else:
            os.environ[_ESMFOLD2_REQUIRE_FAST_ENV] = previous_require
    return result


@esmfold2_kit_fallback_app.function(image=esmfold2_kit_image, gpu="A100-80GB", timeout=60 * 60, volumes=_VOLUMES)
def esmfold2_kit_fallback(mode: str, sequence: str, seeds: list[int], options: dict[str, Any]) -> dict[str, Any]:
    """:func:`_esmfold2_fallback` on the ESMFold2 kit image (``mode`` is ``"exact"`` or ``"fast"``)."""
    return _esmfold2_fallback(mode, sequence, seeds, options)


@esmfold2_stock_fallback_app.function(image=esmfold2_stock_image, gpu="A100-80GB", timeout=60 * 60, volumes=_VOLUMES)
def esmfold2_stock_fallback(sequence: str, seeds: list[int], options: dict[str, Any]) -> dict[str, Any]:
    """:func:`_esmfold2_fallback` of ``optimization="vanilla"`` on the stock ESMFold2 image."""
    return _esmfold2_fallback("vanilla", sequence, seeds, options)


def _guard_report(core: Any, extra: dict[str, Any]) -> dict[str, Any]:
    """Load ``core`` twice the way a Modal server does and describe the failure, as plain data."""
    from boileroom.optimization import OptimizationUnavailableError, initialize_core, is_refusal, retry_initialize

    failure = initialize_core(core)
    again = retry_initialize(core, failure)
    reload = initialize_core(core) if failure is not None else None
    return {
        **extra,
        "failure_type": type(failure).__name__ if failure is not None else None,
        "failure_is_typed": isinstance(failure, OptimizationUnavailableError),
        "failure_is_refusal": failure is not None and is_refusal(failure),
        "failure": str(failure) if failure is not None else None,
        "retry_kept_failure": again is failure,
        "reload_failure": f"{type(reload).__name__}: {reload}" if reload is not None else None,
        "ready": bool(getattr(core, "ready", False)),
    }


@esmfold2_kit_guard_app.function(image=esmfold2_guard_image, gpu="A100-40GB", timeout=20 * 60)
def esmfold2_kit_guard() -> dict[str, Any]:
    """Load ``optimization="exact"`` on the kit image without flash-attn, with an empty ``HF_HOME`` and no volume.

    Returns
    -------
    dict[str, Any]
        Whether ``flash_attn`` is importable, the load failure and its retry (see :func:`_guard_report`), the weight
        fetches the core attempted and the files that appeared under ``HF_HOME``.
    """
    import importlib.util
    import os
    import tempfile

    hf_home = Path(tempfile.mkdtemp(prefix="kit-guard-hf-"))
    os.environ["HF_HOME"] = str(hf_home)

    from boileroom.models.esmfold2.core import ESMFold2Core

    core: Any = ESMFold2Core({"optimization": "exact"})
    fetches: list[str] = []
    fetch = core._ensure_kit_weights

    def recording_fetch(hf_home: Path) -> None:
        fetches.append(str(hf_home))
        fetch(hf_home)

    core._ensure_kit_weights = recording_fetch
    report = _guard_report(core, {"flash_attn_importable": importlib.util.find_spec("flash_attn") is not None})
    report["fetches"] = fetches
    report["hf_home_files"] = sorted(str(p.relative_to(hf_home)) for p in hf_home.rglob("*") if p.is_file())
    return report


@protenix_kit_guard_app.function(image=protenix_guard_image, gpu="A100-40GB", timeout=20 * 60)
def protenix_kit_guard() -> dict[str, Any]:
    """Load ``optimization="exact"`` on the Protenix kit image without ``cuequivariance_ops_torch`` and no volume.

    Returns
    -------
    dict[str, Any]
        Whether ``cuequivariance_ops_torch`` is importable and the load failure and its retry (see
        :func:`_guard_report`).
    """
    import importlib.util

    from boileroom.models.protenix.core import ProtenixCore

    core = ProtenixCore({"optimization": "exact"})
    importable = importlib.util.find_spec("cuequivariance_ops_torch") is not None
    report = _guard_report(core, {"cuequivariance_ops_torch_importable": importable})
    core.close()
    return report
