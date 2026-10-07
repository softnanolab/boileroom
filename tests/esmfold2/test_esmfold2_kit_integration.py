"""Opt-in GPU tests of ESMFold2's ``optimization`` modes (vanilla, exact, fast) on the bakeoff MDM2 heterodimers.

Skipped unless ``--run-kit`` or ``BOILEROOM_RUN_KIT=1`` is given; they fold on paid GPUs. Pull the kit image by its
pinned digest (``BOILEROOM_KIT_IMAGE_SOURCE=registry``): without it Modal builds the ESMFold2 kit image from its
Dockerfile first (about 45 minutes of a 48-CPU builder). All commands run from the repository root::

    # everything (~45 A100-80GB minutes plus ~3 A100-40GB minutes, about $2.2)
    BOILEROOM_KIT_IMAGE_SOURCE=registry UV_CACHE_DIR=$TMPDIR/uv timeout 7200 uv run pytest \
        tests/esmfold2/test_esmfold2_kit_integration.py --run-kit --gpu A100-80GB -s -rs
    # one test: add -k heterodimer | -k "heterodimer and exact" | -k fallback | -k "fallback and fast" | -k guard

Tests and their cost on Modal (the first load of a mode reads ~27 GB of kit weights from the ``model-weights`` volume,
or downloads them into it on the very first run):

- ``test_kit_parity_heterodimer[vanilla|exact|fast]``: one live model per mode folds p53 at seeds 0, 1, 2 and the
  decoy at seed 0 through the public wrapper; ~4-5 A100-80GB minutes per mode, ~15 for all three. Vanilla is folded
  once per module and reused by the exact / fast comparisons (``-k "heterodimer and exact"`` alone folds vanilla too).
- ``test_fallback_parity[vanilla|exact|fast]``: in the container, folds p53 at three seeds as served, then with every
  flash-attention switch forced off, on one loaded core; ~6-8 A100-80GB minutes per mode.
- ``test_kit_kernel_guard``: the kit image minus flash-attn (a one-layer image built on top of it); the load must
  refuse before any weight fetch. ~2-3 A100-40GB minutes.

On Apptainer only the parity tests run (``--backend apptainer --device cuda:0``); the in-container tests skip.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from _kit_parity import (
    FoldCache,
    FoldSet,
    assert_parity,
    check_shared_kit_runtime,
    fold_heterodimers,
    heterodimer,
    ipsae_min,
    require_modal,
    wrapper_device,
)
from _structure_metrics import BINDERS, GOLDENS, OPTIMIZATION_MODES, SEEDS, TARGET, TOLERANCE, chain_index_from_lengths

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.gpu,
    pytest.mark.kit,
    pytest.mark.xdist_group("esmfold2-kit"),
]

FAMILY = "esmfold2"
#: The bakeoff's ESMFold2 fold parameters, as the goldens were measured (no MSA).
FOLD_OPTIONS = {
    "num_loops": 3,
    "num_sampling_steps": 50,
    "num_diffusion_samples": 1,
    "msa_max_depth": 1024,
    "msa_column_mask_rate": 0.1,
    "include_fields": ["plddt", "ptm", "iptm", "pae"],
}
#: The kit's accelerated attention / MLP words on the pinned stack (``esmfold2_opt.attn.REQUIRED`` plus esmc_attn).
KIT_WORDS = {"atom_attn": "flash_attn", "esmc_mlp": "te", "esmc_rope": "flash_attn_triton"}


def _chain_index(binder: str) -> np.ndarray:
    return chain_index_from_lengths(len(TARGET), len(BINDERS[binder]))


@pytest.fixture(scope="module")
def folds(backend_option: str, device_option: str | None, output_ctx) -> FoldCache:
    """Fold sets per mode, each on its own live model, folded on first use."""
    from boileroom import ESMFold2

    device = wrapper_device(FAMILY, backend_option, device_option)

    def make(mode: str) -> FoldSet:
        with output_ctx(), ESMFold2(backend=backend_option, device=device, config={"optimization": mode}) as model:

            def fold(binder: str, seed: int) -> tuple[float, object]:
                result = model.fold(heterodimer(binder), options={**FOLD_OPTIONS, "seed": seed})
                length = len(TARGET) + len(BINDERS[binder])
                assert result.pae is not None and result.pae[0].shape == (length, length)
                return ipsae_min(result.pae[0], _chain_index(binder)), result.metadata

            return fold_heterodimers(FAMILY, mode, fold)

    return FoldCache(make)


def _check_runtime(mode: str, metadata: Any) -> None:
    """``metadata.optimization`` names the mode and GPU; ``metadata.runtime`` records the GPU and the kit's levers."""
    optimization = metadata.optimization
    runtime = metadata.runtime
    assert optimization is not None and optimization["mode"] == mode, optimization
    assert runtime is not None, "metadata.runtime is missing"
    assert runtime.get("gpu", "none") != "none" and runtime.get("gpu_capability", "none").startswith("sm"), runtime
    check_shared_kit_runtime(mode, runtime)
    if mode == "vanilla":
        assert optimization["kit_config"] is None, optimization
        assert "esm.FLASH_ATTN_AVAILABLE" in runtime, sorted(runtime)
        return
    assert optimization["kit_config"] in ("a100", "h100"), optimization
    assert optimization["gpu_name"] == runtime["gpu"], (optimization, runtime["gpu"])
    for key in ("kit.stack", "kit.package", "kit.variant", "kit.weights", "kit.attn", "kit.scope"):
        assert key in runtime, f"metadata.runtime lacks {key}: {sorted(runtime)}"
    assert runtime["kit.variant"] == "full_nomsa", runtime["kit.variant"]
    assert runtime["kit.require_fast_env"] == "1", runtime["kit.require_fast_env"]
    for word, value in KIT_WORDS.items():
        key = f"kit.attn.{word}"
        assert runtime.get(key) == value, f"{key}={runtime.get(key)!r}, expected {value!r}"
    print(
        f"esmfold2 {mode} runtime: gpu={runtime['gpu']} kit.commit={runtime['kit.commit']} "
        f"levers={runtime['kit.levers_applied']} gated={runtime.get('kit.levers_gated')} attn={runtime['kit.attn']}"
    )


@pytest.mark.parametrize("mode", OPTIMIZATION_MODES)
def test_kit_parity_heterodimer(mode: str, folds: FoldCache) -> None:
    """Seed-averaged ipSAE of MDM2 + p53 matches the golden and vanilla; the binder separates from the decoy."""
    fold_set = folds.get(mode)
    for metadata in fold_set.metadata:
        _check_runtime(mode, metadata)
    assert_parity(fold_set, None if mode == "vanilla" else folds.get("vanilla"))


@pytest.mark.parametrize("mode", OPTIMIZATION_MODES)
def test_fallback_parity(mode: str, backend_option: str) -> None:
    """With flash-attention forced off on a loaded model, ipSAE stays within the tolerance of the served path.

    Guards the regression where the fork's non-flash attention dropped the sliding window (p53 ipSAE ≈ 0.29 instead of
    ≈ 0.39): a card or image without flash-attn must fold the same complex. In a kit mode the post-fold check must
    also refuse the forced-off state (``after the fold``).
    """
    require_modal(backend_option)
    import _kit_remote
    import modal

    sequence = heterodimer("p53")
    options = {**FOLD_OPTIONS, "include_fields": ["pae"]}
    with modal.enable_output():
        if mode == "vanilla":
            with _kit_remote.esmfold2_stock_fallback_app.run():
                result = _kit_remote.esmfold2_stock_fallback.remote(sequence, list(SEEDS), options)
        else:
            with _kit_remote.esmfold2_kit_fallback_app.run():
                result = _kit_remote.esmfold2_kit_fallback.remote(mode, sequence, list(SEEDS), options)

    flipped = result["flipped"]
    print(f"esmfold2 {mode} flash switches (module, flag, served value): {flipped}")
    if not any(previous for _, _, previous in flipped):
        if mode == "vanilla":
            pytest.skip("the stock image serves vanilla without flash-attention already; nothing to force off")
        pytest.fail(f"no flash-attention switch was on in the {mode} kit image: {flipped}")
    chains = _chain_index("p53")
    served = [ipsae_min(pae, chains) for pae in result["served"]]
    forced = [ipsae_min(pae, chains) for pae in result["forced"]]
    line = f"esmfold2 {mode}: served {np.round(served, 5).tolist()} forced-off {np.round(forced, 5).tolist()}"
    print(line)
    assert len(served) == len(forced) == len(SEEDS), line
    # Identical PAEs would mean the switches did not reach the forward pass, and the comparison proved nothing.
    assert not all(np.array_equal(a, b) for a, b in zip(result["served"], result["forced"], strict=True)), line
    gap = abs(float(np.mean(forced)) - float(np.mean(served)))
    assert gap <= TOLERANCE, f"forced-off mean is {gap:.4f} from the served mean: {line}"
    golden = GOLDENS[FAMILY][mode]["p53"]
    assert golden is not None and abs(float(np.mean(served)) - golden) <= TOLERANCE, f"golden {golden}: {line}"
    if mode != "vanilla":
        assert result["served_words"]["atom_attn"] == "flash_attn", result["served_words"]
        assert result["forced_words"]["atom_attn"] != "flash_attn", result["forced_words"]
        settle_error = result["settle_error"]
        assert settle_error is not None, "the post-fold check accepted folds run without flash-attention"
        assert settle_error.startswith("OptimizationUnavailableError") and "after the fold" in settle_error, (
            settle_error
        )


def test_kit_kernel_guard(backend_option: str) -> None:
    """Without flash-attn in the kit image, ``exact`` is refused at load, by name, before any weight is fetched."""
    require_modal(backend_option)
    import _kit_remote
    import modal

    with modal.enable_output(), _kit_remote.esmfold2_kit_guard_app.run():
        report = _kit_remote.esmfold2_kit_guard.remote()
    print(f"esmfold2 kit guard: {report}")

    assert report["flash_attn_importable"] is False, "the guard image still has flash_attn"
    assert report["failure_type"] == "OptimizationUnavailableError", report
    assert report["failure_is_typed"] and report["failure_is_refusal"], report
    message = report["failure"]
    assert "atom_attn" in message and "flash_attn" in message and "before the weight fetch" in message, message
    assert report["fetches"] == [] and report["hf_home_files"] == [], report
    assert report["retry_kept_failure"] is True, "a refusal was retried"
    assert report["reload_failure"] is not None and "before the weight fetch" in report["reload_failure"], report
    assert report["ready"] is False
