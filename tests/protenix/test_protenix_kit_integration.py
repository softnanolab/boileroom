"""Opt-in GPU tests of Protenix's ``optimization`` modes (vanilla, exact, fast) on the bakeoff MDM2 heterodimers.

Skipped unless ``--run-kit`` or ``BOILEROOM_RUN_KIT=1`` is given; they fold on paid GPUs. Pull the kit image by its
pinned digest (``BOILEROOM_KIT_IMAGE_SOURCE=registry``) rather than building it on Modal. From the repository root::

    # everything (~25 A100-40GB minutes, about $1)
    BOILEROOM_KIT_IMAGE_SOURCE=registry UV_CACHE_DIR=$TMPDIR/uv timeout 5400 uv run pytest \
        tests/protenix/test_protenix_kit_integration.py --run-kit --gpu A100-40GB -s -rs
    # one test: add -k heterodimer | -k "heterodimer and fast" | -k fallback | -k guard

Every fold uses the golden protocol: protenix-v2, cycle 10, step 200, one diffusion sample, bf16, the MDM2 MSA from
``tests/data/mdm2.a3m`` for the target and none for the 15-residue binder, no templates (3 seeds x 1 sample, not the
bakeoff's 5 samples).

- ``test_kit_parity_heterodimer[vanilla|exact|fast]``: one live model per mode folds p53 at seeds 0, 1, 2 and the
  decoy at seed 0; ~5 A100-40GB minutes per mode. Vanilla is folded once per module and reused.
- ``test_fallback_parity``: vanilla with the torch triangle kernels instead of cuEquivariance, three p53 seeds, against
  the vanilla mean; ~5 minutes (plus the vanilla set if not folded yet). The kit modes refuse torch kernels, and their
  fused LayerNorm is forced per mode (``LAYERNORM_BY_MODE``), so the fast-vs-torch LayerNorm half of the fallback is
  not reachable through the public config; vanilla runs openfold LayerNorm and the kit modes fast_layernorm, so the
  parity test compares those two.
- ``test_kit_kernel_guard``: the kit image minus cuequivariance-ops-torch (a one-layer image built on top of it); the
  ``exact`` load must refuse naming cuEquivariance. ~2 A100-40GB minutes, Modal only.
"""

from __future__ import annotations

import pytest
from _kit_parity import (
    PROTENIX_FAMILY_CONFIG,
    TORCH_KERNELS,
    FoldCache,
    assert_parity,
    check_worker_runtime,
    fold_protenix_family,
    require_modal,
    wrapper_device,
)
from _structure_metrics import OPTIMIZATION_MODES, TOLERANCE

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.gpu,
    pytest.mark.kit,
    pytest.mark.xdist_group("protenix-kit"),
]

FAMILY = "protenix"
CONFIG = {**PROTENIX_FAMILY_CONFIG, "model_name": "protenix-v2"}
#: ``LAYERNORM_TYPE`` the worker must report per mode (``PROTENIX_LAYERNORM`` of the core).
LAYERNORM = {"vanilla": "openfold", "exact": "fast_layernorm", "fast": "fast_layernorm"}


@pytest.fixture(scope="module")
def folds(backend_option: str, device_option: str | None, output_ctx) -> FoldCache:
    """Fold sets per mode, each on its own live model, folded on first use."""
    from boileroom import Protenix

    device = wrapper_device(FAMILY, backend_option, device_option)
    return FoldCache(
        lambda mode: fold_protenix_family(
            FAMILY, Protenix, backend_option, device, output_ctx, {**CONFIG, "optimization": mode}, mode
        )
    )


@pytest.mark.parametrize("mode", OPTIMIZATION_MODES)
def test_kit_parity_heterodimer(mode: str, folds: FoldCache) -> None:
    """Seed-averaged ipSAE of MDM2 + p53 matches the golden and vanilla; the binder separates from the decoy."""
    fold_set = folds.get(mode)
    for metadata in fold_set.metadata:
        check_worker_runtime(mode, metadata, LAYERNORM[mode])
    assert_parity(fold_set, None if mode == "vanilla" else folds.get("vanilla"))


def test_fallback_parity(backend_option: str, device_option: str | None, output_ctx, folds: FoldCache) -> None:
    """Vanilla with the torch triangle kernels folds p53 within the tolerance of vanilla with cuEquivariance."""
    from boileroom import Protenix

    device = wrapper_device(FAMILY, backend_option, device_option)
    config = {**CONFIG, "optimization": "vanilla", **TORCH_KERNELS}
    torch_set = fold_protenix_family(
        FAMILY, Protenix, backend_option, device, output_ctx, config, "vanilla-torch-kernels", with_decoy=False
    )
    for metadata in torch_set.metadata:
        check_worker_runtime("vanilla", metadata, LAYERNORM["vanilla"], kernels="torch")
    vanilla = folds.get("vanilla")
    gap = abs(torch_set.p53_mean - vanilla.p53_mean)
    assert gap <= TOLERANCE, (
        f"torch kernels {gap:.4f} from cuEquivariance: {torch_set.describe()} | {vanilla.describe()}"
    )


def test_kit_kernel_guard(backend_option: str) -> None:
    """Without cuequivariance-ops-torch in the kit image, ``exact`` is refused at load, naming cuEquivariance."""
    require_modal(backend_option)
    import _kit_remote
    import modal

    with modal.enable_output(), _kit_remote.protenix_kit_guard_app.run():
        report = _kit_remote.protenix_kit_guard.remote()
    print(f"protenix kit guard: {report}")

    assert report["cuequivariance_ops_torch_importable"] is False, "the guard image still has cuequivariance_ops_torch"
    assert report["failure_type"] == "OptimizationUnavailableError", report
    assert report["failure_is_typed"] and report["failure_is_refusal"], report
    assert "cuequivariance" in report["failure"].lower(), report["failure"]
    assert report["retry_kept_failure"] is True, "a refusal was retried"
    assert report["reload_failure"] is not None and "refused earlier" in report["reload_failure"], report
    assert report["ready"] is False
