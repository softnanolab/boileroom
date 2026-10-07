"""Opt-in GPU tests of OpenDDE's ``optimization`` modes and of caller templates on one live model.

Skipped unless ``--run-kit`` or ``BOILEROOM_RUN_KIT=1`` is given; they fold on paid GPUs. OpenDDE runs every mode on its
one runtime image (the kit is part of it), so no kit-image source applies. From the repository root::

    # everything (~30 A100-40GB minutes, about $1.1)
    UV_CACHE_DIR=$TMPDIR/uv timeout 5400 uv run pytest \
        tests/opendde/test_opendde_kit_integration.py --run-kit --gpu A100-40GB -s -rs
    # one test: add -k heterodimer | -k "heterodimer and exact" | -k fallback | -k template

Parity folds use the Protenix golden protocol on ``opendde_v1``: cycle 10, step 200, one diffusion sample, bf16, the
MDM2 MSA from ``tests/data/mdm2.a3m`` for the target and none for the binder, no templates (3 seeds x 1 sample).

- ``test_kit_parity_heterodimer[vanilla|exact|fast]``: one live model per mode folds p53 at seeds 0, 1, 2 and the
  decoy at seed 0; ~6 A100-40GB minutes per mode. Checks the runtime metadata, the separation, a kit mode within the
  tolerance of vanilla and every mode within the tolerance of its golden (``GOLDENS["opendde"]``, measured 2026-10-06 on
  image ``sha-06a2b81593d2``). The vanilla set is folded once and reused.
- ``test_fallback_parity``: vanilla with the torch triangle kernels against vanilla with cuEquivariance, three p53
  seeds; ~5 minutes. The kit modes refuse torch kernels and force fast_layernorm, so the LayerNorm half of the
  fallback is not reachable through the public config (vanilla runs torch LayerNorm; parity compares the two).
- ``test_template_switch_on_live_model`` / ``test_self_template_featurized``: one live vanilla model folds the
  413-residue query of ``tests/data/chai/pred.rank_0.cif`` with that structure as its template (A), then with a copy
  without the atoms of residues 207+ (H), then with A again, then without a template, all at seed 0 without an MSA
  (cycle 4, step 50); ~2 minutes. A stale per-request template cache would fold H, or the untemplated request, like A.
  OpenDDE follows a template's per-domain geometry, not a rigid move of one part, so H hides structure instead of
  moving it.
"""

from __future__ import annotations

import io
import pathlib
from typing import Any

import numpy as np
import pytest
from _kit_parity import (
    PROTENIX_FAMILY_CONFIG,
    TORCH_KERNELS,
    FoldCache,
    assert_parity,
    check_worker_runtime,
    fold_protenix_family,
    wrapper_device,
)
from _structure_metrics import (
    OPTIMIZATION_MODES,
    TOLERANCE,
    ca_coordinates,
    cif_protein_sequence,
    drop_residues_in_cif,
    mean_distance_difference,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.gpu,
    pytest.mark.kit,
    pytest.mark.xdist_group("opendde-kit"),
]

FAMILY = "opendde"
CONFIG = {**PROTENIX_FAMILY_CONFIG, "model_name": "opendde_v1"}
#: ``LAYERNORM_TYPE`` the worker must report per mode (``OPENDDE_LAYERNORM`` of the core).
LAYERNORM = {"vanilla": "torch", "exact": "fast_layernorm", "fast": "fast_layernorm"}

TEMPLATE_CIF = pathlib.Path(__file__).parents[1] / "data" / "chai" / "pred.rank_0.cif"
#: Template H: template A without the atoms of residues ``HALF_FROM`` on (SEQRES unchanged).
HALF_FROM = 207
TEMPLATE_SEED = 0
TEMPLATE_CONFIG = {
    "model_name": "opendde_v1",
    "optimization": "vanilla",
    "use_msa": False,
    "use_template": False,
    "sample": 1,
    "cycle": 4,
    "step": 50,
    "dtype": "bf16",
}
#: Distances are mean CA distance differences (Å). Measured 2026-10-06 (A100-40GB, image ``sha-06a2b81593d2``):
#: d(A, A again) 0.012, d(A, H) 2.72, d(pred with A, template A) 0.63, d(pred without a template, template A) 9.74.
#: A switched template must move the prediction by more than this beyond twice the repeat noise of the same request.
TEMPLATE_SWITCH_MARGIN = 1.0
#: The prediction with the query's own structure as its template stays within this of that structure...
SELF_TEMPLATE_MAX = 1.5
#: ...and the prediction without a template lands at least this far from it.
NO_TEMPLATE_MIN = 5.0


@pytest.fixture(scope="module")
def folds(backend_option: str, device_option: str | None, output_ctx) -> FoldCache:
    """Fold sets per mode, each on its own live model, folded on first use."""
    from boileroom import OpenDDE

    device = wrapper_device(FAMILY, backend_option, device_option)
    return FoldCache(
        lambda mode: fold_protenix_family(
            FAMILY, OpenDDE, backend_option, device, output_ctx, {**CONFIG, "optimization": mode}, mode
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
    from boileroom import OpenDDE

    device = wrapper_device(FAMILY, backend_option, device_option)
    config = {**CONFIG, "optimization": "vanilla", **TORCH_KERNELS}
    torch_set = fold_protenix_family(
        FAMILY, OpenDDE, backend_option, device, output_ctx, config, "vanilla-torch-kernels", with_decoy=False
    )
    for metadata in torch_set.metadata:
        check_worker_runtime("vanilla", metadata, LAYERNORM["vanilla"], kernels="torch")
    vanilla = folds.get("vanilla")
    gap = abs(torch_set.p53_mean - vanilla.p53_mean)
    assert gap <= TOLERANCE, (
        f"torch kernels {gap:.4f} from cuEquivariance: {torch_set.describe()} | {vanilla.describe()}"
    )


def _template_ca(cif_text: str) -> np.ndarray:
    from biotite.structure.io import pdbx

    return ca_coordinates(pdbx.get_structure(pdbx.CIFFile.read(io.StringIO(cif_text)), model=1))


@pytest.fixture(scope="module")
def template_runs(backend_option: str, device_option: str | None, output_ctx) -> dict[str, Any]:
    """Fold the query with template A, then H, then A again, then without a template, on one live model at one seed.

    Returns
    -------
    dict[str, Any]
        ``template``: CA coordinates of template A; ``runs``: ``label -> (predicted CA, metadata.runtime)`` for
        ``"A"``, ``"H"``, ``"A again"`` and ``"none"``, folded in that order.
    """
    from boileroom import OpenDDE

    template_a = TEMPLATE_CIF.read_text()
    templates = {"A": template_a, "H": drop_residues_in_cif(template_a, HALF_FROM), "A again": template_a, "none": None}
    query = cif_protein_sequence(template_a)
    device = wrapper_device(FAMILY, backend_option, device_option)
    runs: dict[str, tuple[np.ndarray, dict[str, str]]] = {}
    with output_ctx(), OpenDDE(backend=backend_option, device=device, config=TEMPLATE_CONFIG) as model:
        for label, template in templates.items():
            options: dict[str, Any] = {"seeds": str(TEMPLATE_SEED), "include_fields": ["plddt"]}
            if template is not None:
                options |= {"templates": {"caller": template}, "templates_chain": 0}
            result = model.fold(query, options=options)
            assert result.atom_array is not None
            runs[label] = (ca_coordinates(result.atom_array[0]), dict(result.metadata.runtime or {}))
    template_ca = _template_ca(template_a)
    for label, (ca, runtime) in runs.items():
        assert ca.shape == template_ca.shape == (len(query), 3), (label, ca.shape, template_ca.shape)
        counts = {key: runtime.get(f"predict.templates.{key}") for key in ("staged", "featurized", "other_chains")}
        print(f"opendde template {label}: {counts}")
    return {"template": template_ca, "runs": runs}


def test_template_switch_on_live_model(template_runs: dict[str, Any]) -> None:
    """A different caller template changes the prediction of a live model; repeating the first does not."""
    runs = template_runs["runs"]
    for label in ("A", "H", "A again"):
        runtime = runs[label][1]
        counts = {key: runtime.get(f"predict.templates.{key}") for key in ("staged", "featurized", "other_chains")}
        assert counts == {"staged": "1", "featurized": "1", "other_chains": "0"}, (label, counts)
    first, half, repeat = runs["A"][0], runs["H"][0], runs["A again"][0]
    noise = mean_distance_difference(first, repeat)
    switch = mean_distance_difference(first, half)
    back = mean_distance_difference(repeat, half)
    line = f"d(A, A again)={noise:.3f} Å, d(A, H)={switch:.3f} Å, d(A again, H)={back:.3f} Å"
    print(f"opendde template switch: {line}")
    assert switch > 2 * noise + TEMPLATE_SWITCH_MARGIN, f"template H folded like A (stale template): {line}"
    assert back > 2 * noise + TEMPLATE_SWITCH_MARGIN, f"template A again folded like H (stale template): {line}"


def test_self_template_featurized(template_runs: dict[str, Any]) -> None:
    """The query's own structure as a template pulls the prediction onto it; a later request without one does not."""
    runs, template = template_runs["runs"], template_runs["template"]
    untemplated = runs["none"][1]
    leaked = {key: value for key, value in untemplated.items() if key.startswith("predict.templates.")}
    assert not leaked, f"a request without templates reports template metadata: {leaked}"
    with_a = mean_distance_difference(runs["A"][0], template)
    without = mean_distance_difference(runs["none"][0], template)
    line = f"d(pred with A, template A)={with_a:.3f} Å, d(pred without a template, template A)={without:.3f} Å"
    print(f"opendde self template: {line}")
    assert with_a < SELF_TEMPLATE_MAX, f"the self template did not pull the prediction onto it: {line}"
    assert without > NO_TEMPLATE_MIN, f"the request without a template folded like the templated one: {line}"
