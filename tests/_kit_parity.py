"""Shared fold-set bookkeeping and parity assertions of the kit integration tests (``tests/*/test_*_kit_integration.py``).

Imports only ``numpy``, ``pytest`` and :mod:`_structure_metrics` at module scope; the model wrappers are imported by the
test modules inside their fixtures.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from _structure_metrics import (
    BINDERS,
    DECOY_SEED,
    GOLDENS,
    SEEDS,
    SEPARATION,
    TARGET,
    TOLERANCE,
    chain_index_from_token_chain_ids,
    ipsae,
)

#: GPU each family's kit goldens were measured on, used on Modal when ``--gpu`` is not given. ESMFold2's vanilla Modal
#: default is a T4, which neither matches its golden nor serves the kit modes, so the class is always explicit.
DEFAULT_MODAL_GPU: Mapping[str, str] = {"esmfold2": "A100-80GB", "protenix": "A100-40GB", "opendde": "A100-40GB"}


def heterodimer(binder: str) -> str:
    """Return the ``target:binder`` sequence string of a bakeoff complex (``"p53"`` or ``"decoy"``)."""
    return f"{TARGET}:{BINDERS[binder]}"


def backend_family(backend: str) -> str:
    """Return ``"modal"`` or ``"apptainer"`` from a ``--backend`` value."""
    return backend.split(":", 1)[0].strip()


def wrapper_device(family: str, backend: str, device_option: str | None) -> str | None:
    """Return the ``device`` a kit test passes to a wrapper: ``--gpu``/``--device``, else the family's golden GPU.

    Parameters
    ----------
    family : str
        ``"esmfold2"``, ``"protenix"`` or ``"opendde"``.
    backend : str
        The ``--backend`` value.
    device_option : str | None
        The ``device_option`` fixture (``--gpu`` on Modal, ``--device`` on Apptainer).

    Returns
    -------
    str | None
        The device for the wrapper.
    """
    if backend_family(backend) == "apptainer":
        return device_option
    return device_option or DEFAULT_MODAL_GPU[family]


def require_modal(backend: str) -> None:
    """Skip an in-container test unless the backend is Modal (it runs a variant of the runtime image as a function)."""
    if backend_family(backend) != "modal":
        pytest.skip("in-container kit test: runs on the Modal backend only (it builds a variant of the runtime image)")


def ipsae_min(pae: Any, chain_index: np.ndarray) -> float:
    """Return the headline ipSAE (min of both directions) of one prediction."""
    return ipsae(np.asarray(pae, dtype=float), chain_index).min


@dataclass
class FoldSet:
    """The bakeoff heterodimer folds of one family and mode.

    Attributes
    ----------
    family, mode : str
        Model family and optimization mode (or a label such as ``"vanilla-torch-kernels"``).
    p53 : dict[int, float]
        ipSAE of MDM2 + p53 per seed.
    decoy : float | None
        ipSAE of MDM2 + the scrambled decoy at :data:`DECOY_SEED`, or ``None`` when it was not folded.
    metadata : list[Any]
        ``PredictionMetadata`` of every fold, in fold order.
    """

    family: str
    mode: str
    p53: dict[int, float] = field(default_factory=dict)
    decoy: float | None = None
    metadata: list[Any] = field(default_factory=list)

    @property
    def p53_mean(self) -> float:
        """Mean p53 ipSAE over the folded seeds."""
        if not self.p53:
            raise ValueError(f"{self.family} {self.mode}: no p53 fold")
        return float(np.mean(list(self.p53.values())))

    def describe(self) -> str:
        """One line with every measured value, printed by the tests (``-s``) so goldens can be recorded."""
        seeds = " / ".join(f"s{seed}={value:.5f}" for seed, value in sorted(self.p53.items()))
        decoy = "not folded" if self.decoy is None else f"{self.decoy:.5f}"
        return f"{self.family} {self.mode}: p53 {seeds} mean={self.p53_mean:.5f}; decoy s{DECOY_SEED}={decoy}"


def fold_heterodimers(
    family: str, mode: str, fold: Callable[[str, int], tuple[float, Any]], *, with_decoy: bool = True
) -> FoldSet:
    """Fold p53 at every seed of :data:`SEEDS` and, optionally, the decoy at :data:`DECOY_SEED`.

    Parameters
    ----------
    family, mode : str
        Labels of the fold set.
    fold : Callable[[str, int], tuple[float, Any]]
        Folds ``(binder name, seed)`` on a live model and returns ``(ipSAE, metadata)``.
    with_decoy : bool
        Whether to fold the decoy too.

    Returns
    -------
    FoldSet
        The scores, with the metadata of every fold.
    """
    folds = FoldSet(family=family, mode=mode)
    for seed in SEEDS:
        score, metadata = fold("p53", seed)
        folds.p53[seed] = score
        folds.metadata.append(metadata)
    if with_decoy:
        folds.decoy, metadata = fold("decoy", DECOY_SEED)
        folds.metadata.append(metadata)
    print(folds.describe())
    return folds


class FoldCache:
    """Fold sets of one test module, each folded once: the vanilla set is shared by every comparison against it.

    Parameters
    ----------
    make : Callable[[str], FoldSet]
        Folds the set of a mode on a fresh model.
    """

    def __init__(self, make: Callable[[str], FoldSet]) -> None:
        self._make = make
        self._sets: dict[str, FoldSet] = {}

    def get(self, mode: str) -> FoldSet:
        """Return the fold set of ``mode``, folding it on first use."""
        if mode not in self._sets:
            self._sets[mode] = self._make(mode)
        return self._sets[mode]


def assert_parity(folds: FoldSet, vanilla: FoldSet | None) -> None:
    """Assert separation, closeness to vanilla and closeness to the golden; skip the golden check when it is unset.

    Parameters
    ----------
    folds : FoldSet
        The fold set under test (with its decoy).
    vanilla : FoldSet | None
        The vanilla set of the same family, compared against for a kit mode.
    """
    assert folds.decoy is not None, f"{folds.family} {folds.mode}: the decoy was not folded"
    line = folds.describe()
    separation = SEPARATION[folds.family]
    assert folds.p53_mean - folds.decoy >= separation.margin, f"binder does not separate from the decoy: {line}"
    if separation.p53_min is not None:
        assert folds.p53_mean >= separation.p53_min, f"p53 mean below {separation.p53_min}: {line}"
    if separation.decoy_max is not None:
        assert folds.decoy <= separation.decoy_max, f"decoy above {separation.decoy_max}: {line}"
    if vanilla is not None and vanilla is not folds:
        gap = abs(folds.p53_mean - vanilla.p53_mean)
        assert gap <= TOLERANCE, f"{folds.mode} p53 mean is {gap:.4f} from vanilla ({vanilla.describe()}): {line}"
    golden = GOLDENS[folds.family][folds.mode]["p53"]
    if golden is None:
        pytest.skip(
            f"no {folds.family} {folds.mode} golden yet; every other check passed. Measured: {line}. Record the p53 "
            "mean and the decoy in GOLDENS (tests/_structure_metrics.py)."
        )
    else:
        gap = abs(folds.p53_mean - golden)
        assert gap <= TOLERANCE, f"p53 mean is {gap:.4f} from the golden {golden}: {line}"


# --------------------------------------------------------------------------- Protenix and OpenDDE (one worker design)

#: Caller MSA of the MDM2 target (the bakeoff's), given for chain A; the 15-residue binder gets none.
MDM2_MSA_PATH = Path(__file__).parent / "data" / "mdm2.a3m"
#: The golden fold protocol of Protenix and OpenDDE; ``model_name`` and ``optimization`` are added per family / test.
PROTENIX_FAMILY_CONFIG: Mapping[str, Any] = {
    "cycle": 10,
    "step": 200,
    "sample": 1,
    "dtype": "bf16",
    "use_msa": True,
    "use_template": False,
    "use_tfg_guidance": False,
    "use_default_params": False,
    "use_seeds_in_json": False,
}
PROTENIX_FAMILY_FIELDS: tuple[str, ...] = ("plddt", "ptm", "iptm", "pae", "token_chain_ids")
#: The kernel config that runs the triangle sites in plain torch (vanilla only; the kit modes refuse it).
TORCH_KERNELS: Mapping[str, str] = {"trimul_kernel": "torch", "triatt_kernel": "torch"}
TRIANGLE_SITES: tuple[str, ...] = ("triangle_attention", "triangle_multiplicative")


def fold_protenix_family(
    family: str,
    wrapper: type,
    backend: str,
    device: str | None,
    output_ctx: Callable[[], Any],
    config: Mapping[str, Any],
    label: str,
    *,
    with_decoy: bool = True,
) -> FoldSet:
    """Fold the heterodimers on one live Protenix or OpenDDE model with the golden protocol.

    Parameters
    ----------
    family : str
        ``"protenix"`` or ``"opendde"``.
    wrapper : type
        ``boileroom.Protenix`` or ``boileroom.OpenDDE``.
    backend, device : str, str | None
        Passed to the wrapper.
    output_ctx : Callable[[], Any]
        The ``output_ctx`` fixture (Modal output while the model is live).
    config : Mapping[str, Any]
        Static model config (the protocol plus ``model_name``, ``optimization`` and any kernels).
    label : str
        Label of the fold set (the mode, or e.g. ``"vanilla-torch-kernels"``).
    with_decoy : bool
        Whether to fold the decoy too.

    Returns
    -------
    FoldSet
        The scores and the metadata of every fold.
    """
    msa = MDM2_MSA_PATH.read_text()
    with output_ctx(), wrapper(backend=backend, device=device, config=dict(config)) as model:

        def fold(binder: str, seed: int) -> tuple[float, Any]:
            result = model.fold(
                heterodimer(binder),
                options={"include_fields": list(PROTENIX_FAMILY_FIELDS), "seeds": str(seed), "msa": [msa, None]},
            )
            assert list(result.seeds) == [seed], result.seeds
            assert result.pae is not None and result.token_chain_ids is not None
            chains = chain_index_from_token_chain_ids(result.token_chain_ids[0])
            assert len(chains) == len(TARGET) + len(BINDERS[binder]), len(chains)
            return ipsae_min(result.pae[0], chains), result.metadata

        return fold_heterodimers(family, label, fold, with_decoy=with_decoy)


def check_shared_kit_runtime(mode: str, runtime: Mapping[str, str]) -> None:
    """Check the ``metadata.runtime`` keys every kit family records the same way (``kit.commit``, ``kit.levers_*``).

    Vanilla records none of them; a kit mode records all of them, with levers applied, none fallen back and no
    partial set.
    """
    from boileroom.provenance import KIT_RUNTIME_KEYS

    if mode == "vanilla":
        assert not any(key.startswith("kit.") for key in runtime), sorted(runtime)
        return
    missing = [key for key in KIT_RUNTIME_KEYS if key not in runtime]
    assert not missing, f"metadata.runtime lacks {missing}: {sorted(runtime)}"
    assert runtime["kit.partial"] == "false", runtime["kit.partial"]
    assert runtime["kit.levers_applied"] != "none", "the kit applied no lever"
    assert runtime["kit.levers_fallback"] == "none", runtime["kit.levers_fallback"]


def check_worker_runtime(mode: str, metadata: Any, layernorm: str, kernels: str = "cuequivariance") -> None:
    """Assert what a Protenix or OpenDDE fold's metadata records about the GPU, LayerNorm, kernels and kit.

    Parameters
    ----------
    mode : str
        The optimization mode the model was built with.
    metadata : Any
        ``PredictionMetadata`` of one fold.
    layernorm : str
        ``LAYERNORM_TYPE`` the worker must report.
    kernels : str
        Kernel both triangle sites must resolve to, in the worker and in the request.
    """
    optimization = metadata.optimization
    runtime = metadata.runtime
    assert optimization is not None and optimization["mode"] == mode, optimization
    assert runtime is not None, "metadata.runtime is missing"
    assert runtime.get("gpu", "none") != "none" and runtime.get("gpu_capability", "none").startswith("sm"), runtime
    assert runtime.get("worker.optimization") == mode, runtime.get("worker.optimization")
    assert runtime.get("worker.layernorm_type") == layernorm, runtime.get("worker.layernorm_type")
    for site in TRIANGLE_SITES:
        for source in ("worker", "predict"):
            key = f"{source}.kernel.resolved.{site}"
            assert runtime.get(key) == kernels, f"{key}={runtime.get(key)!r}, expected {kernels!r}"
    kit_keys = sorted(key for key in runtime if key.startswith(("worker.kit.", "predict.kit.")))
    check_shared_kit_runtime(mode, runtime)
    if mode == "vanilla":
        assert optimization["kit_config"] is None and not kit_keys, (optimization, kit_keys)
        return
    assert optimization["kit_config"] in ("a100", "h100"), optimization
    assert optimization["gpu_name"] == runtime["gpu"], (optimization, runtime["gpu"])
    assert runtime.get("worker.kit.active") == "true", kit_keys
    late = {key: value for key, value in runtime.items() if key.startswith("predict.kit.")}
    print(
        f"{mode} runtime: gpu={runtime['gpu']} kit.commit={runtime['kit.commit']} "
        f"levers={runtime['kit.levers_applied']} late={late}"
    )
