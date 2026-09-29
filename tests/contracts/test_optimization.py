"""Contract tests for the static ``optimization`` option."""

import pytest

from boileroom.models.registry import ESMFOLD2_SPEC, PROTENIX_SPEC
from boileroom.optimization import (
    DEFAULT_OPTIMIZATION,
    GpuInfo,
    OptimizationUnavailableError,
    resolve_optimization,
    validate_optimization,
)

L4 = GpuInfo("NVIDIA L4", (8, 9))
L40S = GpuInfo("NVIDIA L40S", (8, 9))
A100 = GpuInfo("NVIDIA A100-SXM4-80GB", (8, 0))
H100 = GpuInfo("NVIDIA H100 80GB HBM3", (9, 0))


def test_default_is_vanilla() -> None:
    assert DEFAULT_OPTIMIZATION == "vanilla"


@pytest.mark.parametrize("spec", [ESMFOLD2_SPEC, PROTENIX_SPEC])
def test_optimization_is_a_static_option(spec) -> None:
    assert "optimization" in spec.contract.static_config_keys


def test_unknown_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="optimization must be one of"):
        validate_optimization("turbo")


@pytest.mark.parametrize("family", ["esmfold2", "protenix"])
@pytest.mark.parametrize("gpu", [L4, L40S, A100, H100, None])
def test_vanilla_resolves_on_any_gpu(family: str, gpu: GpuInfo | None) -> None:
    resolution = resolve_optimization(family, "vanilla", gpu)
    assert resolution.active == "vanilla"
    assert resolution.kit_config is None


@pytest.mark.parametrize("family", ["esmfold2", "protenix"])
@pytest.mark.parametrize("mode", ["exact", "fast"])
@pytest.mark.parametrize("gpu", [A100, H100])
def test_kit_modes_resolve_on_a100_and_h100(family: str, mode: str, gpu: GpuInfo) -> None:
    resolution = resolve_optimization(family, mode, gpu)
    assert resolution.active == mode
    assert resolution.kit_config == ("a100" if gpu is A100 else "h100")


@pytest.mark.parametrize("family", ["esmfold2", "protenix"])
@pytest.mark.parametrize("mode", ["exact", "fast"])
@pytest.mark.parametrize("gpu", [L4, L40S])
def test_kit_modes_refused_by_name_on_sm89(family: str, mode: str, gpu: GpuInfo) -> None:
    with pytest.raises(OptimizationUnavailableError, match=rf"{mode!r} cannot run {family} on {gpu.name} \(sm89\)"):
        resolve_optimization(family, mode, gpu)


def test_kit_mode_needs_a_gpu() -> None:
    with pytest.raises(OptimizationUnavailableError, match="needs a GPU"):
        resolve_optimization("protenix", "exact", None)


def test_unknown_family_has_no_kit() -> None:
    with pytest.raises(OptimizationUnavailableError, match="no kit for model family"):
        resolve_optimization("boltz2", "exact", A100)


def test_esmfold2_core_rejects_unknown_mode() -> None:
    from boileroom.models.esmfold2.core import ESMFold2Core

    with pytest.raises(ValueError, match="optimization must be one of"):
        ESMFold2Core({"optimization": "turbo"})


def test_esmfold2_core_refuses_before_loading_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    from boileroom.models.esmfold2 import core as esmfold2_core

    monkeypatch.setattr(esmfold2_core, "detect_gpu", lambda device=None: L4)
    core = esmfold2_core.ESMFold2Core({"optimization": "fast"})
    with pytest.raises(OptimizationUnavailableError, match="cannot run esmfold2 on NVIDIA L4"):
        core._activate_optimization()
    assert core.model is None


def test_esmfold2_core_vanilla_needs_no_gpu_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    from boileroom.models.esmfold2 import core as esmfold2_core

    def fail(device=None):
        raise AssertionError("vanilla must not probe the GPU")

    monkeypatch.setattr(esmfold2_core, "detect_gpu", fail)
    core = esmfold2_core.ESMFold2Core({})
    core._activate_optimization()
    assert core.optimization is not None and core.optimization.active == "vanilla"


def test_esmfold2_kit_variant_follows_checkpoint() -> None:
    from boileroom.models.esmfold2.core import ESMFold2Core

    assert ESMFold2Core({"model_name": "biohub/ESMFold2-Fast"})._kit_variant() == "fast"
    assert ESMFold2Core({})._kit_variant() == "full_nomsa"


def test_protenix_core_rejects_unknown_mode() -> None:
    from boileroom.models.protenix.core import ProtenixCore

    with pytest.raises(ValueError, match="optimization must be one of"):
        ProtenixCore({"optimization": "turbo"})


def test_protenix_core_refuses_before_starting_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    from boileroom.models.protenix import core as protenix_core

    monkeypatch.setattr(protenix_core, "detect_gpu", lambda device=None: L40S)
    core = protenix_core.ProtenixCore({"optimization": "exact"})
    with pytest.raises(OptimizationUnavailableError, match="cannot run protenix on NVIDIA L40S"):
        core._load()
    assert core._worker is None
