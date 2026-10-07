"""Contract tests for the static ``optimization`` option."""

import subprocess
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from boileroom import optimization as optimization_module
from boileroom.models.registry import ESMFOLD2_SPEC, OPENDDE_SPEC, PROTENIX_SPEC, ModelSpec
from boileroom.optimization import (
    DEFAULT_OPTIMIZATION,
    KIT_FAMILIES,
    GpuInfo,
    OptimizationResolution,
    OptimizationUnavailableError,
    describe_gpu,
    detect_gpu,
    initialize_core,
    is_refusal,
    resolve_optimization,
    retry_initialize,
    validate_optimization,
)

L4 = GpuInfo("NVIDIA L4", (8, 9))
L40S = GpuInfo("NVIDIA L40S", (8, 9))
A100 = GpuInfo("NVIDIA A100-SXM4-80GB", (8, 0))
H100 = GpuInfo("NVIDIA H100 80GB HBM3", (9, 0))


H200 = GpuInfo("NVIDIA H200", (9, 0))
A30 = GpuInfo("NVIDIA A30", (8, 0))
GH200 = GpuInfo("NVIDIA GH200 480GB", (9, 0))
A10G = GpuInfo("NVIDIA A10G", (8, 6))
A100_PCIE = GpuInfo("NVIDIA A100 80GB PCIe", (8, 0))
FAMILIES = sorted(KIT_FAMILIES)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("mode", ["exact", "fast"])
@pytest.mark.parametrize("gpu", [A30, GH200, A10G], ids=lambda gpu: gpu.name)
def test_unvalidated_card_is_refused_by_name(family: str, mode: str, gpu: GpuInfo) -> None:
    """A30 shares sm80 and GH200 sm90 with served cards, and GH200 contains the substring "H200"."""
    with pytest.raises(OptimizationUnavailableError, match=rf"{gpu.name} \({gpu.sm}\): the kit is only validated on"):
        resolve_optimization(family, mode, gpu)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("mode", ["exact", "fast"])
@pytest.mark.parametrize(
    ("gpu", "config"), [(A100, "a100"), (A100_PCIE, "a100"), (H100, "h100"), (H200, "h100")], ids=lambda x: str(x)
)
def test_served_card_names_resolve_to_one_config_for_every_family(
    family: str, mode: str, gpu: GpuInfo, config: str
) -> None:
    assert resolve_optimization(family, mode, gpu) == OptimizationResolution(mode, config, gpu.name, gpu.sm)


def test_default_is_vanilla() -> None:
    assert DEFAULT_OPTIMIZATION == "vanilla"


@pytest.mark.parametrize("spec", [ESMFOLD2_SPEC, PROTENIX_SPEC, OPENDDE_SPEC])
def test_optimization_is_a_static_option(spec: ModelSpec) -> None:
    assert "optimization" in spec.contract.static_config_keys


def test_unknown_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="optimization must be one of"):
        validate_optimization("turbo")


@pytest.mark.parametrize("family", [*FAMILIES, "boltz2"])
@pytest.mark.parametrize("gpu", [L4, L40S, A100, H100, GH200, None])
def test_vanilla_resolves_on_any_gpu_and_records_it(family: str, gpu: GpuInfo | None) -> None:
    resolution = resolve_optimization(family, "vanilla", gpu)
    assert resolution.mode == "vanilla" and not resolution.kit
    assert resolution.kit_config is None
    assert resolution.gpu_name == (gpu.name if gpu else None)
    assert resolution.capability == (gpu.sm if gpu else None)


def test_resolution_record_has_one_mode_field() -> None:
    """``requested`` always equalled ``active`` (a mode that cannot run raises), so the record keeps one ``mode``."""
    assert resolve_optimization("protenix", "fast", A100).to_dict() == {
        "mode": "fast",
        "kit_config": "a100",
        "gpu_name": "NVIDIA A100-SXM4-80GB",
        "capability": "sm80",
    }
    assert resolve_optimization("protenix", "fast", A100).kit


@pytest.mark.parametrize("family", ["esmfold2", "protenix", "opendde"])
@pytest.mark.parametrize("mode", ["exact", "fast"])
@pytest.mark.parametrize("gpu", [A100, H100, H200])
def test_kit_modes_resolve_on_a100_and_h100(family: str, mode: str, gpu: GpuInfo) -> None:
    resolution = resolve_optimization(family, mode, gpu)
    assert resolution.mode == mode and resolution.kit
    assert resolution.kit_config == ("a100" if gpu is A100 else "h100")


@pytest.mark.parametrize("family", ["esmfold2", "protenix", "opendde"])
@pytest.mark.parametrize("mode", ["exact", "fast"])
@pytest.mark.parametrize("gpu", [L4, L40S])
def test_kit_modes_refused_by_name_on_sm89(family: str, mode: str, gpu: GpuInfo) -> None:
    with pytest.raises(OptimizationUnavailableError, match=rf"{mode!r} cannot run {family} on {gpu.name} \(sm89\)"):
        resolve_optimization(family, mode, gpu)


def test_kit_mode_rejects_cpu_device() -> None:
    with pytest.raises(OptimizationUnavailableError, match="not on device='cpu'"):
        detect_gpu("cpu")


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


def _install_weight_loaders_that_must_not_run(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Stand in for every ESMFold2 weight loader; each records its call instead of loading."""
    from boileroom.models.esmfold2 import loading

    calls: list[str] = []

    class _Model:
        @classmethod
        def from_pretrained(cls, *args: Any, **kwargs: Any) -> None:
            calls.append(cls.__qualname__)

    def module(name: str, **attributes: Any) -> ModuleType:
        stub = ModuleType(name)
        stub.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, stub)
        return stub

    esm_model = type("EsmFold2Model", (_Model,), {})
    module("esm")
    module("esm.models")
    module("esm.models.esmfold2", EsmFold2Model=esm_model, ESMFold2InputBuilder=lambda **kwargs: None)
    module("transformers")
    module("transformers.models")
    module("transformers.models.esmfold2")
    module("transformers.models.esmfold2.modeling_esmfold2", ESMFold2Model=type("ESMFold2Model", (_Model,), {}))
    monkeypatch.setattr(
        loading, "load_pretrained", lambda model_class, *args, **kwargs: calls.append("load_pretrained")
    )
    return calls


@pytest.mark.parametrize("mode", ["exact", "fast"])
def test_esmfold2_core_refuses_before_loading_weights(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    """Driven through ``_load()``: a reorder that loads weights before resolving the mode fails here."""
    pytest.importorskip("torch")
    from boileroom.models.esmfold2 import core as esmfold2_core

    calls = _install_weight_loaders_that_must_not_run(monkeypatch)
    monkeypatch.setattr(esmfold2_core, "detect_gpu", lambda device=None: L4)
    core = esmfold2_core.ESMFold2Core({"optimization": mode})
    with pytest.raises(OptimizationUnavailableError, match="cannot run esmfold2 on NVIDIA L4"):
        core._load()
    assert calls == []
    assert core.model is None and not core.ready


def test_esmfold2_core_vanilla_needs_no_kit_probe_and_records_the_card(monkeypatch: pytest.MonkeyPatch) -> None:
    from boileroom.models.esmfold2 import core as esmfold2_core

    def fail(device: str | None = None) -> GpuInfo:
        raise AssertionError("vanilla must not demand a GPU")

    monkeypatch.setattr(esmfold2_core, "detect_gpu", fail)
    monkeypatch.setattr(esmfold2_core, "describe_gpu", lambda device=None: L4)
    core = esmfold2_core.ESMFold2Core({})
    core._activate_optimization()
    assert core.optimization == OptimizationResolution("vanilla", None, "NVIDIA L4", "sm89")


def test_esmfold2_core_vanilla_without_a_gpu_records_none(monkeypatch: pytest.MonkeyPatch) -> None:
    from boileroom.models.esmfold2 import core as esmfold2_core

    core = esmfold2_core.ESMFold2Core({"device": "cpu"})
    core._activate_optimization()
    assert core.optimization == OptimizationResolution("vanilla", None, None, None)


def test_esmfold2_vanilla_keeps_revision_and_cache_dirs() -> None:
    from boileroom.models.esmfold2.core import ESMFold2Core

    core = ESMFold2Core({"revision": "abc", "cache_dir": "/tmp/x", "device": "cpu"})
    core._activate_optimization()
    assert core.optimization is not None and core.optimization.mode == "vanilla"


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


def test_protenix_core_vanilla_records_the_card(monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import Mock

    from boileroom.models.protenix import core as protenix_core

    monkeypatch.setattr(protenix_core, "ModelWorker", Mock())
    monkeypatch.setattr(protenix_core, "describe_gpu", lambda device=None: L40S)
    core = protenix_core.ProtenixCore({})
    core._load()
    assert core.optimization == OptimizationResolution("vanilla", None, "NVIDIA L40S", "sm89")


class _FailingCore:
    """Stand-in core whose load fails, to see how the Modal wrappers' init helper treats it."""

    def __init__(self, optimization: str, error: BaseException | None) -> None:
        self.config = {"optimization": optimization}
        self.error = error
        self.calls = 0

    def _initialize(self) -> None:
        self.calls += 1
        if self.error is not None:
            raise self.error


@pytest.mark.parametrize("mode", ["vanilla", "exact", "fast"])
@pytest.mark.parametrize(
    "error", [OptimizationUnavailableError("needs the kit image"), ValueError("bad config"), TypeError("boom")]
)
def test_init_failure_is_returned_not_raised(mode: str, error: Exception) -> None:
    """Raising in ``@modal.enter()`` makes Modal restart the container silently, so the failure is handed back."""
    assert initialize_core(_FailingCore(mode, error)) is error


def test_init_success_returns_none() -> None:
    assert initialize_core(_FailingCore("exact", None)) is None


def test_init_turns_the_kit_refusal_exit_into_a_permanent_refusal() -> None:
    """The kits leave with ``SystemExit(3)``; raised from ``@modal.enter()`` it would kill the container silently."""
    core = _FailingCore("fast", SystemExit(3))
    refusal = initialize_core(core)
    assert isinstance(refusal, OptimizationUnavailableError)
    assert str(refusal) == "_FailingCore load refused: the optimization kit exited with code 3"
    assert isinstance(refusal.__cause__, SystemExit)
    assert retry_initialize(core, refusal) is refusal
    assert core.calls == 1


def test_init_turns_the_kernel_census_exit_into_a_permanent_refusal() -> None:
    """OpenDDE's ``KernelsRefused`` is ``SystemExit(5)``: unmapped, it would be retried as an ordinary failure."""
    core = _FailingCore("exact", _KernelsRefused(["triangle_attention: cuequivariance absent"]))
    refusal = initialize_core(core)
    assert isinstance(refusal, OptimizationUnavailableError)
    assert str(refusal) == "_FailingCore load refused: the optimization kit exited with code 5"
    assert retry_initialize(core, refusal) is refusal
    assert core.calls == 1


def test_process_and_in_process_refusal_codes_are_distinct_contracts() -> None:
    """A process reports a refusal only as exit 3; the kits' in-process ``SystemExit`` codes are 3 and 5."""
    from boileroom.models import _worker

    assert optimization_module.REFUSAL_PROCESS_EXIT_CODE == _worker._REFUSAL_PROCESS_EXIT_CODE == 3
    assert optimization_module.KIT_REFUSAL_EXIT_CODES == _worker._KIT_REFUSAL_EXIT_CODES == {3, 5}


@pytest.mark.parametrize(
    ("code", "detail"),
    [(1, "exited with code 1"), (2, "exited with code 2"), (None, "exited with code None"), ("no", "exited: no")],
)
def test_init_turns_any_other_exit_into_a_load_failure(code: object, detail: str) -> None:
    """Only the kits' refusal code is a refusal; ``sys.exit(1)`` in a vanilla load is a failure, retried later."""
    core = _FailingCore("vanilla", SystemExit(code))
    failure = initialize_core(core)
    assert type(failure) is RuntimeError
    assert str(failure) == f"_FailingCore load failed: it {detail}"
    assert isinstance(failure.__cause__, SystemExit)
    core.error = None
    assert retry_initialize(core, failure) is None
    assert core.calls == 2


def test_init_lets_a_keyboard_interrupt_through() -> None:
    with pytest.raises(KeyboardInterrupt):
        initialize_core(_FailingCore("fast", KeyboardInterrupt()))


@pytest.mark.parametrize("error", [OptimizationUnavailableError("wrong GPU"), ValueError("bad config")])
def test_permanent_failures_stand_without_a_retry(error: Exception) -> None:
    core = _FailingCore("exact", None)
    core.calls = 0
    assert retry_initialize(core, error) is error
    assert core.calls == 0


def test_transient_failure_gets_another_load_attempt() -> None:
    """A dropped download must not poison the container for the rest of its life."""
    core = _FailingCore("exact", None)
    assert retry_initialize(core, OSError("connection reset")) is None
    core.error = OSError("still down")
    assert isinstance(retry_initialize(core, OSError("connection reset")), OSError)
    assert retry_initialize(core, None) is None


class _RuntimeLocalRefusal(RuntimeError):
    """Stands in for the class a kit runtime defines locally because it cannot import boileroom."""


_RuntimeLocalRefusal.__name__ = "OptimizationUnavailableError"


class _KernelsRefused(SystemExit):
    """The shape of ``opendde_opt.lncensus.KernelsRefused``: a ``SystemExit`` with the census's exit code 5."""

    def __init__(self, problems: list[str]) -> None:
        super().__init__(5)
        self.problems = problems


@pytest.mark.parametrize(
    ("error", "refused"),
    [
        (OptimizationUnavailableError("wrong GPU"), True),
        (_RuntimeLocalRefusal("kit inactive"), True),
        (SystemExit(3), True),
        (_KernelsRefused(["trimul: reference path served"]), True),
        (SystemExit(5), True),
        (SystemExit(1), False),
        (SystemExit(2), False),
        (SystemExit(None), False),
        (SystemExit([3]), False),
        (RuntimeError("CUDA out of memory"), False),
        (ValueError("bad config"), False),
    ],
)
def test_refusals_are_classified_alike_in_boileroom_and_the_worker_child(error: BaseException, refused: bool) -> None:
    """The server, the Apptainer backend and the worker child must agree on what a refusal is."""
    from boileroom.models import _worker

    assert is_refusal(error) is refused
    assert _worker._is_refusal(error) is refused


# --- GPU detection without torch: nvidia-smi and CUDA_VISIBLE_DEVICES ---------------------------------------------

_SMI_ROWS = {
    "same": "0, GPU-aaaa1111, NVIDIA A100-SXM4-80GB, 8.0\n1, GPU-bbbb2222, NVIDIA A100-SXM4-80GB, 8.0\n",
    "mixed": "0, GPU-aaaa1111, NVIDIA L4, 8.9\n1, GPU-bbbb2222, NVIDIA H100 80GB HBM3, 9.0\n",
    "comma": "0, GPU-aaaa1111, NVIDIA Weird, Card, 9.0\n",
}


@pytest.fixture
def smi(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Fake ``nvidia-smi`` output; set ``state["rows"]`` (or ``state["error"]``) before calling."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)
    state: dict[str, Any] = {"rows": _SMI_ROWS["same"], "error": None, "argv": None}

    def run(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        state["argv"] = argv
        if state["error"] is not None:
            raise state["error"]
        return SimpleNamespace(stdout=state["rows"])

    monkeypatch.setattr(optimization_module.subprocess, "run", run)
    return state


def test_nvidia_smi_queries_index_uuid_name_and_capability(smi: dict[str, Any]) -> None:
    assert optimization_module._detect_gpu_nvidia_smi(1) == GpuInfo("NVIDIA A100-SXM4-80GB", (8, 0))
    assert smi["argv"][1] == "--query-gpu=index,uuid,name,compute_cap"


def test_nvidia_smi_maps_the_logical_index_through_cuda_visible_devices(
    smi: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """``cuda:0`` under ``CUDA_VISIBLE_DEVICES=1`` is physical GPU 1, not the L4 nvidia-smi lists first."""
    smi["rows"] = _SMI_ROWS["mixed"]
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")  # integers follow nvidia-smi's order only under PCI_BUS_ID
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    assert optimization_module._detect_gpu_nvidia_smi(0) == GpuInfo("NVIDIA H100 80GB HBM3", (9, 0))
    monkeypatch.delenv("CUDA_DEVICE_ORDER")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-bbbb")
    assert optimization_module._detect_gpu_nvidia_smi(0) == GpuInfo("NVIDIA H100 80GB HBM3", (9, 0))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-bbbb, GPU-aaaa")
    assert optimization_module._detect_gpu_nvidia_smi(1) == GpuInfo("NVIDIA L4", (8, 9))


@pytest.mark.parametrize(("visible", "index"), [("", 0), ("-1", 0), ("0,-1,1", 1), ("0", 1), ("7", 0), ("GPU-zz", 0)])
def test_nvidia_smi_refuses_a_device_cuda_cannot_see(
    smi: dict[str, Any], monkeypatch: pytest.MonkeyPatch, visible: str, index: int
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    with pytest.raises(OptimizationUnavailableError):
        optimization_module._detect_gpu_nvidia_smi(index)


def test_nvidia_smi_refuses_a_mig_slice(smi: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "MIG-1234")
    with pytest.raises(OptimizationUnavailableError, match="MIG slice"):
        optimization_module._detect_gpu_nvidia_smi(0)


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError("nvidia-smi"),
        subprocess.CalledProcessError(9, "nvidia-smi"),
        subprocess.TimeoutExpired("x", 60),
    ],
)
def test_nvidia_smi_failure_is_a_refusal(smi: dict[str, Any], error: Exception) -> None:
    smi["error"] = error
    with pytest.raises(OptimizationUnavailableError, match="nvidia-smi failed"):
        optimization_module._detect_gpu_nvidia_smi(0)


def test_nvidia_smi_unparseable_output_is_a_refusal(smi: dict[str, Any]) -> None:
    smi["rows"] = "garbage\n"
    with pytest.raises(OptimizationUnavailableError, match="nvidia-smi failed"):
        optimization_module._detect_gpu_nvidia_smi(0)


def test_nvidia_smi_reads_a_name_with_a_comma(smi: dict[str, Any]) -> None:
    smi["rows"] = _SMI_ROWS["comma"]
    assert optimization_module._detect_gpu_nvidia_smi(0) == GpuInfo("NVIDIA Weird, Card", (9, 0))


def test_nvidia_smi_refuses_integer_selection_on_a_mixed_node_without_pci_order(
    smi: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """CUDA's default device order is fastest-first, nvidia-smi's is PCI bus order; on mixed cards they can differ."""
    smi["rows"] = _SMI_ROWS["mixed"]
    with pytest.raises(OptimizationUnavailableError, match="CUDA_DEVICE_ORDER=PCI_BUS_ID"):
        optimization_module._detect_gpu_nvidia_smi(0)
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    assert optimization_module._detect_gpu_nvidia_smi(0) == GpuInfo("NVIDIA L4", (8, 9))
    monkeypatch.delenv("CUDA_DEVICE_ORDER")
    # CUDA reads a CUDA_VISIBLE_DEVICES integer in its own (fastest-first) order over every GPU of the node, so one
    # visible card is just as ambiguous; a UUID or CUDA_DEVICE_ORDER=PCI_BUS_ID settles it.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    with pytest.raises(OptimizationUnavailableError, match="CUDA_DEVICE_ORDER=PCI_BUS_ID"):
        optimization_module._detect_gpu_nvidia_smi(0)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-bbbb2222")
    assert optimization_module._detect_gpu_nvidia_smi(0) == GpuInfo("NVIDIA H100 80GB HBM3", (9, 0))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    assert optimization_module._detect_gpu_nvidia_smi(0) == GpuInfo("NVIDIA H100 80GB HBM3", (9, 0))


def test_detect_gpu_without_torch_falls_back_to_nvidia_smi(
    smi: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    smi["rows"] = _SMI_ROWS["mixed"]
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setitem(sys.modules, "torch", None)
    assert detect_gpu("cuda:1") == GpuInfo("NVIDIA H100 80GB HBM3", (9, 0))
    with pytest.raises(OptimizationUnavailableError, match="cannot read a CUDA device index"):
        detect_gpu("cuda:x")


def test_describe_gpu_never_raises(smi: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)
    assert describe_gpu("cpu") is None
    assert describe_gpu("cuda:0") == GpuInfo("NVIDIA A100-SXM4-80GB", (8, 0))
    smi["error"] = FileNotFoundError("nvidia-smi")
    assert describe_gpu("cuda:0") is None
    monkeypatch.setattr(optimization_module, "detect_gpu", lambda device=None: 1 / 0)
    assert describe_gpu(None) is None
