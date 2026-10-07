"""Contract tests for runtime provenance recorded in prediction metadata."""

import platform
import subprocess
import sys
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, cast

import pytest

from boileroom.base import PredictionMetadata
from boileroom.optimization import GpuInfo
from boileroom.provenance import (
    GPU_MEMORY_KEYS,
    IMAGE_REF_ENV,
    KIT_RUNTIME_KEYS,
    gpu_memory_facts,
    kit_provenance,
    record_gpu_memory,
    runtime_provenance,
)

BASE_KEYS = ["boileroom", "python", "image_ref", "torch", "cuda", "gpu", "gpu_capability"]


@pytest.fixture
def no_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.delenv(IMAGE_REF_ENV, raising=False)


def test_defaults_without_torch_image_or_gpu(no_torch: None) -> None:
    record = runtime_provenance()
    assert list(record) == BASE_KEYS
    assert record["python"] == platform.python_version()
    assert record["image_ref"] == "unknown"
    assert record["torch"] == record["cuda"] == "not-loaded"
    assert record["gpu"] == record["gpu_capability"] == "none"
    assert record["boileroom"] not in ("", "unknown")


def test_torch_is_read_from_loaded_modules_only(no_torch: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(__version__="2.13.0+cu130", version=SimpleNamespace(cuda="13.0"))
    )
    record = runtime_provenance()
    assert record["torch"] == "2.13.0+cu130" and record["cuda"] == "13.0"
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(__version__="2.7.1", version=SimpleNamespace(cuda=None)))
    record = runtime_provenance()
    assert record["torch"] == "2.7.1" and record["cuda"] == "none"


def test_image_gpu_packages_and_extra(no_torch: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(IMAGE_REF_ENV, "docker.io/org/boileroom-protenix-kit:sha-abc")
    record = runtime_provenance(
        ["numpy", "surely-not-an-installed-distribution"],
        gpu=GpuInfo("NVIDIA H100 80GB HBM3", (9, 0)),
        extra={"kit.commit": "f4f62fa", "levers": 7, "python": "3.11.9"},
    )
    assert list(record) == [*BASE_KEYS, "numpy", "surely-not-an-installed-distribution", "kit.commit", "levers"]
    assert record["image_ref"] == "docker.io/org/boileroom-protenix-kit:sha-abc"
    assert record["gpu"] == "NVIDIA H100 80GB HBM3" and record["gpu_capability"] == "sm90"
    import numpy

    assert record["numpy"] == numpy.__version__
    assert record["surely-not-an-installed-distribution"] == "absent"
    assert record["levers"] == "7"
    assert record["python"] == "3.11.9"  # extra wins, e.g. the worker interpreter's own version
    assert all(isinstance(value, str) for value in record.values())


def test_metadata_carries_runtime_provenance() -> None:
    metadata = PredictionMetadata(model_name="m", model_version="v", sequence_lengths=None)
    assert metadata.runtime is None
    metadata.runtime = runtime_provenance()
    assert metadata.runtime["python"] == platform.python_version()


A100 = GpuInfo("NVIDIA A100-SXM4-40GB", (8, 0))
#: One healthy kit report, as the kit hands it to boileroom (lists and a bool) ...
KIT_REPORT = {"active": True, "partial": [], "levers_applied": ["lnstream", "cueq_tri"], "levers_fallback": []}
#: ... and as a Protenix / OpenDDE worker flattens it into ``describe()`` (``kit.<field>``, ``""`` for an empty list).
FLAT_KIT_REPORT = {"kit.active": "true", "kit.partial": "false", "kit.levers_applied": "lnstream,cueq_tri"}


def test_kit_provenance_names_and_formats_the_shared_keys() -> None:
    record = kit_provenance(commit="f4f62fa", levers_applied=["a", "b"], levers_fallback=[], partial=False)
    assert tuple(record) == KIT_RUNTIME_KEYS
    assert record == {
        "kit.commit": "f4f62fa",
        "kit.levers_applied": "a,b",
        "kit.levers_fallback": "none",
        "kit.partial": "false",
    }
    record = kit_provenance(commit=None, levers_applied=(), levers_fallback=("mk",), partial=True)
    assert record["kit.commit"] == "unknown" and record["kit.levers_applied"] == "none"
    assert record["kit.levers_fallback"] == "mk" and record["kit.partial"] == "true"


def _protenix_family_record(family: str, mode: str, info: dict[str, str], payload: dict[str, str]) -> dict[str, str]:
    """``metadata.runtime`` of a Protenix-family core whose worker reported ``info`` and the request ``payload``."""
    from boileroom.models.opendde.core import OpenDDECore
    from boileroom.models.protenix.core import ProtenixCore
    from boileroom.optimization import resolve_optimization

    core = {"protenix": ProtenixCore, "opendde": OpenDDECore}[family]({"optimization": mode})
    core.optimization = resolve_optimization(family, mode, A100)
    core._gpu = A100
    core._worker = cast(Any, SimpleNamespace(info=info))
    return core._runtime_record(payload)


@pytest.mark.parametrize("family", ["protenix", "opendde"])
def test_protenix_family_records_the_shared_kit_keys_from_the_settled_report(family: str) -> None:
    """The request's late (settled) report wins over the activation report the worker described at load."""
    info = {"python": "3.11.5", "kit.commit": "f4f62fa", **FLAT_KIT_REPORT}
    payload = {"kit.levers_applied": "lnstream", "kit.levers_fallback": "cueq_tri", "kit.partial": "true"}
    runtime = _protenix_family_record(family, "exact", info, payload)
    assert {key: runtime[key] for key in KIT_RUNTIME_KEYS} == {
        "kit.commit": "f4f62fa",
        "kit.levers_applied": "lnstream",
        "kit.levers_fallback": "cueq_tri",
        "kit.partial": "true",
    }
    # The family's own prefixed detail stays.
    assert runtime["worker.kit.levers_applied"] == "lnstream,cueq_tri" and runtime["predict.kit.partial"] == "true"
    runtime = _protenix_family_record(family, "fast", info, {})
    assert runtime["kit.levers_applied"] == "lnstream,cueq_tri" and runtime["kit.levers_fallback"] == "none"


@pytest.mark.parametrize("family", ["protenix", "opendde"])
def test_protenix_family_vanilla_records_no_kit_keys(family: str) -> None:
    runtime = _protenix_family_record(family, "vanilla", {"python": "3.11.5"}, {"kernel.resolved.dtype": "bf16"})
    assert not any(key.startswith("kit.") for key in runtime), sorted(runtime)


def test_every_kit_family_records_one_report_under_the_same_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """ESMFold2 and the Protenix family write the same report as the same ``kit.*`` entries, so no per-family lookup."""
    from boileroom.models.esmfold2.core import KIT_COMMIT_ENV, ESMFold2Core

    monkeypatch.setenv(KIT_COMMIT_ENV, "f4f62fa")
    esmfold2 = ESMFold2Core._kit_lever_words(KIT_REPORT, 1, SimpleNamespace())
    protenix = _protenix_family_record("protenix", "exact", {"kit.commit": "f4f62fa", **FLAT_KIT_REPORT}, {})
    expected = kit_provenance(
        commit="f4f62fa", levers_applied=["lnstream", "cueq_tri"], levers_fallback=[], partial=False
    )
    assert {key: esmfold2[key] for key in KIT_RUNTIME_KEYS} == expected
    assert {key: protenix[key] for key in KIT_RUNTIME_KEYS} == expected
    # ESMFold2's own detail uses the same dotted scheme; the old underscore keys are gone.
    assert {"kit.levers_gated", "kit.gated", "kit.scope"} <= set(esmfold2)
    assert not any(key.startswith("kit_") for key in esmfold2), sorted(esmfold2)


GIB = 2**30


def _cuda(initialized: bool = True, *, free: int = 70 * GIB, total: int = 80 * GIB) -> SimpleNamespace:
    """A stand-in for ``torch.cuda`` in a process holding 3 GiB of tensors in a 4 GiB cache."""
    return SimpleNamespace(
        is_initialized=lambda: initialized,
        mem_get_info=lambda: (free, total),
        memory_allocated=lambda: 3 * GIB + 1,
        memory_reserved=lambda: 4 * GIB,
    )


class _NvidiaSmi:
    """A stand-in for ``subprocess.run`` that records ``nvidia-smi`` calls and answers with ``reply``."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.reply: str | BaseException = "12345, 81920\n"

    def __call__(self, command: list[str], **kwargs: Any) -> SimpleNamespace:
        self.calls.append(command)
        if isinstance(self.reply, BaseException):
            raise self.reply
        return SimpleNamespace(stdout=self.reply)


@pytest.fixture
def smi(monkeypatch: pytest.MonkeyPatch) -> _NvidiaSmi:
    """Fake ``nvidia-smi`` with every GPU visible."""
    fake = _NvidiaSmi()
    monkeypatch.setattr(subprocess, "run", fake)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    return fake


def test_gpu_memory_of_a_process_using_cuda(monkeypatch: pytest.MonkeyPatch, smi: _NvidiaSmi) -> None:
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=_cuda()))
    facts = gpu_memory_facts()
    assert tuple(facts) == GPU_MEMORY_KEYS
    assert facts == {
        "gpu.mem.used_mib": "10240",
        "gpu.mem.total_mib": "81920",
        "gpu.mem.allocated_mib": "3072",
        "gpu.mem.reserved_mib": "4096",
    }
    assert smi.calls == []


@pytest.mark.parametrize(
    ("torch", "visible", "device"),
    [
        (None, None, "0"),
        (SimpleNamespace(cuda=_cuda(initialized=False)), "3,1", "3"),
        (SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: True, mem_get_info=None)), "GPU-ab12", "GPU-ab12"),
    ],
    ids=["no-torch", "cuda-not-initialized", "cuda-errors"],
)
def test_gpu_memory_of_the_visible_device_through_nvidia_smi(
    monkeypatch: pytest.MonkeyPatch, smi: _NvidiaSmi, torch: Any, visible: str | None, device: str
) -> None:
    """A parent whose model runs in a worker child has no CUDA context; the device total still counts the child."""
    if torch is None:
        monkeypatch.delitem(sys.modules, "torch", raising=False)
    else:
        monkeypatch.setitem(sys.modules, "torch", torch)
    if visible is not None:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)

    assert gpu_memory_facts() == {"gpu.mem.used_mib": "12345", "gpu.mem.total_mib": "81920"}
    (command,) = smi.calls
    assert command[:2] == ["nvidia-smi", f"--id={device}"]


@pytest.mark.parametrize(
    "reply",
    [FileNotFoundError("nvidia-smi"), subprocess.TimeoutExpired("nvidia-smi", 10), "", "N/A, N/A\n", "12345\n"],
    ids=["no-nvidia-smi", "timeout", "no-output", "not-a-number", "one-column"],
)
def test_gpu_memory_is_empty_when_nvidia_smi_cannot_answer(
    monkeypatch: pytest.MonkeyPatch, smi: _NvidiaSmi, reply: Any
) -> None:
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    smi.reply = reply
    assert gpu_memory_facts() == {}


@pytest.mark.parametrize("visible", ["", "-1"])
def test_gpu_memory_is_empty_when_no_gpu_is_visible(
    monkeypatch: pytest.MonkeyPatch, smi: _NvidiaSmi, visible: str
) -> None:
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    assert gpu_memory_facts() == {}
    assert smi.calls == []


@dataclass
class _Output:
    metadata: Any
    extra: list[int] = field(default_factory=list)


def test_record_gpu_memory_adds_the_facts_without_touching_shared_metadata(
    monkeypatch: pytest.MonkeyPatch, smi: _NvidiaSmi
) -> None:
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=_cuda()))
    shared = PredictionMetadata(model_name="m", model_version="v", sequence_lengths=[3], runtime={"python": "3.12"})
    first, second = _Output(shared), _Output(shared)

    record_gpu_memory(first)

    assert first.metadata.runtime == {"python": "3.12", **gpu_memory_facts()}
    assert first.metadata.sequence_lengths == [3] and first.metadata.model_name == "m"
    assert second.metadata is shared and shared.runtime == {"python": "3.12"}
    bare = _Output(PredictionMetadata(model_name="m", model_version="v", sequence_lengths=None))
    record_gpu_memory(bare)
    assert bare.metadata.runtime == gpu_memory_facts()


@pytest.mark.parametrize(
    "output",
    [
        SimpleNamespace(),
        SimpleNamespace(metadata=None),
        SimpleNamespace(metadata={"runtime": {}}),
        SimpleNamespace(metadata=PredictionMetadata),
    ],
    ids=["no-metadata", "none", "mapping", "class"],
)
def test_record_gpu_memory_skips_outputs_without_runtime_metadata(
    monkeypatch: pytest.MonkeyPatch, smi: _NvidiaSmi, output: Any
) -> None:
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=_cuda()))
    before = dict(vars(output))
    record_gpu_memory(output)
    assert vars(output) == before


def test_record_gpu_memory_on_a_frozen_output_keeps_the_output(
    monkeypatch: pytest.MonkeyPatch, smi: _NvidiaSmi, caplog: pytest.LogCaptureFixture
) -> None:
    @dataclass(frozen=True)
    class Frozen:
        metadata: PredictionMetadata

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=_cuda()))
    output = Frozen(PredictionMetadata(model_name="m", model_version="v", sequence_lengths=None))
    record_gpu_memory(output)
    assert output.metadata.runtime is None
    assert "GPU memory not recorded on Frozen" in caplog.text


def test_record_gpu_memory_leaves_metadata_alone_without_a_gpu(
    monkeypatch: pytest.MonkeyPatch, smi: _NvidiaSmi
) -> None:
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    smi.reply = FileNotFoundError("nvidia-smi")
    metadata = PredictionMetadata(model_name="m", model_version="v", sequence_lengths=None)
    output = _Output(metadata)
    record_gpu_memory(output)
    assert output.metadata is metadata and metadata.runtime is None
