"""Opt-in GPU test that no model leaks GPU memory across calls on one live runtime.

Every model of the registry runs on one live model through its public wrapper, calling ``fold()`` / ``embed()`` with
sequences of a few lengths for :data:`CYCLES` cycles. Every output records the GPU memory in use right after its call
(``metadata.runtime["gpu.mem.used_mib"]``, see ``docs/optimization.md``); the peak of the last cycle may not exceed the
peak of the second one by :data:`GROWTH_LIMIT_MIB`. The first cycle is the warm-up: allocator pools and caches keyed by
length fill there, and a bounded cache stops growing once every length has been seen.

A leak grows on every call instead. The kits' TriMul geometry caches, which the kit's LRU evicted but whose weight
finalizers still held them, grew ESMFold2 ``fast`` on an A100 by about 0.3 GB per new length (``docs/optimization.md``,
"Memory across calls"); three lengths cycling past its two-entry LRU make every call a new one. The kit cases
(``exact`` / ``fast``) fold lengths 280-360, where the H100 adapter runs its native kernel too.

Run from the repository root (kit cases also need ``--run-kit``; pull the kit images by their pinned digests)::

    # everything, about $4 on Modal (~25 A100-80GB, ~30 A100-40GB and ~20 small-GPU minutes)
    BOILEROOM_KIT_IMAGE_SOURCE=registry UV_CACHE_DIR=$TMPDIR/uv timeout 7200 uv run pytest \
        tests/test_memory_stability.py --run-kit -s -rs
    # one model or mode: -k esm2 | -k "esmfold2 and fast" | -k "not kit"

``--gpu`` / ``--device`` set the device of every case. On Modal the ESMFold2 / Protenix / OpenDDE cases default to
their kit golden GPUs (``DEFAULT_MODAL_GPU``) and the others to their wrapper's default. Cases run one at a time
(one ``xdist_group``) so that a shared Apptainer GPU only carries one model.
"""

from __future__ import annotations

import importlib
import random
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import pytest
from _kit_parity import DEFAULT_MODAL_GPU, wrapper_device

from boileroom.models.registry import MODEL_SPECS, MODEL_SPECS_BY_KEY

#: Cycles over a case's lengths; the first is the warm-up.
CYCLES = 4
#: Largest rise of the per-cycle peak of ``gpu.mem.used_mib`` from the second cycle to the last. A leak of the kit
#: caches' size (about 0.3 GB per call) exceeds it within one call; allocator rounding stays far below it.
GROWTH_LIMIT_MIB = 256
_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

#: Short lengths for models whose memory does not depend on the length beyond its activations.
SHORT = (60, 90, 120)
#: Lengths inside the native range of the kits' TriMul adapter on both A100 and H100.
KIT = (280, 320, 360)


@dataclass(frozen=True)
class MemoryCase:
    """One model and mode whose GPU memory is checked across calls.

    Attributes
    ----------
    model : str
        Registry key of the model (:data:`boileroom.models.registry.MODEL_SPECS_BY_KEY`).
    lengths : tuple[int, ...]
        Sequence lengths of one cycle; each call gets a new sequence.
    config : Mapping[str, Any]
        Wrapper config.
    options : Mapping[str, Any]
        Options of every call.
    mode : str
        ``optimization`` mode; ``exact`` / ``fast`` cases run only with ``--run-kit``.
    xfail : str | None
        Reason of a known failure the case runs into before it can measure anything (a strict ``xfail``).
    """

    model: str
    lengths: tuple[int, ...]
    config: Mapping[str, Any] = field(default_factory=dict)
    options: Mapping[str, Any] = field(default_factory=dict)
    mode: str = "vanilla"
    xfail: str | None = None

    @property
    def id(self) -> str:
        return self.model if self.mode == "vanilla" else f"{self.model}-{self.mode}"


_ESMFOLD2_OPTIONS = {"num_loops": 3, "num_sampling_steps": 50, "num_diffusion_samples": 1, "include_fields": ["plddt"]}
#: No MSA (no MSA server call), the reduced sampling budget of the Protenix-family integration tests.
_PROTENIX_CONFIG = {"use_msa": False, "sample": 1, "cycle": 4, "step": 50}
_PROTENIX_OPTIONS = {"include_fields": ["plddt"]}
_MODES = ("vanilla", "exact", "fast")

CASES: tuple[MemoryCase, ...] = (
    MemoryCase("esmfold", SHORT),
    MemoryCase("esm2", SHORT),
    MemoryCase("esmc", SHORT),
    MemoryCase("esm3", SHORT),
    MemoryCase(
        "sae",
        SHORT,
        config={"feature_source": "local"},
        xfail="the local SAE checkpoint has no b_enc, so the model fails to load (softnanolab/boileroom#115)",
    ),
    *(
        MemoryCase("esmfold2", KIT, config={"optimization": mode}, options=_ESMFOLD2_OPTIONS, mode=mode)
        for mode in _MODES
    ),
    MemoryCase(
        "chai1",
        SHORT,
        options={"num_diffn_samples": 1, "num_trunk_samples": 1, "num_trunk_recycles": 1, "num_diffn_timesteps": 10},
    ),
    MemoryCase(
        "boltz2",
        SHORT,
        config={"use_msa_server": False, "recycling_steps": 1, "sampling_steps": 20, "diffusion_samples": 1},
    ),
    *(
        MemoryCase(model, KIT, config={**_PROTENIX_CONFIG, "optimization": mode}, options=_PROTENIX_OPTIONS, mode=mode)
        for model in ("protenix", "opendde")
        for mode in _MODES
    ),
    MemoryCase(
        "alphafold2_multimer",
        SHORT,
        config={"use_msa_server": False, "num_models": 1, "num_recycle": 1, "num_seeds": 1},
        options={"include_fields": ["plddt"]},
    ),
)


def _param(case: MemoryCase) -> Any:
    marks = [pytest.mark.kit] if case.mode != "vanilla" else []
    if case.xfail is not None:
        marks.append(pytest.mark.xfail(reason=case.xfail, strict=True))
    return pytest.param(case, id=case.id, marks=marks)


def _sequence(length: int, cycle: int) -> str:
    """A new sequence per length and cycle, so a cache keyed by the sequence would grow too."""
    rng = random.Random(f"{length}-{cycle}")
    return "".join(rng.choice(_AMINO_ACIDS) for _ in range(length))


def _wrapper(model: str) -> type:
    module_name, class_name = MODEL_SPECS_BY_KEY[model].wrapper_class_path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), class_name)


def _device(case: MemoryCase, backend: str, device_option: str | None) -> str | None:
    family = MODEL_SPECS_BY_KEY[case.model].family
    if family in DEFAULT_MODAL_GPU:
        return wrapper_device(family, backend, device_option)
    return device_option


def test_every_model_has_a_memory_case() -> None:
    assert {case.model for case in CASES} == {spec.key for spec in MODEL_SPECS}
    assert len({case.id for case in CASES}) == len(CASES)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.xdist_group("gpu-memory")
@pytest.mark.parametrize("case", [_param(case) for case in CASES])
def test_gpu_memory_stays_flat_across_calls(
    case: MemoryCase, backend_option: str, device_option: str | None, output_ctx
) -> None:
    device = _device(case, backend_option, device_option)
    if device is not None and device.startswith("cpu"):
        pytest.skip("GPU memory test: needs a GPU device")
    task_method = MODEL_SPECS_BY_KEY[case.model].contract.task_method
    used: list[list[int]] = []
    with output_ctx(), _wrapper(case.model)(backend=backend_option, device=device, config=dict(case.config)) as model:
        task = getattr(model, task_method)
        for cycle in range(CYCLES):
            used.append([])
            for length in case.lengths:
                runtime = task(_sequence(length, cycle), options=dict(case.options)).metadata.runtime or {}
                assert "gpu.mem.used_mib" in runtime, f"{case.id}: the output records no GPU memory: {sorted(runtime)}"
                used[-1].append(int(runtime["gpu.mem.used_mib"]))

    print(f"{case.id} gpu.mem.used_mib per cycle of {case.lengths}: {used}")
    growth = max(used[-1]) - max(used[1])
    assert growth < GROWTH_LIMIT_MIB, (
        f"{case.id}: GPU memory grew {growth} MiB from cycle 2 to cycle {CYCLES} (limit {GROWTH_LIMIT_MIB} MiB): {used}"
    )
