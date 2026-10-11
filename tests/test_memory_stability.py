"""Opt-in GPU test that no model leaks GPU memory across calls on one live runtime.

Every model of the registry runs on one live model through its public wrapper, calling ``fold()`` / ``embed()`` with
sequences of a few lengths for :data:`CYCLES` cycles. Every output records the GPU memory in use right after its call
(``metadata.runtime["gpu.mem.used_mib"]``, see ``docs/optimization.md``). After warm-up no allocated or reserved cycle peak may rise
by :data:`GROWTH_LIMIT_MIB`; total device use has a separate sustained-growth guard. Short traces start at cycle two; traces of ten or more cycles measure the entire second
half, allowing bounded shape-dependent cache fill while rejecting continued growth.

A leak grows on every call instead. The kits' TriMul geometry caches, which the kit's LRU evicted but whose weight
finalizers still held them, grew ESMFold2 ``fast`` on an A100 by about 0.3 GB per new length (``docs/optimization.md``,
"Memory across calls"); three lengths cycling past its two-entry LRU make every call a new one. The kit cases
(``exact``) fold lengths 280-360, where the H100 adapter runs its native kernel too.

Run from the repository root (kit cases also need ``--run-kit``; pull the kit images by their pinned digests)::

    # everything, about $4 on Modal (~25 A100-80GB, ~30 A100-40GB and ~20 small-GPU minutes)
    BOILEROOM_KIT_IMAGE_SOURCE=registry UV_CACHE_DIR=$TMPDIR/uv timeout 7200 uv run pytest \
        tests/test_memory_stability.py --run-kit -s -rs
    # H100 PAD8, 60 folds; about $1 on Modal
    BOILEROOM_MEMORY_CYCLES=20 BOILEROOM_KIT_IMAGE_SOURCE=registry uv run pytest \
        tests/test_memory_stability.py --run-kit --gpu H100 -k padding -s
    # one model or mode: -k esm2 | -k "esmfold2 and exact" | -k "not kit"

``--gpu`` / ``--device`` set the device of every case. On Modal the ESMFold2 / Protenix / OpenDDE cases default to
their kit golden GPUs (``DEFAULT_MODAL_GPU``) and the others to their wrapper's default. Cases run one at a time
(one ``xdist_group``) so that a shared Apptainer GPU only carries one model.
"""

from __future__ import annotations

import importlib
import os
import random
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest
from _kit_parity import DEFAULT_MODAL_GPU, wrapper_device

from boileroom.models.registry import MODEL_SPECS, MODEL_SPECS_BY_KEY

#: Cycles over a case's lengths; use 20 for the long plateau gate.
CYCLES = int(os.environ.get("BOILEROOM_MEMORY_CYCLES", "4"))
if CYCLES < 3:
    raise ValueError("BOILEROOM_MEMORY_CYCLES must be at least 3")
#: Largest permitted rise of any settled per-cycle ``gpu.mem.used_mib`` peak.
#: A 40 MiB/cycle leak fails the second-half window of a 20-cycle trace.
GROWTH_LIMIT_MIB = 256
_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

#: Short lengths for models whose memory does not depend on the length beyond its activations.
SHORT = (60, 90, 120)
#: Lengths inside the native range of the kits' TriMul adapter on both A100 and H100.
KIT = (280, 320, 360)
# Non-aligned sizes exercise the H100 PAD8 path, whose buffers used to accumulate.
PADDED = (529, 577, 625)


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
        name = self.model if self.mode == "vanilla" else f"{self.model}-{self.mode}"
        return f"{name}-padding" if self.lengths == PADDED else name


_ESMFOLD2_OPTIONS = {"num_loops": 3, "num_sampling_steps": 50, "num_diffusion_samples": 1, "include_fields": ["plddt"]}
#: No MSA (no MSA server call), the reduced sampling budget of the Protenix-family integration tests.
_PROTENIX_CONFIG = {"use_msa": False, "sample": 1, "cycle": 4, "step": 50}
_PROTENIX_OPTIONS = {"include_fields": ["plddt", "pae"]}
_MODES = ("vanilla", "exact")

CASES: tuple[MemoryCase, ...] = (
    MemoryCase("minifold", SHORT, options={"include_fields": ["plddt"]}),
    MemoryCase("esmfold", SHORT),
    MemoryCase("esm2", SHORT),
    MemoryCase("esmc", SHORT),
    MemoryCase("esm3", SHORT),
    MemoryCase("sae", SHORT, config={"feature_source": "local"}),
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
        "protenix",
        PADDED,
        config={**_PROTENIX_CONFIG, "optimization": "exact"},
        options=_PROTENIX_OPTIONS,
        mode="exact",
    ),
    *(
        MemoryCase(
            "rf3",
            SHORT,
            config={"optimization": mode, "diffusion_batch_size": 1, "n_recycles": 1, "num_steps": 20},
            options={"include_fields": ["plddt", "pae"]},
            mode=mode,
        )
        for mode in ("vanilla", "exact", "fast", "big")
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
    """Unique sequences with fixed composition and termini, controlling allocation shapes at each length."""
    rng = random.Random(f"{length}-{cycle}")
    residues = list((_AMINO_ACIDS * ((length + 19) // 20))[: length - 2])
    rng.shuffle(residues)
    return "A" + "".join(residues) + "A"


def test_memory_sequences_change_identity_without_changing_composition() -> None:
    for length in (*SHORT, *KIT, *PADDED):
        sequences = [_sequence(length, cycle) for cycle in range(20)]
        assert len(set(sequences)) == 20
        assert all(len(sequence) == length for sequence in sequences)
        assert all(Counter(sequence) == Counter(sequences[0]) for sequence in sequences)
        assert all(sequence[0] == sequence[-1] == "A" for sequence in sequences)


def _wrapper(model: str) -> type:
    module_name, class_name = MODEL_SPECS_BY_KEY[model].wrapper_class_path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), class_name)


def _device(case: MemoryCase, backend: str, device_option: str | None) -> str | None:
    family = MODEL_SPECS_BY_KEY[case.model].family
    if case.lengths == PADDED and backend == "modal" and device_option is None:
        return "H100"
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
    allocator: dict[str, list[list[int]]] = {"allocated": [], "reserved": []}
    with output_ctx(), _wrapper(case.model)(backend=backend_option, device=device, config=dict(case.config)) as model:
        task = getattr(model, task_method)
        for cycle in range(CYCLES):
            used.append([])
            for trace in allocator.values():
                trace.append([])
            for length in case.lengths:
                result = task(_sequence(length, cycle), options=dict(case.options))
                runtime = result.metadata.runtime or {}
                if case.model == "opendde" and case.mode != "vanilla":
                    assert runtime["worker.allocator.alloc_effective"] == "true"
                    assert runtime["predict.allocator.alloc_effective"] == "true"
                    assert runtime["predict.allocator.source"] == "boileroom.OpenDDE"
                for field_name in ("pae", "plddt", "embeddings", "pooled_features"):
                    values = getattr(result, field_name, None)
                    if values is None:
                        continue
                    for value in values if isinstance(values, list) else [values]:
                        if value is not None:
                            assert np.isfinite(value).all(), (
                                f"{case.id}: non-finite {field_name} at cycle {cycle}, length {length}"
                            )
                assert "gpu.mem.used_mib" in runtime, f"{case.id}: the output records no GPU memory: {sorted(runtime)}"
                used[-1].append(int(runtime["gpu.mem.used_mib"]))
                for metric, trace in allocator.items():
                    key = f"gpu.mem.{metric}_mib"
                    value = runtime.get(f"predict.{key}", runtime.get(key))
                    if value is not None:
                        trace[-1].append(int(value))

    print(f"{case.id} gpu.mem.used_mib per cycle of {case.lengths}: {used}")
    _check_memory_plateau(used, case.id, device=True)
    for metric, trace in allocator.items():
        if any(trace):
            assert all(len(row) == len(case.lengths) for row in trace), f"partial {metric} trace: {trace}"
            print(f"{case.id} {metric}_mib: {trace}")
            _check_memory_plateau(trace, f"{case.id} {metric}")


def _check_memory_plateau(used: list[list[int]], name: str, *, device: bool = False) -> None:
    """A shape-dependent allocator may warm for several cycles; persistent growth must stop in the second half.

    Allocator traces retain the original 256 MiB bound, applied to every settled peak.
    Device traces additionally reject sustained growth while tolerating transient variation.
    This catches even 40 MiB/cycle leaks over a 20-cycle run while tolerating bounded initial cache fill.
    """
    start = len(used) // 2 if len(used) >= 10 else 1
    settled = [max(cycle) for cycle in used[start:]]
    # Device usage includes direct CUDA allocations and transient driver state.
    # Retain a sustained-growth guard alongside strict allocator peak bounds.
    growth = (
        float(np.median(settled[-3:]) - np.median(settled[:3]))
        if device and len(used) >= 10
        else max(settled) - settled[0]
    )
    assert growth < GROWTH_LIMIT_MIB, (
        f"{name}: GPU memory grew {growth} MiB after warm-up, cycles {start + 1}–{len(used)} "
        f"(limit {GROWTH_LIMIT_MIB} MiB): {used}"
    )


def test_memory_gate_accepts_bounded_cache_fill_and_rejects_continued_growth() -> None:
    _check_memory_plateau([[4000 + min(i, 6) * 100] for i in range(20)], "bounded warm-up")
    for per_cycle in (40, 75, 900):
        with pytest.raises(AssertionError, match="GPU memory grew"):
            _check_memory_plateau([[4000 + i * per_cycle] for i in range(20)], "leak")
        with pytest.raises(AssertionError, match="GPU memory grew"):
            _check_memory_plateau([[4000 + i * per_cycle] for i in range(20)], "device leak", device=True)


def test_allocator_peak_gate_remains_strict_when_device_usage_fluctuates() -> None:
    trace = [[4000] for _ in range(20)]
    trace[13] = [4700]
    _check_memory_plateau(trace, "bounded device variation", device=True)
    with pytest.raises(AssertionError, match="GPU memory grew"):
        _check_memory_plateau(trace, "retained allocator spike")
