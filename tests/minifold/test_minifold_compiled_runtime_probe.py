"""Offline contracts for the explicit compiled probe; no model/paid GPU launch."""

import base64
import hashlib
import io
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest

pytestmark = pytest.mark.contract


def _assert_native_json_tree(value: Any) -> None:
    """Reject subclasses too: JSON accepts TorchVersion, but pickle needs Torch."""
    if type(value) is dict:
        for key, item in value.items():
            assert type(key) is str
            _assert_native_json_tree(item)
    elif type(value) is list:
        for item in value:
            _assert_native_json_tree(item)
    else:
        assert type(value) in (str, int, float, bool, type(None)), type(value)


@pytest.mark.parametrize("model_size", ["12L", "48L"])
def test_default_dispatch_keeps_uncompiled_path_and_environment(monkeypatch, model_size):
    import minifold_runtime_validation as validation

    implementation = Mock(return_value={"passed": True})
    monkeypatch.setattr(validation, "_run_validation", implementation)
    environment = dict(os.environ)
    assert validation.run_validation(model_size) == {"passed": True}
    implementation.assert_called_once_with(model_size)
    assert dict(os.environ) == environment


@pytest.mark.parametrize("value", [None, "true", 1])
def test_compiled_flag_is_strict_boolean(value):
    from minifold_runtime_validation import run_validation

    with pytest.raises(ValueError, match="boolean"):
        run_validation(compiled=value)


def test_withdrawn_compiled_probe_refuses_without_launch(monkeypatch):
    import minifold_runtime_validation as validation

    monkeypatch.setitem(sys.modules, "torch", object())
    implementation = Mock(side_effect=AssertionError("must not run"))
    monkeypatch.setattr(validation, "_run_validation", implementation)
    result = validation.run_validation(compiled=True)
    assert result["passed"] is False and result["compiled"] is True
    assert "withdrawn" in result["phases"]["compiler_preflight"]["message"]
    implementation.assert_not_called()
    _assert_native_json_tree(result)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("fails", [False, True])
def test_compiler_environment_precedes_import_and_restores_previous_values(monkeypatch, tmp_path, fails):
    import minifold_runtime_validation as validation

    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setattr(validation.tempfile, "mkdtemp", lambda **_: str(tmp_path))
    monkeypatch.setenv("TORCHINDUCTOR_COMPILE_THREADS", "9")
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", "/unused-prior-cache")
    monkeypatch.delenv("TRITON_CACHE_DIR", raising=False)

    def implementation(model_size, compiled, compiler_environment):
        assert model_size == "12L" and compiled is True
        assert "torch" not in sys.modules
        assert os.environ["TORCHINDUCTOR_COMPILE_THREADS"] == "1"
        assert compiler_environment == {
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
            "TORCHINDUCTOR_CACHE_DIR": str(tmp_path / "inductor"),
            "TRITON_CACHE_DIR": str(tmp_path / "triton"),
        }
        if fails:
            raise RuntimeError("probe failed")
        return {"passed": True}

    if fails:
        with pytest.raises(RuntimeError, match="probe failed"), validation._compiler_environment() as environment:
            implementation("12L", True, environment)
    else:
        with validation._compiler_environment() as environment:
            assert implementation("12L", True, environment)["passed"] is True
    assert os.environ["TORCHINDUCTOR_COMPILE_THREADS"] == "9"
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == "/unused-prior-cache"
    assert "TRITON_CACHE_DIR" not in os.environ
    assert tmp_path.exists()  # No cleanup/deletion of compiler caches.


def test_pinned_torch_reads_single_compile_thread_in_fresh_process():
    """Exercise real Torch config/counter APIs on CPU, without compiling a model."""
    script = """
import json
from minifold_runtime_validation import _compiler_environment, _compiler_counters
with _compiler_environment() as environment:
    import torch
    from torch._inductor import config
    assert config.compile_threads == 1
    assert torch.__version__.startswith('2.6.')
    assert _compiler_counters()['unique_graphs'] == 0
    assert _compiler_counters()['generated_kernel_count'] == 0
    print(json.dumps(environment))
"""
    environment = {**os.environ, "PYTHONPATH": str(Path(__file__).parent)}
    process = subprocess.run(
        [sys.executable, "-c", script], env=environment, capture_output=True, text=True, timeout=60
    )
    assert process.returncode == 0, process.stderr
    configured = json.loads(process.stdout)
    _assert_native_json_tree(configured)
    assert configured["TORCHINDUCTOR_COMPILE_THREADS"] == "1"
    assert Path(configured["TORCHINDUCTOR_CACHE_DIR"]).parent == Path(configured["TRITON_CACHE_DIR"]).parent


def test_real_compiler_evidence_unpickles_without_torch(monkeypatch):
    """Audit actual pinned Torch values, not just JSON encoding or mocked settings."""
    import minifold_runtime_validation as validation
    import torch
    from torch.torch_version import TorchVersion

    monkeypatch.setattr(torch, "__version__", TorchVersion("2.6.0+cu124"))
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 2**20)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 2**21)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 2**20)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 2**21)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (2**28, 2**29))
    report = {
        "compiled_runtime": {
            "settings": validation._compiler_settings(),
            "counters": validation._compiler_counters(),
            "peak_memory": validation._peak_memory(),
            "memory": validation._memory(),
            "numeric_deltas": validation._numeric_deltas([1, 3], [1, 2]),
        }
    }
    _assert_native_json_tree(report)
    assert report["compiled_runtime"]["settings"]["torch"] == str(torch.__version__)
    script = """
import importlib.util
import json
import pickle
import sys
assert importlib.util.find_spec('torch') is None
value = pickle.loads(sys.stdin.buffer.read())
assert type(value['compiled_runtime']['settings']['torch']) is str
print(json.dumps(value, allow_nan=False))
"""
    process = subprocess.run(
        [sys.executable, "-I", "-S", "-c", script],
        input=pickle.dumps(report),
        capture_output=True,
        timeout=30,
    )
    assert process.returncode == 0, process.stderr.decode()
    assert json.loads(process.stdout) == report


def test_numeric_deltas_are_raw_and_json_safe():
    from minifold_runtime_validation import _numeric_deltas

    result = _numeric_deltas([1, 3], [1, 2])
    assert result["max_absolute"] == 1
    assert result["p95_absolute"] == pytest.approx(0.95)
    assert result["rms_absolute"] == pytest.approx(np.sqrt(0.5))
    assert result["max_relative"] == 0.5
    assert result["p95_relative"] == pytest.approx(0.475)
    assert _numeric_deltas([1], [1, 2])["comparable"] is False
    nonfinite = _numeric_deltas([np.nan], [1])
    assert nonfinite["nonfinite_count"] == 1
    _assert_native_json_tree(result)
    _assert_native_json_tree(nonfinite)
    json.dumps(nonfinite, allow_nan=False)


@pytest.fixture
def parity_input():
    mask = np.zeros((2, 37))
    mask[:, 0] = 1
    reference = {
        "final_atom_positions": np.zeros((2, 37, 3)),
        "final_atom_mask": mask,
        "plddt": np.full(2, 50.0),
    }
    result = SimpleNamespace(
        atom_array=[
            SimpleNamespace(
                coord=np.ones((2, 3)),
                b_factor=np.full(2, 50.0),
                atom_name=np.array(["N", "N"]),
                res_id=np.array([1, 2]),
                res_name=np.array(["ALA", "ALA"]),
                chain_id=np.array(["A", "A"]),
            )
        ],
        plddt=[np.full(2, 0.5)],
        metadata=SimpleNamespace(runtime={"minifold.compile": "True"}),
    )
    return result, reference


def test_raw_deltas_survive_strict_parity_failure(monkeypatch, parity_input):
    import _minifold_cases
    from minifold_runtime_validation import _check_parity

    output, reference = parity_input
    report: dict[str, Any] = {}

    def strict_assertion(*args):
        assert report["raw_parity_deltas"][0]["coordinates_angstrom"]["max_absolute"] == 1
        raise AssertionError("unchanged tight gate failed")

    monkeypatch.setattr(_minifold_cases, "_assert_stock_parity", strict_assertion)
    with pytest.raises(AssertionError, match="tight gate"):
        _check_parity(output, 0, {"id": "fixture"}, reference, {"compile": False}, report, True, "cold")
    assert report["raw_parity_deltas"][0]["thresholds"]["coordinates_and_atom_plddt"] == {"atol": 1e-4, "rtol": 1e-4}
    content = base64.b64decode(report["compiled_output_artifacts_base64"]["compiled-first-output.npz"])
    assert (
        hashlib.sha256(content).hexdigest() == report["compiled_output_manifest"]["files"]["compiled-first-output.npz"]
    )
    with np.load(io.BytesIO(content), allow_pickle=False) as artifact:
        np.testing.assert_array_equal(artifact["atom_coordinates"], output.atom_array[0].coord)
        np.testing.assert_array_equal(artifact["atom_names"], ["N", "N"])
        np.testing.assert_array_equal(artifact["residue_ids"], [1, 2])
        np.testing.assert_array_equal(artifact["residue_plddt_unit"], [0.5, 0.5])
    _assert_native_json_tree(report)
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("compiled", [False, True])
def test_compile_metadata_must_match_requested_mode(monkeypatch, parity_input, compiled):
    import _minifold_cases
    from minifold_runtime_validation import _check_parity

    output, reference = parity_input
    output.metadata.runtime["minifold.compile"] = str(not compiled)
    assertion = Mock()
    monkeypatch.setattr(_minifold_cases, "_assert_stock_parity", assertion)
    with pytest.raises(AssertionError):
        _check_parity(output, 0, {"id": "fixture"}, reference, {"compile": False}, {}, compiled, "probe")
    assertion.assert_not_called()


def test_eager_preflight_deltas_do_not_claim_a_compiled_artifact(monkeypatch, parity_input):
    import _minifold_cases
    from minifold_runtime_validation import _check_parity

    output, reference = parity_input
    output.metadata.runtime["minifold.compile"] = "False"
    report: dict[str, Any] = {}
    monkeypatch.setattr(_minifold_cases, "_assert_stock_parity", Mock())
    _check_parity(
        output,
        0,
        {"id": "fixture"},
        reference,
        {"compile": False},
        report,
        False,
        "eager-adapter-r3",
        record_deltas=True,
    )
    assert report["raw_parity_deltas"][0]["coordinates_angstrom"]["max_absolute"] == 1
    assert "compiled_output_artifacts_base64" not in report
    _assert_native_json_tree(report)


@pytest.mark.parametrize("failure", [None, "parity", "model", "gpu"])
def test_eager_adapter_preflight_is_sequential_and_blocks_failed_compiled_load(monkeypatch, failure):
    import _minifold_cases
    import _minifold_reference
    import minifold_runtime_validation as validation
    import torch

    from boileroom import optimization
    from boileroom.models.minifold import core as core_module

    events = []
    state = {"gpu_leaked": False}

    class Resident:
        pass

    class Core:
        def __init__(self, config):
            self.config = config
            self.model: Any = Resident()
            events.append(("load", config["compile"]))
            if config["compile"]:
                assert ("close", False) in events

        def fold(self, sequence, options):
            assert self.config["compile"] is False
            events.append(("fold", options["num_recycling"]))
            return SimpleNamespace(
                metadata=SimpleNamespace(runtime={"minifold.num_recycling": str(options["num_recycling"])})
            )

        def close(self):
            events.append(("close", self.config["compile"]))
            if failure != "model":
                self.model = None
            if failure == "gpu":
                state["gpu_leaked"] = True

    class Guard:
        def __init__(self, factory):
            self.core = factory()

        def get(self):
            return self.core

        def close(self):
            self.core.close()

    def stock(model_size, cases):
        assert model_size == "12L" and not events
        events.extend([("stock", "load"), ("stock", "close")])
        stream = io.BytesIO()
        np.savez_compressed(stream, placeholder=np.array([1]))
        files = {f"{case['id']}-r3.npz": stream.getvalue() for case in cases}
        files[f"{cases[0]['id']}-r0.npz"] = stream.getvalue()
        return {"manifest": {"compile": False}, "files": files}

    def parity(output, index, case, reference, manifest, report, compiled, label, record_deltas=False):
        assert compiled is False and record_deltas is True
        events.append(("parity", case["id"]))
        if failure == "parity":
            raise AssertionError("eager parity failed")

    def compiled_warmup(*args):
        events.append(("compiled", "forward"))
        raise RuntimeError("stop before real compiled work")

    monkeypatch.setattr(core_module, "MiniFoldCore", Core)
    monkeypatch.setattr(optimization, "GuardedCore", Guard)
    monkeypatch.setattr(_minifold_reference, "run_stock", stock)
    monkeypatch.setattr(validation, "_check_parity", parity)
    monkeypatch.setattr(validation, "_compiled_warmup", compiled_warmup)
    monkeypatch.setattr(validation, "_compiled_cleanup", lambda *args: None)
    monkeypatch.setattr(validation, "_peak_memory", lambda: {})
    monkeypatch.setattr(validation, "_memory", lambda: {"allocated": 17 if state["gpu_leaked"] else 0})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    report = validation._run_validation("12L", compiled=True)
    evidence = report["eager_adapter_preflight"]
    assert events[:3] == [("stock", "load"), ("stock", "close"), ("load", False)]
    assert ("close", False) in events
    if failure is None:
        assert evidence["passed"] is True
        assert [(call["entry"], call["num_recycling"]) for call in evidence["calls"]] == [
            (case["id"], 3) for case in _minifold_cases.CASES
        ] + [(_minifold_cases.CASES[0]["id"], 0)]
        assert events.index(("close", False)) < events.index(("load", True)) < events.index(("compiled", "forward"))
        assert report["phases"]["eager_adapter_original_stock_parity"]["passed"] is True
    else:
        assert evidence["passed"] is False
        assert ("load", True) not in events and ("compiled", "forward") not in events
        assert report["phases"]["eager_adapter_original_stock_parity"]["passed"] is False
    _assert_native_json_tree(report)


@pytest.fixture
def compiled_probe(monkeypatch):
    import minifold_runtime_validation as validation
    import torch
    import torch._dynamo.eval_frame

    class CompiledModule:
        pass

    monkeypatch.setattr(torch._dynamo.eval_frame, "OptimizedModule", CompiledModule)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    settings = {"torch": "2.6.0", "compile_threads": 1, "suppress_errors": False, "disable": False}
    monkeypatch.setattr(validation, "_compiler_settings", lambda: settings)
    counters = {"unique_graphs": 0, "generated_kernel_count": 0, "guard_failure_events": 0}
    monkeypatch.setattr(validation, "_compiler_counters", lambda: dict(counters))
    monkeypatch.setattr(validation, "_peak_memory", lambda: {})
    parity = Mock()
    monkeypatch.setattr(validation, "_check_parity", parity)
    seen = set()

    def fold(sequence, options):
        key = (len(sequence), options["num_recycling"])
        if key not in seen:
            counters["unique_graphs"] += 1
            counters["generated_kernel_count"] += 2
            seen.add(key)
        return SimpleNamespace(metadata=SimpleNamespace(runtime={"minifold.precision": "fp32-lm/bf16-fold"}))

    core = SimpleNamespace(
        model=SimpleNamespace(fold=SimpleNamespace(miniformer=CompiledModule())), fold=Mock(side_effect=fold)
    )
    cases = [{"id": str(length), "sequence": "A" * length} for length in (3, 5, 7)]
    references: dict[str, Any] = {f"{case['id']}-r3.npz": {} for case in cases} | {"3-r0.npz": {}}
    report: dict[str, Any] = {"compiled_runtime": {}}
    return validation, core, cases, references, report, settings, counters, parity


def test_cold_shape_specializations_then_two_warm_repeats(compiled_probe):
    validation, core, cases, references, report, _, _, parity = compiled_probe
    validation._compiled_warmup(core, cases, references, {"compile": False}, report)
    calls = report["compiled_runtime"]["calls"]
    assert len(calls) == core.fold.call_count == parity.call_count == 12
    assert sum(call["kind"] == "warm_repeat" for call in calls) == 8
    assert [(call["length"], call["num_recycling"]) for call in calls[::3]] == [(3, 3), (5, 3), (7, 3), (3, 0)]
    assert all(call["counter_delta"]["unique_graphs"] == (call["repeat"] == 0) for call in calls)
    assert report["compiled_runtime"]["cold_call_seconds"] >= 0
    _assert_native_json_tree(report)
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    "setting,value", [("suppress_errors", True), ("disable", True), ("compile_threads", 2), ("torch", "2.7.0")]
)
def test_compiler_preflight_never_silently_falls_back(compiled_probe, setting, value):
    validation, core, cases, references, report, settings, _, _ = compiled_probe
    settings[setting] = value
    with pytest.raises(AssertionError):
        validation._compiled_warmup(core, cases, references, {"compile": False}, report)
    core.fold.assert_not_called()


@pytest.mark.parametrize("missing", ["unique_graphs", "generated_kernel_count"])
def test_cold_forward_requires_both_compiler_evidence_counters(compiled_probe, missing):
    validation, core, cases, references, report, _, counters, _ = compiled_probe
    implementation = core.fold.side_effect

    def fold(*args):
        output = implementation(*args)
        counters[missing] = 0
        return output

    core.fold.side_effect = fold
    with pytest.raises(AssertionError, match="Cold forward"):
        validation._compiled_warmup(core, cases, references, {"compile": False}, report)
    assert len(report["compiled_runtime"]["calls"]) == 1
    assert report["compiled_runtime"]["cold_call_seconds"] >= 0


def test_new_compilation_on_a_warm_repeat_fails_with_evidence(compiled_probe):
    validation, core, cases, references, report, _, counters, _ = compiled_probe
    implementation = core.fold.side_effect

    def fold(*args):
        output = implementation(*args)
        counters["unique_graphs"] += 1
        return output

    core.fold.side_effect = fold
    with pytest.raises(AssertionError, match="Warm repeat unexpectedly compiled"):
        validation._compiled_warmup(core, cases, references, {"compile": False}, report)
    assert len(report["compiled_runtime"]["calls"]) == 2


@pytest.mark.parametrize("model_retained,gpu_allocated", [(False, 0), (True, 0), (False, 17)])
def test_normal_release_verdict_precedes_dynamo_reset(monkeypatch, model_retained, gpu_allocated):
    import minifold_runtime_validation as validation
    import torch

    events = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append("empty"))
    monkeypatch.setattr(torch._dynamo, "reset", lambda: events.append("reset"))
    monkeypatch.setattr(validation, "_peak_memory", lambda: {})

    def memory():
        events.append("memory")
        return {"allocated": 0 if "reset" in events else gpu_allocated}

    def reference():
        events.append("weakref")
        return object() if model_retained else None

    monkeypatch.setattr(validation, "_memory", memory)
    report: dict[str, Any] = {"passed": True, "phases": {}, "compiled_runtime": {}}
    validation._compiled_cleanup(report, reference, {"allocated": 0})
    assert events.index("memory") < events.index("weakref") < events.index("reset")
    expected = not model_retained and gpu_allocated <= 16
    assert report["passed"] is expected
    assert report["compiled_runtime"]["normal_close_before_dynamo_reset"]["memory_mib"]["allocated"] == gpu_allocated
    assert report["compiled_runtime"]["memory_after_dynamo_reset_mib"]["allocated"] == 0
    if not expected:
        assert report["phases"]["cleanup"]["passed"] is False
    _assert_native_json_tree(report)


def test_dynamo_cleanup_error_preserves_prior_failure_evidence(monkeypatch):
    import minifold_runtime_validation as validation
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch._dynamo, "reset", Mock(side_effect=RuntimeError("reset failed")))
    report: dict[str, Any] = {
        "passed": False,
        "phases": {"parity": {"passed": False}},
        "compiled_runtime": {"calls": ["retained"]},
    }
    validation._compiled_cleanup(report, None, None)
    assert report["compiled_runtime"]["calls"] == ["retained"]
    assert report["phases"]["parity"]["passed"] is False
    assert report["phases"]["post_verdict_dynamo_cleanup"]["message"] == "reset failed"
    _assert_native_json_tree(report)
