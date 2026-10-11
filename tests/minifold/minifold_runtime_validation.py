"""Approved, bounded real-checkpoint MiniFold validation inside its canonical image.

Independent upstream load and adapter load run sequentially. A real signed loopback
HTTP server exercises production serialization/proxy behavior; this is not an
Apptainer SIF launch or the public Modal class lifecycle.
"""

from __future__ import annotations

import base64
import gc
import hashlib
import io
import json
import os
import secrets
import socket
import sys
import tempfile
import threading
import time
import traceback
import weakref
from contextlib import contextmanager
from pathlib import Path
from typing import Any


@contextmanager
def _live_http(core: Any):
    """Real loopback socket and production server/proxy; no second model load."""
    import httpx
    import uvicorn

    from boileroom.backend import server
    from boileroom.backend.apptainer import _ApptainerModelProxy
    from boileroom.backend.transport import TRANSPORT_HMAC_KEY_ENV

    old_model = server._model_instance
    old_secret = os.environ.get(TRANSPORT_HMAC_KEY_ENV)
    secret = secrets.token_hex(32)
    os.environ[TRANSPORT_HMAC_KEY_ENV] = secret
    server._model_instance = core
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    service = uvicorn.Server(uvicorn.Config(server.app, log_level="warning", lifespan="off"))
    worker = threading.Thread(target=service.run, kwargs={"sockets": [sock]}, daemon=True)
    worker.start()
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=300, trust_env=False) as client:
            deadline = time.monotonic() + 20
            while not service.started:
                assert worker.is_alive() and time.monotonic() < deadline, "HTTP startup failed"
                time.sleep(0.02)
            assert client.get("/health").status_code == 200
            yield _ApptainerModelProxy(client, secret), client
    finally:
        service.should_exit = True
        worker.join(timeout=10)
        sock.close()
        server._model_instance = old_model
        if old_secret is None:
            os.environ.pop(TRANSPORT_HMAC_KEY_ENV, None)
        else:
            os.environ[TRANSPORT_HMAC_KEY_ENV] = old_secret
        assert not worker.is_alive(), "HTTP worker did not shut down"


def _memory() -> dict[str, int]:
    import torch

    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return {
        "allocated": torch.cuda.memory_allocated() // 2**20,
        "reserved": torch.cuda.memory_reserved() // 2**20,
        "device_used": (total - free) // 2**20,
    }


def _plateau(trace: list[list[dict[str, int]]]) -> None:
    """Same 256 MiB post-first-cycle guard as the repository's short memory gate."""
    for metric in ("allocated", "reserved", "device_used"):
        peaks = [max(point[metric] for point in cycle) for cycle in trace[1:]]
        assert max(peaks) - peaks[0] < 256, f"{metric} grew after warm-up: {peaks}"


def _memory_sequence(sequence: str, cycle: int) -> str:
    """Change identity at fixed length/composition/termini to defeat result-cache false passes."""
    interior = sequence[1:-1]
    shift = cycle + 1
    return sequence[0] + interior[shift:] + interior[:shift] + sequence[-1]


@contextmanager
def _compiler_environment():
    """Isolate this opt-in probe before Torch initializes its compiler settings.

    The launcher owns the existing 20-minute remote wall limit. Cache directories
    are unique to this invocation and intentionally retained for diagnostics.
    """
    assert "torch" not in sys.modules, "Compiled validation requires a fresh process before importing Torch"
    root = Path(tempfile.mkdtemp(prefix="minifold-compiled-validation-"))
    settings = {
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
        "TORCHINDUCTOR_CACHE_DIR": str(root / "inductor"),
        "TRITON_CACHE_DIR": str(root / "triton"),
    }
    previous = {key: os.environ.get(key) for key in settings}
    os.environ.update(settings)
    try:
        yield settings
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _compiler_counters() -> dict[str, int]:
    """Read Torch 2.6 evidence without resetting or hiding earlier compilation."""
    from torch._dynamo.utils import counters, guard_failures
    from torch._inductor import metrics

    return {
        "unique_graphs": counters["stats"]["unique_graphs"],
        "generated_kernel_count": metrics.generated_kernel_count,
        "guard_failure_events": sum(len(events) for events in guard_failures.values()),
        "frames_total": counters["frames"]["total"],
        "frames_ok": counters["frames"]["ok"],
        "graph_break_events": sum(counters["graph_break"].values()),
    }


def _compiler_settings() -> dict[str, Any]:
    """Capture effective compiler/precision settings; never enable eager fallback."""
    import torch
    from torch._dynamo import config as dynamo_config
    from torch._inductor import config as inductor_config

    return {
        # TorchVersion passes JSON encoding but makes pickle import Torch on the host.
        "torch": str(torch.__version__),
        "backend": "inductor",
        "compiled_component": "model.fold.miniformer",
        "dynamic": True,
        "fullgraph": True,
        "suppress_errors": dynamo_config.suppress_errors,
        "disable": dynamo_config.disable,
        "compile_threads": inductor_config.compile_threads,
        "max_autotune": inductor_config.max_autotune,
        "fx_graph_cache": inductor_config.fx_graph_cache,
        "cache_size_limit": dynamo_config.cache_size_limit,
        "accumulated_cache_size_limit": dynamo_config.accumulated_cache_size_limit,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "bf16_supported": torch.cuda.is_bf16_supported(),
    }


def _peak_memory() -> dict[str, Any]:
    """Report allocator peaks and Linux process/child host high-water marks."""
    import resource

    import torch

    torch.cuda.synchronize()
    return {
        "gpu_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "gpu_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "host_process_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "host_terminated_child_peak_rss_mib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024,
        "host_scope": "Process-lifetime high-water marks; child maximum is separate, not simultaneous aggregate RSS.",
    }


def _numeric_deltas(actual: Any, expected: Any) -> dict[str, Any]:
    """Keep unaligned raw errors, including failures, without changing tolerances."""
    import numpy as np

    actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    summary: dict[str, Any] = {"actual_shape": list(actual.shape), "expected_shape": list(expected.shape)}
    if actual.shape != expected.shape or not actual.size:
        return {**summary, "comparable": False}
    finite = np.isfinite(actual) & np.isfinite(expected)
    summary.update(comparable=True, nonfinite_count=int(finite.size - finite.sum()))
    if not finite.all():
        return summary
    absolute = np.abs(actual - expected)
    relative = absolute / np.maximum(np.abs(expected), 1e-12)
    return {
        **summary,
        "max_absolute": float(absolute.max()),
        "p95_absolute": float(np.percentile(absolute, 95)),
        "rms_absolute": float(np.sqrt(np.mean(absolute**2))),
        "max_relative": float(relative.max()),
        "p95_relative": float(np.percentile(relative, 95)),
        "relative_denominator_floor": 1e-12,
    }


def _check_parity(
    output: Any,
    index: int,
    case: dict,
    reference: dict,
    manifest: dict,
    report: dict,
    compiled: bool,
    label: str,
    record_deltas: bool = False,
) -> None:
    """Record opt-in numeric evidence before applying the unchanged stock gates."""
    import numpy as np
    from _minifold_cases import _assert_stock_parity

    if compiled or record_deltas:
        residues, names = np.where(reference["final_atom_mask"].astype(bool))
        atoms = output.atom_array[index]
        if compiled and "compiled_output_artifacts_base64" not in report:
            # Retain the cold output even if the first tight parity assertion fails.
            # Sparse atom identities preserve the observed mask without rounding,
            # alignment, or another paid reproduction; stock atom37 data is above.
            stream = io.BytesIO()
            np.savez_compressed(
                stream,
                atom_coordinates=atoms.coord,
                atom_plddt_percent=atoms.b_factor,
                atom_names=atoms.atom_name,
                residue_ids=atoms.res_id,
                residue_names=atoms.res_name,
                chain_ids=atoms.chain_id,
                residue_plddt_unit=output.plddt[index],
            )
            content = stream.getvalue()
            name = "compiled-first-output.npz"
            report["compiled_output_artifacts_base64"] = {name: base64.b64encode(content).decode()}
            report["compiled_output_manifest"] = {
                "call": label,
                "entry": case["id"],
                "index": index,
                "format": "Unrounded sparse atom array; atom names/residue IDs identify the observed atom mask.",
                "files": {name: hashlib.sha256(content).hexdigest()},
            }
        report.setdefault("raw_parity_deltas", []).append(
            {
                "call": label,
                "entry": case["id"],
                "index": index,
                "coordinates_angstrom": _numeric_deltas(
                    atoms.coord, reference["final_atom_positions"][residues, names]
                ),
                "atom_plddt_percent": _numeric_deltas(atoms.b_factor, reference["plddt"][residues]),
                "residue_plddt_unit": _numeric_deltas(output.plddt[index], reference["plddt"] / 100),
                "thresholds": {
                    "coordinates_and_atom_plddt": {"atol": 1e-4, "rtol": 1e-4},
                    "residue_plddt": {"atol": 1e-5, "rtol": 1e-4},
                },
            }
        )
    assert output.metadata.runtime["minifold.compile"] == str(compiled)
    assert manifest["compile"] is False, "The independent reference must remain uncompiled"
    _assert_stock_parity(output, index, case, reference, manifest)


def _eager_adapter_preflight(
    model_size: str, cases: list[dict], references: dict, manifest: dict, report: dict
) -> None:
    """Gate the patched default path against original stock, then release it fully."""
    import torch

    from boileroom.models.minifold.core import MiniFoldCore
    from boileroom.optimization import GuardedCore

    evidence: dict[str, Any] = {"passed": False, "calls": []}
    report["eager_adapter_preflight"] = evidence
    before = _memory()
    started = time.monotonic()
    guard = resident_ref = None
    try:
        config = {"model_size": model_size, "max_sequence_length": 128, "compile": False}
        guard = GuardedCore(lambda: MiniFoldCore(config))
        core = guard.get()
        resident_ref = weakref.ref(core.model)
        resident_id = id(core.model)
        evidence["load_seconds"] = time.monotonic() - started
        for case, recycling in [(case, 3) for case in cases] + [(cases[0], 0)]:
            torch.cuda.synchronize()
            call_started = time.monotonic()
            call = {"entry": case["id"], "length": len(case["sequence"]), "num_recycling": recycling}
            evidence["calls"].append(call)
            try:
                output = core.fold(case["sequence"], {"include_fields": ["*"], "num_recycling": recycling})
            finally:
                torch.cuda.synchronize()
                call["seconds"] = time.monotonic() - call_started
            _check_parity(
                output,
                0,
                case,
                references[f"{case['id']}-r{recycling}.npz"],
                manifest,
                report,
                False,
                f"eager-adapter-r{recycling}",
                record_deltas=True,
            )
            assert id(core.model) == resident_id
            assert output.metadata.runtime is not None
            assert output.metadata.runtime["minifold.num_recycling"] == str(recycling)
        evidence["passed"] = True
    except Exception as error:
        evidence["failure"] = {"error_type": type(error).__name__, "message": str(error)}
        raise
    finally:
        if guard is not None:
            guard.close()
        gc.collect()
        torch.cuda.empty_cache()
        memory = _memory()
        released = resident_ref is None or resident_ref() is None
        gpu_released = memory["allocated"] <= before["allocated"] + 16
        evidence["normal_close"] = {"model_released": released, "gpu_released": gpu_released, "memory_mib": memory}
        evidence["peak_memory"] = _peak_memory()
        evidence["seconds"] = time.monotonic() - started
        evidence["passed"] = evidence["passed"] and released and gpu_released
        assert released, "Eager preflight close retained the model"
        assert gpu_released, "Eager preflight close retained GPU tensors"


def _compiled_warmup(core: Any, cases: list[dict], references: dict, manifest: dict, report: dict) -> None:
    """Compile directly, outside HTTP's shorter deadline, then prove warm reuse."""
    import torch
    from torch._dynamo.eval_frame import OptimizedModule

    evidence = report["compiled_runtime"]
    evidence["settings"] = settings = _compiler_settings()
    assert settings["torch"].startswith("2.6."), "Compiler evidence is validated for pinned Torch 2.6 only"
    assert settings["compile_threads"] == 1
    assert not settings["suppress_errors"] and not settings["disable"], "Compiler fallback/disable is forbidden"
    assert isinstance(core.model.fold.miniformer, OptimizedModule), "MiniFormer was not wrapped by torch.compile"
    evidence["calls"] = []
    evidence["guard_failure_note"] = (
        "Guard failures can accompany bounded shape specialization; they are not a recompile count."
    )
    baseline = _compiler_counters()
    probes = [(case, 3) for case in cases] + [(cases[0], 0)]
    for case, recycling in probes:
        # Two warm repeats must reuse each newly warmed length/recycling variant.
        for repeat in range(3):
            before = _compiler_counters()
            torch.cuda.synchronize()
            started = time.monotonic()
            call: dict[str, Any] = {
                "entry": case["id"],
                "length": len(case["sequence"]),
                "num_recycling": recycling,
                "kind": "first_shape_call" if repeat == 0 else "warm_repeat",
                "repeat": repeat,
                "counters_before": before,
            }
            evidence["calls"].append(call)
            try:
                output = core.fold(case["sequence"], {"include_fields": ["*"], "num_recycling": recycling})
            finally:
                torch.cuda.synchronize()
                after = _compiler_counters()
                call.update(
                    seconds=time.monotonic() - started,
                    counters_after=after,
                    counter_delta={key: after[key] - value for key, value in before.items()},
                    peak_memory=_peak_memory(),
                )
                if len(evidence["calls"]) == 1:
                    evidence["cold_call_seconds"] = call["seconds"]
            evidence["precision"] = output.metadata.runtime["minifold.precision"]
            _check_parity(
                output,
                0,
                case,
                references[f"{case['id']}-r{recycling}.npz"],
                manifest,
                report,
                True,
                f"direct-r{recycling}-repeat{repeat}",
            )
            if len(evidence["calls"]) == 1:
                assert after["unique_graphs"] > baseline["unique_graphs"], "Cold forward captured no Dynamo graph"
                assert after["generated_kernel_count"] > baseline["generated_kernel_count"], (
                    "Cold forward generated no Inductor kernels"
                )
            if repeat:
                for key in ("unique_graphs", "generated_kernel_count"):
                    assert after[key] == before[key], (
                        f"Warm repeat unexpectedly compiled {key}: {call['counter_delta']}"
                    )
    evidence["counters_after_warmup"] = _compiler_counters()


def _compiled_cleanup(report: dict, resident_ref: Any, before: dict | None) -> None:
    """Test normal release first; Dynamo reset can only clean up after that verdict."""
    import torch

    evidence = report["compiled_runtime"]
    try:
        if torch.cuda.is_available():
            memory = _memory()
            released = resident_ref is None or resident_ref() is None
            gpu_released = before is None or memory["allocated"] <= before["allocated"] + 16
            evidence["normal_close_before_dynamo_reset"] = {
                "model_released": released,
                "gpu_released": gpu_released,
                "memory_mib": memory,
                "peak_memory": _peak_memory(),
            }
            if not released or not gpu_released:
                report["passed"] = False
                report["phases"]["cleanup"] = {
                    "passed": False,
                    "message": "Normal close retained the model or GPU tensors before Dynamo reset",
                }
    except Exception as error:
        report["passed"] = False
        report["phases"]["cleanup"] = {
            "passed": False,
            "error_type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
    finally:
        # Never move this before normal close/weakref/GPU evidence above.
        try:
            torch._dynamo.reset()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                evidence["memory_after_dynamo_reset_mib"] = _memory()
        except Exception as error:
            report["passed"] = False
            report["phases"]["post_verdict_dynamo_cleanup"] = {
                "passed": False,
                "error_type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }


def run_validation(model_size: str = "12L", compiled: bool = False) -> dict[str, Any]:
    """Opt into a compiled probe in a fresh process; never change the default path."""
    if type(compiled) is not bool:
        raise ValueError("compiled must be a boolean")
    if not compiled:
        return _run_validation(model_size)
    started = time.monotonic()
    try:
        with _compiler_environment() as environment:
            return _run_validation(model_size, compiled=True, compiler_environment=environment)
    except Exception as error:
        return {
            "passed": False,
            "model_size": model_size,
            "compiled": True,
            "seconds": time.monotonic() - started,
            "phases": {
                "compiler_preflight": {
                    "passed": False,
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                }
            },
        }


def _run_validation(
    model_size: str = "12L", compiled: bool = False, compiler_environment: dict | None = None
) -> dict[str, Any]:
    """Return pass/failure, provenance, raw parity artifacts, quality and memory evidence."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import numpy as np
    import torch
    from _minifold_cases import CASES, experimental_ca, quality_metrics
    from _minifold_reference import run_stock

    from boileroom.models.minifold.core import MiniFoldCore
    from boileroom.optimization import GuardedCore

    started = time.monotonic()
    report: dict[str, Any] = {
        "passed": False,
        "model_size": model_size,
        "phases": {},
        "coverage_limits": [
            f"Only {model_size} is executed; the other checkpoint needs a separate run.",
            f"{'Compiled MiniFormer' if compiled else 'Uncompiled'}, no custom kernels, same L4/image/precision only.",
            "Real signed HTTP exercises server/core/proxy; public Modal class and Apptainer SIF launcher need separate gates.",
            "Four memory cycles are bounded repeated-call evidence, not a long-duration leak proof.",
            "Experimental quality metrics are descriptive; these cases are not established training-held-out data.",
        ],
    }
    if compiled:
        report["compiled"] = True
        report["compiled_runtime"] = {"environment": compiler_environment}
    guard = None
    resident_ref = None
    before = None
    phase = "independent_stock_reference"
    try:
        assert model_size in ("12L", "48L")
        assert torch.cuda.is_available(), "Requires the actual approved CUDA runtime"
        before = _memory()
        if compiled:
            torch.cuda.reset_peak_memory_stats()
        reference = run_stock(model_size, CASES)
        manifest = reference["manifest"]
        manifest["files"] = {name: hashlib.sha256(content).hexdigest() for name, content in reference["files"].items()}
        report["reference_manifest"] = manifest
        report["reference_artifacts_base64"] = {
            name: base64.b64encode(content).decode() for name, content in reference["files"].items()
        }
        report["phases"][phase] = {"passed": True, "seconds": time.monotonic() - started}
        references = {}
        for name, content in reference["files"].items():
            with np.load(io.BytesIO(content), allow_pickle=False) as arrays:
                references[name] = {key: arrays[key].copy() for key in arrays.files}
        del reference
        assert _memory()["allocated"] <= before["allocated"] + 16, "Independent reference retained GPU tensors"
        if compiled:
            report["compiled_runtime"]["independent_reference_peak_memory"] = _peak_memory()
            torch.cuda.reset_peak_memory_stats()
            phase = "eager_adapter_original_stock_parity"
            _eager_adapter_preflight(model_size, CASES, references, manifest, report)
            report["phases"][phase] = {"passed": True, "cases": [case["id"] for case in CASES], "recycling": [3, 0]}
            torch.cuda.reset_peak_memory_stats()
        phase = "fresh_adapter_load"
        config = {"model_size": model_size, "max_sequence_length": 128}
        if compiled:
            config["compile"] = True
        guard = GuardedCore(lambda: MiniFoldCore(config))
        core = guard.get()
        resident_ref = weakref.ref(core.model)
        resident_id = id(core.model)
        report["phases"][phase] = {"passed": True}
        if compiled:
            phase = "compiled_cold_and_warm_forward"
            _compiled_warmup(core, CASES, references, manifest, report)
            report["phases"][phase] = {"passed": True, "shapes_and_recycling": 4, "warm_repeats_per_shape": 2}
        phase = "signed_http_stock_parity"
        with _live_http(core) as (model, client):
            saved = []
            qualities = []
            for case in CASES:
                output = model.fold(case["sequence"], {"include_fields": ["*"]})
                _check_parity(
                    output, 0, case, references[f"{case['id']}-r3.npz"], manifest, report, compiled, "http-single-r3"
                )
                ids, observed = experimental_ca(case)
                ca = output.atom_array[0][output.atom_array[0].atom_name == "CA"]
                np.testing.assert_array_equal(ca.res_id, np.arange(1, len(case["sequence"]) + 1))
                qualities.append({"entry": case["id"], **quality_metrics(ca.coord[ids - 1], observed)})
                saved.append(
                    (output, output.atom_array[0].coord.copy(), json.dumps(output.metadata.runtime, sort_keys=True))
                )
            report["experimental_quality"] = qualities
            report["phases"][phase] = {"passed": True, "cases": [case["id"] for case in CASES]}
            phase = "mixed_length_order_and_recycling"
            for cases in (CASES, list(reversed(CASES))):
                output = model.fold([case["sequence"] for case in cases], {"include_fields": ["*"]})
                assert output.metadata.sequence_lengths == [len(case["sequence"]) for case in cases]
                for index, case in enumerate(cases):
                    _check_parity(
                        output,
                        index,
                        case,
                        references[f"{case['id']}-r3.npz"],
                        manifest,
                        report,
                        compiled,
                        "http-mixed-r3",
                    )
                    length = len(case["sequence"])
                    np.testing.assert_array_equal(output.residue_index[index, :length], np.arange(length))
                    assert (output.residue_index[index, length:] == -1).all()
                    assert (output.chain_index[index, :length] == 0).all()
                    assert (output.chain_index[index, length:] == -1).all()
                assert output.pae is None
            changed = model.fold(CASES[0]["sequence"], {"include_fields": ["*"], "num_recycling": 0})
            _check_parity(
                changed,
                0,
                CASES[0],
                references[f"{CASES[0]['id']}-r0.npz"],
                manifest,
                report,
                compiled,
                "http-single-r0",
            )
            assert changed.metadata.runtime["minifold.num_recycling"] == "0"
            report["phases"][phase] = {"passed": True}
            phase = "filtering_refusal_recovery"
            minimal = model.fold(CASES[0]["sequence"])
            assert minimal.plddt is minimal.pdb is minimal.cif is minimal.residue_index is None
            assert minimal.metadata.runtime["minifold.num_recycling"] == "3"
            for sequence in ("AC:DE", "A" * 129, [CASES[0]["sequence"], "AC?"]):
                response = client.post("/fold", json={"sequences": sequence})
                assert response.status_code == 500 and response.json()["error_type"] == "ValueError"
            response = client.post("/fold", json={"sequences": "ACD", "options": {"model_size": model_size}})
            assert response.status_code == 500 and response.json()["error_type"] == "ValueError"
            selected = model.fold(CASES[0]["sequence"], {"include_fields": ["plddt"]})
            assert selected.plddt is not None and selected.pdb is selected.cif is None
            np.testing.assert_allclose(selected.plddt[0], saved[0][0].plddt[0], atol=1e-5, rtol=1e-4)
            report["phases"][phase] = {"passed": True}
            phase = "repeated_varying_length_memory"
            trace = []
            for cycle in range(4):
                points = []
                for case in CASES:
                    output = model.fold(_memory_sequence(case["sequence"], cycle), {"include_fields": ["plddt"]})
                    assert id(core.model) == resident_id
                    assert np.isfinite(output.plddt[0]).all()
                    points.append(_memory())
                trace.append(points)
            report["memory_trace_mib"] = trace
            _plateau(trace)
            for output, coords, metadata in saved:
                np.testing.assert_array_equal(output.atom_array[0].coord, coords)
                assert json.dumps(output.metadata.runtime, sort_keys=True) == metadata
            report["phases"][phase] = {
                "passed": True,
                "cycles": 4,
                "lengths": [len(case["sequence"]) for case in CASES],
            }
        if compiled:
            phase = "compiled_http_graph_reuse"
            evidence = report["compiled_runtime"]
            after = evidence["counters_after_http"] = _compiler_counters()
            evidence["http_counter_delta"] = {
                key: after[key] - value for key, value in evidence["counters_after_warmup"].items()
            }
            for key in ("unique_graphs", "generated_kernel_count"):
                assert evidence["http_counter_delta"][key] == 0, f"Warmed HTTP calls unexpectedly compiled {key}"
            report["phases"][phase] = {"passed": True}
        phase = "cleanup"
        guard.close()
        gc.collect()
        torch.cuda.empty_cache()
        assert resident_ref() is None, "Closed core retained the model"
        report["memory_after_close_mib"] = _memory()
        assert report["memory_after_close_mib"]["allocated"] <= before["allocated"] + 16
        report["phases"][phase] = {"passed": True}
        report["passed"] = True
    except Exception as error:
        report["phases"][phase] = {
            "passed": False,
            "error_type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
    finally:
        if guard is not None:
            guard.close()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if compiled:
            _compiled_cleanup(report, resident_ref, before)
        report["seconds"] = time.monotonic() - started
    return report
