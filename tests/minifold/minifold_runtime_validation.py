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
import threading
import time
import traceback
import weakref
from contextlib import contextmanager
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


def run_validation(model_size: str = "12L") -> dict[str, Any]:
    """Return pass/failure, provenance, raw parity artifacts, quality and memory evidence."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import numpy as np
    import torch
    from _minifold_cases import CASES, _assert_stock_parity, experimental_ca, quality_metrics
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
            "Uncompiled, no custom kernels, same L4/image/precision only.",
            "Real signed HTTP exercises server/core/proxy; public Modal class and Apptainer SIF launcher need separate gates.",
            "Four memory cycles are bounded repeated-call evidence, not a long-duration leak proof.",
            "Experimental quality metrics are descriptive; these cases are not established training-held-out data.",
        ],
    }
    guard = None
    resident_ref = None
    phase = "independent_stock_reference"
    try:
        assert model_size in ("12L", "48L")
        assert torch.cuda.is_available(), "Requires the actual approved CUDA runtime"
        before = _memory()
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
        phase = "fresh_adapter_load"
        guard = GuardedCore(lambda: MiniFoldCore({"model_size": model_size, "max_sequence_length": 128}))
        core = guard.get()
        resident_ref = weakref.ref(core.model)
        resident_id = id(core.model)
        report["phases"][phase] = {"passed": True}
        phase = "signed_http_stock_parity"
        with _live_http(core) as (model, client):
            saved = []
            qualities = []
            for case in CASES:
                output = model.fold(case["sequence"], {"include_fields": ["*"]})
                _assert_stock_parity(output, 0, case, references[f"{case['id']}-r3.npz"], manifest)
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
                    _assert_stock_parity(output, index, case, references[f"{case['id']}-r3.npz"], manifest)
                    length = len(case["sequence"])
                    np.testing.assert_array_equal(output.residue_index[index, :length], np.arange(length))
                    assert (output.residue_index[index, length:] == -1).all()
                    assert (output.chain_index[index, :length] == 0).all()
                    assert (output.chain_index[index, length:] == -1).all()
                assert output.pae is None
            changed = model.fold(CASES[0]["sequence"], {"include_fields": ["*"], "num_recycling": 0})
            _assert_stock_parity(changed, 0, CASES[0], references[f"{CASES[0]['id']}-r0.npz"], manifest)
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
        report["seconds"] = time.monotonic() - started
    return report
