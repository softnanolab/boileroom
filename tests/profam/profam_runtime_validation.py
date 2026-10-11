"""Real-checkpoint validation inside the canonical image; never imported at collection.

The reference is the pinned upstream API on the same resident weights, bypassing
Boileroom's adapter. This measures adapter parity, not independent model accuracy.
PDB sequences are public reference proteins; the explicitly labelled point
variants only exercise family conditioning and do not claim natural homology.
"""

from __future__ import annotations

import copy
import gc
import math
import os
import random
import secrets
import socket
import threading
import time
import traceback
import warnings
import weakref
from contextlib import contextmanager
from dataclasses import asdict
from typing import Any

# RCSB FASTA, checked 2026-10-11:
# https://www.rcsb.org/fasta/entry/1UBQ/display (human ubiquitin, 76 aa)
# https://www.rcsb.org/fasta/entry/1CRN/display (crambin, 46 aa)
UBIQUITIN = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
CRAMBIN = "TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN"
FAMILIES = {
    "ubiquitin": ([UBIQUITIN, UBIQUITIN[:-1] + "A"], ["1UBQ_A", "1UBQ_G76A_synthetic"]),
    "crambin": ([CRAMBIN, CRAMBIN[:-1] + "Q"], ["1CRN_A", "1CRN_N46Q_synthetic"]),
}


def _sampling(options: dict[str, Any]) -> dict[str, Any]:
    """Independent explicit upstream call mapping, not the adapter's helper."""
    return {
        "num_samples": options["num_samples"],
        "max_tokens": options["max_context_tokens"],
        "max_generated_length": options["max_generated_length"],
        "max_sequence_length_multiplier": options["max_sequence_length_multiplier"],
        "temperature": options["temperature"],
        "top_p": options["top_p"],
        "sampler": options["sampler"],
        "num_prompts_in_ensemble": options["num_prompts_in_ensemble"],
        "reduction": options["reduction"],
        "minimum_sequence_length_proportion": options["minimum_sequence_length_proportion"],
        "maximum_retries": options["maximum_retries"],
        "repeat_guard": options["repeat_guard"],
        "seed": options["seed"],
        "prompt_accessions": options["prompt_accessions"],
    }


def _canonical_result(output: Any) -> dict[str, Any]:
    """Sampling result without timing or mutable runtime counters."""
    return {
        "sequences": list(output.sequences),
        "scores": list(getattr(output, "log_likelihoods", getattr(output, "scores", []))),
        "conditioning": [
            {"sequences": list(item.sequences), "accessions": list(item.accessions)}
            for item in output.conditioning_prompts
        ],
    }


def _assert_same(actual: Any, reference: Any) -> None:
    left, right = _canonical_result(actual), _canonical_result(reference)
    assert left["sequences"] == right["sequences"], "seeded upstream sequence parity failed"
    assert left["conditioning"] == right["conditioning"], "paired conditioning parity failed"
    assert len(left["scores"]) == len(right["scores"])
    assert all(
        math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-5) for a, b in zip(left["scores"], right["scores"], strict=True)
    ), "upstream sampling-score parity failed"


def _validate_output(output: Any, prompt: list[str], options: dict[str, Any]) -> None:
    from boileroom.models.profam.core import AMINO_ACIDS, _violates_repeat_guard
    from boileroom.models.profam.types import ProFamOutput

    assert isinstance(output, ProFamOutput)
    assert len(output.sequences) == len(output.log_likelihoods) == options["num_samples"]
    assert all(type(sequence) is str and sequence and set(sequence) <= AMINO_ACIDS for sequence in output.sequences)
    assert all(type(score) is float and math.isfinite(score) and score <= 1e-6 for score in output.log_likelihoods)
    assert output.metadata.sequence_lengths == list(map(len, output.sequences))
    pairs = dict(zip(options["prompt_accessions"], prompt, strict=True))
    for conditioning in output.conditioning_prompts:
        assert conditioning.sequences
        assert all(
            pairs[accession] == sequence
            for accession, sequence in zip(conditioning.accessions, conditioning.sequences, strict=True)
        )
        assert (
            2 + sum(len(sequence) + 1 for sequence in conditioning.sequences) + options["max_generated_length"]
            <= options["max_context_tokens"]
        )
    source = output.conditioning_prompts[0].sequences if options["sampler"] == "single" else prompt
    minimum = int(min(map(len, source)) * options["minimum_sequence_length_proportion"])
    assert all(minimum <= len(sequence) <= options["max_generated_length"] for sequence in output.sequences)
    assert all(not _violates_repeat_guard(sequence) for sequence in output.sequences)


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


def _allocation_diagnostics() -> dict[str, Any]:
    """Report allocator-owned blocks separately from live Python CUDA tensors."""
    import torch

    blocks = [
        {"size": block["size"], "requested_size": block.get("requested_size", 0), "state": block["state"]}
        for segment in torch.cuda.memory_snapshot()
        for block in segment["blocks"]
        if block["state"].startswith("active")
    ]
    tensors = []
    # gc's object list is released before the next memory measurement; the
    # report contains only scalar metadata, never tensors or their contents.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        for value in gc.get_objects():
            if isinstance(value, torch.Tensor) and value.is_cuda:
                tensors.append(
                    {
                        "shape": list(value.shape),
                        "dtype": str(value.dtype),
                        "bytes": value.numel() * value.element_size(),
                    }
                )
    return {"active_blocks": blocks, "live_cuda_tensor_count": len(tensors), "live_cuda_tensor_examples": tensors[:32]}


def run_validation() -> dict[str, Any]:
    """Run bounded real inference and preserve failed-phase evidence for CI artifacts."""
    import torch

    from boileroom.models.profam.core import CHECKPOINT_SHA256, UPSTREAM_COMMIT, ProFamCore
    from boileroom.optimization import initialize_core

    started = time.monotonic()
    report: dict[str, Any] = {
        "passed": False,
        "phases": {},
        "coverage_limits": [
            "Adapter parity shares the upstream weights and code; it is not independent scientific validation.",
            "Signed HTTP uses a real loopback service, not an Apptainer SIF launch.",
            "Four memory cycles are a bounded smoke gate, not a long-duration leak proof.",
            "Sequence validity and conditioning responsiveness do not establish biological function or fitness.",
        ],
    }
    core = ProFamCore({"num_samples": 1, "max_context_tokens": 512})
    resident_ref = None
    before_load_allocated = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    phase = "load"
    try:
        assert torch.cuda.is_available(), "real validation requires one CUDA GPU"
        failure = initialize_core(core)
        if failure is not None:
            raise failure
        resident_id = id(core.model)
        resident_ref = weakref.ref(core.model)
        report["provenance"] = dict(core._runtime)
        assert core._runtime["profam.checkpoint_sha256"] == CHECKPOINT_SHA256
        assert core._runtime["profam.commit"] == UPSTREAM_COMMIT
        report["phases"][phase] = {"passed": True, "seconds": time.monotonic() - started}
        saved = []
        phase = "upstream_parity"
        for family_name, (prompt, accessions) in FAMILIES.items():
            for sampler, reduction in (
                ("single", "mean_probs"),
                ("ensemble", "mean_probs"),
                ("ensemble", "sum_log_probs"),
            ):
                options = dict(core.config) | {
                    "sampler": sampler,
                    "seed": 42,
                    "num_samples": 2 if family_name == "crambin" and sampler == "single" else 1,
                    "temperature": 1.0,
                    "top_p": 0.95,
                    "max_generated_length": int(max(map(len, prompt)) * 1.2),
                    "num_prompts_in_ensemble": 2,
                    "reduction": reduction,
                    "prompt_accessions": accessions,
                }
                options = {key: value for key, value in options.items() if key not in core.STATIC_CONFIG_KEYS}
                actual = core.generate(prompt, options)
                _validate_output(actual, prompt, options)
                with torch.inference_mode():
                    reference = core.model.generate(prompt=prompt, **_sampling(options))
                _assert_same(actual, reference)
                del reference
                saved.append((actual, copy.deepcopy(asdict(actual))))
                report["phases"][f"parity_{family_name}_{sampler}_{reduction}"] = {
                    "passed": True,
                    **_canonical_result(actual),
                }
                assert id(core.model) == resident_id, "weights reloaded between calls"
        phase = "seed_and_isolation"
        first = saved[0][0]
        prompt, accessions = FAMILIES["ubiquitin"]
        seed_options = {"seed": 42, "prompt_accessions": accessions, "max_generated_length": int(len(UBIQUITIN) * 1.2)}
        again = core.generate(prompt, seed_options)
        _assert_same(first, again)
        # Different seed is exercised, but no scientifically invalid promise that every seed differs.
        alternate = core.generate(prompt, seed_options | {"seed": 43})
        report["phases"][phase] = {
            "passed": True,
            "alternate_seed_result": _canonical_result(alternate),
            "same_seed_scope": "same image, device, dtype, attention and sampling options",
        }
        phase = "conditioning_response"
        with torch.inference_mode():
            likelihoods = [
                float(
                    core.model.score(
                        [UBIQUITIN],
                        prompt=family[0],
                        ensemble_size=1,
                        max_tokens=512,
                        scoring_max_tokens=512,
                        use_diversity_weights=False,
                        seed=42,
                    ).scores[0]
                )
                for family in FAMILIES.values()
            ]
        assert all(math.isfinite(value) for value in likelihoods)
        assert abs(likelihoods[0] - likelihoods[1]) > 1e-6, (
            "changing family context did not change candidate likelihood"
        )
        report["phases"][phase] = {
            "passed": True,
            "held_fixed_candidate": "1UBQ_A",
            "mean_log_likelihood_by_context": dict(zip(FAMILIES, likelihoods, strict=True)),
            "interpretation": "Context responsiveness only; these teacher-forced scores differ from generation sampling scores.",
        }
        phase = "live_signed_http"
        with _live_http(core) as (proxy, client):
            received = proxy.generate(prompt, seed_options)
            _assert_same(first, received)
            assert "gpu.mem.used_mib" in received.metadata.runtime
            for payload in (
                {"sequences": "AC-D"},
                {"sequences": prompt, "options": {"sampler": "bad"}},
                {"sequences": prompt, "options": {"max_context_tokens": 8193}},
            ):
                assert client.post("/generate", json=payload).status_code == 422
            # Invalid requests must leave the resident model usable.
            _assert_same(first, proxy.generate(prompt, seed_options))
        report["phases"][phase] = {"passed": True, "invalid_requests": 3, "worker_stopped": True}
        phase = "resident_memory"
        trace: list[list[dict[str, int]]] = []
        for cycle in range(4):
            trace.append([])
            for length in (46, 76, 120):
                # Synthetic composition-controlled varied inputs are for allocation tests only.
                residues = list(("ACDEFGHIKLMNPQRSTVWY" * 6)[:length])
                random.Random(f"{cycle}-{length}").shuffle(residues)
                for sampler in ("single", "ensemble"):
                    sequence = "".join(residues)
                    output = core.generate(
                        [sequence, sequence[:-1] + ("A" if sequence[-1] != "A" else "C")],
                        {
                            "sampler": sampler,
                            "num_prompts_in_ensemble": 2,
                            "seed": 100 + cycle,
                            "max_generated_length": 16,
                            "minimum_sequence_length_proportion": 0.05,
                        },
                    )
                    assert len(output.sequences) == 1
                    del output
                    gc.collect()
                    trace[-1].append(_memory())
                    assert id(core.model) == resident_id
        report["phases"][phase] = {
            "trace_mib": trace,
            "lengths": [46, 76, 120],
            "cycles": 4,
            "samplers": ["single", "ensemble"],
            "growth_limit_mib": 256,
        }
        _plateau(trace)
        report["phases"][phase]["passed"] = True
        assert all(asdict(output) == snapshot for output, snapshot in saved), "later calls mutated returned data"
        first.sequences.append("M")
        first.conditioning_prompts[0].sequences.clear()
        assert again.sequences != first.sequences and again.conditioning_prompts[0].sequences
        report["passed"] = True
    except Exception as error:
        report["failed_phase"] = phase
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
    finally:
        core.close()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        released = resident_ref is None or resident_ref() is None
        report["cleanup"] = {"core_closed": not core.ready and core.model is None, "resident_object_released": released}
        if torch.cuda.is_available():
            after_close_allocated = torch.cuda.memory_allocated()
            report["cleanup"]["allocated_before_load_mib"] = before_load_allocated / 2**20
            report["cleanup"]["allocated_after_close_mib"] = after_close_allocated / 2**20
            report["cleanup"]["after_close_diagnostics"] = _allocation_diagnostics()
            # Torch 2.6 retains allocator-backed cuBLAS workspaces globally per
            # handle/stream (T4 default8.125 MiB each). Main + HTTP threads can
            # leave16.25 MiB even after every model tensor is released. This is
            # a test-only diagnostic of that precise hypothesis, not a relaxed
            # memory gate or a production global-cache purge.
            # https://github.com/pytorch/pytorch/blob/v2.6.0/aten/src/ATen/cuda/CublasHandlePool.cpp
            clear_workspaces = getattr(torch._C, "_cuda_clearCublasWorkspaces", None)
            if not callable(clear_workspaces):
                raise RuntimeError("Pinned torch lacks the cuBLAS workspace diagnostic binding")
            clear_workspaces()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            after_workspace_clear = torch.cuda.memory_allocated()
            report["cleanup"]["test_only_cublas_workspace_bytes_released"] = (
                after_close_allocated - after_workspace_clear
            )
            report["cleanup"]["allocated_after_workspace_clear_mib"] = after_workspace_clear / 2**20
            report["cleanup"]["after_workspace_clear_diagnostics"] = _allocation_diagnostics()
            if report["cleanup"]["after_workspace_clear_diagnostics"]["live_cuda_tensor_count"]:
                report["passed"] = False
                report["cleanup"]["tensor_error"] = "live CUDA tensors remained after model cleanup"
            if after_workspace_clear - before_load_allocated > 16 * 2**20:
                report["passed"] = False
                report["cleanup"]["allocation_error"] = (
                    "more than16 MiB allocations remained after model and test-only workspace cleanup"
                )
        if not released:
            report["passed"] = False
            report["cleanup"]["error"] = "resident model remained referenced after close"
        report["seconds"] = time.monotonic() - started
    if report["passed"]:
        # A fresh load/call/release distinguishes a fixed process-global
        # workspace from growth across model lifetimes. Reuse only verified
        # cached checkpoint bytes, never a retained native model object.
        fresh_ref = None
        try:
            core._load()
            fresh_ref = weakref.ref(core.model)
            output = core.generate(
                UBIQUITIN, {"seed": 42, "max_generated_length": 16, "minimum_sequence_length_proportion": 0.05}
            )
            report["phases"]["fresh_lifetime"] = {"passed": True, **_canonical_result(output)}
            del output
        except Exception as error:
            report["passed"] = False
            report["phases"]["fresh_lifetime"] = {"passed": False, "error": f"{type(error).__name__}: {error}"}
        finally:
            core.close()
            gc.collect()
            torch.cuda.synchronize()
            before_clear = torch.cuda.memory_allocated()
            torch._C._cuda_clearCublasWorkspaces()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            after_clear = torch.cuda.memory_allocated()
            cleanup = {
                "resident_object_released": fresh_ref is None or fresh_ref() is None,
                "allocated_before_workspace_clear_mib": before_clear / 2**20,
                "allocated_after_workspace_clear_mib": after_clear / 2**20,
                "after_workspace_clear_diagnostics": _allocation_diagnostics(),
            }
            report["phases"]["fresh_lifetime"]["cleanup"] = cleanup
            if (
                not cleanup["resident_object_released"]
                or after_clear - before_load_allocated > 16 * 2**20
                or cleanup["after_workspace_clear_diagnostics"]["live_cuda_tensor_count"]
            ):
                report["passed"] = False
                report["phases"]["fresh_lifetime"]["passed"] = False
                cleanup["error"] = "fresh model lifetime failed the unchanged release gate"
    report["seconds"] = time.monotonic() - started
    return report
