"""Bounded real public-backend checks, separate from stock-API scientific comparisons."""

from __future__ import annotations

import copy
import math
import time
from dataclasses import asdict
from typing import Any

PROMPT = [
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGA",
]


def run_public_validation(wrapper_class: type, *, backend: str, device: str | None) -> dict[str, Any]:
    """Exercise the actual public wrapper and backend on a single live instance.

    These short completions test transport, seeded per-call settings and lifecycle;
    upstream parity and full-length sequence checks live in profam_runtime_validation.py.
    The launcher supplies the canonical source image/SIF without changing production
    registry defaults and independently confirms remote app/process termination.
    """
    started = time.monotonic()
    report: dict[str, Any] = {
        "passed": False,
        "backend": backend,
        "device": device,
        "scope": "Real public wrapper/backend transport and lifecycle; short completions are not biological-quality evidence.",
    }
    model = None
    try:
        with wrapper_class(
            backend=backend,
            device=device,
            config={
                "num_samples": 2,
                "max_context_tokens": 512,
                "max_generated_length": 16,
                "minimum_sequence_length_proportion": 0.05,
            },
        ) as model:
            options: dict[str, Any] = {
                "seed": 42,
                "prompt_accessions": ["1UBQ_A", "1UBQ_G76A_synthetic"],
                "num_prompts_in_ensemble": 2,
            }
            first = model.generate(PROMPT, options)
            snapshot = copy.deepcopy(asdict(first))
            second = model.generate(PROMPT, options | {"sampler": "ensemble"})
            for result in (first, second):
                assert len(result.sequences) == len(result.log_likelihoods) == 2
                assert all(
                    type(sequence) is str and 3 <= len(sequence) <= 16 and set(sequence) <= set("ACDEFGHIKLMNPQRSTVWY")
                    for sequence in result.sequences
                )
                assert all(
                    type(value) is float and math.isfinite(value) and value <= 1e-6 for value in result.log_likelihoods
                )
                pairs = dict(zip(options["prompt_accessions"], PROMPT, strict=True))
                assert result.conditioning_prompts
                for selected in result.conditioning_prompts:
                    assert all(
                        pairs[accession] == sequence
                        for accession, sequence in zip(selected.accessions, selected.sequences, strict=True)
                    )
                assert (
                    result.metadata.runtime["profam.checkpoint_sha256"]
                    == "8ec54d80323f188d3454fb4457f63f12158fe55552b919554806b12730c30e08"
                )
                assert result.metadata.runtime["profam.commit"] == "a19061606dc6469424035db3fdcaae6e63688017"
                if backend == "modal":
                    assert "gpu.mem.used_mib" in result.metadata.runtime
            refused = 0
            for sequences, overrides in (
                ("AC-D", {}),
                (PROMPT, {"sampler": "bad"}),
                (PROMPT, {"max_context_tokens": 8193}),
                (PROMPT, {"checkpoint_path": "/tmp/untrusted"}),
            ):
                try:
                    model.generate(sequences, overrides)
                except ValueError:
                    refused += 1
                else:
                    raise AssertionError("invalid public request was not refused")
            repeated = model.generate(PROMPT, options)
            assert repeated.sequences == first.sequences
            assert repeated.conditioning_prompts == first.conditioning_prompts
            assert all(
                math.isclose(left, right, abs_tol=1e-5, rel_tol=1e-5)
                for left, right in zip(first.log_likelihoods, repeated.log_likelihoods, strict=True)
            )
            assert asdict(first) == snapshot, "later calls mutated returned output"
            first.sequences.append("M")
            first.conditioning_prompts[0].sequences.clear()
            assert len(repeated.sequences) == 2 and repeated.conditioning_prompts[0].sequences
            report["results"] = [snapshot, asdict(second), asdict(repeated)]
            report["reported_image_reference"] = repeated.metadata.runtime.get("image_ref", "unknown")
            report["invalid_inputs_refused"] = refused
            report["passed"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        if model is not None:
            worker = model._backend
            report["client_backend_stopped"] = (
                getattr(worker, "model", None) is None and getattr(worker, "_process", None) is None
            )
            if not report["client_backend_stopped"]:
                report["passed"] = False
        report["seconds"] = time.monotonic() - started
    return report


def run_public_modal_validation(wrapper_class: type) -> dict[str, Any]:
    """Run the public Modal backend with the launcher's bounded source-image class."""
    return run_public_validation(wrapper_class, backend="modal", device=None)
