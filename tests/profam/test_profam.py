"""Backend-free ProFam configuration, checkpoint, sampling and lifecycle tests."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from boileroom.models.profam.types import ConditioningPrompt, ProFamOutput

PROMPT = ["ACDEFGHIKLMNPQRSTVWY", "ACDEFGHIKLMNPQRSTVWF"]


def _core(config: dict | None = None) -> Any:
    from boileroom.models.profam.core import ProFamCore

    return ProFamCore(config)


@pytest.mark.parametrize(
    "sequences", ["", [], [""], ["ACD", ""], "A:C", "acde", "AC-D", "AC.X", ">fasta\nACD", [None], None, {"ACD": 1}]
)
def test_invalid_prompt_fails_without_loading(sequences: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    core = _core()
    monkeypatch.setattr(core, "_load", lambda: pytest.fail("invalid request loaded model"))
    with pytest.raises(ValueError):
        core.generate(sequences)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("num_samples", 0),
        ("num_samples", True),
        ("num_samples", 1.2),
        ("sampler", "unknown"),
        ("reduction", "invalid"),
        ("temperature", float("nan")),
        ("temperature", -1),
        ("top_p", 1.1),
        ("max_context_tokens", 8193),
        ("max_context_tokens", 0),
        ("max_generated_length", -1),
        ("seed", -1),
        ("seed", 2**32),
        ("seed", True),
        ("maximum_retries", 0),
        ("repeat_guard", 1),
        ("dtype", "half"),
        ("attn_implementation", "magic"),
        ("max_sequence_length_multiplier", float("inf")),
        ("prompt_accessions", ["same", "same"]),
        ("prompt_accessions", "name"),
        ("checkpoint_path", "/tmp/untrusted.ckpt"),
        ("repo_id", "attacker/model"),
        ("revision", "main"),
        ("match_representative_length", True),
        ("include_fields", ["*"]),
        ("bogus", 1),
    ],
)
def test_invalid_config_and_call_options_fail(key: str, value: Any) -> None:
    with pytest.raises(ValueError):
        _core({key: value})
    with pytest.raises(ValueError):
        _core().validate_request(PROMPT, {key: value})


@pytest.mark.parametrize("key", ["device", "cache_dir", "dtype", "attn_implementation", "optimization"])
def test_static_overrides_are_refused(key: str) -> None:
    with pytest.raises(ValueError, match="only be set at initialization"):
        _core().validate_request(PROMPT, {key: "override"})


def test_budget_and_accessions_are_checked() -> None:
    with pytest.raises(ValueError, match="cannot fit"):
        _core().validate_request(PROMPT, {"max_context_tokens": 40})
    with pytest.raises(ValueError, match="number of prompt"):
        _core().validate_request(PROMPT, {"prompt_accessions": ["one"]})
    assert _core().validate_request(tuple(PROMPT))[0] == PROMPT


def _install_runtime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, valid_hash: bool = True
) -> tuple[Any, list[dict], list[dict]]:
    import boileroom.models.profam.core as module

    checkpoint = tmp_path / "last.ckpt"
    checkpoint.write_bytes(b"fake official artifact")
    if valid_hash:
        monkeypatch.setattr(module, "CHECKPOINT_SHA256", hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    downloads: list[dict] = []
    loads: list[dict] = []

    def download(**kwargs: Any) -> str:
        downloads.append(kwargs)
        return str(checkpoint)

    def upstream(**kwargs: Any) -> Any:
        loads.append(kwargs)
        return SimpleNamespace(generate=lambda **kw: None)

    device = SimpleNamespace(type="cpu", __str__=lambda self: "cpu")
    fake_torch = SimpleNamespace(inference_mode=nullcontext, cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "profam", SimpleNamespace(ProFam=upstream))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download))
    core = _core({"cache_dir": str(tmp_path)})
    monkeypatch.setattr(core, "_resolve_device", lambda: device)
    return core, downloads, loads


def test_pinned_verified_checkpoint_loads_once_and_close_releases(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    core, downloads, loads = _install_runtime(monkeypatch, tmp_path)
    core._load()
    core._load()
    assert len(downloads) == len(loads) == 1
    assert downloads[0]["repo_id"] == "judewells/ProFam-1"
    assert downloads[0]["revision"] == "fe4084f356480fd048a23097fcc133f1e5ba019d"
    assert downloads[0]["token"] is False
    assert loads[0]["auto_download"] is False
    assert loads[0]["dtype"] == "float32"
    assert core.ready and core.model is not None
    core.close()
    assert not core.ready and core.model is None
    core._load()
    assert len(loads) == 2


def test_corrupt_checkpoint_never_reaches_pickle_loader(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    core, _, loads = _install_runtime(monkeypatch, tmp_path, valid_hash=False)
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        core._load()
    assert loads == [] and core.model is None and not core.ready


@pytest.mark.parametrize("sampler", ["single", "ensemble"])
def test_generation_maps_api_options_and_returns_independent_cpu_outputs(
    sampler: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    core = _core({"num_samples": 2})
    calls: list[dict] = []
    result = SimpleNamespace(
        sequences=["ACDE", "ACDF"],
        scores=[-0.1, -0.5],
        conditioning_prompts=[SimpleNamespace(sequences=[PROMPT[1]], accessions=["second"])],
    )

    def generate(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return result

    core.model = SimpleNamespace(generate=generate)
    core.ready = True
    core._runtime = {"profam.commit": "pin"}
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))
    options = {"sampler": sampler, "seed": 42, "num_prompts_in_ensemble": 2, "prompt_accessions": ["first", "second"]}
    first = core.generate(PROMPT, options)
    second = core.generate(PROMPT, options)
    assert isinstance(first, ProFamOutput)
    assert calls[0]["prompt"] == PROMPT
    assert calls[0]["max_tokens"] == 4096 and "max_context_tokens" not in calls[0]
    assert calls[0]["sampler"] == sampler and calls[0]["num_prompts_in_ensemble"] == 2
    assert first.sequences == ["ACDE", "ACDF"]
    assert first.log_likelihoods == [-0.1, -0.5]
    assert first.metadata.sequence_lengths == [4, 4]
    assert first.conditioning_prompts == [ConditioningPrompt([PROMPT[1]], ["second"])]
    first.sequences.append("M")
    first.conditioning_prompts[0].sequences.clear()
    assert first.metadata.runtime is not None and second.metadata.runtime is not None
    first.metadata.runtime["profam.commit"] = "mutated"
    assert second.sequences == result.sequences and second.conditioning_prompts[0].sequences == [PROMPT[1]]
    assert second.metadata.runtime["profam.commit"] == "pin"
    assert core.config["sampler"] == "single" and core.config["seed"] is None


@pytest.mark.parametrize(
    ("sequences", "scores"), [([], []), (["ACD"], []), (["A?D"], [-0.2]), (["ACD"], [float("nan")])]
)
def test_invalid_or_incomplete_upstream_results_are_errors(
    sequences: list, scores: list, monkeypatch: pytest.MonkeyPatch
) -> None:
    core = _core({"num_samples": 1})
    core.ready = True
    core.model = SimpleNamespace(
        generate=lambda **kwargs: SimpleNamespace(sequences=sequences, scores=scores, conditioning_prompts=[])
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))
    with pytest.raises(RuntimeError):
        core.generate(PROMPT)


def test_public_types_and_core_do_not_import_model_dependencies() -> None:
    script = """
import sys
from boileroom import ProFam
from boileroom.models.profam.types import ProFamOutput
from boileroom.models.profam.core import ProFamCore
for name in ("torch", "transformers", "profam", "huggingface_hub", "lightning"):
    assert name not in sys.modules, name
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)


def test_wrapper_rejects_invalid_request_without_backend_call(monkeypatch: pytest.MonkeyPatch) -> None:
    from boileroom import ProFam
    from boileroom.base import ModelWrapper

    started = []
    monkeypatch.setattr(ModelWrapper, "_initialize_backend_from_spec", lambda *args, **kwargs: started.append(True))
    with pytest.raises(ValueError, match="does not accept"):
        ProFam(config={"checkpoint_path": "/tmp/custom.ckpt"})
    assert not started
    wrapper = ProFam()
    monkeypatch.setattr(
        wrapper, "_call_backend_method", lambda *args, **kwargs: pytest.fail("invalid request dispatched")
    )
    with pytest.raises(ValueError):
        wrapper.generate("AC-D")
    with pytest.raises(ValueError):
        wrapper.generate(PROMPT, {"max_context_tokens": 8193})


def test_canonical_image_pins_upstream_without_baking_weights_or_secrets() -> None:
    from boileroom.images.metadata import get_model_image_spec, get_supported_cuda, get_supported_platforms
    from boileroom.models.profam.core import UPSTREAM_COMMIT

    spec = get_model_image_spec("profam")
    assert get_supported_cuda(spec) == ("12.6",)
    assert get_supported_platforms(spec) == ("linux/amd64",)
    dockerfile = spec.dockerfile_path.read_text()
    requirements = (spec.context_path / "requirements.txt").read_text()
    assert UPSTREAM_COMMIT in dockerfile and UPSTREAM_COMMIT in requirements
    assert f"git+https://github.com/alex-hh/profam.git@{UPSTREAM_COMMIT}" in requirements
    assert "ARG HF_TOKEN" not in dockerfile and "snapshot_download" not in dockerfile
    assert ":latest" not in dockerfile
