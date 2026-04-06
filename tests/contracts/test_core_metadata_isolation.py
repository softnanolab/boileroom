"""Focused regressions for per-call metadata isolation in core implementations."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from boileroom.models.chai.types import Chai1Output
from boileroom.models.esm.types import ESM2Output, ESMFoldOutput

pytestmark = pytest.mark.contract


class _FakeBatch(dict):
    """Minimal BatchEncoding-like mapping for core tests."""

    def to(self, _device: object) -> _FakeBatch:
        return self


def test_chai1_fold_uses_fresh_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each Chai fold call should return independent metadata."""
    pytest.importorskip("chai_lab", reason="requires chai backend dependencies")
    torch = pytest.importorskip("torch", reason="requires torch")
    from boileroom.models.chai import core as chai_core_module
    from boileroom.models.chai.core import Chai1Core

    core = Chai1Core(config={"device": "cpu"})
    core._device = torch.device("cpu")

    monkeypatch.setattr(core, "_write_fasta", lambda sequences, buffer_path: buffer_path / "input.fasta")
    monkeypatch.setattr(core, "_write_constraint", lambda buffer_path, config: None)
    monkeypatch.setattr(chai_core_module, "run_inference", lambda **kwargs: object())

    def fake_convert_outputs(
        candidate: object,
        metadata,
        preprocessing_time: float,
        inference_time: float,
        postprocessing_time: float,
        effective_config: dict,
    ) -> Chai1Output:
        metadata.preprocessing_time = preprocessing_time
        metadata.inference_time = inference_time
        metadata.postprocessing_time = postprocessing_time
        return Chai1Output(metadata=metadata)

    monkeypatch.setattr(core, "_convert_outputs", fake_convert_outputs)

    out1 = core.fold("ACDE")
    out2 = core.fold("ACDEFG")

    assert out1.metadata is not out2.metadata
    assert out1.metadata.sequence_lengths == [4]
    assert out2.metadata.sequence_lengths == [6]


def test_esm2_embed_uses_fresh_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each ESM2 embed call should return independent metadata."""
    pytest.importorskip("transformers", reason="requires transformers")
    torch = pytest.importorskip("torch", reason="requires torch")
    from boileroom.models.esm.core import ESM2Core

    core = ESM2Core(config={"device": "cpu", "model_name": "esm2_t6_8M_UR50D"})
    core._device = torch.device("cpu")

    def fake_tokenizer(sequences: list[str], **kwargs: object) -> _FakeBatch:
        max_length = max(len(sequence) for sequence in sequences) + 2
        return _FakeBatch({"input_ids": torch.zeros((len(sequences), max_length), dtype=torch.int64)})

    class FakeModel:
        def __call__(self, **kwargs: object) -> SimpleNamespace:
            input_ids = kwargs["input_ids"]
            assert isinstance(input_ids, torch.Tensor)
            batch_size, token_count = input_ids.shape
            return SimpleNamespace(
                last_hidden_state=torch.ones((batch_size, token_count, 3), dtype=torch.float32),
                hidden_states=None,
            )

    core.tokenizer = cast(Any, fake_tokenizer)
    core.model = cast(Any, FakeModel())

    out1 = core.embed("ACDE")
    out2 = core.embed("ACDEFG")

    assert isinstance(out1, ESM2Output)
    assert isinstance(out2, ESM2Output)
    assert out1.metadata is not out2.metadata
    assert out1.metadata.sequence_lengths == [4]
    assert out2.metadata.sequence_lengths == [6]


def test_esmfold_fold_uses_fresh_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each ESMFold fold call should return independent metadata."""
    pytest.importorskip("transformers", reason="requires transformers")
    torch = pytest.importorskip("torch", reason="requires torch")
    from boileroom.models.esm.core import ESMFoldCore

    core = ESMFoldCore(config={"device": "cpu"})
    core._device = torch.device("cpu")
    core.tokenizer = cast(Any, object())

    class FakeModel:
        def __call__(self, **kwargs: object) -> dict:
            return {}

    core.model = cast(Any, FakeModel())
    monkeypatch.setattr(core, "_tokenize_sequences", lambda sequences, config: ({}, None))

    def fake_convert_outputs(
        outputs: dict,
        multimer_properties: dict | None,
        metadata,
        preprocessing_time: float,
        inference_time: float,
        postprocessing_time: float,
        config: dict,
    ) -> ESMFoldOutput:
        metadata.preprocessing_time = preprocessing_time
        metadata.inference_time = inference_time
        metadata.postprocessing_time = postprocessing_time
        return ESMFoldOutput(metadata=metadata)

    monkeypatch.setattr(core, "_convert_outputs", fake_convert_outputs)

    out1 = core.fold("ACDE")
    out2 = core.fold("ACDEFG")

    assert out1.metadata is not out2.metadata
    assert out1.metadata.sequence_lengths == [4]
    assert out2.metadata.sequence_lengths == [6]
