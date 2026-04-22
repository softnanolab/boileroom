from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from boileroom.models.esm.linker import compute_position_ids, store_multimer_properties
from boileroom.models.esm.parsing import parse_esm2_sequence


class _FakeBatch(dict):
    """Minimal BatchEncoding-like mapping for core tests."""

    def to(self, _device: object) -> _FakeBatch:
        return self


class FakeTokenizer:
    """Tokenizer stub that counts ESM2 residue tokens including <mask>."""

    def __call__(self, sequences: str | list[str], **kwargs: Any) -> _FakeBatch:
        sequence_list = [sequences] if isinstance(sequences, str) else sequences
        add_special_tokens = kwargs.get("add_special_tokens", True)
        lengths = [parse_esm2_sequence(sequence).residue_count + (2 if add_special_tokens else 0) for sequence in sequence_list]
        max_length = max(lengths)

        input_ids = torch.zeros((len(sequence_list), max_length), dtype=torch.int64)
        attention_mask = torch.zeros((len(sequence_list), max_length), dtype=torch.int64)
        for index, length in enumerate(lengths):
            input_ids[index, :length] = torch.arange(length)
            attention_mask[index, :length] = 1

        return _FakeBatch({"input_ids": input_ids, "attention_mask": attention_mask})


class FakeBaseModel:
    """Backbone stub returning deterministic hidden states."""

    def __init__(self, hidden_dim: int = 5, hidden_state_count: int = 3) -> None:
        self.hidden_dim = hidden_dim
        self.hidden_state_count = hidden_state_count
        self.calls: list[dict[str, Any]] = []

    def to(self, _device: object) -> FakeBaseModel:
        return self

    def eval(self) -> FakeBaseModel:
        return self

    def __call__(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        input_ids = kwargs["input_ids"]
        assert isinstance(input_ids, torch.Tensor)
        batch_size, token_count = input_ids.shape
        last_hidden_state = torch.arange(
            batch_size * token_count * self.hidden_dim,
            dtype=torch.float32,
        ).reshape(batch_size, token_count, self.hidden_dim)

        hidden_states = None
        if kwargs.get("output_hidden_states"):
            hidden_states = tuple(last_hidden_state + layer for layer in range(self.hidden_state_count - 1)) + (
                last_hidden_state,
            )

        return SimpleNamespace(
            last_hidden_state=last_hidden_state,
            hidden_states=hidden_states,
        )


class FakeLMHead:
    """LM head stub that expands hidden states to a fixed vocabulary size."""

    def __init__(self, vocab_size: int = 33) -> None:
        self.vocab_size = vocab_size
        self.calls = 0

    def __call__(self, last_hidden_state: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        vocab_offsets = torch.arange(self.vocab_size, dtype=last_hidden_state.dtype, device=last_hidden_state.device)
        return last_hidden_state[..., :1] + vocab_offsets.view(1, 1, self.vocab_size)


class FakeMaskedLMModel:
    """MLM wrapper stub exposing `.esm` and `.lm_head` like Hugging Face."""

    def __init__(self, hidden_dim: int = 5, hidden_state_count: int = 3, vocab_size: int = 33) -> None:
        self.esm = FakeBaseModel(hidden_dim=hidden_dim, hidden_state_count=hidden_state_count)
        self.lm_head = FakeLMHead(vocab_size=vocab_size)

    def to(self, _device: object) -> FakeMaskedLMModel:
        self.esm.to(_device)
        return self

    def eval(self) -> FakeMaskedLMModel:
        self.esm.eval()
        return self


def _install_fake_loaders(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, list[FakeBaseModel], list[FakeMaskedLMModel]]:
    pytest.importorskip("transformers", reason="requires transformers")
    from boileroom.models.esm import core as esm_core_module

    base_models: list[FakeBaseModel] = []
    masked_lm_models: list[FakeMaskedLMModel] = []

    def load_tokenizer(*_args: Any, **_kwargs: Any) -> FakeTokenizer:
        return FakeTokenizer()

    def load_base_model(*_args: Any, **_kwargs: Any) -> FakeBaseModel:
        model = FakeBaseModel()
        base_models.append(model)
        return model

    def load_masked_lm_model(*_args: Any, **_kwargs: Any) -> FakeMaskedLMModel:
        model = FakeMaskedLMModel()
        masked_lm_models.append(model)
        return model

    monkeypatch.setattr(esm_core_module, "AutoTokenizer", SimpleNamespace(from_pretrained=load_tokenizer))
    monkeypatch.setattr(esm_core_module, "EsmModel", SimpleNamespace(from_pretrained=load_base_model))
    monkeypatch.setattr(esm_core_module, "EsmForMaskedLM", SimpleNamespace(from_pretrained=load_masked_lm_model))
    return esm_core_module, base_models, masked_lm_models


def _make_core(monkeypatch: pytest.MonkeyPatch):
    esm_core_module, base_models, masked_lm_models = _install_fake_loaders(monkeypatch)
    core = esm_core_module.ESM2Core(config={"device": "cpu", "model_name": "esm2_t6_8M_UR50D"})
    return core, base_models, masked_lm_models


def test_parse_esm2_sequence_rejects_invalid_fragment() -> None:
    """Invalid ESM2 token fragments should raise a clear parsing error."""
    with pytest.raises(ValueError, match=r"Invalid ESM2 token fragment '<foo>'"):
        parse_esm2_sequence("AC<foo>D")


def test_esm2_default_embed_does_not_return_lm_logits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default ESM2 requests should stay on the base-model path."""
    core, base_models, masked_lm_models = _make_core(monkeypatch)

    result = core.embed("ACDE")

    assert result.embeddings.shape == (1, 4, 5)
    assert result.hidden_states is None
    assert result.lm_logits is None
    assert len(base_models) == 1
    assert len(masked_lm_models) == 0
    assert core._model_mode == "base"
    assert base_models[0].calls[0]["output_hidden_states"] is False


def test_esm2_embed_with_lm_logits_returns_embeddings_and_logits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Requesting lm_logits should return full-vocabulary logits and embeddings together."""
    core, base_models, masked_lm_models = _make_core(monkeypatch)

    result = core.embed("ACDE", options={"include_fields": ["lm_logits"]})

    assert result.embeddings.shape == (1, 4, 5)
    assert result.hidden_states is None
    assert result.lm_logits is not None
    assert result.lm_logits.shape == (1, 4, 33)
    assert len(base_models) == 0
    assert len(masked_lm_models) == 1
    assert core._model_mode == "masked_lm"


def test_esm2_embed_with_hidden_states_and_lm_logits_returns_both(monkeypatch: pytest.MonkeyPatch) -> None:
    """Requesting hidden_states and lm_logits should return both optional outputs."""
    core, _, masked_lm_models = _make_core(monkeypatch)

    result = core.embed("ACDE", options={"include_fields": ["hidden_states", "lm_logits"]})

    assert result.hidden_states is not None
    assert result.hidden_states.shape == (1, 3, 4, 5)
    assert result.lm_logits is not None
    assert result.lm_logits.shape == (1, 4, 33)
    assert masked_lm_models[0].lm_head.calls == 1


def test_esm2_embed_with_all_optional_fields_returns_hidden_states_and_logits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wildcard include_fields should return all optional ESM2 outputs."""
    core, _, masked_lm_models = _make_core(monkeypatch)

    result = core.embed("ACDE", options={"include_fields": ["*"]})

    assert result.hidden_states is not None
    assert result.hidden_states.shape == (1, 3, 4, 5)
    assert result.lm_logits is not None
    assert result.lm_logits.shape == (1, 4, 33)
    assert masked_lm_models[0].lm_head.calls == 1


def test_esm2_upgrades_to_masked_lm_once_and_reuses_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """A later logits request should upgrade once and keep the MLM model resident."""
    core, base_models, masked_lm_models = _make_core(monkeypatch)

    default_result = core.embed("ACDE")
    logits_result = core.embed("ACDE", options={"include_fields": ["lm_logits"]})
    hidden_only_result = core.embed("ACDE", options={"include_fields": ["hidden_states"]})

    assert default_result.lm_logits is None
    assert logits_result.lm_logits is not None
    assert hidden_only_result.hidden_states is not None
    assert hidden_only_result.lm_logits is None
    assert len(base_models) == 1
    assert len(masked_lm_models) == 1
    assert masked_lm_models[0].lm_head.calls == 1
    assert core.model is masked_lm_models[0]
    assert core._model_mode == "masked_lm"


def test_esm2_invalid_logits_request_does_not_upgrade_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid requests should fail before mutating the resident model mode."""
    core, base_models, masked_lm_models = _make_core(monkeypatch)

    with pytest.raises(ValueError, match="Invalid ESM2 token fragment"):
        core.embed("AC<foo>D", options={"include_fields": ["lm_logits"]})

    assert len(base_models) == 0
    assert len(masked_lm_models) == 0
    assert core.model is None
    assert core._model_mode is None

    result = core.embed("ACDE")
    assert result.lm_logits is None
    assert len(base_models) == 1
    assert len(masked_lm_models) == 0
    assert core._model_mode == "base"


def test_esm2_masked_monomer_outputs_align_to_residue_axis(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inline <mask> should count as one residue in monomer outputs."""
    core, _, _ = _make_core(monkeypatch)

    result = core.embed("AC<mask>D", options={"include_fields": ["lm_logits"]})

    assert result.metadata.sequence_lengths == [4]
    assert result.embeddings.shape == (1, 4, 5)
    assert result.lm_logits is not None
    assert result.lm_logits.shape == (1, 4, 33)
    assert np.array_equal(result.lm_logits[0, :, 0], np.array([5, 10, 15, 20], dtype=np.float32))


def test_esm2_masked_multimer_outputs_align_to_residue_axis(monkeypatch: pytest.MonkeyPatch) -> None:
    """Multimer bookkeeping should treat <mask> as one residue and remove internal linkers."""
    core, _, _ = _make_core(monkeypatch)

    result = core.embed(
        "A<mask>:CD",
        options={"include_fields": ["lm_logits"], "glycine_linker": "GG", "position_ids_skip": 512},
    )

    assert result.metadata.sequence_lengths == [4]
    assert result.embeddings.shape == (1, 4, 5)
    assert result.lm_logits is not None
    assert result.lm_logits.shape == (1, 4, 33)
    assert result.chain_index.dtype.kind in {"i", "u"}
    assert result.residue_index.dtype.kind in {"i", "u"}
    assert np.array_equal(result.chain_index[0], np.array([0, 0, 1, 1]))
    assert np.array_equal(result.residue_index[0], np.array([0, 1, 0, 1]))
    assert np.array_equal(result.lm_logits[0, :, 0], np.array([5, 10, 25, 30], dtype=np.float32))


def test_linker_helpers_treat_raw_mask_tokens_as_single_residues() -> None:
    """Shared linker helpers should count inline <mask> as one residue even for raw strings."""
    linker_map, residue_index, chain_index = store_multimer_properties(["A<mask>:CD"], "GG")
    position_ids = compute_position_ids(["A<mask>:CD"], "GG", 512, add_special_tokens=True)

    assert linker_map.shape == (1, 6)
    assert residue_index.shape == (1, 6)
    assert chain_index.shape == (1, 6)
    assert position_ids.shape == (1, 8)
    assert torch.equal(linker_map[0], torch.tensor([1, 1, 0, 0, 1, 1]))
    assert torch.equal(chain_index[0], torch.tensor([0, 0, 0, 0, 1, 1]))
    assert torch.equal(residue_index[0], torch.tensor([0, 1, 3, 4, 0, 1]))


def test_esm2_mask_linker_region_pads_hidden_states_and_logits_on_sequence_axis() -> None:
    """Linker masking should pad logits on the residue axis like embeddings and hidden states."""
    pytest.importorskip("transformers", reason="requires transformers")
    from boileroom.models.esm.core import ESM2Core

    core = ESM2Core(config={"device": "cpu", "model_name": "esm2_t6_8M_UR50D"})
    embeddings = np.arange(2 * 5 * 3, dtype=np.float32).reshape(2, 5, 3)
    hidden_states = np.arange(2 * 2 * 5 * 3, dtype=np.float32).reshape(2, 2, 5, 3)
    lm_logits = np.arange(2 * 5 * 7, dtype=np.float32).reshape(2, 5, 7)
    linker_map = torch.tensor([[1, 1, 1, -1, -1], [1, 1, 1, 1, -1]])
    residue_index = torch.tensor([[0, 1, 2, -1, -1], [0, 1, 2, 3, -1]])
    chain_index = torch.tensor([[0, 0, 1, -1, -1], [0, 0, 1, 1, -1]])

    (
        filtered_embeddings,
        filtered_hidden_states,
        filtered_lm_logits,
        filtered_chain_index,
        filtered_residue_index,
    ) = core._mask_linker_region(
        embeddings,
        hidden_states,
        lm_logits,
        linker_map,
        residue_index,
        chain_index,
    )

    assert filtered_embeddings.shape == (2, 4, 3)
    assert filtered_hidden_states is not None
    assert filtered_hidden_states.shape == (2, 2, 4, 3)
    assert filtered_lm_logits is not None
    assert filtered_lm_logits.shape == (2, 4, 7)
    assert np.all(filtered_embeddings[0, 3] == 0)
    assert np.all(filtered_hidden_states[0, :, 3] == 0)
    assert np.all(filtered_lm_logits[0, 3] == 0)
    assert filtered_chain_index.shape == (2, 4)
    assert filtered_residue_index.shape == (2, 4)
    assert filtered_chain_index[0, 3] == -1
    assert filtered_residue_index[0, 3] == -1
