"""Contract/fake-SDK tests for ESM-C and ESM3 embedding support."""

from __future__ import annotations

import ast
import sys
import types
from dataclasses import fields
from pathlib import Path
from typing import Any, ClassVar, cast

import numpy as np
import pytest


def test_esm3_types_are_lightweight() -> None:
    source = Path("boileroom/models/esm3/types.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    assert not ({"torch", "esm", "transformers", "modal", "biotite"} & imports)

    from boileroom.models.esm3.types import ESM3Output, ESMCOutput, ESMEmbeddingOutput

    assert ESMCOutput is ESMEmbeddingOutput
    assert ESM3Output is ESMEmbeddingOutput
    assert {field.name for field in fields(ESMEmbeddingOutput)} >= {
        "metadata",
        "embeddings",
        "chain_index",
        "residue_index",
        "hidden_states",
        "lm_logits",
    }


def test_parse_sequences_preserves_residue_and_chain_indices() -> None:
    from boileroom.models.esm3.core import parse_esm3_sequences

    parsed = parse_esm3_sequences("ACD:EF")

    assert len(parsed) == 1
    assert parsed[0].original == "ACD:EF"
    assert parsed[0].sdk_sequence == "ACD|EF"
    assert parsed[0].residue_count == 5
    assert parsed[0].chain_index.tolist() == [0, 0, 0, 1, 1]
    assert parsed[0].residue_index.tolist() == [0, 1, 2, 0, 1]


def test_parse_sequences_rejects_empty_batches() -> None:
    from boileroom.models.esm3.core import ESMCCore, parse_esm3_sequences

    with pytest.raises(ValueError, match="at least one sequence"):
        parse_esm3_sequences([])

    with pytest.raises(ValueError, match="at least one sequence"):
        ESMCCore(config={"device": "cpu"}).embed([])


def test_pad_residue_arrays_zero_and_minus_one_padding() -> None:
    from boileroom.models.esm3.core import pad_residue_arrays

    embeddings, hidden_states, lm_logits, chain_index, residue_index = pad_residue_arrays(
        embeddings=[np.ones((5, 2), dtype=np.float32), np.full((2, 2), 2, dtype=np.float32)],
        chain_index=[np.array([0, 0, 0, 1, 1]), np.array([0, 0])],
        residue_index=[np.array([0, 1, 2, 0, 1]), np.array([0, 1])],
        hidden_states=None,
        lm_logits=None,
    )

    assert embeddings.shape == (2, 5, 2)
    assert embeddings[1, 2:].tolist() == [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    assert chain_index.tolist() == [[0, 0, 0, 1, 1], [0, 0, -1, -1, -1]]
    assert residue_index.tolist() == [[0, 1, 2, 0, 1], [0, 1, -1, -1, -1]]
    assert hidden_states is None
    assert lm_logits is None


def test_pad_residue_arrays_rejects_empty_batches() -> None:
    from boileroom.models.esm3.core import pad_residue_arrays

    with pytest.raises(ValueError, match="empty batch"):
        pad_residue_arrays(embeddings=[], chain_index=[], residue_index=[])


class _FakeProtein:
    def __init__(self, sequence: str) -> None:
        self.sequence = sequence


class _FakeLogitsConfig:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _FakeEncoded:
    def __init__(self, sequence: str) -> None:
        self.sequence = sequence


class _FakeForwardTrackData:
    def __init__(self, sequence: Any | None) -> None:
        self.sequence = sequence


class _FakeLogitsOutput:
    def __init__(self, sequence: str, include_hidden: bool, include_logits: bool) -> None:
        torch = pytest.importorskip("torch")
        token_count = len(sequence) + 2
        values = torch.arange(token_count * 3, dtype=torch.float32).reshape(token_count, 3)
        self.embeddings = values
        self.hidden_states = torch.stack([values + 100, values + 200])[:, None, :, :] if include_hidden else None
        sequence_logits = (
            torch.arange(token_count * 4, dtype=torch.float32).reshape(token_count, 4) if include_logits else None
        )
        self.logits = _FakeForwardTrackData(sequence_logits)


class _FakeSDKModel:
    requested_model_names: ClassVar[list[str]] = []
    logits_configs: ClassVar[list[dict[str, Any]]] = []
    encoded_sequences: ClassVar[list[str]] = []

    @classmethod
    def from_pretrained(cls, model_name: str) -> _FakeSDKModel:
        cls.requested_model_names.append(model_name)
        return cls()

    def to(self, device: str) -> _FakeSDKModel:
        self.device = device
        return self

    def eval(self) -> _FakeSDKModel:
        return self

    def encode(self, protein: _FakeProtein) -> _FakeEncoded:
        self.encoded_sequences.append(protein.sequence)
        return _FakeEncoded(protein.sequence)

    def logits(self, encoded: _FakeEncoded, config: _FakeLogitsConfig) -> _FakeLogitsOutput:
        self.logits_configs.append(config.kwargs)
        return _FakeLogitsOutput(
            encoded.sequence,
            include_hidden=bool(config.kwargs.get("return_hidden_states")),
            include_logits=bool(config.kwargs.get("sequence")),
        )


@pytest.fixture()
def fake_esm_sdk(monkeypatch: pytest.MonkeyPatch) -> type[_FakeSDKModel]:
    _FakeSDKModel.requested_model_names = []
    _FakeSDKModel.logits_configs = []
    _FakeSDKModel.encoded_sequences = []

    esm = types.ModuleType("esm")
    cast(Any, esm).__version__ = "fake-version"
    esm_models = types.ModuleType("esm.models")
    esmc = types.ModuleType("esm.models.esmc")
    esm3 = types.ModuleType("esm.models.esm3")
    sdk = types.ModuleType("esm.sdk")
    api = types.ModuleType("esm.sdk.api")
    cast(Any, esmc).ESMC = _FakeSDKModel
    cast(Any, esm3).ESM3 = _FakeSDKModel
    cast(Any, api).ESMProtein = _FakeProtein
    cast(Any, api).LogitsConfig = _FakeLogitsConfig
    monkeypatch.setitem(sys.modules, "esm", esm)
    monkeypatch.setitem(sys.modules, "esm.models", esm_models)
    monkeypatch.setitem(sys.modules, "esm.models.esmc", esmc)
    monkeypatch.setitem(sys.modules, "esm.models.esm3", esm3)
    monkeypatch.setitem(sys.modules, "esm.sdk", sdk)
    monkeypatch.setitem(sys.modules, "esm.sdk.api", api)
    return _FakeSDKModel


def test_esmc_core_uses_sdk_and_strips_special_chain_break_tokens(fake_esm_sdk: type[_FakeSDKModel]) -> None:
    from boileroom.models.esm3.core import ESMCCore

    core = ESMCCore(config={"device": "cpu", "model_name": "esmc_300m"})
    result = core.embed(["ACD:EF", "GH", "I"], options={"include_fields": ["hidden_states", "lm_logits"]})

    assert fake_esm_sdk.requested_model_names == ["esmc_300m"]
    assert fake_esm_sdk.encoded_sequences == ["ACD|EF", "GH", "I"]
    assert result.metadata.model_name == "ESM-C"
    assert result.metadata.model_version == "fake-version"
    assert result.metadata.sequence_lengths == [5, 2, 1]
    assert result.embeddings.shape == (3, 5, 3)
    # Fake output rows are [BOS, A, C, D, |, E, F, EOS]; chain break row (index 4) is stripped.
    assert result.embeddings[0, :, 0].tolist() == [3.0, 6.0, 9.0, 15.0, 18.0]
    assert result.embeddings[1, 2:].tolist() == [[0.0, 0.0, 0.0]] * 3
    assert result.chain_index.tolist() == [[0, 0, 0, 1, 1], [0, 0, -1, -1, -1], [0, -1, -1, -1, -1]]
    assert result.residue_index.tolist() == [[0, 1, 2, 0, 1], [0, 1, -1, -1, -1], [0, -1, -1, -1, -1]]
    assert result.hidden_states is not None and result.hidden_states.shape == (2, 3, 5, 3)
    assert result.lm_logits is not None and result.lm_logits.shape == (3, 5, 4)


def test_esmc_static_config_and_invalid_model_validation(fake_esm_sdk: type[_FakeSDKModel]) -> None:
    from boileroom.models.esm3.core import ESMCCore

    with pytest.raises(ValueError, match="Unsupported ESM-C model"):
        ESMCCore(config={"model_name": "nope"})

    core = ESMCCore(config={"device": "cpu"})
    with pytest.raises(ValueError, match="model_name"):
        core.embed("ACD", options={"model_name": "esmc_600m"})


def test_esm3_core_hidden_states_are_rejected(fake_esm_sdk: type[_FakeSDKModel]) -> None:
    from boileroom.models.esm3.core import ESM3Core

    with pytest.raises(ValueError, match=r"hidden_states.*ESM3"):
        ESM3Core(config={"device": "cpu"}).embed("ACD", options={"include_fields": ["hidden_states"]})


def test_esm3_wildcard_requests_supported_logits_only(fake_esm_sdk: type[_FakeSDKModel]) -> None:
    from boileroom.models.esm3.core import ESM3Core

    result = ESM3Core(config={"device": "cpu"}).embed("ACD", options={"include_fields": ["*"]})

    assert result.lm_logits is not None and result.lm_logits.shape == (1, 3, 4)
    assert result.hidden_states is None


def test_esm3_alias_model_names_are_valid(fake_esm_sdk: type[_FakeSDKModel]) -> None:
    from boileroom.models.esm3.core import ESM3Core

    core = ESM3Core(config={"device": "cpu", "model_name": "esm3-sm-open-v1"})
    result = core.embed("ACD")

    assert fake_esm_sdk.requested_model_names == ["esm3_sm_open_v1"]
    assert result.embeddings.shape == (1, 3, 3)
    assert result.lm_logits is None


def test_esm3_package_types_import_has_no_modal_wrapper_side_effects(monkeypatch: pytest.MonkeyPatch) -> None:
    for module_name in [
        "boileroom.models.esm3",
        "boileroom.models.esm3.types",
        "boileroom.models.esm3.esmc",
        "boileroom.models.esm3.esm3",
        "modal",
    ]:
        monkeypatch.delitem(sys.modules, module_name, raising=False)

    import boileroom.models.esm3.types  # noqa: F401, PLC0415

    assert "boileroom.models.esm3.esmc" not in sys.modules
    assert "boileroom.models.esm3.esm3" not in sys.modules
    assert "modal" not in sys.modules
