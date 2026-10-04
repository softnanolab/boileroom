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
    """Residue arrays pad with zeros; chain/residue indices pad with -1."""
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


def test_to_numpy_upcasts_bfloat16_tensors() -> None:
    # The Biohub ESM-C/ESM3 SDK returns bfloat16 tensors on CUDA, which NumPy
    # cannot convert directly; the core must upcast them to float32 first.
    torch = pytest.importorskip("torch")
    from boileroom.models.esm3.core import _BaseESM3EmbeddingCore

    array = _BaseESM3EmbeddingCore._to_numpy(torch.ones(2, 3, dtype=torch.bfloat16))

    assert array.dtype == np.float32
    assert array.tolist() == [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]


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
    def __init__(
        self,
        sequence: Any | None,
        sasa: Any | None = None,
        secondary_structure: Any | None = None,
        function: Any | None = None,
    ) -> None:
        """Store fake per-track logits for a ForwardTrackData stand-in."""
        self.sequence = sequence
        self.sasa = sasa
        self.secondary_structure = secondary_structure
        self.function = function


class _FakeLogitsOutput:
    # Per-track fake vocab sizes, so tests can assert distinct output shapes.
    _TRACK_VOCAB: ClassVar[dict[str, int]] = {
        "sasa": 5,
        "secondary_structure": 6,
        "function": 7,
        "residue_annotations": 8,
    }

    def __init__(
        self, sequence: str, include_hidden: bool, include_logits: bool, tracks: set[str] | None = None
    ) -> None:
        """Build fake embeddings, hidden states, and logits for the requested tracks."""
        torch = pytest.importorskip("torch")
        tracks = tracks or set()
        token_count = len(sequence) + 2
        values = torch.arange(token_count * 3, dtype=torch.float32).reshape(token_count, 3)
        self.embeddings = values
        self.hidden_states = torch.stack([values + 100, values + 200])[:, None, :, :] if include_hidden else None
        sequence_logits = (
            torch.arange(token_count * 4, dtype=torch.float32).reshape(token_count, 4) if include_logits else None
        )

        def _track(name: str) -> Any | None:
            """Return fake logits for track ``name`` when requested, else ``None``."""
            if name not in tracks:
                return None
            vocab = self._TRACK_VOCAB[name]
            return torch.arange(token_count * vocab, dtype=torch.float32).reshape(token_count, vocab)

        self.logits = _FakeForwardTrackData(
            sequence_logits,
            sasa=_track("sasa"),
            secondary_structure=_track("secondary_structure"),
            function=_track("function"),
        )
        # Residue-annotation logits live at the top level of LogitsOutput.
        self.residue_annotation_logits = _track("residue_annotations")


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
        """Record the requested LogitsConfig and return fake logits for its tracks."""
        self.logits_configs.append(config.kwargs)
        tracks = {
            name
            for name in ("sasa", "secondary_structure", "function", "residue_annotations")
            if bool(config.kwargs.get(name))
        }
        return _FakeLogitsOutput(
            encoded.sequence,
            include_hidden=bool(config.kwargs.get("return_hidden_states")),
            include_logits=bool(config.kwargs.get("sequence")),
            tracks=tracks,
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


# (output field, LogitsConfig kwarg, fake vocab size) for each ESM3 track logit.
_ESM3_TRACK_CASES = [
    ("sasa_logits", "sasa", 5),
    ("secondary_structure_logits", "secondary_structure", 6),
    ("function_logits", "function", 7),
    ("residue_annotation_logits", "residue_annotations", 8),
]


def test_esm3_wildcard_requests_all_supported_tracks(fake_esm_sdk: type[_FakeSDKModel]) -> None:
    """``["*"]`` requests every ESM3-supported track logit, but not hidden states."""
    from boileroom.models.esm3.core import ESM3Core

    result = ESM3Core(config={"device": "cpu"}).embed("ACD", options={"include_fields": ["*"]})

    assert result.lm_logits is not None and result.lm_logits.shape == (1, 3, 4)
    assert result.sasa_logits is not None and result.sasa_logits.shape == (1, 3, 5)
    assert result.secondary_structure_logits is not None and result.secondary_structure_logits.shape == (1, 3, 6)
    assert result.function_logits is not None and result.function_logits.shape == (1, 3, 7)
    assert result.residue_annotation_logits is not None and result.residue_annotation_logits.shape == (1, 3, 8)
    # hidden_states is not an ESM3-supported optional field, so "*" must not request it.
    assert result.hidden_states is None


@pytest.mark.parametrize(("field", "kwarg", "vocab"), _ESM3_TRACK_CASES)
def test_esm3_returns_requested_track_logits(
    fake_esm_sdk: type[_FakeSDKModel], field: str, kwarg: str, vocab: int
) -> None:
    """Each ESM3 track can be requested on its own and returns its own logits shape."""
    from boileroom.models.esm3.core import ESM3Core

    result = ESM3Core(config={"device": "cpu"}).embed("ACD", options={"include_fields": [field]})

    # Only the requested track (and not sequence logits) must be asked of the SDK.
    assert fake_esm_sdk.logits_configs[-1][kwarg] is True
    assert fake_esm_sdk.logits_configs[-1]["sequence"] is False
    assert getattr(result, field) is not None and getattr(result, field).shape == (1, 3, vocab)
    assert result.lm_logits is None


@pytest.mark.parametrize(("field", "kwarg", "vocab"), _ESM3_TRACK_CASES)
def test_esmc_rejects_esm3_track_logits(fake_esm_sdk: type[_FakeSDKModel], field: str, kwarg: str, vocab: int) -> None:
    """ESM-C rejects ESM3-only track requests with a clear ``ValueError``."""
    from boileroom.models.esm3.core import ESMCCore

    with pytest.raises(ValueError, match=rf"{field}.*ESM3"):
        ESMCCore(config={"device": "cpu"}).embed("ACD", options={"include_fields": [field]})


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


class _FakeSequenceTokenizer:
    """Maps amino acids to non-contiguous ids so tests catch id/column mix-ups."""

    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        return [5 + 2 * index for index, _ in enumerate(tokens)]


class _FakeTokenizers:
    sequence = _FakeSequenceTokenizer()


class _FakeIFModel(_FakeSDKModel):
    tokenizers = _FakeTokenizers()
    proteins: ClassVar[list[Any]] = []

    def encode(self, protein: Any) -> _FakeEncoded:
        self.proteins.append(protein)
        return _FakeEncoded(protein.sequence)

    def logits(self, encoded: _FakeEncoded, config: _FakeLogitsConfig) -> Any:
        torch = pytest.importorskip("torch")
        n_tokens = len(encoded.sequence) + 2
        vocab = 64
        # logits[token, v] = 1000 * token + v, so selected values identify (row, column) exactly.
        values = torch.arange(vocab, dtype=torch.float32)[None, :] + 1000 * torch.arange(n_tokens)[:, None]
        return types.SimpleNamespace(logits=types.SimpleNamespace(sequence=values[None]))


@pytest.fixture()
def fake_if_sdk(fake_esm_sdk: type[_FakeSDKModel], monkeypatch: pytest.MonkeyPatch) -> type[_FakeIFModel]:
    _FakeIFModel.proteins = []
    cast(Any, sys.modules["esm.models.esm3"]).ESM3 = _FakeIFModel

    class _Protein(_FakeProtein):
        def __init__(self, sequence: str, coordinates: Any = None) -> None:
            super().__init__(sequence)
            self.coordinates = coordinates

    cast(Any, sys.modules["esm.sdk.api"]).ESMProtein = _Protein
    return _FakeIFModel


def test_esm3_inverse_fold_masks_positions_and_selects_amino_acid_logits(fake_if_sdk: type[_FakeIFModel]) -> None:
    from boileroom.models.esm3.core import INVERSE_FOLDING_AMINO_ACIDS, ESM3Core

    core = ESM3Core(config={"device": "cpu"})
    coords = np.arange(5 * 3 * 3, dtype=np.float32).reshape(5, 3, 3)
    result = core.inverse_fold("ACD:EF", coords, positions=[3, 1])

    protein = fake_if_sdk.proteins[-1]
    assert protein.sequence == "A_D|_F"
    assert tuple(protein.coordinates.shape) == (6, 37, 3)
    # Backbone placed at residue rows, chain-break row and unused atoms are NaN.
    assert protein.coordinates[:3, :3].numpy().tolist() == coords[:3].tolist()
    assert protein.coordinates[4:, :3].numpy().tolist() == coords[3:].tolist()
    assert np.isnan(protein.coordinates[3].numpy()).all()
    assert np.isnan(protein.coordinates[0, 3:].numpy()).all()

    assert result.amino_acids == INVERSE_FOLDING_AMINO_ACIDS
    assert result.positions.tolist() == [3, 1]
    assert result.logits.shape == (2, 20)
    # residue 3 -> sdk index 4 -> token row 5; residue 1 -> sdk index 1 -> token row 2
    expected_columns = [5 + 2 * i for i in range(20)]
    assert result.logits[0].tolist() == [5000 + c for c in expected_columns]
    assert result.logits[1].tolist() == [2000 + c for c in expected_columns]


@pytest.mark.parametrize(
    ("positions", "coords_shape", "match"),
    [
        ([], (3, 3, 3), "non-empty"),
        ([3], (3, 3, 3), "lie in"),
        ([-1], (3, 3, 3), "lie in"),
        ([1, 1], (3, 3, 3), "duplicates"),
        ([1], (2, 3, 3), "shape"),
    ],
)
def test_esm3_inverse_fold_rejects_invalid_inputs(
    fake_if_sdk: type[_FakeIFModel], positions: list[int], coords_shape: tuple[int, ...], match: str
) -> None:
    from boileroom.models.esm3.core import ESM3Core

    core = ESM3Core(config={"device": "cpu"})
    with pytest.raises(ValueError, match=match):
        core.inverse_fold("ACD", np.zeros(coords_shape, dtype=np.float32), positions=positions)
