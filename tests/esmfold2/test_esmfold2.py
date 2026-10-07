"""Fast ESMFold2 unit tests that do not import Biohub runtime dependencies."""

import os
import re
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from boileroom.base import ModelWrapper
from boileroom.inputs import MSAInput
from boileroom.models.esmfold2 import MSAInput as ESMFold2MSAInput
from boileroom.models.esmfold2.types import (
    DNAInput,
    LigandInput,
    PocketConditioning,
    ProteinInput,
    StructurePredictionInput,
)
from boileroom.optimization import GpuInfo

pytestmark = pytest.mark.contract


def _core_cls() -> Any:
    """Import the ESMFold2 core only inside tests that need it."""
    return pytest.importorskip("boileroom.models.esmfold2.core").ESMFold2Core


def _wrapper_cls() -> Any:
    """Import the Modal wrapper module only inside tests that need it."""
    return pytest.importorskip("boileroom.models.esmfold2.esmfold2").ESMFold2


def _payloads() -> Any:
    """Import ESMFold2 payload helpers only inside tests that need them."""
    return pytest.importorskip("boileroom.models.esmfold2.payloads")


def test_esmfold2_string_multimer_becomes_one_complex() -> None:
    """Colon-separated protein strings should represent one multichain complex."""
    core = _core_cls()(config={"device": "cpu"})

    requests = core._coerce_requests("ACD:EFG")

    assert len(requests) == 1
    assert requests[0].sequence_length == 6
    chains = requests[0].input.sequences
    assert chains == [ProteinInput(id="A", sequence="ACD"), ProteinInput(id="B", sequence="EFG")]


def test_esmfold2_string_list_is_batch() -> None:
    """A list of strings should remain a batch of independent proteins."""
    core = _core_cls()(config={"device": "cpu"})

    requests = core._coerce_requests(["ACD", "EFGH"])

    assert [request.sequence_length for request in requests] == [3, 4]
    assert [request.input.sequences[0] for request in requests] == [
        ProteinInput(id="A", sequence="ACD"),
        ProteinInput(id="A", sequence="EFGH"),
    ]


def test_esmfold2_molecule_inputs_become_one_structure_input() -> None:
    """A list of molecule input dataclasses should describe one all-atom complex."""
    core = _core_cls()(config={"device": "cpu"})
    inputs: list[ProteinInput | DNAInput | LigandInput] = [
        ProteinInput(id="A", sequence="ACD"),
        DNAInput(id="B", sequence="GATA"),
        LigandInput(id="L", ccd=["SAH"]),
    ]

    requests = core._coerce_requests(inputs)

    assert len(requests) == 1
    assert requests[0].sequence_length == 8
    assert isinstance(requests[0].input, StructurePredictionInput)


def test_esmfold2_encoded_structure_input_round_trips() -> None:
    """Apptainer payload encoding should preserve all-atom molecule inputs."""
    payloads = _payloads()
    payload = payloads.encode_fold_input([ProteinInput(id="A", sequence="ACD"), LigandInput(id="L", ccd=["SAH"])])

    assert isinstance(payload, dict)
    decoded = payloads.decode_structure_input(payload)

    assert list(decoded.sequences) == [ProteinInput(id="A", sequence="ACD"), LigandInput(id="L", ccd=["SAH"])]


def test_esmfold2_core_accepts_encoded_structure_payload() -> None:
    """The core should accept the JSON payload shape used by Apptainer."""
    core = _core_cls()(config={"device": "cpu"})
    payload = _payloads().encode_fold_input([ProteinInput(id="A", sequence="ACD"), DNAInput(id="B", sequence="GATA")])

    requests = core._coerce_requests(payload)

    assert len(requests) == 1
    assert requests[0].sequence_length == 7


def test_esmfold2_encoded_structure_input_preserves_pocket_conditioning() -> None:
    """Apptainer payload encoding should preserve ESMFold2 pocket conditioning."""
    payloads = _payloads()
    structure_input = StructurePredictionInput(
        sequences=[ProteinInput(id="A", sequence="ACD"), ProteinInput(id="B", sequence="EFG")],
        pocket=PocketConditioning(binder_chain_id="A", contacts=[("B", 1)]),
    )

    payload = payloads.encode_fold_input(structure_input)

    assert isinstance(payload, dict)
    assert payload["pocket"] == {"binder_chain_id": "A", "contacts": [["B", 1]]}
    assert payloads.decode_structure_input(payload).pocket == structure_input.pocket


def test_esmfold2_msa_input_uses_shared_type_and_round_trips() -> None:
    """ESMFold2 should re-export and serialize the shared MSAInput abstraction."""
    payloads = _payloads()
    assert ESMFold2MSAInput is MSAInput
    structure_input = StructurePredictionInput(
        sequences=[ProteinInput(id="A", sequence="ACD", msa=MSAInput(sequences=["ACD", "ACE"], remove_insertions=True))]
    )

    payload = payloads.encode_fold_input(structure_input)

    assert isinstance(payload, dict)
    assert payload["sequences"][0]["msa"] == {
        "sequences": ["ACD", "ACE"],
        "remove_insertions": True,
    }
    decoded = payloads.decode_structure_input(payload)
    assert isinstance(decoded.sequences[0], ProteinInput)
    assert decoded.sequences[0].msa == MSAInput(sequences=["ACD", "ACE"], remove_insertions=True)


def test_esmfold2_path_msa_input_round_trips_through_payloads() -> None:
    """Path-backed MSA payloads should round-trip before model-specific rejection."""
    payloads = _payloads()
    structure_input = StructurePredictionInput(
        sequences=[ProteinInput(id="A", sequence="ACD", msa=MSAInput(path="some/path"))]
    )

    payload = payloads.encode_fold_input(structure_input)

    assert isinstance(payload, dict)
    assert payload["sequences"][0]["msa"] == {"path": "some/path", "remove_insertions": False}
    decoded = payloads.decode_structure_input(payload)
    assert isinstance(decoded.sequences[0], ProteinInput)
    assert decoded.sequences[0].msa == MSAInput(path="some/path")


def test_shared_msa_input_rejects_non_boolean_remove_insertions() -> None:
    """The shared MSAInput should reject ambiguous truthy/falsy flag values."""
    with pytest.raises(TypeError, match="remove_insertions"):
        MSAInput(sequences=["ACD"], remove_insertions="false")  # type: ignore[arg-type]


def test_esmfold2_payload_decode_rejects_silent_coercions() -> None:
    """Encoded Apptainer payloads should reject malformed scalar types."""
    payloads = _payloads()
    base_payload = {
        "kind": "structure_prediction_input",
        "sequences": [{"kind": "protein", "id": "A", "sequence": "ACD", "modifications": None, "msa": None}],
    }

    invalid_id_payload = {
        **base_payload,
        "sequences": [{"kind": "protein", "id": 1, "sequence": "ACD", "modifications": None, "msa": None}],
    }
    with pytest.raises(TypeError, match="id"):
        payloads.decode_structure_input(invalid_id_payload)

    invalid_sequence_payload = {
        **base_payload,
        "sequences": [{"kind": "protein", "id": "A", "sequence": None, "modifications": None, "msa": None}],
    }
    with pytest.raises(TypeError, match="protein.sequence"):
        payloads.decode_structure_input(invalid_sequence_payload)

    invalid_msa_payload = {
        **base_payload,
        "sequences": [
            {
                "kind": "protein",
                "id": "A",
                "sequence": "ACD",
                "modifications": None,
                "msa": {"sequences": ["ACD"], "remove_insertions": "false"},
            }
        ],
    }
    with pytest.raises(TypeError, match="remove_insertions"):
        payloads.decode_structure_input(invalid_msa_payload)

    invalid_msa_path_payload = {
        **base_payload,
        "sequences": [
            {
                "kind": "protein",
                "id": "A",
                "sequence": "ACD",
                "modifications": None,
                "msa": {"path": 123, "remove_insertions": False},
            }
        ],
    }
    with pytest.raises(TypeError, match="path"):
        payloads.decode_structure_input(invalid_msa_path_payload)


def test_esmfold2_rejects_file_backed_msa_until_supported() -> None:
    """Shared path-backed MSA inputs should fail clearly for ESMFold2 for now."""
    with pytest.raises(ValueError, match="requires in-memory MSA sequences"):
        _core_cls()._to_esm_msa(MSAInput(path="/tmp/example.a3m"))


def test_esmfold2_rejects_invalid_dynamic_options() -> None:
    """Invalid dynamic inference options should fail before remote execution."""
    core = _core_cls()(config={"device": "cpu"})

    with pytest.raises(ValueError, match="num_diffusion_samples"):
        core.fold("ACD", options={"num_diffusion_samples": 0})

    with pytest.raises(ValueError, match="seed"):
        core.fold("ACD", options={"seed": -1})

    with pytest.raises(ValueError, match="noise_scale"):
        core.fold("ACD", options={"noise_scale": float("nan")})

    with pytest.raises(ValueError, match="msa_max_depth"):
        core.fold("ACD", options={"msa_max_depth": 0})


def test_esmfold2_forwards_msa_sampling_options(monkeypatch: pytest.MonkeyPatch) -> None:
    """Core should expose Biohub's MSA-depth controls, including full-MSA mode."""
    core = _core_cls()(config={"device": "cpu"})
    captured: dict[str, object] = {}

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(no_grad=nullcontext))
    monkeypatch.setitem(sys.modules, "esm", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "esm.models", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "esm.models.esmfold2", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "esm.models.esmfold2.processor",
        SimpleNamespace(_seed_context=lambda seed: nullcontext()),
    )

    class FakeModel:
        device = "cpu"

        def __call__(self, **kwargs: object) -> object:
            captured.update(kwargs)
            return object()

    class FakeInputBuilder:
        def prepare_input(self, prediction_input: object, *, seed: int | None, device: str) -> tuple[dict, list]:
            return {"feature": object()}, []

        def decode(self, output: object, features: dict, chain_infos: list, **kwargs: object) -> object:
            return object()

    core.model = FakeModel()
    core.input_builder = FakeInputBuilder()
    monkeypatch.setattr(core, "_to_esm_structure_prediction_input", lambda prediction_input: prediction_input)
    prediction_input = StructurePredictionInput(sequences=[ProteinInput(id="A", sequence="ACD")])

    core._fold_one(
        prediction_input,
        {
            **core.config,
            "msa_max_depth": None,
            "msa_column_mask_rate": 0.0,
        },
        request_index=0,
    )

    assert captured["msa_max_depth"] is None
    assert captured["msa_column_mask_rate"] == 0.0
    # esm>=3.4.1 removed these forward() kwargs and raises TypeError on unknown ones.
    for removed in ("early_exit", "lm_dropout", "msa_subsample_at_inference"):
        assert removed not in captured


def test_esmfold2_apptainer_wrapper_encodes_rich_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    """The public wrapper should keep rich input handling out of the shared Apptainer transport."""
    ESMFold2 = _wrapper_cls()
    model = ESMFold2.__new__(ESMFold2)
    ModelWrapper.__init__(model, backend="apptainer:test", device="cuda:0", config={})
    captured: dict[str, object] = {}

    def fake_call(method_name: str, sequences: object, options: dict | None = None) -> object:
        captured["method_name"] = method_name
        captured["sequences"] = sequences
        captured["options"] = options
        return object()

    monkeypatch.setattr(model, "_call_backend_method", fake_call)

    model.fold([ProteinInput(id="A", sequence="ACD")], options={"num_loops": 1})

    assert captured["method_name"] == "fold"
    assert captured["options"] == {"num_loops": 1}
    encoded = captured["sequences"]
    assert isinstance(encoded, dict)
    assert encoded["kind"] == "structure_prediction_input"


def test_esmfold2_rejects_empty_chain() -> None:
    """Empty chains should fail before reaching the Biohub tokenizer."""
    core = _core_cls()(config={"device": "cpu"})

    with pytest.raises(ValueError, match="empty chain"):
        core._coerce_requests("A::B")


def test_esmfold2_ccd_cache_ignores_legacy_file_and_reuses_pinned_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An old unpinned CCD must never satisfy a new revision-pinned load."""
    from boileroom.models.esmfold2.core import ESMFOLD2_HF_REPO, ESMFOLD2_HF_REVISION

    legacy = tmp_path / "ccd.pkl"
    legacy.write_text("legacy snapshot")
    calls = []

    def download(**kwargs: str) -> None:
        calls.append(kwargs)
        directory = Path(kwargs["local_dir"])
        directory.mkdir(parents=True)
        (directory / "ccd.pkl").write_text("pinned snapshot")

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download))
    directory = _core_cls()._ensure_ccd_cache(tmp_path)
    assert directory == tmp_path / ESMFOLD2_HF_REVISION
    assert (directory / "ccd.pkl").read_text() == "pinned snapshot"
    assert legacy.read_text() == "legacy snapshot"
    assert _core_cls()._ensure_ccd_cache(tmp_path) == directory
    assert calls == [
        {
            "repo_id": ESMFOLD2_HF_REPO,
            "filename": "ccd.pkl",
            "revision": ESMFOLD2_HF_REVISION,
            "local_dir": str(directory),
        }
    ]


def test_esmfold2_ccd_download_failure_is_not_treated_as_cache_hit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A failed pinned download must surface even when a legacy file exists."""
    (tmp_path / "ccd.pkl").write_text("legacy snapshot")

    def download(**kwargs: str) -> None:
        raise OSError("download failed")

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download))
    with pytest.raises(OSError, match="download failed"):
        _core_cls()._ensure_ccd_cache(tmp_path)


@pytest.mark.parametrize("fail", [False, True])
def test_esmfold2_buffered_loading_preserves_validation_and_restores_loader(
    monkeypatch: pytest.MonkeyPatch, fail: bool
) -> None:
    """Only the I/O backend changes; arguments and upstream errors survive."""
    from unittest.mock import Mock

    from boileroom.models.esmfold2.loading import load_pretrained

    original = Mock()
    reader = Mock(return_value={"tensor": "weights"})
    hub = SimpleNamespace(load_file=original)
    monkeypatch.setitem(sys.modules, "esm", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "esm.models", SimpleNamespace(hub=hub))
    monkeypatch.setitem(sys.modules, "safetensors.torch", SimpleNamespace(load_file=reader))

    def from_pretrained(name: str, **kwargs: str) -> str:
        assert name == "biohub/ESMFold2"
        assert kwargs == {"revision": "fixed-sha", "cache_dir": "/cache"}
        assert hub.load_file("shard.safetensors") == {"tensor": "weights"}
        if fail:
            raise RuntimeError("unexpected checkpoint key")
        return "loaded model"

    model = SimpleNamespace(from_pretrained=from_pretrained)
    if fail:
        with pytest.raises(RuntimeError, match="unexpected checkpoint key"):
            load_pretrained(model, "biohub/ESMFold2", revision="fixed-sha", cache_dir="/cache")
    else:
        assert load_pretrained(model, "biohub/ESMFold2", revision="fixed-sha", cache_dir="/cache") == "loaded model"
    reader.assert_called_once_with("shard.safetensors", backend="pread")
    assert hub.load_file is original


A3M = ">q\nACD\n>h1\nAcCD\n>h2\nA-D\n"


def test_esmfold2_declares_msa_support_and_refuses_templates() -> None:
    """The core supports a user MSA but not templates, and refuses templates before loading."""
    core_cls = _core_cls()
    assert core_cls.SUPPORTS_USER_MSA is True and core_cls.SUPPORTS_USER_TEMPLATES is False
    with pytest.raises(ValueError, match="does not support user-supplied 'templates'"):
        core_cls(config={"device": "cpu"}).fold("ACD", options={"templates": {"t": "data_"}})


def test_esmfold2_a3m_rows_drop_insertions_and_validate() -> None:
    """A3M text becomes aligned rows; mismatched queries and ragged rows fail clearly."""
    from boileroom.inputs import a3m_rows as rows

    assert rows(A3M, "ACD") == ["ACD", "ACD", "A-D"]
    with pytest.raises(ValueError, match="first A3M row"):
        rows(A3M, "AAA")
    with pytest.raises(ValueError, match="aligned length"):
        rows(">q\nACD\n>h\nAC\n", "ACD")
    with pytest.raises(ValueError, match="A3M text"):
        rows("ACD", "ACD")


def test_esmfold2_attaches_user_msa_per_entry() -> None:
    """options['msa'] lands on the matching protein entries; None entries stay MSA-free."""
    core = _core_cls()(config={"device": "cpu"})
    request = core._coerce_requests("ACD:EFG")[0]
    attached = core._attach_user_msa(request, [A3M, None]).input.sequences
    assert attached[0].msa == MSAInput(sequences=["ACD", "ACD", "A-D"])
    assert attached[1].msa is None
    with pytest.raises(ValueError, match="one A3M string or None per input sequence entry"):
        core._attach_user_msa(request, [A3M])
    ligand = _core_cls()(config={"device": "cpu"})._request_from_structure_input(
        StructurePredictionInput(sequences=[ProteinInput(id="A", sequence="ACD"), LigandInput(id="L", ccd=["ATP"])])
    )
    with pytest.raises(ValueError, match="non-protein"):
        core._attach_user_msa(ligand, [None, A3M])
    with pytest.raises(ValueError, match="already carries an MSA"):
        core._attach_user_msa(
            core._request_from_structure_input(
                StructurePredictionInput(
                    sequences=[ProteinInput(id="A", sequence="ACD", msa=MSAInput(sequences=["ACD"]))]
                )
            ),
            [A3M],
        )


@pytest.mark.parametrize(
    ("config", "variant"),
    [
        ({}, "full_nomsa"),
        ({"kit_msa": True}, "full_msa"),
        ({"model_name": "biohub/ESMFold2-Fast", "kit_msa": True}, "fast"),
    ],
)
def test_esmfold2_kit_variant_follows_checkpoint_and_kit_msa(config: dict, variant: str) -> None:
    """The kit variant is fixed at construction: the Fast checkpoint arms ``fast``; ``kit_msa`` picks the full kernels."""
    assert _core_cls()(config={"device": "cpu", **config})._kit_variant() == variant


def _capture_fold(monkeypatch: pytest.MonkeyPatch, core: Any) -> list[StructurePredictionInput]:
    """Stand in for the loaded runtime: record each input ``_fold_one`` receives and skip result conversion."""
    received: list[StructurePredictionInput] = []

    def fold_one(prediction_input: StructurePredictionInput, config: dict, request_index: int) -> tuple[list, dict]:
        received.append(prediction_input)
        return [object()], {"preprocessing": 0.0, "inference": 0.0, "postprocessing": 0.0}

    core.model, core.input_builder = object(), object()
    monkeypatch.setattr(core, "_fold_one", fold_one)
    monkeypatch.setattr(core, "_convert_results", lambda results, metadata, config: metadata)
    return received


@pytest.mark.parametrize("config", [{}, {"kit_msa": True}])
def test_esmfold2_kit_accepts_a_user_msa_like_an_inline_one(monkeypatch: pytest.MonkeyPatch, config: dict) -> None:
    """options['msa'] reaches the kit fold on both full variants, exactly as an MSA carried on the input entry does."""
    core = _core_cls()(config={"device": "cpu", "optimization": "exact", **config})
    received = _capture_fold(monkeypatch, core)

    core.fold("ACD", options={"msa": [A3M]})
    core.fold([ProteinInput(id="A", sequence="ACD", msa=MSAInput(sequences=["ACD", "ACD", "A-D"]))])

    options_entry, inline_entry = received[0].sequences[0], received[1].sequences[0]
    assert isinstance(options_entry, ProteinInput) and isinstance(inline_entry, ProteinInput)
    assert options_entry.msa == inline_entry.msa == MSAInput(sequences=["ACD", "ACD", "A-D"])


@pytest.mark.parametrize("optimization", ["vanilla", "fast"])
def test_esmfold2_checkpoint_without_an_msa_encoder_refuses_an_msa(
    monkeypatch: pytest.MonkeyPatch, optimization: str
) -> None:
    """ESMFold2-Fast has no MSA encoder and would fold single-sequence; both MSA routes are refused the same way."""
    core = _core_cls()(config={"device": "cpu", "optimization": optimization, "model_name": "biohub/ESMFold2-Fast"})
    received = _capture_fold(monkeypatch, core)
    core.model = SimpleNamespace(msa_encoder=None)

    with pytest.raises(ValueError, match=r"has no MSA encoder.*\['0:A'\].*'biohub/ESMFold2'"):
        core.fold("ACD", options={"msa": [A3M]})
    with pytest.raises(ValueError, match=r"has no MSA encoder.*\['0:A'\]"):
        core.fold([ProteinInput(id="A", sequence="ACD", msa=MSAInput(sequences=["ACD"]))])
    assert received == []

    core.fold("ACD", options={"msa": [None]})
    core.model = SimpleNamespace(msa_encoder=object())
    core.fold("ACD", options={"msa": [A3M]})
    assert len(received) == 2


def test_esmfold2_all_none_msa_option_is_no_msa(monkeypatch: pytest.MonkeyPatch) -> None:
    """``msa=[None, ...]`` names no alignment, so it is accepted for a batch and changes no input."""
    core = _core_cls()(config={"device": "cpu", "optimization": "exact"})
    received = _capture_fold(monkeypatch, core)

    core.fold(["ACD", "EFG"], options={"msa": [None]})

    assert [getattr(item.sequences[0], "msa", "missing") for item in received] == [None, None]


def test_esmfold2_kit_msa_must_be_a_bool() -> None:
    """A string such as "false" would otherwise arm the full_msa kernels."""
    with pytest.raises(ValueError, match="kit_msa"):
        _core_cls()(config={"device": "cpu", "kit_msa": "false"})


def test_esmfold2_msa_rejects_batches() -> None:
    """A single options['msa'] cannot be shared by several inputs."""
    core = _core_cls()(config={"device": "cpu"})
    core.model, core.input_builder = object(), object()
    with pytest.raises(ValueError, match="exactly one input structure"):
        core.fold(["ACD", "EFG"], options={"msa": [A3M]})


def test_esmfold2_wrapper_forwards_msa_option_to_the_core(monkeypatch: pytest.MonkeyPatch) -> None:
    """The public ESMFold2.fold() hands the unified msa key to the backend unchanged."""
    ESMFold2 = _wrapper_cls()
    model = ESMFold2.__new__(ESMFold2)
    ModelWrapper.__init__(model, backend="modal", device="cuda:0", config={})
    captured: dict[str, object] = {}

    def fake_call(method_name: str, sequences: object, options: dict | None = None) -> object:
        captured["options"] = options
        return object()

    monkeypatch.setattr(model, "_call_backend_method", fake_call)
    model.fold("ACD", options={"msa": [A3M]})
    assert captured["options"] == {"msa": [A3M]}


@pytest.fixture
def offline_calls(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Count ``_go_offline`` calls without flipping the real Hugging Face switches for the rest of the session."""
    calls: list[int] = []
    core_module = pytest.importorskip("boileroom.models.esmfold2.core")
    monkeypatch.setattr(core_module.ESMFold2Core, "_go_offline", staticmethod(lambda: calls.append(1)))
    return calls


@pytest.fixture
def kit_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Let a kit activation set its process-wide switches, then restore them (monkeypatch only restores what it set)."""
    for var in (
        "ESMFOLD2_OPT_REQUIRE_FAST_ENV",
        "ESMFOLD2_OPT_WEIGHTS_MEMO_DIR",
        "BOILEROOM_KIT_COMMIT",
        "BOILEROOM_KIT_STACK",
    ):
        monkeypatch.setenv(var, "placeholder")
        monkeypatch.delenv(var)
    monkeypatch.setenv("HF_HOME", str(tmp_path))


A100_GPU = GpuInfo(name="NVIDIA A100-SXM4-80GB", capability=(8, 0))
#: What the kit reports when every accelerated path is live.
FAST_ATTN = {
    "atom_attn": "flash_attn",
    "esmc_mlp": "te",
    "esmc_attn": "varlen",
    "esmc_rope": "flash_attn_triton",
}
KIT_REQUIRED = (
    ("atom_attn", "flash_attn", "flash_attn"),
    ("esmc_mlp", "te", "transformer_engine.pytorch"),
    ("esmc_rope", "flash_attn_triton", "flash_attn.ops.triton.rotary"),
)


class ActivationError(RuntimeError):
    """Stand-in for ``esmfold2_opt.ActivationError``: a refusal by its class name, as the real one is."""


def _fake_kit(
    monkeypatch: pytest.MonkeyPatch,
    files: dict[str, list[str]],
    install_rc: int = 0,
    enable_report: dict | None = None,
) -> SimpleNamespace:
    """Install a stand-in ``esmfold2_opt`` whose pins name ``files`` (repo -> file names).

    Returns the call log; ``log.kit`` is the fake package, whose entry points a test may replace. Its ``attn`` follows
    the kit's semantics: ``failing_words`` names every required word that is not at its accelerated value.
    """
    log = SimpleNamespace(
        installs=[],
        enables=[],
        applies=[],
        refs=[],
        checks=[],
        fetches=[],
        env_at_pregate=[],
        fetch_error=None,
        offline_at_fetch=[],
        settles=[],
    )
    #: Stand-ins for ``stack.settle_mk`` / ``settle_guards``: each maps the report it is given to the settled one.
    log.settle_mk = lambda rep: rep
    log.settle_guards = lambda rep: rep
    log.check_result = ([], [])
    log.pregate_state = dict(FAST_ATTN)
    #: What ``attn.state(model=...)`` reads on the loaded model after a fold; ``status()`` keeps the load-time words.
    log.model_state = dict(FAST_ATTN)
    log.status_report = None
    pins = {
        "ccd": {"repo": "biohub/ESMFold2", "file": "ccd.pkl"},
        "weights": {
            repo: {"snapshot_commit": "c" * 40, "files": dict.fromkeys(names, {})} for repo, names in files.items()
        },
    }
    report: dict[str, Any] = {}

    def pinned_weight_files(pins_: dict, variant: str | None) -> list[tuple[str, str, int]]:
        repos = {"biohub/ESMC-6B", "biohub/ESMFold2-Fast" if variant == "fast" else "biohub/ESMFold2"}
        return [
            (repo, f"hub/models--{repo.replace('/', '--')}/snapshots/{'c' * 40}/{name}", 1)
            for repo in sorted(repos & set(pins_["weights"]))
            for name in pins_["weights"][repo]["files"]
        ]

    def upstream_fetch(hf_home: str) -> Any:
        constants = sys.modules.get("huggingface_hub.constants")
        log.offline_at_fetch.append((os.environ.get("HF_HUB_OFFLINE"), getattr(constants, "HF_HUB_OFFLINE", None)))

        def fetch(repo: str, commit: str, name: str) -> str:
            log.fetches.append((repo, commit, name))
            if log.fetch_error is not None:
                raise log.fetch_error
            return f"{hf_home}/{repo}/{name}"

        return fetch

    def install_weights(hf_home: str, fetch: Any = None, pins: dict | None = None, log: Any = print) -> int:
        kit_log.installs.append((hf_home, pins))
        # The kit catches a failed transfer, logs it and returns 1, as it does for a file off its pin.
        try:
            for repo, entry in sorted((pins or {})["weights"].items()):
                for name in entry["files"]:
                    fetch(repo, entry["snapshot_commit"], name)
        except Exception:
            return 1
        return install_rc

    kit_log = log

    def weight_files_check(hf: str, pins_: dict, variant: str | None) -> tuple[list[str], list[str]]:
        log.checks.append((hf, sorted(pins_["weights"]), variant))
        return log.check_result

    def enable(mode: str, variant: str | None = None) -> dict:
        log.enables.append((mode, variant))
        report.clear()
        report.update(enable_report or {"active": True, "partial": [], "attn": dict(FAST_ATTN)})
        return dict(report)

    def apply_to(model: Any, builder: Any = None, trigger: str = "explicit", samples: int = 1, out_dir: Any = None):
        log.applies.append((model, builder, trigger, samples))
        if not report.get("active"):
            raise ActivationError("apply_to: no active mode")
        report.update(
            levers_planned=["atom_flash", "esmc_te", "mk"],
            levers_applied=["atom_flash", "esmc_te", "mk"],
            levers_fallback=[],
            attn=dict(FAST_ATTN),
        )
        return {"applied": True}

    def settle(step: str) -> Any:
        def run(rep: dict) -> dict:
            log.settles.append(step)
            return getattr(log, step)(rep)

        return run

    def mk_plan_note(levers_planned: Any, num_diffusion_samples: int) -> str | None:
        if "mk" not in (levers_planned or ()):
            return None
        return "mk: installed; inactive at num_diffusion_samples>1 (kit guard)" if num_diffusion_samples > 1 else None

    def status() -> dict:
        return dict(log.status_report if log.status_report is not None else report)

    def state(model: Any = None, common: Any = None, load: bool = False) -> dict:
        if model is not None:
            return dict(log.model_state)
        log.env_at_pregate.append(os.environ.get("ESMFOLD2_OPT_REQUIRE_FAST_ENV"))
        return dict(log.pregate_state)

    attn = SimpleNamespace(
        ENV_REQUIRE="ESMFOLD2_OPT_REQUIRE_FAST_ENV",
        REQUIRED=KIT_REQUIRED,
        failing_words=lambda st: [
            (word, str(st.get(word, "unread")), want, module)
            for word, want, module in KIT_REQUIRED
            if st.get(word, "unread") != want
        ],
        metadata_words=lambda vers=None: "flash_attn=2.8.3.post1 transformer_engine=2.15.0 xformers=0.0.35",
        state=state,
        words=lambda st: " ".join(f"{word}={st.get(word, 'unread')}" for word in FAST_ATTN),
    )
    kit = SimpleNamespace(
        __version__="0.4.0",
        ActivationError=ActivationError,
        attn=attn,
        stack=SimpleNamespace(
            pins=lambda: pins,
            pinned_weight_files=pinned_weight_files,
            weight_files_check=weight_files_check,
            apply_to=apply_to,
            status=status,
            settle_mk=settle("settle_mk"),
            settle_guards=settle("settle_guards"),
            mk_plan_note=mk_plan_note,
            guards_plan_note=lambda levers_planned, num_diffusion_samples: None,
        ),
        weights=SimpleNamespace(
            install_weights=install_weights,
            upstream_fetch=upstream_fetch,
            point_ref=lambda hf_home, repo, commit: log.refs.append((hf_home, repo, commit)),
        ),
        enable=enable,
    )
    log.kit = kit
    monkeypatch.setitem(sys.modules, "esmfold2_opt", kit)
    return log


FILES = {
    "biohub/ESMC-6B": ["model.safetensors"],
    "biohub/ESMFold2": ["model.safetensors", "ccd.pkl"],
    "biohub/ESMFold2-Fast": ["model.safetensors"],
}


def _place_weights(hf_home: Path, repos: tuple[str, ...] = ("biohub/ESMC-6B", "biohub/ESMFold2")) -> None:
    for repo in repos:
        for name in FILES[repo]:
            path = hf_home / "hub" / f"models--{repo.replace('/', '--')}" / "snapshots" / ("c" * 40) / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"")


def _kit_core(monkeypatch: pytest.MonkeyPatch, **config: Any) -> Any:
    """A kit-mode core on a stand-in A100."""
    core_module = pytest.importorskip("boileroom.models.esmfold2.core")
    monkeypatch.setattr(core_module, "detect_gpu", lambda device: A100_GPU)
    return core_module.ESMFold2Core(config={"device": "cpu", "optimization": "exact", **config})


def test_esmfold2_kit_weights_are_fetched_for_the_variant_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A fresh model directory gets the pinned snapshots of the variant's repos and the language model, nothing else."""
    log = _fake_kit(monkeypatch, FILES)
    core = _core_cls()(config={"device": "cpu", "optimization": "exact"})

    core._ensure_kit_weights(tmp_path)

    assert len(log.installs) == 1
    hf_home, pins = log.installs[0]
    assert hf_home == str(tmp_path)
    assert set(pins["weights"]) == {"biohub/ESMC-6B", "biohub/ESMFold2"}
    assert set(pins["weights"]["biohub/ESMFold2"]["files"]) == {"model.safetensors", "ccd.pkl"}
    assert sorted(name for _, _, name in log.fetches) == ["ccd.pkl", "model.safetensors", "model.safetensors"]


def test_esmfold2_kit_weights_for_the_fast_checkpoint_skip_the_full_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    log = _fake_kit(monkeypatch, FILES)
    core = _core_cls()(config={"device": "cpu", "optimization": "fast", "model_name": "biohub/ESMFold2-Fast"})

    core._ensure_kit_weights(tmp_path)

    weights = log.installs[0][1]["weights"]
    assert set(weights) == {"biohub/ESMC-6B", "biohub/ESMFold2-Fast", "biohub/ESMFold2"}
    # The Fast repository ships no ccd.pkl, so the full repository contributes that file alone.
    assert set(weights["biohub/ESMFold2"]["files"]) == {"ccd.pkl"}
    assert set(weights["biohub/ESMFold2-Fast"]["files"]) == {"model.safetensors"}


def test_esmfold2_kit_weights_already_in_place_are_repointed_and_verified(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Present files are not fetched again, but refs/main is pointed back at the pin and the digests are checked."""
    log = _fake_kit(monkeypatch, FILES)
    core = _core_cls()(config={"device": "cpu", "optimization": "exact"})
    _place_weights(tmp_path)

    core._ensure_kit_weights(tmp_path)

    assert log.installs == [] and log.fetches == []
    assert log.refs == [(str(tmp_path), "biohub/ESMC-6B", "c" * 40), (str(tmp_path), "biohub/ESMFold2", "c" * 40)]
    assert log.checks == [(str(tmp_path), ["biohub/ESMC-6B", "biohub/ESMFold2"], None)]


def test_esmfold2_kit_weights_in_place_but_off_their_pin_are_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from boileroom.optimization import OptimizationUnavailableError

    log = _fake_kit(monkeypatch, FILES)
    log.check_result = ([], ["biohub/ESMFold2 model.safetensors"])
    core = _core_cls()(config={"device": "cpu", "optimization": "exact"})
    _place_weights(tmp_path)

    with pytest.raises(OptimizationUnavailableError, match="off their pin.*model.safetensors"):
        core._ensure_kit_weights(tmp_path)


def test_esmfold2_kit_weight_transfer_failure_is_retryable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A dropped connection must not be cached as a permanent refusal of the mode."""
    from boileroom.optimization import OptimizationUnavailableError

    log = _fake_kit(monkeypatch, FILES)
    log.fetch_error = OSError("connection reset by peer")
    core = _core_cls()(config={"device": "cpu", "optimization": "exact"})

    with pytest.raises(RuntimeError, match="connection reset by peer.*tries again") as raised:
        core._ensure_kit_weights(tmp_path)
    assert not isinstance(raised.value, OptimizationUnavailableError)


def test_esmfold2_kit_weight_fetch_lifts_the_offline_switches_already_imported(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The pre-gate imports huggingface_hub first, so the fetch must flip its frozen switch, not only the env vars."""
    log = _fake_kit(monkeypatch, FILES)
    monkeypatch.setitem(sys.modules, "huggingface_hub.constants", SimpleNamespace(HF_HUB_OFFLINE=True))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    core = _core_cls()(config={"device": "cpu", "optimization": "exact"})

    core._ensure_kit_weights(tmp_path)

    assert log.offline_at_fetch == [(None, False)]
    assert "TRANSFORMERS_OFFLINE" not in os.environ


def test_esmfold2_kit_weights_off_their_pin_after_the_fetch_refuse_the_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from boileroom.optimization import OptimizationUnavailableError

    _fake_kit(monkeypatch, FILES, install_rc=1)
    core = _core_cls()(config={"device": "cpu", "optimization": "exact"})

    with pytest.raises(OptimizationUnavailableError, match="absent or off their pins"):
        core._ensure_kit_weights(tmp_path)


def test_esmfold2_kit_weight_install_that_exits_is_a_typed_refusal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A SystemExit from the kit must not escape the core (it would end the server process)."""
    from boileroom.optimization import OptimizationUnavailableError

    log = _fake_kit(monkeypatch, FILES)

    def exits(*args: Any, **kwargs: Any) -> int:
        raise SystemExit(3)

    log.kit.weights.install_weights = exits
    core = _core_cls()(config={"device": "cpu", "optimization": "exact"})

    with pytest.raises(OptimizationUnavailableError, match="install_weights exited with code 3") as raised:
        core._ensure_kit_weights(tmp_path)
    assert isinstance(raised.value.__cause__, SystemExit)


def test_esmfold2_kit_home_defaults_to_the_model_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, offline_calls: list[int], kit_env: None
) -> None:
    """The kit's HF_HOME is set before the kit is imported, under the model volume so the weights persist."""
    core_module = pytest.importorskip("boileroom.models.esmfold2.core")
    log = _fake_kit(monkeypatch, FILES)
    monkeypatch.delenv("HF_HOME")
    core = _kit_core(monkeypatch)
    core.model_dir = str(tmp_path)
    monkeypatch.setattr(core, "_ensure_kit_weights", lambda hf_home: log.installs.append(hf_home))

    core._activate_optimization()

    expected = tmp_path / core_module.KIT_HF_SUBDIR
    assert log.installs == [expected]
    assert log.enables == [("exact", "full_nomsa")]
    assert core.optimization is not None and core.optimization.mode == "exact" and core.optimization.kit
    assert os.environ["HF_HOME"] == str(expected)
    # The digest memo sits on the model volume too, so a cold start reuses it instead of re-hashing the weights.
    assert os.environ["ESMFOLD2_OPT_WEIGHTS_MEMO_DIR"] == str(tmp_path / core_module.KIT_WEIGHTS_MEMO_SUBDIR)
    assert offline_calls == [1]  # the kit's snapshots are served from disk once they are fetched


def test_esmfold2_kit_home_respects_a_configured_hf_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, offline_calls: list[int], kit_env: None
) -> None:
    log = _fake_kit(monkeypatch, FILES)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "mine"))
    core = _kit_core(monkeypatch)
    core.model_dir = str(tmp_path)
    monkeypatch.setattr(core, "_ensure_kit_weights", lambda hf_home: log.installs.append(hf_home))

    core._activate_optimization()

    assert log.installs == [tmp_path / "mine"]


def test_esmfold2_kit_weights_memo_is_set_before_the_weight_check_and_respects_a_configured_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, offline_calls: list[int], kit_env: None
) -> None:
    """The kit reads its digest-memo directory when it checks the weights: an explicit setting wins over the default."""
    log = _fake_kit(monkeypatch, FILES)
    monkeypatch.setenv("ESMFOLD2_OPT_WEIGHTS_MEMO_DIR", str(tmp_path / "digests"))
    core = _kit_core(monkeypatch)
    core.model_dir = str(tmp_path / "models")
    monkeypatch.setattr(
        core, "_ensure_kit_weights", lambda hf_home: log.installs.append(os.environ["ESMFOLD2_OPT_WEIGHTS_MEMO_DIR"])
    )

    core._activate_optimization()

    assert log.installs == [str(tmp_path / "digests")]


def test_esmfold2_kit_arms_the_fail_loud_switch_before_reading_the_kit(
    monkeypatch: pytest.MonkeyPatch, offline_calls: list[int], kit_env: None
) -> None:
    """Even with the switch explicitly off in the environment, a kit mode turns it on before the kit reads it."""
    log = _fake_kit(monkeypatch, FILES)
    monkeypatch.setenv("ESMFOLD2_OPT_REQUIRE_FAST_ENV", "0")
    core = _kit_core(monkeypatch)
    monkeypatch.setattr(core, "_ensure_kit_weights", lambda hf_home: None)

    core._activate_optimization()

    assert log.env_at_pregate == ["1"]
    assert os.environ["ESMFOLD2_OPT_REQUIRE_FAST_ENV"] == "1"


def test_esmfold2_vanilla_never_touches_the_kit_or_hf_home(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("HF_HOME", "ESMFOLD2_OPT_REQUIRE_FAST_ENV", "ESMFOLD2_OPT_WEIGHTS_MEMO_DIR"):
        monkeypatch.setenv(var, "placeholder")
        monkeypatch.delenv(var)
    monkeypatch.delitem(sys.modules, "esmfold2_opt", raising=False)
    core = _core_cls()(config={"device": "cpu"})

    core._activate_optimization()

    assert "HF_HOME" not in os.environ and "ESMFOLD2_OPT_REQUIRE_FAST_ENV" not in os.environ
    assert "ESMFOLD2_OPT_WEIGHTS_MEMO_DIR" not in os.environ
    assert "esmfold2_opt" not in sys.modules


def test_esmfold2_kit_mode_without_the_kit_installed_is_a_typed_refusal(
    monkeypatch: pytest.MonkeyPatch, kit_env: None
) -> None:
    from boileroom.optimization import OptimizationUnavailableError

    monkeypatch.setitem(sys.modules, "esmfold2_opt", None)  # makes `import esmfold2_opt` raise ImportError
    core = _kit_core(monkeypatch)

    with pytest.raises(OptimizationUnavailableError, match="esmfold2_opt is not installed"):
        core._activate_optimization()


def test_esmfold2_kit_with_slow_attention_is_refused_before_the_weight_fetch(
    monkeypatch: pytest.MonkeyPatch, kit_env: None
) -> None:
    """An image without flash-attn must not cost a ~27 GB download before it is refused."""
    from boileroom.optimization import OptimizationUnavailableError

    log = _fake_kit(monkeypatch, FILES)
    log.pregate_state = {**FAST_ATTN, "atom_attn": "sdpa"}
    core = _kit_core(monkeypatch)

    with pytest.raises(
        OptimizationUnavailableError, match=r"before the weight fetch: atom_attn=sdpa \(expected flash_attn"
    ):
        core._activate_optimization()
    assert log.installs == [] and log.fetches == [] and log.enables == []


def test_esmfold2_kit_whose_paths_cannot_be_read_is_refused(monkeypatch: pytest.MonkeyPatch, kit_env: None) -> None:
    from boileroom.optimization import OptimizationUnavailableError

    log = _fake_kit(monkeypatch, FILES)

    def broken_state(**kwargs: Any) -> dict:
        raise ImportError("libcudart.so.13: cannot open shared object file")

    log.kit.attn.state = broken_state
    core = _kit_core(monkeypatch)

    with pytest.raises(OptimizationUnavailableError, match="libcudart.so.13") as raised:
        core._activate_optimization()
    assert isinstance(raised.value.__cause__, ImportError)
    assert log.installs == []


def test_esmfold2_kit_reading_another_switch_is_refused(monkeypatch: pytest.MonkeyPatch, kit_env: None) -> None:
    """If the kit renamed its fail-loud switch, setting ours would arm nothing."""
    from boileroom.optimization import OptimizationUnavailableError

    log = _fake_kit(monkeypatch, FILES)
    log.kit.attn.ENV_REQUIRE = "ESMFOLD2_OPT_STRICT"
    core = _kit_core(monkeypatch)

    with pytest.raises(OptimizationUnavailableError, match="ESMFOLD2_OPT_STRICT"):
        core._activate_optimization()


def test_esmfold2_kit_that_does_not_activate_is_refused_with_its_reason(
    monkeypatch: pytest.MonkeyPatch, offline_calls: list[int], kit_env: None
) -> None:
    from boileroom.optimization import OptimizationUnavailableError

    _fake_kit(monkeypatch, FILES, enable_report={"active": False, "reason": "pins not met: esm 3.4.1"})
    core = _kit_core(monkeypatch)
    monkeypatch.setattr(core, "_ensure_kit_weights", lambda hf_home: None)

    with pytest.raises(OptimizationUnavailableError, match="not active.*pins not met: esm 3.4.1"):
        core._activate_optimization()


@pytest.mark.parametrize(
    ("report", "match"),
    [
        # The kit activates but reports SDPA atom attention: it would fold, slowly, without this check.
        ({"active": True, "partial": [], "attn": {**FAST_ATTN, "atom_attn": "sdpa"}}, "at activation: atom_attn=sdpa"),
        ({"active": True, "partial": [], "attn": {**FAST_ATTN, "esmc_mlp": "torch"}}, "esmc_mlp=torch"),
        ({"active": True, "partial": ["esmc_te"], "attn": dict(FAST_ATTN)}, r"only part of its lever set.*esmc_te"),
        ({"active": True, "partial": []}, "reported no attention paths"),
    ],
    ids=["slow-atom-attention", "torch-mlp", "partial", "no-attn"],
)
def test_esmfold2_kit_activation_short_of_the_fast_stack_is_refused(
    monkeypatch: pytest.MonkeyPatch, offline_calls: list[int], kit_env: None, report: dict, match: str
) -> None:
    from boileroom.optimization import OptimizationUnavailableError

    _fake_kit(monkeypatch, FILES, enable_report=report)
    core = _kit_core(monkeypatch)
    monkeypatch.setattr(core, "_ensure_kit_weights", lambda hf_home: None)

    with pytest.raises(OptimizationUnavailableError, match=match):
        core._activate_optimization()


def test_esmfold2_kit_enable_that_exits_is_a_typed_refusal(
    monkeypatch: pytest.MonkeyPatch, offline_calls: list[int], kit_env: None
) -> None:
    from boileroom.optimization import OptimizationUnavailableError

    log = _fake_kit(monkeypatch, FILES)

    def exits(mode: str, variant: str | None = None) -> dict:
        raise SystemExit(3)

    log.kit.enable = exits
    core = _kit_core(monkeypatch)
    monkeypatch.setattr(core, "_ensure_kit_weights", lambda hf_home: None)

    with pytest.raises(
        OptimizationUnavailableError, match="'exact': esmfold2_opt enable exited with code 3 on NVIDIA A100"
    ):
        core._activate_optimization()


def _armed_core(monkeypatch: pytest.MonkeyPatch, log: SimpleNamespace, **config: Any) -> Any:
    """A kit core whose activation passed, with a stand-in model and builder (as ``_load_kit`` leaves them)."""
    core = _kit_core(monkeypatch, **config)
    monkeypatch.setattr(core, "_ensure_kit_weights", lambda hf_home: None)
    core._activate_optimization()
    core.model, core.input_builder = "model", "builder"
    return core


def test_esmfold2_kit_levers_on_the_loaded_model_pass_the_gate_and_are_recorded(
    monkeypatch: pytest.MonkeyPatch, offline_calls: list[int], kit_env: None
) -> None:
    log = _fake_kit(monkeypatch, FILES)
    monkeypatch.setenv("BOILEROOM_KIT_COMMIT", "f" * 40)
    monkeypatch.setenv("BOILEROOM_KIT_STACK", "img_esmfold2_a100")
    core = _armed_core(monkeypatch, log, num_diffusion_samples=2)
    assert core._kernel_gate_passed is False

    core._configure_optimization()

    assert log.applies == [("model", "builder", "boileroom", 2)]
    assert core._kernel_gate_passed is True
    runtime = core._runtime
    assert runtime["kit.commit"] == "f" * 40 and runtime["kit.stack"] == "img_esmfold2_a100"
    assert runtime["kit.variant"] == "full_nomsa" and runtime["kit.package"] == "0.4.0"
    assert runtime["kit.weights"] == f"biohub/ESMC-6B@{'c' * 40} biohub/ESMFold2@{'c' * 40}"
    assert runtime["kit.require_fast_env"] == "1"
    assert runtime["kit.attn.atom_attn"] == "flash_attn" and runtime["kit.attn.esmc_mlp"] == "te"
    assert runtime["kit.attn.esmc_rope"] == "flash_attn_triton" and runtime["kit.attn.esmc_attn"] == "varlen"
    assert not {"atom_attn", "esmc_mlp", "esmc_attn", "esmc_rope"} & set(runtime), "kernel words outside kit.*"
    assert runtime["kit.levers_applied"] == "atom_flash,esmc_te,mk" and runtime["kit.levers_fallback"] == "none"
    assert runtime["kit.partial"] == "false"
    assert runtime["kit.levers_gated"] == runtime["kit.gated"] == "none"
    assert runtime["kit.scope"] == "mk: installed; inactive at num_diffusion_samples>1 (kit guard)"
    assert runtime["gpu"] == "NVIDIA A100-SXM4-80GB"


@pytest.mark.parametrize(
    ("status", "match"),
    [
        ({"active": True, "partial": ["atom_flash"], "attn": dict(FAST_ATTN)}, "only part of its lever set"),
        (
            {"active": True, "partial": [], "attn": {**FAST_ATTN, "esmc_rope": "torch"}},
            "on the loaded model: esmc_rope",
        ),
    ],
    ids=["partial", "slow-rope"],
)
def test_esmfold2_kit_levers_short_of_the_fast_stack_are_refused(
    monkeypatch: pytest.MonkeyPatch, offline_calls: list[int], kit_env: None, status: dict, match: str
) -> None:
    from boileroom.optimization import OptimizationUnavailableError

    log = _fake_kit(monkeypatch, FILES)
    core = _armed_core(monkeypatch, log)
    log.status_report = status

    with pytest.raises(OptimizationUnavailableError, match=match):
        core._configure_optimization()
    assert core._kernel_gate_passed is False and core._runtime is None


@pytest.mark.parametrize(
    ("error", "expected", "match"),
    [
        (SystemExit(3), "OptimizationUnavailableError", "apply_to exited with code 3"),
        (SystemExit(1), "RuntimeError", "apply_to exited with code 1 .*a failure, not a refusal"),
        (SystemExit(2), "RuntimeError", "apply_to exited with code 2 .*a failure, not a refusal"),
        (SystemExit(5), "OptimizationUnavailableError", "apply_to exited with code 5"),
        (SystemExit([3]), "RuntimeError", r"apply_to exited with code \[3\] .*a failure, not a refusal"),
        (ActivationError("configure failed: atom_flash"), "OptimizationUnavailableError", "configure failed"),
        (type("NotLoaded", (RuntimeError,), {})("lnstream not bound"), "OptimizationUnavailableError", "not bound"),
        (RuntimeError("CUDA out of memory"), "RuntimeError", "CUDA out of memory"),
    ],
    ids=[
        "exit",
        "exit-1-fails",
        "exit-2-fails",
        "exit-5",
        "unhashable-code-fails",
        "activation-error",
        "named-refusal-class",
        "oom-propagates",
    ],
)
def test_esmfold2_kit_apply_failures(
    monkeypatch: pytest.MonkeyPatch,
    offline_calls: list[int],
    kit_env: None,
    error: BaseException,
    expected: str,
    match: str,
) -> None:
    """The kit's refusals become typed refusals; anything else (an out-of-memory error) propagates unchanged."""
    log = _fake_kit(monkeypatch, FILES)
    core = _armed_core(monkeypatch, log)

    def apply_to(*args: Any, **kwargs: Any) -> None:
        raise error

    log.kit.stack.apply_to = apply_to

    with pytest.raises(BaseException, match=match) as raised:
        core._configure_optimization()
    assert type(raised.value).__name__ == expected
    assert core._kernel_gate_passed is False


def test_esmfold2_kit_refusal_classes_are_the_shared_set_only(
    monkeypatch: pytest.MonkeyPatch, offline_calls: list[int], kit_env: None
) -> None:
    """The kit's ``ActivationError`` is a refusal through the shared class-name set alone, not a family-local check."""
    import boileroom.optimization as optimization

    log = _fake_kit(monkeypatch, FILES)
    core = _armed_core(monkeypatch, log)
    monkeypatch.setattr(optimization, "KIT_REFUSAL_CLASS_NAMES", frozenset({"NotLoaded", "OpenModeError"}))

    def apply_to(*args: Any, **kwargs: Any) -> None:
        raise ActivationError("configure failed: atom_flash")

    log.kit.stack.apply_to = apply_to

    with pytest.raises(ActivationError, match="configure failed"):
        core._configure_optimization()
    assert core._kernel_gate_passed is False


def test_esmfold2_kit_runtime_names_what_it_could_not_establish(
    monkeypatch: pytest.MonkeyPatch, offline_calls: list[int], kit_env: None
) -> None:
    """A kernel word the kit did not report is ``unknown``; an unset environment switch is ``absent``."""
    log = _fake_kit(monkeypatch, FILES)
    core = _armed_core(monkeypatch, log)
    log.status_report = {"active": True, "partial": [], "attn": {**FAST_ATTN, "esmc_attn": None}}
    del log.status_report["attn"]["esmc_attn"]
    monkeypatch.delenv("ESMFOLD2_OPT_REQUIRE_FAST_ENV")
    monkeypatch.delenv("ESMFOLD2_OPT_WEIGHTS_MEMO_DIR")

    core._configure_optimization()

    runtime = core._runtime
    assert runtime["kit.attn.esmc_attn"] == "unknown" and runtime["kit.attn.atom_attn"] == "flash_attn"
    assert runtime["kit.require_fast_env"] == "absent" and runtime["kit.weights_memo"] == "absent"
    assert runtime["kit.stack"] == "unknown"
    assert runtime["kit.guards"] == "none"


class _KitBuilder:
    """Stand-in for esm 3.3.0's ``ESMFold2InputBuilder`` with the kit's ``fold()``."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, Any, dict]] = []

    def fold(self, model: Any, esm_input: Any, **kwargs: Any) -> list[str]:
        self.calls.append((model, esm_input, kwargs))
        return ["decoded"] * int(kwargs["num_diffusion_samples"])


def _kit_fold_core(monkeypatch: pytest.MonkeyPatch, **config: Any) -> tuple[Any, _KitBuilder]:
    from boileroom.optimization import resolve_optimization

    core = _core_cls()(config={"device": "cpu", "optimization": "exact", **config})
    core.optimization = resolve_optimization("esmfold2", "exact", A100_GPU)
    builder = _KitBuilder()
    core.model, core.input_builder = "kit-model", builder
    monkeypatch.setattr(core, "_to_esm_structure_prediction_input", lambda prediction_input: prediction_input)
    return core, builder


def test_esmfold2_kit_fold_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    """The kit fold gets the integer depth, keeps the checkpoint's LM dropout, forwards lm_mask_pct, times one call."""
    core, builder = _kit_fold_core(monkeypatch, num_diffusion_samples=2, seed=7, lm_mask_pct=0.1, msa_max_depth=64)
    core._kernel_gate_passed = True
    prediction_input = StructurePredictionInput(sequences=[ProteinInput(id="A", sequence="ACD")])

    results, timing = core._fold_one(prediction_input, dict(core.config), request_index=1)

    assert results == ["decoded", "decoded"]
    ((model, esm_input, kwargs),) = builder.calls
    assert model == "kit-model" and esm_input is prediction_input
    assert kwargs == {
        "num_loops": 3,
        "num_sampling_steps": 50,
        "num_diffusion_samples": 2,
        "seed": 7,
        "lm_dropout": None,
        "msa_max_depth": 64,
        "msa_column_mask_rate": 0.1,
        "complex_id": "pred_1",
        "lm_mask_pct": 0.1,
    }
    assert timing["preprocessing"] == timing["postprocessing"] == 0.0 and timing["inference"] >= 0.0


@pytest.mark.parametrize("fails", [False, True], ids=["served", "failed"])
def test_esmfold2_kit_fold_releases_evicted_trimul_caches(monkeypatch: pytest.MonkeyPatch, fails: bool) -> None:
    """Every kit fold, a failed one too, frees the TriMul geometries the kit's LRU evicted (see ``_worker``)."""
    core_module = pytest.importorskip("boileroom.models.esmfold2.core")
    core, builder = _kit_fold_core(monkeypatch)
    released: list[int] = []

    def release() -> int:
        released.append(len(builder.calls))
        return 0

    monkeypatch.setattr(core_module, "release_evicted_kit_caches", release)
    core._kernel_gate_passed = True
    prediction_input = StructurePredictionInput(sequences=[ProteinInput(id="A", sequence="ACD")])
    if fails:

        def out_of_memory(model: Any, esm_input: Any, **kwargs: Any) -> list[str]:
            builder.calls.append((model, esm_input, kwargs))
            raise RuntimeError("CUDA out of memory")

        monkeypatch.setattr(builder, "fold", out_of_memory)

    with pytest.raises(RuntimeError, match="out of memory") if fails else nullcontext():
        core._fold_one(prediction_input, dict(core.config), request_index=0)

    assert released == [1]


def test_esmfold2_kit_fold_needs_the_kernel_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """A kit runtime whose levers were never confirmed on the loaded model must not fold."""
    from boileroom.optimization import OptimizationUnavailableError

    core, builder = _kit_fold_core(monkeypatch)
    prediction_input = StructurePredictionInput(sequences=[ProteinInput(id="A", sequence="ACD")])

    with pytest.raises(OptimizationUnavailableError, match="kernels were not confirmed"):
        core._fold_one(prediction_input, dict(core.config), request_index=0)
    assert builder.calls == []


def test_esmfold2_kit_and_vanilla_folds_get_the_same_request(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both branches see the same converted input and the same shared sampling controls."""
    core, builder = _kit_fold_core(monkeypatch, seed=3, num_loops=2, msa_max_depth=16, msa_column_mask_rate=0.0)
    core._kernel_gate_passed = True
    prediction_input = StructurePredictionInput(sequences=[ProteinInput(id="A", sequence="ACD")])
    core._fold_one(prediction_input, dict(core.config), request_index=0)
    kit_input, kit_kwargs = builder.calls[0][1], builder.calls[0][2]

    vanilla = _core_cls()(
        config={"device": "cpu", "seed": 3, "num_loops": 2, "msa_max_depth": 16, "msa_column_mask_rate": 0.0}
    )
    captured: dict[str, Any] = {}
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(no_grad=nullcontext))
    monkeypatch.setitem(sys.modules, "esm", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "esm.models", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "esm.models.esmfold2", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules, "esm.models.esmfold2.processor", SimpleNamespace(_seed_context=lambda seed: nullcontext())
    )

    class VanillaModel:
        device = "cpu"

        def __call__(self, **kwargs: Any) -> object:
            captured.update(kwargs)
            return object()

    class VanillaBuilder:
        def prepare_input(self, esm_input: Any, *, seed: int | None, device: str) -> tuple[dict, list]:
            captured["input"], captured["seed"] = esm_input, seed
            return {}, []

        def decode(self, output: Any, features: dict, chain_infos: list, **kwargs: Any) -> object:
            captured["complex_id"] = kwargs["complex_id"]
            return object()

    vanilla.model, vanilla.input_builder = VanillaModel(), VanillaBuilder()
    monkeypatch.setattr(vanilla, "_to_esm_structure_prediction_input", lambda prediction_input: prediction_input)
    vanilla._fold_one(prediction_input, dict(vanilla.config), request_index=0)

    assert captured["input"] is kit_input
    for key in ("seed", "num_loops", "num_sampling_steps", "num_diffusion_samples", "msa_max_depth", "complex_id"):
        assert captured[key] == kit_kwargs[key], key
    assert captured["msa_column_mask_rate"] == kit_kwargs["msa_column_mask_rate"] == 0.0


def _folding_kit_core(monkeypatch: pytest.MonkeyPatch, log: SimpleNamespace, **config: Any) -> tuple[Any, _KitBuilder]:
    """A kit core past its kernel gate whose ``fold()`` runs ``_fold_kit`` on a stand-in builder."""
    core = _armed_core(monkeypatch, log, **config)
    core._configure_optimization()
    builder = _KitBuilder()
    core.input_builder = builder
    monkeypatch.setattr(core, "_to_esm_structure_prediction_input", lambda prediction_input: prediction_input)
    monkeypatch.setattr(core, "_convert_results", lambda results, metadata, config: metadata)
    return core, builder


def test_esmfold2_kit_fold_records_the_levers_as_settled_after_it(
    monkeypatch: pytest.MonkeyPatch, offline_calls: list[int], kit_env: None
) -> None:
    """At 5 samples the kit's mk guard sends every step to the previous chain; the output says so, not the load snapshot."""
    log = _fake_kit(monkeypatch, FILES)
    core, builder = _folding_kit_core(monkeypatch, log)
    note = "mk: every step fell back to the stock forward"
    log.settle_mk = lambda rep: {
        **rep,
        "levers_applied": ["atom_flash", "esmc_te"],
        "levers_fallback": ["mk"],
        "gated": [note],
    }
    log.settle_guards = lambda rep: {**rep, "levers_gated": ["dit"]}

    metadata = core.fold("ACD", options={"num_diffusion_samples": 5})

    assert len(builder.calls) == 1 and log.settles == ["settle_mk", "settle_guards"]
    runtime = metadata.runtime
    assert runtime["kit.levers_applied"] == "atom_flash,esmc_te" and runtime["kit.levers_fallback"] == "mk"
    assert runtime["kit.levers_gated"] == "dit" and runtime["kit.gated"] == note
    assert runtime["kit.scope"] == "mk: installed; inactive at num_diffusion_samples>1 (kit guard)"
    assert runtime["kit.attn.atom_attn"] == "flash_attn"
    assert core._runtime["kit.levers_applied"] == "atom_flash,esmc_te,mk"


@pytest.mark.parametrize(
    ("settled", "model_state", "match", "stands"),
    [
        ({"partial": ["ro"]}, FAST_ATTN, r"only part of its lever set after the fold: \['ro'\]", False),
        ({}, {**FAST_ATTN, "atom_attn": "sdpa"}, "after the fold: atom_attn=sdpa", True),
    ],
    ids=["unreached-lever", "attention-slow-since-the-load"],
)
def test_esmfold2_kit_fold_whose_levers_did_not_all_run_is_refused(
    monkeypatch: pytest.MonkeyPatch,
    offline_calls: list[int],
    kit_env: None,
    settled: dict,
    model_state: dict,
    match: str,
    stands: bool,
) -> None:
    """A lever no fold reached is partial, which the kit's CLI refuses with exit 3; so does the core.

    The attention words are read again on the model after the fold: ``status()`` still carries the load-time words,
    which a GPU run showed stay at ``flash_attn`` after the process's flash-attention flag is cleared.
    """
    from boileroom.optimization import OptimizationUnavailableError

    log = _fake_kit(monkeypatch, FILES)
    core, builder = _folding_kit_core(monkeypatch, log)
    log.settle_guards = lambda rep: {**rep, **settled}
    log.model_state = dict(model_state)

    with pytest.raises(OptimizationUnavailableError, match=match) as first:
        core.fold("ACD")

    # Lost attention paths belong to the process, so that refusal stands and a later fold never reaches the GPU. A
    # partial lever set depends on the call: a later fold that runs every lever is served.
    log.settle_guards = lambda rep: rep
    log.model_state = dict(FAST_ATTN)
    if stands:
        with pytest.raises(OptimizationUnavailableError) as second:
            core.fold("ACD")
        assert second.value is first.value and len(builder.calls) == 1
    else:
        core.fold("ACD")
        assert len(builder.calls) == 2


def test_esmfold2_kit_fold_whose_attention_cannot_be_read_is_refused(
    monkeypatch: pytest.MonkeyPatch, offline_calls: list[int], kit_env: None
) -> None:
    """An attention state the kit cannot read after the fold is a refusal, as it is before the weight fetch."""
    from boileroom.optimization import OptimizationUnavailableError

    log = _fake_kit(monkeypatch, FILES)
    core, builder = _folding_kit_core(monkeypatch, log)
    pregate_state = log.kit.attn.state

    def state(model: Any = None, **kwargs: Any) -> dict:
        if model is not None:
            raise ImportError("flash_attn_2_cuda: undefined symbol")
        return pregate_state(model, **kwargs)

    log.kit.attn.state = state

    with pytest.raises(OptimizationUnavailableError, match="cannot be read in this image after the fold") as raised:
        core.fold("ACD")
    assert isinstance(raised.value.__cause__, ImportError)
    log.kit.attn.state = pregate_state
    with pytest.raises(OptimizationUnavailableError) as again:
        core.fold("ACD")
    assert again.value is raised.value and len(builder.calls) == 1


T16_UNREACHED = (
    "no Transition / PairTransition call reached the t16 kernel (transition_calls + pair_transition_calls = 0)"
)


def _t16_unreached(rep: dict) -> dict:
    """What the kit's ``settle_guards`` returns when no Transition call was served by the t16 kernel."""
    return {
        **rep,
        "levers_applied": [lever for lever in rep.get("levers_applied") or [] if lever != "t16"],
        "levers_fallback": [*(rep.get("levers_fallback") or []), "t16"],
        "fallback_reasons": {"t16": T16_UNREACHED},
        "partial": ["t16"],
        "guards": {"t16": {"kind": "unreached", "note": T16_UNREACHED}},
    }


def _t16_module(monkeypatch: pytest.MonkeyPatch, **stats: int) -> None:
    """Install a stand-in ``ef2_transition_cute`` whose public counters are ``stats``."""
    refused = {"transition:cell_names_torch_swiglu": stats.get("fallthrough_transition", 0)}
    module = SimpleNamespace(STATS=dict(stats), describe=lambda: {"refused": refused})
    monkeypatch.setitem(sys.modules, "ef2_transition_cute", module)


def test_esmfold2_kit_t16_that_every_call_stepped_past_by_name_is_gated_not_refused(
    monkeypatch: pytest.MonkeyPatch, offline_calls: list[int], kit_env: None
) -> None:
    """Every Transition call reached t16 and fell through to the previous statements: recorded, not refused."""
    log = _fake_kit(monkeypatch, FILES)
    core, _ = _folding_kit_core(monkeypatch, log)
    _t16_module(monkeypatch, transition_calls=0, pair_transition_calls=0, fallthrough_transition=48, face_refused=48)
    log.settle_guards = _t16_unreached

    runtime = core.fold("ACD").runtime

    assert runtime["kit.partial"] == "false" and runtime["kit.levers_fallback"] == "none"
    assert runtime["kit.levers_gated"] == "t16"
    assert runtime["kit.gated"].startswith(
        "t16: all 48 Transition / PairTransition call(s) took the previous statements"
    )
    assert "transition:cell_names_torch_swiglu=48" in runtime["kit.gated"]
    assert runtime["kit.guards"].startswith("t16=inactive: all 48 ")


@pytest.mark.parametrize(
    "stats",
    [
        {"transition_calls": 0, "pair_transition_calls": 0, "fallthrough_transition": 0},
        None,
    ],
    ids=["no-call-reached-the-wrapper", "counters-unreadable"],
)
def test_esmfold2_kit_t16_without_evidence_it_ran_is_refused(
    monkeypatch: pytest.MonkeyPatch, offline_calls: list[int], kit_env: None, stats: dict | None
) -> None:
    """No call reached t16's wrapper (or its counters are gone): nothing shows the lever ran, so it stays partial."""
    from boileroom.optimization import OptimizationUnavailableError

    log = _fake_kit(monkeypatch, FILES)
    core, _ = _folding_kit_core(monkeypatch, log)
    if stats is None:
        monkeypatch.delitem(sys.modules, "ef2_transition_cute", raising=False)
    else:
        _t16_module(monkeypatch, **stats)
    log.settle_guards = _t16_unreached

    with pytest.raises(OptimizationUnavailableError, match=r"only part of its lever set after the fold: \['t16'\]"):
        core.fold("ACD")


def test_esmfold2_kit_guards_the_kit_settled_are_recorded(
    monkeypatch: pytest.MonkeyPatch, offline_calls: list[int], kit_env: None
) -> None:
    """A guard the kit itself settled as gated keeps its kind and note in ``kit.guards``; boileroom leaves it alone."""
    log = _fake_kit(monkeypatch, FILES)
    core, _ = _folding_kit_core(monkeypatch, log)
    _t16_module(monkeypatch, transition_calls=40, pair_transition_calls=0, fallthrough_transition=8)
    note = "8 call(s) outside the kernel's declared scope took the previous statements; 40 served"
    log.settle_guards = lambda rep: {
        **rep,
        "gated": [f"t16: {note}"],
        "guards": {"trimul": {"kind": "gated", "note": "2 served"}, "t16": {"kind": "gated", "note": note}},
    }

    runtime = core.fold("ACD").runtime

    assert runtime["kit.guards"] == f"t16=gated: {note}; trimul=gated: 2 served"
    assert runtime["kit.gated"] == f"t16: {note}" and runtime["kit.levers_gated"] == "none"


def test_esmfold2_kit_without_the_settle_step_is_refused(
    monkeypatch: pytest.MonkeyPatch, offline_calls: list[int], kit_env: None
) -> None:
    from boileroom.optimization import OptimizationUnavailableError

    log = _fake_kit(monkeypatch, FILES)
    core, _ = _folding_kit_core(monkeypatch, log)
    del log.kit.stack.settle_mk

    with pytest.raises(OptimizationUnavailableError, match="no stack.settle_mk"):
        core.fold("ACD")


@pytest.mark.parametrize("key", ["noise_scale", "step_scale", "max_inference_sigma"])
def test_kit_refuses_sampler_overrides(monkeypatch: pytest.MonkeyPatch, key: str) -> None:
    """The kit's fold() would not apply these, so they are refused at construction and per call, not dropped."""
    with pytest.raises(ValueError, match=key):
        _core_cls()(config={"device": "cpu", "optimization": "exact", key: 1.0})
    core = _core_cls()(config={"device": "cpu", "optimization": "fast", "model_name": "biohub/ESMFold2-Fast"})
    with pytest.raises(ValueError, match=key):
        core.fold("ACD", options={key: 1.0})
    assert core.model is None
    # Vanilla applies them.
    assert _core_cls()(config={"device": "cpu", key: 1.0})._sampler_kwargs(
        {**_core_cls().DEFAULT_CONFIG, key: 1.0}
    ) == {key: 1.0}


def test_kit_refuses_unbounded_msa_depth() -> None:
    """None is the checkpoint's depth under esm 3.4.1 but every row under the kit's esm 3.3.0, so kit modes refuse it."""
    with pytest.raises(ValueError, match="integer 'msa_max_depth'"):
        _core_cls()(config={"device": "cpu", "optimization": "exact", "msa_max_depth": None})
    core = _core_cls()(config={"device": "cpu", "optimization": "exact"})
    with pytest.raises(ValueError, match="integer 'msa_max_depth'"):
        core.fold("ACD", options={"msa_max_depth": None})
    _core_cls()(config={"device": "cpu", "msa_max_depth": None})


@pytest.mark.parametrize(
    ("config", "match"),
    [
        ({"revision": "abc"}, "config 'revision' does not apply to optimization='exact'"),
        ({"cache_dir": "/weights"}, "config 'cache_dir' does not apply"),
        ({"ccd_cache_dir": "/ccd"}, "config 'ccd_cache_dir' does not apply"),
        ({"model_name": "someone/ESMFold2-finetune"}, r"serves \['biohub/ESMFold2', 'biohub/ESMFold2-Fast'\]"),
    ],
    ids=["revision", "cache-dir", "ccd-cache-dir", "model-name"],
)
def test_kit_refuses_config_it_would_ignore_at_construction(config: dict, match: str) -> None:
    """The kit loads its own pinned snapshots: these are static, so they fail at construction, not at a GPU cold start."""
    with pytest.raises(ValueError, match=match):
        _core_cls()(config={"device": "cpu", "optimization": "exact", **config})
    # Vanilla honours every one of them.
    _core_cls()(config={"device": "cpu", **config})


@pytest.mark.parametrize("model_name", ["biohub/ESMFold2", "biohub/ESMFold2-Fast"])
def test_kit_accepts_the_checkpoints_it_serves(model_name: str) -> None:
    core = _core_cls()(config={"device": "cpu", "optimization": "fast", "model_name": model_name})
    assert core.config["model_name"] == model_name


def test_esmfold2_kit_stack_without_kernels_for_the_card_is_refused_before_the_kit_loads(
    monkeypatch: pytest.MonkeyPatch, kit_env: None
) -> None:
    """An sm_90-only kit image on an A100 must fail typed, before ~27 GB of weights, not at the first kernel launch."""
    from boileroom.optimization import OptimizationUnavailableError

    log = _fake_kit(monkeypatch, FILES)
    monkeypatch.setenv("BOILEROOM_KIT_STACK", "img_ef2_fa")
    core = _kit_core(monkeypatch)
    weights: list[Path] = []
    monkeypatch.setattr(core, "_ensure_kit_weights", weights.append)

    with pytest.raises(
        OptimizationUnavailableError,
        match=r"BOILEROOM_KIT_STACK=img_ef2_fa, which has kernels for sm_90 only, not NVIDIA A100-SXM4-80GB \(sm_80\)",
    ):
        core._activate_optimization()
    assert weights == [] and log.installs == [] and log.fetches == [] and log.enables == []
    assert log.env_at_pregate == []


@pytest.mark.parametrize(
    ("stack", "gpu"),
    [
        ("img_ef2_fa", GpuInfo(name="NVIDIA H100 80GB HBM3", capability=(9, 0))),
        ("img_esmfold2_a100", A100_GPU),
        ("img_esmfold2_a100", GpuInfo(name="NVIDIA H200", capability=(9, 0))),
        ("a-stack-built-elsewhere", A100_GPU),
        (None, A100_GPU),
    ],
    ids=["sm90-on-h100", "a100-stack-on-a100", "a100-stack-on-h200", "unknown-stack", "unset"],
)
def test_esmfold2_kit_stack_with_kernels_for_the_card_activates(
    monkeypatch: pytest.MonkeyPatch, offline_calls: list[int], kit_env: None, stack: str | None, gpu: GpuInfo
) -> None:
    """A healthy card for its stack is never refused; an unset or unknown stack is left to the kernel checks."""
    log = _fake_kit(monkeypatch, FILES)
    if stack is not None:
        monkeypatch.setenv("BOILEROOM_KIT_STACK", stack)
    core = _kit_core(monkeypatch)
    monkeypatch.setattr(pytest.importorskip("boileroom.models.esmfold2.core"), "detect_gpu", lambda device: gpu)
    monkeypatch.setattr(core, "_ensure_kit_weights", lambda hf_home: None)

    core._activate_optimization()

    assert log.enables and core.optimization is not None and core.optimization.mode == "exact"
    assert core.optimization.gpu_name == gpu.name


def test_esmfold2_kit_stack_table_matches_the_kit_build() -> None:
    """Every stack kit_wheels.sh can build is in the core's table, with the capabilities it compiles for."""
    core_module = pytest.importorskip("boileroom.models.esmfold2.core")
    script = (Path(core_module.__file__).parent / "kit" / "kit_wheels.sh").read_text()
    built = {
        stack: frozenset((int(arch.split(".")[0]), int(arch.split(".")[1])) for arch in archs.split(";"))
        for stack, archs in re.findall(r'^\s*(img_\w+)\)\s+TORCH_CUDA_ARCH_LIST="([0-9.;]+)"', script, re.MULTILINE)
    }
    assert built == core_module.KIT_STACK_CAPABILITIES


def test_esmfold2_kit_runtime_packages_extend_the_vanilla_ones() -> None:
    core_module = pytest.importorskip("boileroom.models.esmfold2.core")
    assert ("esmfold2-opt", *core_module.VANILLA_RUNTIME_PACKAGES) == core_module.KIT_RUNTIME_PACKAGES


@pytest.mark.parametrize(
    ("attributes", "env", "expected"),
    [
        ({"__commit__": "a" * 40}, "f" * 40, "a" * 40),
        ({"KIT_COMMIT": "b" * 40}, None, "b" * 40),
        ({"COMMIT": "", "__commit__": None}, "f" * 40, "f" * 40),
        ({}, None, "unknown"),
    ],
    ids=["attribute-over-env", "other-attribute", "empty-attributes-then-env", "nothing"],
)
def test_esmfold2_kit_commit_reads_the_package_then_the_environment(
    monkeypatch: pytest.MonkeyPatch,
    offline_calls: list[int],
    kit_env: None,
    attributes: dict,
    env: str | None,
    expected: str,
) -> None:
    """The same order as the Protenix runtime: an image built without the ENV still reports the kit's own commit."""
    log = _fake_kit(monkeypatch, FILES)
    for name, value in attributes.items():
        setattr(log.kit, name, value)
    if env is not None:
        monkeypatch.setenv("BOILEROOM_KIT_COMMIT", env)
    core = _armed_core(monkeypatch, log)

    core._configure_optimization()

    assert core._runtime["kit.commit"] == expected


def test_esmfold2_output_records_runtime_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    """metadata.runtime carries the loaded runtime's provenance, a copy per output."""
    core = _core_cls()(config={"device": "cpu"})
    _capture_fold(monkeypatch, core)
    core._runtime = {"esm": "3.4.1.post1", "esm.FLASH_ATTN_AVAILABLE": "False"}

    first = core.fold("ACD")
    first.runtime["esm"] = "edited"
    second = core.fold("ACD")

    assert second.runtime == {"esm": "3.4.1.post1", "esm.FLASH_ATTN_AVAILABLE": "False"}


def test_esmfold2_vanilla_load_records_its_kernel_switches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Vanilla never refuses a slow path, but every output says which esm kernels this process imported."""
    from boileroom.models.esmfold2 import loading

    core_module = pytest.importorskip("boileroom.models.esmfold2.core")

    class Model:
        def to(self, device: Any) -> "Model":
            return self

        def eval(self) -> None:
            return None

    loads: list[tuple[str, dict]] = []

    def load_pretrained(cls: Any, name: str, **kwargs: Any) -> Model:
        loads.append((name, kwargs))
        return Model()

    monkeypatch.setattr(loading, "load_pretrained", load_pretrained)
    monkeypatch.setitem(sys.modules, "esm", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "esm.models", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "esm.models.esmfold2",
        SimpleNamespace(EsmFold2Model=Model, ESMFold2InputBuilder=lambda ccd_cache: SimpleNamespace(ccd=ccd_cache)),
    )
    monkeypatch.setitem(
        sys.modules,
        "esm.models.esmfold2.layers",
        SimpleNamespace(FLASH_ATTN_AVAILABLE=False, CUE_AVAILABLE=True, TRITON_KERNELS_AVAILABLE=True),
    )
    monkeypatch.delitem(sys.modules, "esm.models.esmfold2.model", raising=False)
    monkeypatch.delitem(sys.modules, "esm.models.esmc.kernels", raising=False)
    monkeypatch.setattr(core_module, "describe_gpu", lambda device: A100_GPU)
    core = core_module.ESMFold2Core(
        config={"device": "cpu", "cache_dir": str(tmp_path), "ccd_cache_dir": str(tmp_path)}
    )
    monkeypatch.setattr(core, "_resolve_device", lambda: "cpu")
    monkeypatch.setattr(core, "_ensure_ccd_cache", lambda directory: directory)

    core._load()

    assert loads == [("biohub/ESMFold2", {"cache_dir": str(tmp_path), "revision": core_module.ESMFOLD2_HF_REVISION})]
    runtime = core._runtime
    assert runtime["weights"] == f"biohub/ESMFold2@{core_module.ESMFOLD2_HF_REVISION}"
    assert runtime["esm.FLASH_ATTN_AVAILABLE"] == "False" and runtime["esm.CUE_AVAILABLE"] == "True"
    assert runtime["esm.TE_AVAILABLE"] == "not-loaded" and runtime["esm.XFORMERS_INSTALLED"] == "not-loaded"
    assert runtime["gpu"] == "NVIDIA A100-SXM4-80GB"
    assert core.ready and not core._kernel_gate_passed


def _bond_input(*bonds: Any, protein: str = "ACDE") -> StructurePredictionInput:
    return StructurePredictionInput(
        sequences=[ProteinInput(id="A", sequence=protein), LigandInput(id=["L", "M"], ccd=["SAH", "HEM"])],
        covalent_bonds=list(bonds),
    )


def test_esmfold2_valid_covalent_bonds_pass() -> None:
    from boileroom.models.esmfold2.types import CovalentBond

    _core_cls()._check_covalent_bonds(
        _bond_input(CovalentBond("A", 3, 5, "L", 1, 0), CovalentBond("M", 0, 2, "A", 0, 0))
    )


@pytest.mark.parametrize(
    ("bond", "protein", "match"),
    [
        (("B", 0, 0, "L", 0, 0), "ACDE", r"chain_id 'B' does not exist; available chain ids: \['A', 'L', 'M'\]"),
        (("A", 4, 0, "L", 0, 0), "ACDE", r"residue index 4 is not in chain 'A' \(0-based, 0-3\)"),
        (("A", 0, 0, "L", 2, 0), "ACDE", r"residue index 2 is not in chain 'L'"),
        (("A", -1, 0, "L", 0, 0), "ACDE", "residue index -1"),
        (("A", True, 0, "L", 0, 0), "ACDE", "residue index True"),
        (("A", 0, -1, "L", 0, 0), "ACDE", "atom index -1 must be a non-negative integer"),
        (
            ("A", 0, 5, "L", 0, 0),
            "ACDE",
            r"atom index 5 is past residue 0 of chain 'A' \(protein A: 5 atoms, 0-based 0-4\)",
        ),
        (("L", 0, 0, "A", 3, 9), "ACDE", r"atom index 9 is past residue 3 of chain 'A' \(protein E: 9 atoms"),
        (("A", 0, 0, "L", 0, 0), "AC:DE", "cannot be combined with chainbreaks"),
    ],
    ids=[
        "unknown-chain",
        "residue-past-end",
        "ligand-residue",
        "negative-residue",
        "bool-residue",
        "negative-atom",
        "atom-past-residue",
        "atom-past-residue-second-end",
        "chainbreak",
    ],
)
def test_esmfold2_invalid_covalent_bonds_fail_before_loading(bond: tuple, protein: str, match: str) -> None:
    """esm 3.3.0 (the kit) silently drops these bonds; they must fail the same way on both runtimes, before any load."""
    from boileroom.models.esmfold2.types import CovalentBond

    for mode in ("vanilla", "exact"):
        core = _core_cls()(config={"device": "cpu", "optimization": mode})
        with pytest.raises(ValueError, match=match):
            core.fold(_bond_input(CovalentBond(*bond), protein=protein))
        assert core.model is None


def test_esmfold2_covalent_atom_bounds_follow_esm_heavy_atoms() -> None:
    """The last heavy atom of each residue passes; a modified residue, an unknown nucleotide and SMILES stay with esm."""
    from boileroom.models.esmfold2.types import CovalentBond, Modification, RNAInput

    check = _core_cls()._check_covalent_bonds
    protein = "ARNDCQEGHILKMFPSTWYVX"
    last_atoms = [4, 10, 7, 7, 5, 8, 8, 3, 9, 7, 7, 8, 7, 10, 6, 5, 6, 13, 11, 6, 3]
    sequences: list[ProteinInput | DNAInput | RNAInput | LigandInput] = [
        ProteinInput(id="A", sequence=protein, modifications=[Modification(position=0, ccd="SEP")]),
        DNAInput(id="D", sequence="ACGT"),
        RNAInput(id="R", sequence="ACGUN"),
        LigandInput(id="S", smiles="CCO"),
    ]
    bonds = [CovalentBond("A", index, atom, "S", 0, 50) for index, atom in enumerate(last_atoms) if index]
    bonds += [
        CovalentBond("A", 0, 30, "D", 0, 20),  # SEP replaces A: its atoms come from the CCD
        CovalentBond("D", 3, 19, "R", 3, 19),
        CovalentBond("R", 2, 22, "R", 4, 99),  # N is no standard nucleotide
    ]
    check(StructurePredictionInput(sequences=sequences, covalent_bonds=bonds))
    for bond, match in [
        (CovalentBond("D", 0, 21, "S", 0, 0), r"\(DNA A: 21 atoms"),
        (CovalentBond("S", 0, 0, "R", 1, 20), r"\(RNA C: 20 atoms"),
        (CovalentBond("A", 17, 14, "S", 0, 0), r"\(protein W: 14 atoms"),
    ]:
        with pytest.raises(ValueError, match=match):
            check(StructurePredictionInput(sequences=sequences, covalent_bonds=[bond]))


def test_esmfold2_residue_atom_tables_match_esm() -> None:
    """The static tables are esm's own heavy-atom lists, counted (runs where esm is installed)."""
    constants = pytest.importorskip("esm.models.esmfold2.constants")
    core_module = pytest.importorskip("boileroom.models.esmfold2.core")
    protein = {letter: len(constants.PROTEIN_HEAVY_ATOMS[name]) for letter, name in constants.PROTEIN_1TO3.items()}
    unknown = protein.pop("X")
    assert protein == core_module.PROTEIN_RESIDUE_ATOM_COUNTS
    assert unknown == core_module.UNKNOWN_PROTEIN_RESIDUE_ATOMS
    for one_to_three, atoms, table in [
        (constants.DNA_1TO3, constants.DNA_HEAVY_ATOMS, core_module.DNA_RESIDUE_ATOM_COUNTS),
        (constants.RNA_1TO3, constants.RNA_HEAVY_ATOMS, core_module.RNA_RESIDUE_ATOM_COUNTS),
    ]:
        assert {letter: len(atoms[name]) for letter, name in one_to_three.items()} == table


def _fake_ccd(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stand-in ``esm.models.esmfold2.conformers``: SAH has 26 atoms, one of them leaving; HEM is not in the CCD."""
    atoms = {"SAH": [(f"A{index}", "C", 0) for index in range(25)] + [("OXT", "O", 0)]}
    conformers = SimpleNamespace(
        get_ligand_ccd_atoms_with_charges=atoms.get,
        get_ccd_leaving_atoms=lambda code: {"OXT"} if code == "SAH" else set(),
    )
    monkeypatch.setitem(sys.modules, "esm.models.esmfold2.conformers", conformers)


class _ReachedTheBuilder(Exception):
    """Raised by a stand-in ``_fold_one``: the input got past every check to the esm builder."""


@pytest.mark.parametrize("mode", ["vanilla", "exact"])
def test_esmfold2_ccd_ligand_atom_past_its_component_fails_before_the_builder(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """A bond on a CCD ligand atom past the component's atoms (less its leaving atoms) fails in both modes alike."""
    from boileroom.models.esmfold2.types import CovalentBond

    _fake_ccd(monkeypatch)
    if mode == "exact":
        core, _ = _kit_fold_core(monkeypatch)
    else:
        core = _core_cls()(config={"device": "cpu"})
        core.model, core.input_builder = SimpleNamespace(), SimpleNamespace()
    reached: list[StructurePredictionInput] = []

    def fold_one(prediction_input: StructurePredictionInput, config: dict, request_index: int) -> tuple[list, dict]:
        reached.append(prediction_input)
        raise _ReachedTheBuilder

    monkeypatch.setattr(core, "_fold_one", fold_one)

    with pytest.raises(ValueError, match=r"atom index 25 is past residue 0 of chain 'L' \(CCD SAH: 25 atoms"):
        core.fold(_bond_input(CovalentBond("A", 0, 0, "L", 0, 25)))
    assert reached == []

    # The last kept atom, and a code the CCD lacks (esm refuses that itself), both reach the builder.
    with pytest.raises(_ReachedTheBuilder):
        core.fold(_bond_input(CovalentBond("A", 0, 0, "L", 0, 24), CovalentBond("M", 1, 99, "A", 1, 0)))
    assert len(reached) == 1


def test_esmfold2_go_offline_flips_already_imported_hub_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    """The weight fetch imports huggingface_hub online, so setting the env vars alone would come too late."""
    core_module = pytest.importorskip("boileroom.models.esmfold2.core")
    constants = SimpleNamespace(HF_HUB_OFFLINE=False)
    hub = SimpleNamespace(_is_offline_mode=False)
    monkeypatch.setitem(sys.modules, "huggingface_hub.constants", constants)
    monkeypatch.setitem(sys.modules, "transformers.utils.hub", hub)
    # setenv first so that monkeypatch restores the session's values even though _go_offline sets them directly.
    for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.setenv(var, "placeholder")
        monkeypatch.delenv(var)

    core_module.ESMFold2Core._go_offline()

    assert os.environ["HF_HUB_OFFLINE"] == os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert constants.HF_HUB_OFFLINE is True and hub._is_offline_mode is True


@pytest.mark.parametrize("model_name", ["biohub/ESMFold2", "biohub/ESMFold2-Fast"])
def test_esmfold2_kit_ccd_comes_from_the_pinned_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, model_name: str
) -> None:
    """Kit mode reads ccd.pkl beside the kit weights, never fetching from the hub (it is offline by then)."""
    _fake_kit(monkeypatch, FILES)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    core = _core_cls()(config={"device": "cpu", "optimization": "fast", "model_name": model_name})

    assert core._kit_ccd_dir() == tmp_path / "hub" / "models--biohub--ESMFold2" / "snapshots" / ("c" * 40)
