"""Fast ESMFold2 unit tests that do not import Biohub runtime dependencies."""

import os
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
    """The kit variant is fixed at construction: Fast never uses an MSA; full uses it only with kit_msa."""
    assert _core_cls()(config={"device": "cpu", **config})._kit_variant() == variant


@pytest.mark.parametrize("config", [{}, {"model_name": "biohub/ESMFold2-Fast", "kit_msa": True}])
def test_esmfold2_msa_with_kit_variant_that_ignores_it_fails_before_loading(config: dict) -> None:
    """A user MSA must never be silently dropped by a no-MSA kernel."""
    core = _core_cls()(config={"device": "cpu", "optimization": "exact", **config})
    with pytest.raises(ValueError, match="does not consume an MSA"):
        core.fold("ACD", options={"msa": [A3M]})
    assert core.model is None


def test_esmfold2_msa_is_allowed_with_vanilla_and_full_msa_kit() -> None:
    """Vanilla and the full_msa kit variant pass the kernel check."""
    _core_cls()(config={"device": "cpu"})._check_kit_consumes_msa({"optimization": "vanilla"})
    core = _core_cls()(config={"device": "cpu", "optimization": "exact", "kit_msa": True})
    core._check_kit_consumes_msa(core.config)


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


A100_GPU = SimpleNamespace(name="NVIDIA A100-SXM4-80GB", capability=(8, 0))


def _fake_kit(
    monkeypatch: pytest.MonkeyPatch,
    files: dict[str, list[str]],
    install_rc: int = 0,
    enable_report: dict | None = None,
) -> SimpleNamespace:
    """Install a stand-in ``esmfold2_opt`` whose pins name ``files`` (repo -> file names); returns its call log."""
    log = SimpleNamespace(installs=[], enables=[])
    pins = {
        "ccd": {"repo": "biohub/ESMFold2", "file": "ccd.pkl"},
        "weights": {
            repo: {"snapshot_commit": "c" * 40, "files": dict.fromkeys(names, {})} for repo, names in files.items()
        },
    }

    def pinned_weight_files(pins_: dict, variant: str | None) -> list[tuple[str, str, int]]:
        repos = {"biohub/ESMC-6B", "biohub/ESMFold2-Fast" if variant == "fast" else "biohub/ESMFold2"}
        return [
            (repo, f"hub/models--{repo.replace('/', '--')}/snapshots/{'c' * 40}/{name}", 1)
            for repo in sorted(repos & set(pins_["weights"]))
            for name in pins_["weights"][repo]["files"]
        ]

    def install_weights(hf_home: str, pins: dict | None = None) -> int:
        log.installs.append((hf_home, pins))
        return install_rc

    def enable(mode: str, variant: str | None = None) -> dict:
        log.enables.append((mode, variant))
        return enable_report or {"active": True}

    kit = SimpleNamespace(
        stack=SimpleNamespace(pins=lambda: pins, pinned_weight_files=pinned_weight_files),
        weights=SimpleNamespace(install_weights=install_weights),
        enable=enable,
    )
    monkeypatch.setitem(sys.modules, "esmfold2_opt", kit)
    return log


FILES = {
    "biohub/ESMC-6B": ["model.safetensors"],
    "biohub/ESMFold2": ["model.safetensors", "ccd.pkl"],
    "biohub/ESMFold2-Fast": ["model.safetensors"],
}


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


def test_esmfold2_kit_weights_for_the_fast_checkpoint_skip_the_full_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    log = _fake_kit(monkeypatch, FILES)
    core = _core_cls()(config={"device": "cpu", "optimization": "exact", "model_name": "biohub/ESMFold2-Fast"})

    core._ensure_kit_weights(tmp_path)

    weights = log.installs[0][1]["weights"]
    assert set(weights) == {"biohub/ESMC-6B", "biohub/ESMFold2-Fast", "biohub/ESMFold2"}
    # The Fast repository ships no ccd.pkl, so the full repository contributes that file alone.
    assert set(weights["biohub/ESMFold2"]["files"]) == {"ccd.pkl"}
    assert set(weights["biohub/ESMFold2-Fast"]["files"]) == {"model.safetensors"}


def test_esmfold2_kit_weights_already_in_place_are_not_fetched_again(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    log = _fake_kit(monkeypatch, FILES)
    core = _core_cls()(config={"device": "cpu", "optimization": "exact"})
    for repo in ("biohub/ESMC-6B", "biohub/ESMFold2"):
        for name in FILES[repo]:
            path = tmp_path / "hub" / f"models--{repo.replace('/', '--')}" / "snapshots" / ("c" * 40) / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"")

    core._ensure_kit_weights(tmp_path)

    assert log.installs == []


def test_esmfold2_kit_weights_that_fail_to_install_refuse_the_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from boileroom.optimization import OptimizationUnavailableError

    _fake_kit(monkeypatch, FILES, install_rc=1)
    core = _core_cls()(config={"device": "cpu", "optimization": "exact"})

    with pytest.raises(OptimizationUnavailableError, match="pinned ESMFold2 kit weights"):
        core._ensure_kit_weights(tmp_path)


def test_esmfold2_kit_home_defaults_to_the_model_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, offline_calls: list[int]
) -> None:
    """The kit's HF_HOME is set before the kit is imported, under the model volume so the weights persist."""
    core_module = pytest.importorskip("boileroom.models.esmfold2.core")
    log = _fake_kit(monkeypatch, FILES)
    monkeypatch.setenv("HF_HOME", "placeholder")
    monkeypatch.delenv("HF_HOME")
    monkeypatch.setattr(core_module, "detect_gpu", lambda device: A100_GPU)
    core = core_module.ESMFold2Core(config={"device": "cpu", "optimization": "exact"})
    core.model_dir = str(tmp_path)
    monkeypatch.setattr(core, "_ensure_kit_weights", lambda hf_home: log.installs.append(hf_home))

    core._activate_optimization()

    expected = tmp_path / core_module.KIT_HF_SUBDIR
    assert log.installs == [expected]
    assert log.enables == [("exact", "full_nomsa")]
    assert core.optimization is not None and core.optimization.active == "exact"
    assert os.environ["HF_HOME"] == str(expected)
    assert offline_calls == [1]  # the kit's snapshots are served from disk once they are fetched


def test_esmfold2_kit_home_respects_a_configured_hf_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, offline_calls: list[int]
) -> None:
    core_module = pytest.importorskip("boileroom.models.esmfold2.core")
    log = _fake_kit(monkeypatch, FILES)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "mine"))
    monkeypatch.setattr(core_module, "detect_gpu", lambda device: A100_GPU)
    core = core_module.ESMFold2Core(config={"device": "cpu", "optimization": "exact"})
    core.model_dir = str(tmp_path)
    monkeypatch.setattr(core, "_ensure_kit_weights", lambda hf_home: log.installs.append(hf_home))

    core._activate_optimization()

    assert log.installs == [tmp_path / "mine"]


def test_esmfold2_vanilla_never_touches_the_kit_or_hf_home(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HOME", "placeholder")
    monkeypatch.delenv("HF_HOME")
    monkeypatch.delitem(sys.modules, "esmfold2_opt", raising=False)
    core = _core_cls()(config={"device": "cpu"})

    core._activate_optimization()

    assert "HF_HOME" not in os.environ
    assert "esmfold2_opt" not in sys.modules


def test_esmfold2_kit_that_does_not_activate_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, offline_calls: list[int]
) -> None:
    from boileroom.optimization import OptimizationUnavailableError

    core_module = pytest.importorskip("boileroom.models.esmfold2.core")
    _fake_kit(monkeypatch, FILES, enable_report={"active": False, "reason": "no flash-attn"})
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.setattr(core_module, "detect_gpu", lambda device: A100_GPU)
    core = core_module.ESMFold2Core(config={"device": "cpu", "optimization": "exact"})
    monkeypatch.setattr(core, "_ensure_kit_weights", lambda hf_home: None)

    with pytest.raises(OptimizationUnavailableError, match="no flash-attn"):
        core._activate_optimization()


def test_esmfold2_go_offline_flips_already_imported_hub_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    """The weight fetch imports huggingface_hub online, so setting the env vars alone would come too late."""
    core_module = pytest.importorskip("boileroom.models.esmfold2.core")
    constants = SimpleNamespace(HF_HUB_OFFLINE=False)
    hub = SimpleNamespace(_is_offline_mode=False)
    monkeypatch.setitem(sys.modules, "huggingface_hub.constants", constants)
    monkeypatch.setitem(sys.modules, "transformers.utils.hub", hub)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    core_module.ESMFold2Core._go_offline()

    assert os.environ["HF_HUB_OFFLINE"] == os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert constants.HF_HUB_OFFLINE is True and hub._is_offline_mode is True
    monkeypatch.delenv("HF_HUB_OFFLINE")
    monkeypatch.delenv("TRANSFORMERS_OFFLINE")


@pytest.mark.parametrize("model_name", ["biohub/ESMFold2", "biohub/ESMFold2-Fast"])
def test_esmfold2_kit_ccd_comes_from_the_pinned_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, model_name: str
) -> None:
    """Kit mode reads ccd.pkl beside the kit weights, never fetching from the hub (it is offline by then)."""
    _fake_kit(monkeypatch, FILES)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    core = _core_cls()(config={"device": "cpu", "optimization": "exact", "model_name": model_name})

    assert core._kit_ccd_dir() == tmp_path / "hub" / "models--biohub--ESMFold2" / "snapshots" / ("c" * 40)
