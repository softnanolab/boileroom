"""Fast ESMFold2 unit tests that do not import Biohub runtime dependencies."""

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
