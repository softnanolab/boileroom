"""Fast ESMFold2 unit tests that do not import Biohub runtime dependencies."""

import pytest

from boileroom.base import ModelWrapper
from boileroom.models.esmfold2.esmfold2 import ESMFold2
from boileroom.models.esmfold2.core import ESMFold2Core
from boileroom.models.esmfold2.payloads import decode_structure_input, encode_fold_input
from boileroom.models.esmfold2.types import DNAInput, LigandInput, ProteinInput, StructurePredictionInput

pytestmark = pytest.mark.contract


def test_esmfold2_string_multimer_becomes_one_complex() -> None:
    """Colon-separated protein strings should represent one multichain complex."""
    core = ESMFold2Core(config={"device": "cpu"})

    requests = core._coerce_requests("ACD:EFG")

    assert len(requests) == 1
    assert requests[0].sequence_length == 6
    chains = requests[0].input.sequences
    assert chains == [ProteinInput(id="A", sequence="ACD"), ProteinInput(id="B", sequence="EFG")]


def test_esmfold2_string_list_is_batch() -> None:
    """A list of strings should remain a batch of independent proteins."""
    core = ESMFold2Core(config={"device": "cpu"})

    requests = core._coerce_requests(["ACD", "EFGH"])

    assert [request.sequence_length for request in requests] == [3, 4]
    assert [request.input.sequences[0] for request in requests] == [
        ProteinInput(id="A", sequence="ACD"),
        ProteinInput(id="A", sequence="EFGH"),
    ]


def test_esmfold2_molecule_inputs_become_one_structure_input() -> None:
    """A list of molecule input dataclasses should describe one all-atom complex."""
    core = ESMFold2Core(config={"device": "cpu"})
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
    payload = encode_fold_input([ProteinInput(id="A", sequence="ACD"), LigandInput(id="L", ccd=["SAH"])])

    assert isinstance(payload, dict)
    decoded = decode_structure_input(payload)

    assert list(decoded.sequences) == [ProteinInput(id="A", sequence="ACD"), LigandInput(id="L", ccd=["SAH"])]


def test_esmfold2_core_accepts_encoded_structure_payload() -> None:
    """The core should accept the JSON payload shape used by Apptainer."""
    core = ESMFold2Core(config={"device": "cpu"})
    payload = encode_fold_input([ProteinInput(id="A", sequence="ACD"), DNAInput(id="B", sequence="GATA")])

    requests = core._coerce_requests(payload)

    assert len(requests) == 1
    assert requests[0].sequence_length == 7


def test_esmfold2_apptainer_wrapper_encodes_rich_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    """The public wrapper should keep rich input handling out of the shared Apptainer transport."""
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
    core = ESMFold2Core(config={"device": "cpu"})

    with pytest.raises(ValueError, match="empty chain"):
        core._coerce_requests("A::B")
