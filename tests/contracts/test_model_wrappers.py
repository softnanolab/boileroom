"""Fast contract tests for public model wrappers."""

from typing import Any

import numpy as np
import pytest

from boileroom.backend.modal import ModalBackend, modal_app_of
from boileroom.base import ModelWrapper, PredictionMetadata
from boileroom.images.metadata import get_default_image_tag
from boileroom.models.boltz.types import Boltz2Output
from boileroom.models.chai.types import Chai1Output
from boileroom.models.esm.types import ESM2Output, ESMFoldOutput
from boileroom.models.registry import CHAI1_SPEC, ESM2_SPEC, MODEL_SPECS, ModelSpec, get_model_spec, resolve_object

pytestmark = pytest.mark.contract

SAMPLE_INPUTS: dict[str, Any] = {
    "esmfold": "MLKNVHVLVLGAGDVGSVVVRLLEK",
    "esm2": ["MALWMRLLPLLALLALWGPDPAAA"],
    "chai1": (
        "ICLQKTSNQILKPKLISYTLGQSGTCITDPLLAMDEGYFAYSHLERIGSCSRGVSKQRIIGVGEVLDRGDEVPSLFMTNVWTPPNPNTVYHCSAVYNNEFYYVLCAVSTVGD"
        "PILNSTYWSGSLMMTRLAVKPKSNGGGYNQHQLALRSIEKGRYDKVMPYGPSGIKQGDTLYFPAVGFLVRTEFKYNDSNCPITKCQYSKPENCRLSMGIRPNSHYILRSGLLKYN"
        "LSDGENPKVVFIEISDQRLSIGSPSKIYDSLGQPVFYQASFSWDTMIKFGDVLTVNPLVVNWRNNTVISRPGQSQCPRFNTCPEICWEGVYNDAFLIDRINWISAGVFLDSNQTAE"
        "NPVFTVFKDNEILYRAQLASEDTNAQKTITNCFLLKNKIWCISLVEIYDTGDNVIRPKLFAVKIPEQCTH"
    ),
    "boltz2": "MLKNVHVLVLGAGDVGSVVVRLLEK",
}
EXPECTED_MODAL_APP_NAMES = {spec.key: f"boileroom-{spec.key}" for spec in MODEL_SPECS}


def _make_metadata(model_name: str) -> PredictionMetadata:
    return PredictionMetadata(model_name=model_name, model_version="test", sequence_lengths=[3])


def _make_output(spec: ModelSpec) -> object:
    if spec.key == "esm2":
        return ESM2Output(
            embeddings=np.zeros((1, 3, 8), dtype=np.float32),
            metadata=_make_metadata(spec.public_name),
            chain_index=np.zeros((1, 3), dtype=np.int64),
            residue_index=np.arange(3, dtype=np.int64)[None, :],
            hidden_states=None,
            lm_logits=None,
        )

    if spec.key == "esmfold":
        return ESMFoldOutput(metadata=_make_metadata(spec.public_name), atom_array=[object()])

    if spec.key == "chai1":
        return Chai1Output(metadata=_make_metadata(spec.public_name), atom_array=[object()])

    if spec.key == "boltz2":
        return Boltz2Output(metadata=_make_metadata(spec.public_name), atom_array=[object()])

    raise KeyError(f"Unsupported contract spec: {spec.key}")


class _FakeModel:
    def __init__(self, method_name: str, result: object, records: dict[str, Any]) -> None:
        self._method_name = method_name
        self._result = result
        self._records = records

    def fold(self, *args: Any, **kwargs: Any) -> object:
        self._records["method"] = "fold"
        self._records["call_args"] = args
        self._records["call_kwargs"] = kwargs
        return self._result

    def embed(self, *args: Any, **kwargs: Any) -> object:
        self._records["method"] = "embed"
        self._records["call_args"] = args
        self._records["call_kwargs"] = kwargs
        return self._result


class _FakeBackend:
    def __init__(self, model_spec: ModelSpec, result: object, records: dict[str, Any]) -> None:
        self._model_spec = model_spec
        self._result = result
        self._records = records
        self._model = _FakeModel(model_spec.contract.task_method, result, records)

    def start(self) -> None:
        self._records["started"] = True

    def stop(self) -> None:
        self._records["stopped"] = True

    def get_model(self) -> _FakeModel:
        return self._model


def _install_fake_initializer(monkeypatch: pytest.MonkeyPatch, records: dict[str, Any], result: object) -> None:
    def fake_initialize(
        self: ModelWrapper,
        model_spec: ModelSpec,
        backend: str | None = None,
        device: str | None = None,
        config: dict | None = None,
    ) -> None:
        records["model_spec"] = model_spec
        records["backend"] = backend
        records["device"] = device
        records["config"] = config
        self.backend = backend or model_spec.default_backend
        self.device = device
        self.config = config or {}
        self.model_spec = model_spec
        self._backend = _FakeBackend(model_spec, result, records)
        self._backend.start()

    monkeypatch.setattr(ModelWrapper, "_initialize_backend_from_spec", fake_initialize)


def test_model_registry_entries_are_resolvable() -> None:
    """Every registered spec should resolve to the declared public wrapper class."""
    for spec in MODEL_SPECS:
        assert get_model_spec(spec.key) is spec
        assert get_model_spec(spec.public_name) is spec
        assert resolve_object(spec.wrapper_class_path).__name__ == spec.public_name
        if spec.modal_class_path is not None:
            assert resolve_object(spec.modal_class_path).__name__ == spec.modal_class_path.rsplit(".", 1)[-1]


def test_modal_classes_are_registered_on_model_specific_apps() -> None:
    """Modal GPU functions should not all register on one shared app."""
    app_names: list[str] = []

    for spec in MODEL_SPECS:
        modal_class_path = spec.modal_class_path
        if modal_class_path is None:
            continue
        modal_cls = resolve_object(modal_class_path)
        app_name = modal_app_of(modal_cls).name
        app_names.append(app_name)
        assert app_name == EXPECTED_MODAL_APP_NAMES[spec.key]

    assert len(app_names) == len(set(app_names))


@pytest.mark.parametrize("spec", MODEL_SPECS, ids=lambda spec: spec.public_name)
def test_modal_backend_uses_selected_class_app(spec: ModelSpec) -> None:
    """ModalBackend should run the app owned by the concrete Modal class."""
    modal_class_path = spec.modal_class_path
    if modal_class_path is None:
        pytest.skip("Model has no Modal backend class.")
    assert modal_class_path is not None
    modal_cls = resolve_object(modal_class_path)

    backend = ModalBackend(modal_cls, config={}, device=None)

    assert backend._app is modal_app_of(modal_cls)


def test_modal_app_of_rejects_undecorated_class() -> None:
    """Modal app lookup should reject classes not registered on a Modal app."""

    class NotAModalClass:
        pass

    with pytest.raises(TypeError, match="Modal-decorated class"):
        modal_app_of(NotAModalClass)

    with pytest.raises(TypeError, match="Modal-decorated class"):
        ModalBackend(NotAModalClass, config={}, device=None)


@pytest.mark.parametrize("spec", MODEL_SPECS, ids=lambda spec: spec.public_name)
def test_public_wrapper_exposes_registry_spec(spec: ModelSpec) -> None:
    """Each public wrapper should expose the same ModelSpec object as the registry."""
    wrapper_cls = resolve_object(spec.wrapper_class_path)
    assert wrapper_cls.MODEL_SPEC is spec
    assert wrapper_cls.MODEL_SPEC.contract.task_method == spec.contract.task_method
    assert wrapper_cls.MODEL_SPEC.contract.minimal_output_fields == spec.contract.minimal_output_fields


@pytest.mark.parametrize("spec", MODEL_SPECS, ids=lambda spec: spec.public_name)
def test_public_wrapper_dispatch_uses_shared_initializer(monkeypatch: pytest.MonkeyPatch, spec: ModelSpec) -> None:
    """Public wrappers should delegate backend construction to the shared registry-driven initializer."""
    wrapper_cls = resolve_object(spec.wrapper_class_path)
    output = _make_output(spec)
    records: dict[str, Any] = {}
    _install_fake_initializer(monkeypatch, records, output)

    wrapper = wrapper_cls(backend="apptainer:dev", device="cuda:0", config={"include_fields": ["*"]})

    assert records["model_spec"] is spec
    assert records["backend"] == "apptainer:dev"
    assert records["device"] == "cuda:0"
    assert records["config"] == {"include_fields": ["*"]}
    assert wrapper.model_spec is spec
    assert wrapper.backend == "apptainer:dev"
    assert wrapper.device == "cuda:0"
    assert wrapper.config == {"include_fields": ["*"]}
    assert records["started"] is True

    call_input = SAMPLE_INPUTS[spec.key]
    result = getattr(wrapper, spec.contract.task_method)(call_input)
    assert result is output
    assert records["method"] == spec.contract.task_method
    assert records["call_args"] == (call_input,)
    assert records["call_kwargs"] == {"options": None}


def test_parse_backend_apptainer_tag_handling() -> None:
    """Apptainer backend tags should keep the explicit tag or default to the package version."""
    assert ModelWrapper.parse_backend("modal") == ("modal", None)
    assert ModelWrapper.parse_backend("modal:dev") == ("modal", None)
    assert ModelWrapper.parse_backend("apptainer") == ("apptainer", get_default_image_tag())
    assert ModelWrapper.parse_backend("apptainer:dev") == ("apptainer", "dev")


def test_chai1_contract_declares_single_input_only() -> None:
    """Chai1 should advertise that it does not support top-level batching."""
    assert CHAI1_SPEC.contract.supports_batch is False


def test_esm2_contract_declares_lm_logits_optional_field() -> None:
    """ESM2 should advertise lm_logits as an optional output field."""
    assert ESM2_SPEC.contract.optional_output_fields == ("hidden_states", "lm_logits")


def test_chai1_wrapper_rejects_multiple_top_level_sequences(monkeypatch: pytest.MonkeyPatch) -> None:
    """Chai1 should fail early with a clear error when multiple top-level sequences are provided."""
    records: dict[str, Any] = {}
    _install_fake_initializer(monkeypatch, records, _make_output(CHAI1_SPEC))

    wrapper_cls = resolve_object(CHAI1_SPEC.wrapper_class_path)
    wrapper = wrapper_cls(backend="apptainer:dev")

    with pytest.raises(ValueError, match="exactly one top-level sequence"):
        wrapper.fold(["AAAA", "BBBB"])

    assert "method" not in records
