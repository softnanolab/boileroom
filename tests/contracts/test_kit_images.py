"""Contract tests for the optimization-kit images (ESMFold2 and Protenix ``exact`` / ``fast``)."""

import re
from pathlib import Path
from typing import Any

import pytest

from boileroom.backend.modal import modal_app_of
from boileroom.base import ModelWrapper
from boileroom.images.metadata import (
    DEFAULT_PYTHON_VERSION,
    KIT_COMMIT,
    KIT_IMAGE_SOURCE_ENV,
    KIT_IMAGE_SPECS,
    MODEL_IMAGE_SPECS,
    format_image_reference,
    get_kit_image_source,
    get_kit_image_spec,
    render_modal_runtime_env,
)
from boileroom.models.registry import (
    ESMFOLD2_SPEC,
    MODEL_SPECS,
    OPENDDE_SPEC,
    PROTENIX_SPEC,
    RF3_SPEC,
    ModelSpec,
    resolve_object,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
KIT_SPECS = [ESMFOLD2_SPEC, PROTENIX_SPEC, RF3_SPEC]


def _install_fake_backends(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace both backends so that ``_initialize_backend_from_spec`` records its choice and starts nothing."""
    import boileroom.backend as backend_package
    import boileroom.backend.apptainer as apptainer_module

    records: dict[str, Any] = {}

    class FakeModal:
        def __init__(self, model_cls: Any, config: dict | None = None, device: str | None = None) -> None:
            records["modal_cls"] = model_cls

        def start(self) -> None:
            records["started"] = True

    class FakeApptainer:
        def __init__(self, core_class_path: str, image_uri: str, config: dict | None = None, **kwargs: Any) -> None:
            records["image_uri"] = image_uri
            records["kwargs"] = kwargs

        def start(self) -> None:
            records["started"] = True

    monkeypatch.setattr(backend_package, "ModalBackend", FakeModal)
    monkeypatch.setattr(apptainer_module, "ApptainerBackend", FakeApptainer)
    return records


def _initialize(spec: ModelSpec, backend: str, config: dict | None) -> None:
    ModelWrapper()._initialize_backend_from_spec(spec, backend=backend, config=config)


def test_kit_images_are_not_part_of_the_published_image_set() -> None:
    """Nothing publishes the kit images: they stay out of the CI matrix and the tag tooling."""
    stock_names = {spec.image_name for spec in MODEL_IMAGE_SPECS}
    stock_dockerfiles = {spec.dockerfile_relative_path for spec in MODEL_IMAGE_SPECS}
    for spec in KIT_IMAGE_SPECS:
        assert spec.image_name not in stock_names
        assert spec.dockerfile_relative_path not in stock_dockerfiles
        assert spec.image_name.endswith("-kit")


@pytest.mark.parametrize("spec", KIT_IMAGE_SPECS, ids=lambda spec: spec.key)
def test_kit_image_files_exist(spec: Any) -> None:
    assert spec.dockerfile_path.is_file()
    assert (REPO_ROOT / spec.config_relative_path).is_file()
    assert spec.context_path.is_dir()
    assert spec.dockerfile_path.is_relative_to(spec.context_path)
    assert get_kit_image_spec(spec.key) is spec
    assert get_kit_image_spec(spec.image_name) is spec


def test_unknown_kit_image_is_rejected() -> None:
    with pytest.raises(KeyError, match="Unknown kit image spec"):
        get_kit_image_spec("boltz2")


@pytest.mark.parametrize("spec", KIT_IMAGE_SPECS, ids=lambda spec: spec.key)
def test_kit_dockerfile_pins_the_commit_of_the_metadata(spec: Any) -> None:
    """The Dockerfile and the metadata name the same kit commit, and it is a full SHA."""
    match = re.search(r"^ARG KIT_COMMIT=(\S+)$", spec.dockerfile_path.read_text(), flags=re.MULTILINE)
    assert match is not None
    assert match.group(1) == KIT_COMMIT
    assert re.fullmatch(r"[0-9a-f]{40}", KIT_COMMIT)


def test_opendde_dockerfile_pins_the_same_kit_commit_and_verifies_it() -> None:
    """OpenDDE bakes the kit into its own image, so its Dockerfile is held to the same pin."""
    text = (REPO_ROOT / "boileroom/models/opendde/Dockerfile").read_text()
    assert re.search(rf"^ARG KIT_COMMIT={KIT_COMMIT}$", text, flags=re.MULTILINE)
    assert 'rev-parse HEAD)" = "${KIT_COMMIT}"' in text


@pytest.mark.parametrize("spec", KIT_IMAGE_SPECS, ids=lambda spec: spec.key)
def test_kit_dockerfile_uses_only_what_modal_parses(spec: Any) -> None:
    """Modal's Dockerfile parser rejects BuildKit mounts and COPY globs that match nothing."""
    instructions = [line for line in spec.dockerfile_path.read_text().splitlines() if not line.lstrip().startswith("#")]
    assert not any(line.startswith("RUN --mount") for line in instructions)
    for line in instructions:
        if line.startswith(("COPY ", "ADD ")):
            assert not re.search(r"[*?\[]", line), f"glob in {line!r}"
            assert "_jitcache" not in line


def test_esmfold2_kit_dockerfile_defers_or_builds_the_wheels() -> None:
    """The same Dockerfile serves ``docker build`` (compiles inline) and Modal (compiles in a sized build step)."""
    text = get_kit_image_spec("esmfold2").dockerfile_path.read_text()
    assert "ARG WHEELS_FROM=build" in text
    assert re.search(r'case "\$WHEELS_FROM" in\s*\\\s*build\) sh /opt/boileroom-kit/kit_wheels\.sh', text)
    assert re.search(r"\sdefer\) ", text)
    assert "kit_wheels.sh" in text and "kit_finish.sh" in text


def test_protenix_kit_runs_python_311_and_the_fast_layernorm() -> None:
    spec = get_kit_image_spec("protenix")
    assert spec.python_version == "3.11"
    assert "python-build-standalone" in spec.dockerfile_path.read_text()
    assert dict(spec.modal_runtime_env)["LAYERNORM_TYPE"] == "fast_layernorm"
    assert "LAYERNORM_TYPE=fast_layernorm" in spec.dockerfile_path.read_text()
    assert get_kit_image_spec("esmfold2").python_version == DEFAULT_PYTHON_VERSION


def test_kit_runtime_env_points_at_the_model_directory() -> None:
    env = dict(render_modal_runtime_env(get_kit_image_spec("protenix"), "/mnt/models"))
    assert env["MODEL_DIR"] == "/mnt/models"
    assert env["LAYERNORM_TYPE"] == "fast_layernorm"


def test_kit_image_source_defaults_to_building_from_the_dockerfile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(KIT_IMAGE_SOURCE_ENV, raising=False)
    assert get_kit_image_source() == "build"
    monkeypatch.setenv(KIT_IMAGE_SOURCE_ENV, " Registry ")
    assert get_kit_image_source() == "registry"


def test_kit_image_source_rejects_unknown_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(KIT_IMAGE_SOURCE_ENV, "docker")
    with pytest.raises(ValueError, match=KIT_IMAGE_SOURCE_ENV):
        get_kit_image_source()


@pytest.mark.parametrize("spec", KIT_SPECS, ids=lambda spec: spec.key)
def test_modal_kit_classes_live_on_their_own_app(spec: ModelSpec) -> None:
    """The kit image must not be built when only the vanilla class runs, so it needs its own app."""
    assert spec.kit_modal_class_path is not None
    assert spec.modal_class_path is not None
    kit_app = modal_app_of(resolve_object(spec.kit_modal_class_path))
    vanilla_app = modal_app_of(resolve_object(spec.modal_class_path))
    assert kit_app is not vanilla_app
    assert kit_app.name == f"boileroom-{spec.key}-kit"


@pytest.mark.parametrize("spec", KIT_SPECS, ids=lambda spec: spec.key)
def test_modal_kit_class_serves_the_same_methods_as_the_vanilla_class(spec: ModelSpec) -> None:
    """A wrapper calls the same remote methods whichever class it runs."""
    assert spec.kit_modal_class_path is not None
    assert spec.modal_class_path is not None
    kit_cls = resolve_object(spec.kit_modal_class_path)
    vanilla_cls = resolve_object(spec.modal_class_path)
    kit_methods = set(kit_cls._get_partial_functions())
    assert spec.contract.task_method in kit_methods
    assert kit_methods == set(vanilla_cls._get_partial_functions())


def test_only_families_with_a_kit_image_declare_one() -> None:
    for spec in MODEL_SPECS:
        has_kit = spec in KIT_SPECS
        assert (spec.kit_image_key is not None) is has_kit
        assert (spec.kit_modal_class_path is not None) is has_kit
    # OpenDDE ships its kit inside its own image.
    assert OPENDDE_SPEC.kit_image_key is None


@pytest.mark.parametrize("spec", KIT_SPECS, ids=lambda spec: spec.key)
def test_vanilla_runs_on_the_stock_modal_class(monkeypatch: pytest.MonkeyPatch, spec: ModelSpec) -> None:
    records = _install_fake_backends(monkeypatch)
    for config in (None, {}, {"optimization": "vanilla"}):
        _initialize(spec, "modal", config)
        assert records["modal_cls"] is resolve_object(spec.modal_class_path)  # type: ignore[arg-type]


@pytest.mark.parametrize("spec", KIT_SPECS, ids=lambda spec: spec.key)
@pytest.mark.parametrize("mode", ["exact", "fast"])
def test_kit_modes_run_on_the_kit_modal_class(monkeypatch: pytest.MonkeyPatch, spec: ModelSpec, mode: str) -> None:
    records = _install_fake_backends(monkeypatch)
    _initialize(spec, "modal", {"optimization": mode})
    assert records["modal_cls"] is resolve_object(spec.kit_modal_class_path)  # type: ignore[arg-type]
    assert records["started"] is True


@pytest.mark.parametrize("spec", KIT_SPECS, ids=lambda spec: spec.key)
def test_unknown_optimization_is_refused_before_a_backend_starts(
    monkeypatch: pytest.MonkeyPatch, spec: ModelSpec
) -> None:
    records = _install_fake_backends(monkeypatch)
    with pytest.raises(ValueError, match="optimization must be one of"):
        _initialize(spec, "modal", {"optimization": "turbo"})
    assert "started" not in records


@pytest.mark.parametrize("spec", KIT_SPECS, ids=lambda spec: spec.key)
def test_vanilla_apptainer_keeps_the_stock_image_and_interpreter(
    monkeypatch: pytest.MonkeyPatch, spec: ModelSpec
) -> None:
    records = _install_fake_backends(monkeypatch)
    _initialize(spec, "apptainer:dev", {"optimization": "vanilla"})
    assert spec.apptainer_image_name is not None
    assert records["image_uri"] == f"docker://{format_image_reference(spec.apptainer_image_name, 'dev')}"
    assert records["kwargs"]["python_version"] == DEFAULT_PYTHON_VERSION


@pytest.mark.parametrize("spec", KIT_SPECS, ids=lambda spec: spec.key)
def test_kit_apptainer_uses_the_kit_image_and_its_interpreter(monkeypatch: pytest.MonkeyPatch, spec: ModelSpec) -> None:
    records = _install_fake_backends(monkeypatch)
    kit_spec = get_kit_image_spec(spec.key)
    _initialize(spec, "apptainer:dev", {"optimization": "fast"})
    assert records["image_uri"] == f"docker://{format_image_reference(kit_spec.image_name, 'dev')}"
    assert records["kwargs"]["python_version"] == kit_spec.python_version


@pytest.mark.parametrize("spec", KIT_SPECS, ids=lambda spec: spec.key)
def test_kit_apptainer_without_a_published_image_is_refused(monkeypatch: pytest.MonkeyPatch, spec: ModelSpec) -> None:
    """No kit image is published, so a bare ``apptainer`` backend would fail late on a registry pull."""
    records = _install_fake_backends(monkeypatch)
    monkeypatch.delenv(KIT_IMAGE_SOURCE_ENV, raising=False)
    with pytest.raises(ValueError, match="none is published"):
        _initialize(spec, "apptainer", {"optimization": "exact"})
    assert "started" not in records
    monkeypatch.setenv(KIT_IMAGE_SOURCE_ENV, "registry")
    _initialize(spec, "apptainer", {"optimization": "exact"})
    assert records["started"] is True


def test_opendde_checks_optimization_in_the_caller_and_keeps_its_one_image(monkeypatch: pytest.MonkeyPatch) -> None:
    records = _install_fake_backends(monkeypatch)
    with pytest.raises(ValueError, match="optimization must be one of"):
        _initialize(OPENDDE_SPEC, "modal", {"optimization": "turbo"})
    assert "started" not in records
    _initialize(OPENDDE_SPEC, "modal", {"optimization": "fast"})
    assert records["modal_cls"] is resolve_object(OPENDDE_SPEC.modal_class_path)  # type: ignore[arg-type]


def test_apptainer_service_runs_under_the_image_interpreter() -> None:
    from boileroom.backend.apptainer import _build_ld_library_path

    assert "/usr/local/lib/python3.12/site-packages/nvidia" in _build_ld_library_path()
    assert "/usr/local/lib/python3.11/site-packages/nvidia" in _build_ld_library_path("3.11")
    assert "python3.12" not in _build_ld_library_path("3.11")


class _FakeImage:
    """Records the calls ``boileroom.images.modal`` makes on Modal's ``Image``."""

    calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    @classmethod
    def from_registry(cls, *args: Any, **kwargs: Any) -> "_FakeImage":
        cls.calls.append(("from_registry", args, kwargs))
        return cls()

    @classmethod
    def from_dockerfile(cls, *args: Any, **kwargs: Any) -> "_FakeImage":
        cls.calls.append(("from_dockerfile", args, kwargs))
        return cls()

    def run_function(self, *args: Any, **kwargs: Any) -> "_FakeImage":
        self.calls.append(("run_function", args, kwargs))
        return self

    def env(self, *args: Any, **kwargs: Any) -> "_FakeImage":
        self.calls.append(("env", args, kwargs))
        return self


@pytest.fixture
def fake_image(monkeypatch: pytest.MonkeyPatch) -> type[_FakeImage]:
    import boileroom.images.modal as images_modal

    _FakeImage.calls = []
    monkeypatch.setattr(images_modal, "Image", _FakeImage)
    monkeypatch.delenv(KIT_IMAGE_SOURCE_ENV, raising=False)
    return _FakeImage


def test_esmfold2_kit_image_builds_from_the_dockerfile_with_a_sized_compile_step(
    fake_image: type[_FakeImage],
) -> None:
    from boileroom.images import modal as images_modal

    images_modal.get_modal_kit_image("esmfold2")

    names = [name for name, _, _ in fake_image.calls]
    assert names == ["from_dockerfile", "run_function", "env"]
    _, args, kwargs = fake_image.calls[0]
    assert args[0] == get_kit_image_spec("esmfold2").dockerfile_path
    assert kwargs["build_args"] == {"STACK": images_modal.ESMFOLD2_KIT_STACK, "WHEELS_FROM": "defer"}
    _, args, kwargs = fake_image.calls[1]
    assert args == (images_modal.compile_esmfold2_kit,)
    assert kwargs["cpu"] == images_modal.ESMFOLD2_KIT_COMPILE_CPU
    assert kwargs["memory"] == images_modal.ESMFOLD2_KIT_COMPILE_MEMORY_MIB
    assert kwargs["env"]["BUILD_JOBS"] == str(images_modal.ESMFOLD2_KIT_COMPILE_CPU)
    env = dict(fake_image.calls[2][1][0])
    assert env["MODEL_DIR"]


def test_protenix_kit_image_is_the_dockerfile_alone(fake_image: type[_FakeImage]) -> None:
    from boileroom.images import modal as images_modal

    images_modal.get_modal_kit_image("protenix")

    assert [name for name, _, _ in fake_image.calls] == ["from_dockerfile", "env"]
    assert dict(fake_image.calls[1][1][0])["LAYERNORM_TYPE"] == "fast_layernorm"


@pytest.mark.parametrize("family", ["esmfold2", "protenix"])
def test_kit_image_from_a_registry_pulls_the_named_image(
    monkeypatch: pytest.MonkeyPatch, fake_image: type[_FakeImage], family: str
) -> None:
    from boileroom.images import modal as images_modal

    monkeypatch.setenv(KIT_IMAGE_SOURCE_ENV, "registry")
    monkeypatch.setenv("BOILEROOM_IMAGE_TAG", "mytag")
    images_modal.get_modal_kit_image(family)

    assert [name for name, _, _ in fake_image.calls] == ["from_registry", "env"]
    reference = fake_image.calls[0][1][0]
    assert reference.endswith(f"/boileroom-{family}-kit:mytag")


def test_kit_image_without_its_dockerfile_points_at_the_registry_source(
    monkeypatch: pytest.MonkeyPatch, fake_image: type[_FakeImage], tmp_path: Path
) -> None:
    from dataclasses import replace

    from boileroom.images import modal as images_modal

    missing = replace(get_kit_image_spec("protenix"), dockerfile_relative_path="no/such/Dockerfile")
    monkeypatch.setattr(images_modal, "get_kit_image_spec", lambda _: missing)
    monkeypatch.setattr(type(missing), "dockerfile_path", property(lambda self: tmp_path / "Dockerfile"))

    with pytest.raises(FileNotFoundError, match=KIT_IMAGE_SOURCE_ENV):
        images_modal.get_modal_kit_image("protenix")
    assert fake_image.calls == []


def test_esmfold2_compile_step_runs_both_scripts_in_the_kit_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    from boileroom.images import modal as images_modal

    commands: list[tuple[list[str], dict[str, Any]]] = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kwargs: commands.append((cmd, kwargs)))

    images_modal.compile_esmfold2_kit()

    assert [cmd for cmd, _ in commands] == [
        ["chmod", "1777", "/tmp"],
        ["sh", "/opt/boileroom-kit/kit_wheels.sh"],
        ["sh", "/opt/boileroom-kit/kit_finish.sh"],
    ]
    assert all(kwargs["check"] is True for _, kwargs in commands)
    assert commands[1][1]["cwd"] == commands[2][1]["cwd"] == "/kit/esmfold2"
