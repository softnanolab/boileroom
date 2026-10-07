"""Contract tests for the optimization-kit images (ESMFold2 and Protenix ``exact`` / ``fast``)."""

import importlib
import re
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from boileroom.backend.modal import modal_app_of
from boileroom.base import ModelWrapper
from boileroom.images.metadata import (
    DEFAULT_PYTHON_VERSION,
    IMAGE_TAG_ENV,
    KIT_COMMIT,
    KIT_IMAGE_DIGESTS,
    KIT_IMAGE_SOURCE_ENV,
    KIT_IMAGE_SPECS,
    KIT_IMAGE_TAG_ENV,
    MODEL_IMAGE_SPECS,
    format_image_reference,
    get_kit_image_source,
    get_kit_image_spec,
    get_model_image_spec,
    kit_build_reference,
    kit_image_reference,
    render_modal_runtime_env,
)
from boileroom.models.registry import (
    ESMFOLD2_SPEC,
    MODEL_SPECS,
    OPENDDE_SPEC,
    PROTENIX_SPEC,
    ModelSpec,
    resolve_object,
)
from boileroom.provenance import IMAGE_REF_ENV

REPO_ROOT = Path(__file__).resolve().parents[2]
KIT_SPECS = [ESMFOLD2_SPEC, PROTENIX_SPEC]
UPSTREAM_KIT_REPOSITORY = "github.com/anthropics/uplifting-biomolecular-modeling"


@pytest.fixture(autouse=True)
def _no_image_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test from the default image lookup, whatever the shell exports."""
    for name in (IMAGE_TAG_ENV, KIT_IMAGE_TAG_ENV, KIT_IMAGE_SOURCE_ENV, "BOILEROOM_DOCKER_REPOSITORY"):
        monkeypatch.delenv(name, raising=False)


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


def test_kit_images_are_not_part_of_the_stock_build_matrix() -> None:
    """The CI matrix builds MODEL_IMAGE_SPECS only; the kit images are pushed by hand and pulled by digest."""
    stock_names = {spec.image_name for spec in MODEL_IMAGE_SPECS}
    stock_dockerfiles = {spec.dockerfile_relative_path for spec in MODEL_IMAGE_SPECS}
    for spec in KIT_IMAGE_SPECS:
        assert spec.image_name not in stock_names
        assert spec.dockerfile_relative_path not in stock_dockerfiles
        assert spec.image_name.endswith("-kit")


@pytest.mark.parametrize("spec", KIT_IMAGE_SPECS, ids=lambda spec: spec.key)
def test_kit_image_files_exist(spec: Any) -> None:
    assert spec.dockerfile_path.is_file()
    assert spec.config_relative_path is None
    assert spec.context_path.is_dir()
    assert spec.dockerfile_path.is_relative_to(spec.context_path)
    assert get_kit_image_spec(spec.key) is spec


@pytest.mark.parametrize("identifier", ["boltz2", "boileroom-esmfold2-kit"])
def test_unknown_kit_image_is_rejected(identifier: str) -> None:
    """Kit images are looked up by family key only; an image name is not a key."""
    with pytest.raises(KeyError, match="Unknown kit image spec"):
        get_kit_image_spec(identifier)


@pytest.mark.parametrize("spec", KIT_IMAGE_SPECS, ids=lambda spec: spec.key)
def test_every_kit_image_has_a_pinned_sha256_digest(spec: Any) -> None:
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", KIT_IMAGE_DIGESTS[spec.image_name])


def test_kit_digests_cover_exactly_the_kit_images() -> None:
    assert set(KIT_IMAGE_DIGESTS) == {spec.image_name for spec in KIT_IMAGE_SPECS}


@pytest.mark.parametrize("spec", KIT_IMAGE_SPECS, ids=lambda spec: spec.key)
def test_kit_image_reference_defaults_to_the_pinned_digest(spec: Any) -> None:
    assert kit_image_reference(spec) == f"docker.io/jakublala/{spec.image_name}@{KIT_IMAGE_DIGESTS[spec.image_name]}"


@pytest.mark.parametrize("spec", KIT_IMAGE_SPECS, ids=lambda spec: spec.key)
def test_stock_image_tag_does_not_reach_kit_images(monkeypatch: pytest.MonkeyPatch, spec: Any) -> None:
    """BOILEROOM_IMAGE_TAG used to name the kit image too, so a stock override silently swapped the kit image."""
    monkeypatch.setenv(IMAGE_TAG_ENV, "sha-1234567")
    assert kit_image_reference(spec).endswith(f"@{KIT_IMAGE_DIGESTS[spec.image_name]}")
    assert "sha-1234567" not in kit_image_reference(spec)


@pytest.mark.parametrize("spec", KIT_IMAGE_SPECS, ids=lambda spec: spec.key)
def test_kit_image_tag_env_and_explicit_tag_name_a_tag(monkeypatch: pytest.MonkeyPatch, spec: Any) -> None:
    monkeypatch.setenv(KIT_IMAGE_TAG_ENV, " kit-f4f62fa ")
    monkeypatch.setenv("BOILEROOM_DOCKER_REPOSITORY", "myorg")
    assert kit_image_reference(spec) == f"docker.io/myorg/{spec.image_name}:kit-f4f62fa"
    # An explicit tag wins over the environment; kit tags are used verbatim (no CUDA qualifier).
    assert kit_image_reference(spec, "cuda12.6-dev") == f"docker.io/myorg/{spec.image_name}:cuda12.6-dev"


@pytest.mark.parametrize("tag", ["latest", "bad tag", "-leading-dash", "x" * 129])
def test_invalid_kit_image_tags_are_rejected(monkeypatch: pytest.MonkeyPatch, tag: str) -> None:
    spec = get_kit_image_spec("protenix")
    with pytest.raises(ValueError):
        kit_image_reference(spec, tag)
    monkeypatch.setenv(KIT_IMAGE_TAG_ENV, tag)
    with pytest.raises(ValueError):
        kit_image_reference(spec)


def test_kit_image_reference_refuses_a_stock_spec() -> None:
    with pytest.raises(ValueError, match="not a kit image"):
        kit_image_reference(MODEL_IMAGE_SPECS[0])


def test_a_modified_copy_of_a_kit_spec_is_still_a_kit_image() -> None:
    """Kit-ness used to be object identity, so a ``replace``d kit spec resolved as a stock image with the stock tag."""
    from dataclasses import replace

    from boileroom.images.metadata import is_kit_image_spec

    copy = replace(get_kit_image_spec("protenix"), python_version="3.12")
    assert is_kit_image_spec(copy)
    assert kit_image_reference(copy).endswith(f"@{KIT_IMAGE_DIGESTS['boileroom-protenix-kit']}")
    env = render_modal_runtime_env(copy, "/models")
    assert KIT_IMAGE_SOURCE_ENV in env
    assert IMAGE_TAG_ENV not in env
    assert not is_kit_image_spec(replace(copy, image_name="boileroom-protenix"))


@pytest.mark.parametrize("spec", KIT_IMAGE_SPECS, ids=lambda spec: spec.key)
def test_kit_build_reference_is_deterministic_and_names_the_dockerfile(spec: Any) -> None:
    reference = kit_build_reference(spec, {"STACK": "a"})
    assert re.fullmatch(rf"build:{re.escape(spec.dockerfile_relative_path)}@[0-9a-f]{{12}}", reference)
    assert kit_build_reference(spec, {"STACK": "a"}) == reference
    assert kit_build_reference(spec, {"STACK": "b"}) != reference
    assert kit_build_reference(spec) != reference


def test_kit_build_reference_tracks_the_build_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Editing any context file (not a bytecode cache) changes the reference."""
    from dataclasses import replace

    import boileroom.images.metadata as metadata

    context = tmp_path / "kit"
    context.mkdir()
    (context / "Dockerfile").write_text("FROM scratch\n")
    (context / "kit.sh").write_text("echo 1\n")
    monkeypatch.setattr(metadata, "get_repo_root", lambda: tmp_path)
    spec = replace(
        get_kit_image_spec("protenix"), dockerfile_relative_path="kit/Dockerfile", context_relative_path="kit"
    )

    before = kit_build_reference(spec)
    (context / "__pycache__").mkdir()
    (context / "__pycache__" / "x.cpython-312.pyc").write_bytes(b"cache")
    assert kit_build_reference(spec) == before
    (context / "kit.sh").write_text("echo 2\n")
    assert kit_build_reference(spec) != before


def test_kit_dockerfiles_fetch_the_same_upstream_repository() -> None:
    """The two kit Dockerfiles and the OpenDDE Dockerfile all fetch the kit from one upstream repository."""
    for spec in KIT_IMAGE_SPECS:
        match = re.search(r"^ARG KIT_REPOSITORY=(\S+)$", spec.dockerfile_path.read_text(), flags=re.MULTILINE)
        assert match is not None
        assert match.group(1).removesuffix(".git") == f"https://{UPSTREAM_KIT_REPOSITORY}"
    opendde = (REPO_ROOT / "boileroom/models/opendde/Dockerfile").read_text()
    match = re.search(r"^ARG KIT_REPOSITORY=(\S+)$", opendde, flags=re.MULTILINE)
    assert match is not None
    assert match.group(1).removesuffix(".git") == f"https://{UPSTREAM_KIT_REPOSITORY}"
    assert 'fetch --quiet --depth 1 "${KIT_REPOSITORY}" "${KIT_COMMIT}"' in opendde


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
def test_kit_dockerfile_records_the_kit_commit_for_provenance(spec: Any) -> None:
    """Each kit image exports the pinned commit, which the runtime records as ``kit.commit``."""
    text = spec.dockerfile_path.read_text()
    assert re.search(r"^\s*BOILEROOM_KIT_COMMIT=\$\{?KIT_COMMIT\}?\s*\\?$", text, flags=re.MULTILINE)


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


def test_kit_runtime_env_carries_the_kit_lookup_not_the_stock_tag(monkeypatch: pytest.MonkeyPatch) -> None:
    """A container re-import must resolve the same kit image, and the stock tag must not change the kit image."""
    spec = get_kit_image_spec("esmfold2")
    monkeypatch.setenv(IMAGE_TAG_ENV, "sha-1234567")
    env = render_modal_runtime_env(spec, "/mnt/models")
    assert IMAGE_TAG_ENV not in env
    assert env[KIT_IMAGE_SOURCE_ENV] == "build"
    assert KIT_IMAGE_TAG_ENV not in env

    monkeypatch.setenv(KIT_IMAGE_SOURCE_ENV, "registry")
    monkeypatch.setenv(KIT_IMAGE_TAG_ENV, "kit-f4f62fa")
    env = render_modal_runtime_env(spec, "/mnt/models")
    assert env[KIT_IMAGE_SOURCE_ENV] == "registry"
    assert env[KIT_IMAGE_TAG_ENV] == "kit-f4f62fa"


@pytest.fixture
def _fresh_kit_remote() -> Iterator[None]:
    """Import ``tests/_kit_remote.py`` afresh in the test and drop it afterwards."""
    sys.modules.pop("_kit_remote", None)
    yield
    sys.modules.pop("_kit_remote", None)


@pytest.mark.parametrize(
    ("container", "spec_of", "fails_on_own_env"),
    [
        ("esmfold2-kit", lambda: get_kit_image_spec("esmfold2"), True),
        ("protenix-kit", lambda: get_kit_image_spec("protenix"), True),
        ("esmfold2-stock", lambda: get_model_image_spec("esmfold2"), False),
    ],
)
@pytest.mark.usefixtures("_fresh_kit_remote")
def test_kit_remote_imports_in_each_of_its_containers(
    monkeypatch: pytest.MonkeyPatch, container: str, spec_of: Any, fails_on_own_env: bool
) -> None:
    """``tests/_kit_remote.py`` evaluates all four of its images on import, in each container it runs in.

    boileroom is not installed in the container, so the default stock tag cannot be read there: the image must carry the
    whole lookup of the run. A kit image's own runtime environment has no stock tag, which is what made the kit
    fallback and guard runs fail to import before any fold.
    """
    import modal

    from boileroom.images import metadata
    from boileroom.utils import MODAL_MODEL_DIR

    applied: list[dict[str, str]] = []
    original_env = modal.Image.env

    def _recording_env(self: modal.Image, vars: dict[str, str]) -> modal.Image:
        applied.append(dict(vars))
        return original_env(self, vars)

    monkeypatch.setattr(modal.Image, "env", _recording_env)
    monkeypatch.setenv(IMAGE_TAG_ENV, "0.4.4-alpha.9")
    monkeypatch.setenv(KIT_IMAGE_SOURCE_ENV, "registry")
    kit_remote = importlib.import_module("_kit_remote")
    lookup = kit_remote._image_lookup_env()
    # Each of the four images (ESMFold2 kit, stock and guard, Protenix guard) carries it.
    assert applied.count(lookup) == 4
    own_env = render_modal_runtime_env(spec_of(), MODAL_MODEL_DIR)
    baked = {**own_env, **lookup}

    def _not_installed() -> str:
        raise RuntimeError("boileroom is not installed in the container")

    monkeypatch.setattr(metadata, "get_default_image_tag", _not_installed)
    for name in (IMAGE_TAG_ENV, KIT_IMAGE_TAG_ENV, KIT_IMAGE_SOURCE_ENV):
        monkeypatch.delenv(name, raising=False)
    with monkeypatch.context() as container_env:
        for name, value in own_env.items():
            container_env.setenv(name, value)
        if fails_on_own_env:
            with pytest.raises(RuntimeError, match="not installed"):
                importlib.reload(kit_remote)
        else:
            importlib.reload(kit_remote)

    for name, value in baked.items():
        monkeypatch.setenv(name, value)
    importlib.reload(kit_remote)
    assert kit_remote._image_lookup_env()[IMAGE_TAG_ENV] == "0.4.4-alpha.9"


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
        assert (spec.kit_modal_class_path is not None) is has_kit
        # The Apptainer kit image is the kit image of the same family.
        assert (spec.family in {kit_spec.key for kit_spec in KIT_IMAGE_SPECS}) is has_kit
    # OpenDDE ships its kit inside its own image.
    assert OPENDDE_SPEC.kit_modal_class_path is None


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
    kit_spec = get_kit_image_spec(spec.family)
    _initialize(spec, "apptainer:dev", {"optimization": "fast"})
    assert records["image_uri"] == f"docker://docker.io/jakublala/{kit_spec.image_name}:dev"
    assert records["kwargs"]["python_version"] == kit_spec.python_version


@pytest.mark.parametrize("spec", KIT_SPECS, ids=lambda spec: spec.key)
@pytest.mark.parametrize("backend", ["apptainer", "apptainer:", "apptainer: "])
def test_kit_apptainer_without_a_tag_pulls_the_pinned_digest(
    monkeypatch: pytest.MonkeyPatch, spec: ModelSpec, backend: str
) -> None:
    """No tag (or an empty one) used to be refused; the kit image is now pulled by its pinned digest."""
    records = _install_fake_backends(monkeypatch)
    monkeypatch.setenv(IMAGE_TAG_ENV, "sha-1234567")  # the stock tag never names a kit image
    _initialize(spec, backend, {"optimization": "exact"})
    kit_spec = get_kit_image_spec(spec.family)
    assert records["image_uri"] == (
        f"docker://docker.io/jakublala/{kit_spec.image_name}@{KIT_IMAGE_DIGESTS[kit_spec.image_name]}"
    )
    assert records["started"] is True


@pytest.mark.parametrize("spec", KIT_SPECS, ids=lambda spec: spec.key)
def test_kit_apptainer_tag_comes_from_the_backend_then_the_kit_tag_env(
    monkeypatch: pytest.MonkeyPatch, spec: ModelSpec
) -> None:
    records = _install_fake_backends(monkeypatch)
    kit_name = get_kit_image_spec(spec.family).image_name
    monkeypatch.setenv(KIT_IMAGE_TAG_ENV, "kit-env")
    _initialize(spec, "apptainer", {"optimization": "exact"})
    assert records["image_uri"] == f"docker://docker.io/jakublala/{kit_name}:kit-env"
    _initialize(spec, "apptainer:sha-abc1234", {"optimization": "exact"})
    assert records["image_uri"] == f"docker://docker.io/jakublala/{kit_name}:sha-abc1234"


@pytest.mark.parametrize("spec", KIT_SPECS, ids=lambda spec: spec.key)
def test_vanilla_apptainer_ignores_the_kit_tag(monkeypatch: pytest.MonkeyPatch, spec: ModelSpec) -> None:
    records = _install_fake_backends(monkeypatch)
    monkeypatch.setenv(KIT_IMAGE_TAG_ENV, "kit-env")
    monkeypatch.setenv(IMAGE_TAG_ENV, "sha-1234567")
    _initialize(spec, "apptainer", {"optimization": "vanilla"})
    assert spec.apptainer_image_name is not None
    assert records["image_uri"] == f"docker://docker.io/jakublala/{spec.apptainer_image_name}:sha-1234567"


NON_KIT_SPECS = [spec for spec in MODEL_SPECS if "optimization" not in spec.contract.static_config_keys]


def test_every_kit_family_lists_optimization_and_every_other_does_not() -> None:
    from boileroom.optimization import KIT_FAMILIES

    assert {spec.key for spec in MODEL_SPECS if spec not in NON_KIT_SPECS} == set(KIT_FAMILIES)


@pytest.mark.parametrize("spec", NON_KIT_SPECS, ids=lambda spec: spec.key)
@pytest.mark.parametrize("mode", ["exact", "fast"])
def test_non_kit_family_refuses_a_kit_mode_before_a_backend_starts(
    monkeypatch: pytest.MonkeyPatch, spec: ModelSpec, mode: str
) -> None:
    """Without this, ``optimization='fast'`` on e.g. Boltz-2 reached the remote core and ran vanilla, or hung."""
    from boileroom.optimization import OptimizationUnavailableError

    records = _install_fake_backends(monkeypatch)
    with pytest.raises(OptimizationUnavailableError, match=rf"{spec.public_name} has no optimization kit"):
        _initialize(spec, "modal", {"optimization": mode})
    assert "started" not in records


@pytest.mark.parametrize("spec", NON_KIT_SPECS, ids=lambda spec: spec.key)
def test_non_kit_refusal_reads_the_same_from_the_wrapper_and_the_core(
    monkeypatch: pytest.MonkeyPatch, spec: ModelSpec
) -> None:
    """The wrapper refuses before a backend starts and the core refuses at construction, in the same words."""
    from boileroom.base import Algorithm
    from boileroom.optimization import OptimizationUnavailableError

    class Core(Algorithm):
        DISPLAY_NAME = spec.public_name

        def _load(self) -> None:
            pass

    _install_fake_backends(monkeypatch)
    with pytest.raises(OptimizationUnavailableError) as wrapper_refusal:
        _initialize(spec, "modal", {"optimization": "fast"})
    with pytest.raises(OptimizationUnavailableError) as core_refusal:
        Core({"optimization": "fast"})
    assert str(wrapper_refusal.value) == str(core_refusal.value)
    assert "use optimization='vanilla' or leave it unset" in str(core_refusal.value)


@pytest.mark.parametrize("spec", NON_KIT_SPECS, ids=lambda spec: spec.key)
def test_non_kit_family_rejects_an_unknown_mode_and_accepts_vanilla(
    monkeypatch: pytest.MonkeyPatch, spec: ModelSpec
) -> None:
    records = _install_fake_backends(monkeypatch)
    with pytest.raises(ValueError, match="optimization must be one of"):
        _initialize(spec, "modal", {"optimization": "turbo"})
    assert "started" not in records
    _initialize(spec, "modal", {"optimization": "vanilla"})
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


def test_esmfold2_kit_image_builds_from_the_dockerfile_with_a_sized_compile_step_then_a_finish_step(
    fake_image: type[_FakeImage],
) -> None:
    """The compile and the finish used to be one build step, so a failed finish (a transient pip error) recompiled."""
    from boileroom.images import modal as images_modal

    images_modal.get_modal_kit_image("esmfold2")

    names = [name for name, _, _ in fake_image.calls]
    assert names == ["from_dockerfile", "run_function", "run_function", "env", "env"]
    _, args, kwargs = fake_image.calls[0]
    assert args[0] == get_kit_image_spec("esmfold2").dockerfile_path
    assert kwargs["build_args"] == {"STACK": images_modal.ESMFOLD2_KIT_STACK, "WHEELS_FROM": "defer"}
    _, args, kwargs = fake_image.calls[1]
    assert args == (images_modal.compile_esmfold2_kit,)
    assert kwargs["cpu"] == images_modal.ESMFOLD2_KIT_COMPILE_CPU
    assert kwargs["memory"] == images_modal.ESMFOLD2_KIT_COMPILE_MEMORY_MIB
    assert kwargs["timeout"] == images_modal.ESMFOLD2_KIT_COMPILE_TIMEOUT
    assert kwargs["env"]["BUILD_JOBS"] == str(images_modal.ESMFOLD2_KIT_COMPILE_CPU)
    # The finish step compiles nothing: Modal's default builder, no compile environment.
    _, args, kwargs = fake_image.calls[2]
    assert args == (images_modal.finish_esmfold2_kit,)
    assert kwargs["cpu"] is None
    assert kwargs["memory"] is None
    assert kwargs["timeout"] == images_modal.KIT_BUILD_STEP_TIMEOUT
    assert kwargs["env"] == {}
    env = dict(fake_image.calls[3][1][0])
    assert env["MODEL_DIR"]
    spec = get_kit_image_spec("esmfold2")
    assert fake_image.calls[4][1][0] == {
        IMAGE_REF_ENV: kit_build_reference(spec, images_modal._kit_build(spec).reference_inputs())
    }


def test_every_kit_build_step_gets_an_integer_timeout(
    monkeypatch: pytest.MonkeyPatch, fake_image: type[_FakeImage]
) -> None:
    """A step declared without a timeout used to hand Modal ``timeout=None``, which ``run_function`` does not take."""
    from boileroom.images import modal as images_modal

    monkeypatch.setattr(
        images_modal,
        "_kit_build",
        lambda _: images_modal.KitBuild(steps=(images_modal.KitBuildStep(function=_changed_compile_step),)),
    )
    images_modal.get_modal_kit_image("protenix")

    (step_kwargs,) = [kwargs for name, _, kwargs in fake_image.calls if name == "run_function"]
    assert type(step_kwargs["timeout"]) is int
    assert step_kwargs["timeout"] == images_modal.KIT_BUILD_STEP_TIMEOUT


def _esmfold2_build_reference(
    fake_image: type[_FakeImage],
) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    """Build the ESMFold2 kit image on the fake; return its BOILEROOM_IMAGE_REF, the Dockerfile call and the steps."""
    from boileroom.images import modal as images_modal

    _FakeImage.calls = []
    images_modal.get_modal_kit_image("esmfold2")
    calls = fake_image.calls
    steps = [kwargs | {"function": args[0]} for name, args, kwargs in calls if name == "run_function"]
    return calls[-1][1][0][IMAGE_REF_ENV], calls[0][2], steps


def _changed_compile_step() -> None:
    """A compile step that differs from ``compile_esmfold2_kit`` in its source only."""
    import subprocess

    subprocess.run(["sh", "/opt/boileroom-kit/kit_wheels.sh", "--extra"], check=True)


@pytest.mark.parametrize(
    "change",
    ["build-arg", "compile-env", "compile-memory", "compile-step", "finish-step"],
)
def test_esmfold2_build_reference_tracks_every_build_input(
    monkeypatch: pytest.MonkeyPatch, fake_image: type[_FakeImage], change: str
) -> None:
    """The reference used to hash STACK alone: WHEELS_FROM, BUILD_JOBS or the compile step could change unnoticed.

    Each input is changed where the build reads it, so the test also proves the build and the reference share it.
    """
    from types import MappingProxyType

    from boileroom.images import modal as images_modal

    before, _, _ = _esmfold2_build_reference(fake_image)
    if change == "build-arg":
        new_args = MappingProxyType({**images_modal.ESMFOLD2_KIT_BUILD_ARGS, "WHEELS_FROM": "build"})
        monkeypatch.setattr(images_modal, "ESMFOLD2_KIT_BUILD_ARGS", new_args)
    elif change == "compile-env":
        new_env = MappingProxyType({**images_modal.ESMFOLD2_KIT_COMPILE_ENV, "BUILD_JOBS": "8"})
        monkeypatch.setattr(images_modal, "ESMFOLD2_KIT_COMPILE_ENV", new_env)
    elif change == "compile-memory":
        monkeypatch.setattr(images_modal, "ESMFOLD2_KIT_COMPILE_MEMORY_MIB", 65536)
    elif change == "compile-step":
        monkeypatch.setattr(images_modal, "compile_esmfold2_kit", _changed_compile_step)
    else:
        monkeypatch.setattr(images_modal, "finish_esmfold2_kit", _changed_compile_step)

    after, build_kwargs, (compile_kwargs, finish_kwargs) = _esmfold2_build_reference(fake_image)

    assert after != before
    assert build_kwargs["build_args"] == dict(images_modal.ESMFOLD2_KIT_BUILD_ARGS)
    assert compile_kwargs["env"] == dict(images_modal.ESMFOLD2_KIT_COMPILE_ENV)
    assert compile_kwargs["memory"] == images_modal.ESMFOLD2_KIT_COMPILE_MEMORY_MIB
    assert compile_kwargs["function"] is images_modal.compile_esmfold2_kit
    assert finish_kwargs["function"] is images_modal.finish_esmfold2_kit


@pytest.mark.parametrize("family", ["esmfold2", "protenix"])
@pytest.mark.parametrize("changed", [False, True], ids=["current", "protenix-build-arg"])
def test_kit_build_reference_hashes_exactly_what_the_build_uses(
    monkeypatch: pytest.MonkeyPatch, fake_image: type[_FakeImage], family: str, changed: bool
) -> None:
    """The build and the reference used to be two separate ``spec.key`` branches that had to be kept in step by hand.

    A build arg given to the Protenix kit build only would have left its BOILEROOM_IMAGE_REF unchanged, so two
    different images reported one reference. Here the calls the build makes are read back into the inputs they imply,
    which must equal what the reference hashed.
    """
    import inspect

    from boileroom.images import modal as images_modal

    spec = get_kit_image_spec(family)
    before = kit_build_reference(spec, images_modal._kit_build(spec).reference_inputs())
    if changed:
        original = images_modal._kit_build

        def changed_build(build_spec: Any) -> Any:
            if build_spec.key != "protenix":
                return original(build_spec)
            return images_modal.KitBuild(build_args={"KIT_COMMIT": "0" * 40})

        monkeypatch.setattr(images_modal, "_kit_build", changed_build)

    images_modal.get_modal_kit_image(family)

    implied: dict[str, str] = {}
    steps = 0
    for name, args, kwargs in fake_image.calls:
        if name == "from_dockerfile":
            implied.update({f"build-arg {key}": value for key, value in (kwargs.get("build_args") or {}).items()})
        elif name == "run_function":
            implied.update({f"step {steps} env {key}": value for key, value in kwargs["env"].items()})
            if kwargs["memory"] is not None:
                implied[f"step {steps} memory-mib"] = str(kwargs["memory"])
            implied[f"step {steps} function"] = inspect.getsource(args[0])
            steps += 1
    reference = fake_image.calls[-1][1][0][IMAGE_REF_ENV]
    assert reference == kit_build_reference(spec, implied)
    # A different build of the same Dockerfile reports a different reference.
    assert (reference != before) == (changed and family == "protenix")


def test_esmfold2_build_reference_is_stable_across_builds(fake_image: type[_FakeImage]) -> None:
    first, _, _ = _esmfold2_build_reference(fake_image)
    second, _, _ = _esmfold2_build_reference(fake_image)
    assert first == second


def test_protenix_kit_image_is_the_dockerfile_alone(fake_image: type[_FakeImage]) -> None:
    from boileroom.images import modal as images_modal

    images_modal.get_modal_kit_image("protenix")

    assert [name for name, _, _ in fake_image.calls] == ["from_dockerfile", "env", "env"]
    assert dict(fake_image.calls[1][1][0])["LAYERNORM_TYPE"] == "fast_layernorm"
    # BOILEROOM_IMAGE_REF is the last layer and names the Dockerfile build.
    assert fake_image.calls[2][1][0] == {IMAGE_REF_ENV: kit_build_reference(get_kit_image_spec("protenix"))}


@pytest.mark.parametrize("family", ["esmfold2", "protenix"])
def test_kit_image_from_a_registry_pulls_the_pinned_digest(
    monkeypatch: pytest.MonkeyPatch, fake_image: type[_FakeImage], family: str
) -> None:
    from boileroom.images import modal as images_modal

    monkeypatch.setenv(KIT_IMAGE_SOURCE_ENV, "registry")
    monkeypatch.setenv(IMAGE_TAG_ENV, "mytag")  # the stock tag never names a kit image
    images_modal.get_modal_kit_image(family)

    assert [name for name, _, _ in fake_image.calls] == ["from_registry", "env", "env"]
    reference = fake_image.calls[0][1][0]
    assert reference == f"docker.io/jakublala/boileroom-{family}-kit@{KIT_IMAGE_DIGESTS[f'boileroom-{family}-kit']}"
    assert fake_image.calls[-1][1][0] == {IMAGE_REF_ENV: reference}


@pytest.mark.parametrize("family", ["esmfold2", "protenix"])
def test_kit_image_from_a_registry_honours_the_kit_tag(
    monkeypatch: pytest.MonkeyPatch, fake_image: type[_FakeImage], family: str
) -> None:
    from boileroom.images import modal as images_modal

    monkeypatch.setenv(KIT_IMAGE_SOURCE_ENV, "registry")
    monkeypatch.setenv(KIT_IMAGE_TAG_ENV, "sha-dc652b0")
    images_modal.get_modal_kit_image(family)

    reference = fake_image.calls[0][1][0]
    assert reference == f"docker.io/jakublala/boileroom-{family}-kit:sha-dc652b0"
    assert fake_image.calls[-1][1][0] == {IMAGE_REF_ENV: reference}


@pytest.mark.parametrize("family", ["esm", "esmfold2", "protenix", "opendde"])
def test_stock_modal_image_ends_with_its_registry_reference(
    monkeypatch: pytest.MonkeyPatch, fake_image: type[_FakeImage], family: str
) -> None:
    from boileroom.images import modal as images_modal
    from boileroom.images.metadata import get_model_image_spec

    monkeypatch.setenv(IMAGE_TAG_ENV, "sha-1234567")
    images_modal.get_modal_image(family)

    assert [name for name, _, _ in fake_image.calls] == ["from_registry", "env", "env"]
    reference = fake_image.calls[0][1][0]
    assert reference == f"docker.io/jakublala/{get_model_image_spec(family).image_name}:sha-1234567"
    assert fake_image.calls[-1][1][0] == {IMAGE_REF_ENV: reference}


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


@pytest.mark.parametrize(
    ("step", "expected"),
    [
        ("compile_esmfold2_kit", [["chmod", "1777", "/tmp"], ["sh", "/opt/boileroom-kit/kit_wheels.sh"]]),
        ("finish_esmfold2_kit", [["sh", "/opt/boileroom-kit/kit_finish.sh"]]),
    ],
)
def test_esmfold2_build_steps_each_run_one_script_in_the_kit_directory(
    monkeypatch: pytest.MonkeyPatch, step: str, expected: list[list[str]]
) -> None:
    """The compile step must not also run kit_finish.sh, or a failed finish would take the compile down with it."""
    import subprocess

    from boileroom.images import modal as images_modal

    commands: list[tuple[list[str], dict[str, Any]]] = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kwargs: commands.append((cmd, kwargs)))

    getattr(images_modal, step)()

    assert [cmd for cmd, _ in commands] == expected
    assert all(kwargs["check"] is True for _, kwargs in commands)
    assert [kwargs["cwd"] for cmd, kwargs in commands if cmd[0] == "sh"] == ["/kit/esmfold2"]
