"""``LAYERNORM_TYPE`` of the Protenix and OpenDDE workers: one value per family and mode, which no image can move.

The worker environment (the cores' ``_worker_env``) sets ``LAYERNORM_TYPE`` per optimization mode over whatever it inherits. The
images, their Modal runtime environments and their comments must agree with that table: an image may set a value only
when it is the worker's value for every mode the image serves, and a comment that names a value or the table's constant
must name the real ones.
"""

from __future__ import annotations

import importlib
import importlib.util
import re
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from _dockerfile import Dockerfile

from boileroom.images import metadata
from boileroom.optimization import KIT_FAMILIES, OPTIMIZATION_MODES

pytestmark = pytest.mark.contract

REPO = Path(__file__).resolve().parents[2]
RUNTIME = REPO / "boileroom/models/protenix/runtime.py"
#: The decided table: vanilla runs what each stock release runs, the kit modes run the fused LayerNorm they patch.
EXPECTED = {
    "protenix": {"vanilla": "openfold", "exact": "fast_layernorm", "fast": "fast_layernorm"},
    "opendde": {"vanilla": "torch", "exact": "fast_layernorm", "fast": "fast_layernorm"},
}
#: Per family: the core module, its table constant, its core class and the worker runtime class.
FAMILIES = {
    "protenix": ("boileroom.models.protenix.core", "PROTENIX_LAYERNORM", "ProtenixCore", "ProtenixRuntime"),
    "opendde": ("boileroom.models.opendde.core", "OPENDDE_LAYERNORM", "OpenDDECore", "OpenDDERuntime"),
}
INHERITED = ("fast_layernorm", "openfold", "torch", "bogus")


def _core(family: str) -> ModuleType:
    return importlib.import_module(FAMILIES[family][0])


def _table(family: str) -> dict[str, str]:
    return dict(getattr(_core(family), FAMILIES[family][1]))


def _runtime() -> ModuleType:
    """Load the worker runtime standalone, as the worker interpreter does."""
    spec = importlib.util.spec_from_file_location("_layernorm_contract_runtime", RUNTIME)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _served_images() -> list[tuple[str, metadata.RuntimeImageSpec, tuple[str, ...]]]:
    """Return (family, image, modes it serves) for every image a Protenix-runner family runs on."""
    from boileroom.models.registry import MODEL_SPECS

    images: list[tuple[str, metadata.RuntimeImageSpec, tuple[str, ...]]] = []
    for spec in MODEL_SPECS:
        if spec.family not in FAMILIES:
            continue
        stock = metadata.get_model_image_spec(spec.family)
        if spec.kit_modal_class_path is not None:
            images.append((spec.family, stock, ("vanilla",)))
            images.append((spec.family, metadata.get_kit_image_spec(spec.family), ("exact", "fast")))
        else:
            assert spec.family in KIT_FAMILIES
            images.append((spec.family, stock, OPTIMIZATION_MODES))
    return images


def _image_id(value: Any) -> str:
    if isinstance(value, metadata.RuntimeImageSpec):
        return f"{value.key}-kit" if metadata.is_kit_image_spec(value) else value.key
    return value if isinstance(value, str) else "-".join(value) if isinstance(value, tuple) else ""


SERVED = _served_images()


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_worker_layernorm_table(family: str) -> None:
    core = _core(family)
    table = _table(family)
    assert table == EXPECTED[family], (
        f"{FAMILIES[family][1]} changed. The core constant is the source of truth: if the change is intended, update "
        "EXPECTED here and the 'LayerNorm per mode' table in docs/optimization.md (test_layernorm_docs.py checks it)"
    )
    assert tuple(table) == OPTIMIZATION_MODES
    assert table == getattr(core, FAMILIES[family][2]).LAYERNORM_BY_MODE
    # The worker runtime cannot import boileroom, so it carries a literal copy, which must not drift.
    assert dict(getattr(_runtime(), FAMILIES[family][3]).LAYERNORM_BY_MODE) == table


@pytest.mark.parametrize("family", sorted(FAMILIES))
@pytest.mark.parametrize("mode", OPTIMIZATION_MODES)
def test_worker_runtime_accepts_the_value_the_worker_sets(
    family: str, mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _runtime()
    runtime_class = getattr(runtime, FAMILIES[family][3])
    value = _table(family)[mode]
    worker = SimpleNamespace(mode=mode, LABEL=runtime_class.LABEL, LAYERNORM_BY_MODE=runtime_class.LAYERNORM_BY_MODE)
    monkeypatch.setenv("LAYERNORM_TYPE", value)
    assert runtime_class._check_layernorm_env(worker) == value
    # Any other value is refused: a worker started outside the core must not run a mode on the wrong LayerNorm.
    for other in sorted({"fast_layernorm", "openfold", "torch", ""} - {value}):
        monkeypatch.setenv("LAYERNORM_TYPE", other)
        with pytest.raises(runtime.OptimizationUnavailableError, match=f"runs LAYERNORM_TYPE='{value}'"):
            runtime_class._check_layernorm_env(worker)


def test_served_images_cover_every_mode_of_each_family() -> None:
    for family in FAMILIES:
        modes = [mode for served_family, _, served in SERVED if served_family == family for mode in served]
        assert sorted(modes) == sorted(OPTIMIZATION_MODES), family


def _image_layernorm(spec: metadata.RuntimeImageSpec, model_dir: str = "/mnt/models") -> dict[str, str]:
    """Return the LAYERNORM_TYPE an image sets, by source: its Dockerfile ENV and its Modal runtime environment."""
    found = {}
    dockerfile_env = Dockerfile(spec.dockerfile_path).env()
    if "LAYERNORM_TYPE" in dockerfile_env:
        found["dockerfile"] = dockerfile_env["LAYERNORM_TYPE"]
    modal_env = metadata.render_modal_runtime_env(spec, model_dir)
    if "LAYERNORM_TYPE" in modal_env:
        found["modal"] = modal_env["LAYERNORM_TYPE"]
    return found


@pytest.mark.parametrize(("family", "spec", "modes"), SERVED, ids=_image_id)
def test_image_sets_only_the_worker_value_of_the_modes_it_serves(
    family: str, spec: metadata.RuntimeImageSpec, modes: tuple[str, ...]
) -> None:
    table = _table(family)
    values = {table[mode] for mode in modes}
    found = _image_layernorm(spec)
    if len(values) > 1:
        # One image, modes with different LayerNorms: only the worker may set it.
        assert found == {}, f"{spec.dockerfile_relative_path} sets LAYERNORM_TYPE but serves {modes}"
    else:
        assert set(found.values()) <= values, f"{spec.dockerfile_relative_path}: {found}, the worker sets {values}"


@pytest.mark.parametrize(("family", "spec", "modes"), SERVED, ids=_image_id)
def test_worker_sets_the_table_value_whatever_the_image_inherits(
    family: str,
    spec: metadata.RuntimeImageSpec,
    modes: tuple[str, ...],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    core = _core(family)
    core_class = getattr(core, FAMILIES[family][2])
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    image_values = list(_image_layernorm(spec, str(tmp_path)).values())
    for mode in modes:
        for inherited in [*image_values, *INHERITED]:
            monkeypatch.setenv("LAYERNORM_TYPE", inherited)
            env = core_class()._worker_env({**core_class().config, "optimization": mode}, None)
            assert env["LAYERNORM_TYPE"] == _table(family)[mode], (spec.key, mode, inherited)


def _family_dockerfiles() -> list[tuple[str, Path]]:
    return sorted({(family, spec.dockerfile_path) for family, spec, _ in SERVED})


def test_only_protenix_runner_images_mention_layernorm_type() -> None:
    known = {path for _, path in _family_dockerfiles()}
    mentioning = {path for path in (REPO / "boileroom").rglob("Dockerfile") if "LAYERNORM_TYPE" in path.read_text()}
    assert mentioning <= known


_VALUE = re.compile(r"\bLAYERNORM_TYPE=([A-Za-z_]+)")
_VANILLA = re.compile(r"\b([a-z_]+) for vanilla\b|\bvanilla image uses ([a-z_]+)\b")
_CONSTANT = re.compile(r"\b([A-Z][A-Z0-9]*)_LAYERNORM\b")


@pytest.mark.parametrize(
    ("family", "path"), _family_dockerfiles(), ids=lambda value: str(value).split("boileroom/")[-1]
)
def test_dockerfile_comments_name_the_worker_values(family: str, path: Path) -> None:
    table = _table(family)
    (modes,) = {modes for served_family, spec, modes in SERVED if spec.dockerfile_path == path}
    served = {table[mode] for mode in modes}
    comments = " ".join(Dockerfile(path).comments)
    for value in _VALUE.findall(comments):
        assert value in served, f"{path}: a comment says LAYERNORM_TYPE={value}; the worker sets {served} here"
    for match in _VANILLA.finditer(comments):
        assert (match.group(1) or match.group(2)) == table["vanilla"], f"{path}: {match.group(0)!r}"
    for prefix in _CONSTANT.findall(comments):
        assert getattr(_core(prefix.lower()), f"{prefix}_LAYERNORM") == table, f"{path}: {prefix}_LAYERNORM"


def test_comment_checks_find_what_they_look_for() -> None:
    comments = {path: " ".join(Dockerfile(path).comments) for _, path in _family_dockerfiles()}
    assert any(_VALUE.search(text) for text in comments.values())
    assert any(_VANILLA.search(text) for text in comments.values())
    assert any(_CONSTANT.search(text) for text in comments.values())


def test_metadata_comments_name_existing_tables() -> None:
    source = (REPO / "boileroom/images/metadata.py").read_text()
    comments = " ".join(line.split("#", 1)[1] for line in source.splitlines() if "#" in line)
    prefixes = _CONSTANT.findall(comments)
    assert prefixes
    for prefix in prefixes:
        assert prefix.lower() in FAMILIES and hasattr(_core(prefix.lower()), f"{prefix}_LAYERNORM"), prefix
