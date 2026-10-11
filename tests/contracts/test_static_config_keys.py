"""The registry's static config keys must equal each core's, since routing (the kit image) reads the registry copy."""

import ast
import importlib
import importlib.util
from pathlib import Path

import pytest

from boileroom.models.registry import MODEL_SPECS, ModelSpec


def _static_keys_from_source(module_path: str, class_name: str) -> frozenset[str]:
    """Read ``STATIC_CONFIG_KEYS`` of a core whose model dependencies are not installed, without importing it."""
    spec = importlib.util.find_spec(module_path)
    assert spec is not None and spec.origin is not None, module_path
    tree = ast.parse(Path(spec.origin).read_text(encoding="utf-8"))
    for node in tree.body:
        if not (isinstance(node, ast.ClassDef) and node.name == class_name):
            continue
        for statement in node.body:
            target: ast.expr | None = None
            value: ast.expr | None = None
            if isinstance(statement, ast.AnnAssign):
                target, value = statement.target, statement.value
            elif isinstance(statement, ast.Assign) and len(statement.targets) == 1:
                target, value = statement.targets[0], statement.value
            if value is None or not (isinstance(target, ast.Name) and target.id == "STATIC_CONFIG_KEYS"):
                continue
            if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "frozenset":
                value = value.args[0]
            return frozenset(ast.literal_eval(value))
    raise AssertionError(f"{module_path}.{class_name} has no literal STATIC_CONFIG_KEYS this test can read")


def _core_static_keys(spec: ModelSpec) -> frozenset[str]:
    assert spec.apptainer_core_class_path is not None
    module_path, _, class_name = spec.apptainer_core_class_path.rpartition(".")
    try:
        module = importlib.import_module(module_path)
    except ImportError:
        return _static_keys_from_source(module_path, class_name)
    return frozenset(getattr(module, class_name).STATIC_CONFIG_KEYS)


@pytest.mark.parametrize("spec", MODEL_SPECS, ids=lambda spec: spec.key)
def test_registry_static_keys_match_the_core(spec: ModelSpec) -> None:
    assert spec.contract.static_config_keys == _core_static_keys(spec)


def test_source_reader_parses_a_frozenset_literal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The fallback must read the keys, not silently return an empty set."""
    package = tmp_path / "fake_static_core"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "core.py").write_text(
        "import missing_heavy_dependency\n\n"
        "class Core:\n"
        '    STATIC_CONFIG_KEYS: frozenset[str] = frozenset({"device", "cache_dir"})\n'
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    assert _static_keys_from_source("fake_static_core.core", "Core") == {"device", "cache_dir"}
    with pytest.raises(AssertionError, match="no literal STATIC_CONFIG_KEYS"):
        _static_keys_from_source("fake_static_core.core", "Other")
