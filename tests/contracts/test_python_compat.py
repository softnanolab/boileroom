"""The boileroom package must import on Python 3.11: the Protenix kit image runs it on 3.11.5.

The worker runtime files go further: each worker interpreter runs them standalone, so they must stay Python 3.10
compatible and never import boileroom (``_worker.py``, every ``models/*/runtime.py``, and the kit images' build-time
scripts under ``models/*/kit``).

Ruff targets py312 and selects the ``UP`` rules, so nothing else stops 3.12-only syntax from landing.
"""

import ast
import io
import shutil
import subprocess
import sys
import tokenize
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_FILES = sorted((REPO_ROOT / "boileroom").rglob("*.py"))
_PREFIX_CHARS = "rRbBfFuU"


def _quote(token_text: str) -> str:
    """Return the opening quote (``'``, ``"``, ``'''`` or ``\"\"\"``) of a string or f-string start token."""
    body = token_text.lstrip(_PREFIX_CHARS)
    return body[:3] if body[:3] in ("'''", '"""') else body[:1]


def pep701_only_fstrings(source: str) -> list[str]:
    """Return the f-string constructs in ``source`` that only Python 3.12 accepts (PEP 701).

    Python 3.11 rejects a string inside a replacement field that reuses the enclosing f-string's quote, and a comment
    inside a replacement field. The 3.12 tokenizer exposes both.
    """
    if sys.version_info < (3, 12):  # noqa: UP036  # pragma: no cover - the repo runs on 3.12
        return []
    problems: list[str] = []
    enclosing: list[str] = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        kind = tokenize.tok_name[token.type]
        if kind in ("FSTRING_START", "STRING") and enclosing:
            quote = _quote(token.string)
            if quote.startswith(enclosing[-1]):
                problems.append(f"line {token.start[0]}: {token.string!r} reuses the enclosing f-string's quote")
        if kind == "COMMENT" and enclosing:
            problems.append(f"line {token.start[0]}: comment inside an f-string replacement field")
        if kind == "FSTRING_START":
            enclosing.append(_quote(token.string))
        elif kind == "FSTRING_END":
            enclosing.pop()
    return problems


@pytest.mark.parametrize("path", PACKAGE_FILES, ids=lambda path: str(path.relative_to(REPO_ROOT)))
def test_package_source_parses_as_python_311(path: Path) -> None:
    """No PEP 695 syntax (``type X = ...``, ``class X[T]``) or other 3.12-only grammar, and no PEP 701 f-strings."""
    source = path.read_text(encoding="utf-8")
    ast.parse(source, filename=str(path), feature_version=(3, 11))
    assert pep701_only_fstrings(source) == []


@pytest.mark.parametrize(
    "source",
    [
        "type Alias = int\n",
        "class Box[T]:\n    pass\n",
        "def first[T](items: list[T]) -> T:\n    return items[0]\n",
    ],
)
def test_the_parse_check_rejects_pep_695_syntax(source: str) -> None:
    with pytest.raises(SyntaxError):
        ast.parse(source, feature_version=(3, 11))


@pytest.mark.skipif(sys.version_info < (3, 12), reason="needs the 3.12 tokenizer")
@pytest.mark.parametrize(
    ("source", "found"),
    [
        ("x = f\"{', '.join(['a'])}\"\n", False),
        ('x = f"{"a"}"\n', True),
        ("x = f'{d['k']}'\n", True),
        ('x = f"""{"a"}"""\n', False),
        ('x = f"""{"""a"""}"""\n', True),
        ('x = f"""{1 + 2  # three\n}"""\n', True),
        ('x = f"{f"{1}"}"\n', True),
        ("x = f\"{f'{1}'}\"\n", False),
    ],
)
def test_the_fstring_check_finds_pep_701_only_constructs(source: str, found: bool) -> None:
    assert bool(pep701_only_fstrings(source)) is found


def find_python(version: str) -> str | None:
    """Return a CPython ``version`` interpreter: ``python<version>`` on PATH, else one uv already has installed."""
    found = shutil.which(f"python{version}")
    if found is not None:
        return found
    uv = shutil.which("uv")
    if uv is None:
        return None
    try:
        result = subprocess.run(
            [uv, "python", "find", "--no-project", "--no-python-downloads", version],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    candidate = result.stdout.strip().splitlines()[-1] if result.returncode == 0 and result.stdout.strip() else ""
    if not candidate:
        return None
    probe = subprocess.run(
        [candidate, "-I", "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    return candidate if probe.returncode == 0 and probe.stdout.strip() == version else None


def test_package_compiles_with_a_python_311_interpreter() -> None:
    """Byte-compile the whole package with a real 3.11 interpreter, when one is on PATH or installed by uv (it catches
    3.11 grammar the parse check cannot, such as a backslash inside an f-string replacement field)."""
    python311 = find_python("3.11") or ""
    if not python311:
        pytest.skip("no Python 3.11 interpreter on PATH or installed by uv")
    script = (
        "import sys\n"
        "failed = []\n"
        "for path in sys.argv[1:]:\n"
        "    try:\n"
        "        compile(open(path, encoding='utf-8').read(), path, 'exec', dont_inherit=True)\n"
        "    except SyntaxError as error:\n"
        "        failed.append(f'{path}: {error}')\n"
        "print('\\n'.join(failed))\n"
        "sys.exit(1 if failed else 0)\n"
    )
    result = subprocess.run(
        [python311, "-I", "-c", script, *map(str, PACKAGE_FILES)], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr


# --------------------------------------------------------------------------------------------------------------------
# Worker runtime files: Python 3.10, standard library and their own upstream stack only, never boileroom.
# --------------------------------------------------------------------------------------------------------------------

RUNTIME_FILES = sorted(
    [
        REPO_ROOT / "boileroom/models/_worker.py",
        *(REPO_ROOT / "boileroom/models").glob("*/runtime.py"),
        *(REPO_ROOT / "boileroom/models").glob("*/kit/*.py"),
    ]
)
#: Standard-library names added in 3.11 or 3.12 (module, or module attribute), which a 3.10 interpreter lacks.
NEWER_STDLIB = {
    "tomllib": None,
    "typing": {
        "Self",
        "LiteralString",
        "Never",
        "assert_never",
        "reveal_type",
        "Required",
        "NotRequired",
        "Unpack",
        "TypeVarTuple",
        "override",
        "TypeAliasType",
        "assert_type",
        "dataclass_transform",
    },
    "datetime": {"UTC"},
    "enum": {"StrEnum", "verify", "ReprEnum", "EnumCheck"},
    "asyncio": {"TaskGroup", "timeout", "Runner"},
    "itertools": {"batched"},
    "contextlib": {"chdir"},
    "hashlib": {"file_digest"},
}
NEWER_BUILTINS = {"ExceptionGroup", "BaseExceptionGroup"}
#: Classes that run in boileroom's own interpreter, never in the worker: ``_worker.py`` is both the parent's
#: ``ModelWorker`` and the child's script, and only the parent side may import boileroom (lazily, in its methods).
PARENT_SIDE_CLASSES = {"_worker.py": frozenset({"ModelWorker"})}


def runtime_problems(source: str, parent_side: frozenset[str] = frozenset()) -> list[str]:
    """Return what keeps ``source`` from running standalone on Python 3.10 without boileroom.

    Parameters
    ----------
    source : str
        The file's text.
    parent_side : frozenset[str]
        Names of top-level classes that only boileroom's own interpreter runs; boileroom imports inside them are allowed.

    Returns
    -------
    list[str]
        One message per problem: 3.11+ grammar, PEP 701 f-strings, imports of boileroom (by statement, relative, or by
        ``import_module``/``__import__`` of a constant), and standard-library names newer than 3.10.
    """
    try:
        tree = ast.parse(source, feature_version=(3, 10))
    except SyntaxError as error:
        return [f"not Python 3.10 grammar: {error}"]
    problems = pep701_only_fstrings(source)
    parent_nodes = {
        id(inner)
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name in parent_side
        for inner in ast.walk(node)
    }
    for node in ast.walk(tree):
        line = getattr(node, "lineno", 0)
        modules: list[str] = []
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                problems.append(f"line {line}: relative import (the file runs standalone, outside its package)")
            modules = [node.module or ""]
            newer = NEWER_STDLIB.get(node.module or "")
            for alias in node.names:
                if newer and alias.name in newer:
                    problems.append(f"line {line}: {node.module}.{alias.name} is newer than Python 3.10")
        elif isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            if name in {"import_module", "__import__"} and isinstance(node.args[0].value, str):
                modules = [node.args[0].value]
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            newer = NEWER_STDLIB.get(node.value.id)
            if newer and node.attr in newer:
                problems.append(f"line {line}: {node.value.id}.{node.attr} is newer than Python 3.10")
        elif isinstance(node, ast.Name) and node.id in NEWER_BUILTINS:
            problems.append(f"line {line}: {node.id} is newer than Python 3.10")
        for module in modules:
            if module.split(".")[0] == "boileroom" and id(node) not in parent_nodes:
                problems.append(f"line {line}: imports {module}")
            if module in NEWER_STDLIB and NEWER_STDLIB[module] is None:
                problems.append(f"line {line}: {module} is newer than Python 3.10")
    return problems


def test_runtime_files_include_every_worker_entry_point() -> None:
    """Every file a worker interpreter runs is in the checked set: the worker, each core's runtime, the kit smoke."""
    from boileroom.models import _worker
    from boileroom.models.opendde.core import OpenDDECore
    from boileroom.models.protenix.core import ProtenixCore

    entries = {
        Path(_worker.__file__).resolve(),
        ProtenixCore.RUNTIME_PATH.resolve(),
        OpenDDECore.RUNTIME_PATH.resolve(),
        (REPO_ROOT / "boileroom/models/alphafold/runtime.py").resolve(),
        (REPO_ROOT / "boileroom/models/esmfold2/kit/kit_smoke.py").resolve(),
    }
    assert entries <= {path.resolve() for path in RUNTIME_FILES}
    source = (REPO_ROOT / "boileroom/models/alphafold/core.py").read_text(encoding="utf-8")
    assert 'runtime_path=Path(__file__).with_name("runtime.py")' in source


@pytest.mark.parametrize("path", RUNTIME_FILES, ids=lambda path: str(path.relative_to(REPO_ROOT)))
def test_runtime_file_runs_standalone_on_python_310(path: Path) -> None:
    parent_side = PARENT_SIDE_CLASSES.get(path.name, frozenset())
    assert runtime_problems(path.read_text(encoding="utf-8"), parent_side) == []


def test_only_the_parent_side_of_the_worker_may_import_boileroom() -> None:
    source = "class ModelWorker:\n    def f(self):\n        from boileroom.optimization import X\n"
    assert runtime_problems(source, PARENT_SIDE_CLASSES["_worker.py"]) == []
    child = "def _serve():\n    from boileroom.optimization import X\n"
    assert runtime_problems(child, PARENT_SIDE_CLASSES["_worker.py"]) == ["line 2: imports boileroom.optimization"]


@pytest.mark.parametrize(
    ("source", "problem"),
    [
        ("from boileroom.optimization import KIT_FAMILIES\n", "imports boileroom.optimization"),
        ("def f():\n    import boileroom\n", "imports boileroom"),
        ("import importlib\nimportlib.import_module('boileroom.utils')\n", "imports boileroom.utils"),
        ("from .core import ProtenixCore\n", "relative import"),
        ("import tomllib\n", "tomllib is newer"),
        ("from typing import Self\n", "typing.Self is newer"),
        ("import datetime\nnow = datetime.UTC\n", "datetime.UTC is newer"),
        ("try:\n    pass\nexcept* ValueError:\n    pass\n", "not Python 3.10 grammar"),
        ("raise ExceptionGroup('x', [ValueError()])\n", "ExceptionGroup is newer"),
    ],
)
def test_the_runtime_check_finds_what_a_310_worker_cannot_run(source: str, problem: str) -> None:
    assert any(problem in found for found in runtime_problems(source)), runtime_problems(source)


def test_the_runtime_check_accepts_310_standard_library_code() -> None:
    source = "from __future__ import annotations\nimport importlib\nfrom typing import Any\nx: int | None = None\n"
    assert runtime_problems(source) == []


@pytest.mark.parametrize("version", ["3.10", "3.11"])
def test_runtime_files_load_on_a_real_interpreter_without_boileroom(version: str, tmp_path: Path) -> None:
    """Load every runtime file as its worker does (isolated mode, outside the package) and check boileroom stays out."""
    python = find_python(version) or ""
    if not python:
        pytest.skip(f"no Python {version} interpreter on PATH or installed by uv")
    script = (
        "import importlib.util, sys\n"
        "for index, path in enumerate(sys.argv[1:]):\n"
        "    spec = importlib.util.spec_from_file_location(f'_runtime_{index}', path)\n"
        "    module = importlib.util.module_from_spec(spec)\n"
        "    spec.loader.exec_module(module)\n"
        "loaded = sorted(name for name in sys.modules if name.split('.')[0] == 'boileroom')\n"
        "print(sys.version_info[:2], loaded)\n"
        "sys.exit(1 if loaded else 0)\n"
    )
    result = subprocess.run(
        [python, "-I", "-c", script, *map(str, RUNTIME_FILES)],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stdout + result.stderr
