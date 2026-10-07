"""Contracts of the kit Dockerfiles and the OpenDDE Dockerfile, read the way Docker reads them.

The Dockerfiles are parsed (instructions, stages, ARG scope, ENV resolution; ``_dockerfile.py``) and their build-time
smoke steps are executed with ``/bin/sh`` against stand-in interpreters, packages and ``ldd``: no Docker, no GPU, no
network. Every build route is covered: ``docker build`` (``WHEELS_FROM=build``), the Modal build of the ESMFold2 kit
(``WHEELS_FROM=defer``, then the build steps of ``boileroom.images.modal._kit_build``), the Protenix kit image and the
OpenDDE image.
"""

from __future__ import annotations

import ast
import importlib.util
import shlex
import subprocess
import sys
import textwrap
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest
from _dockerfile import Dockerfile, Instruction, expand

from boileroom.images import metadata

pytestmark = pytest.mark.contract

REPO = Path(__file__).resolve().parents[2]
ESMFOLD2_KIT = metadata.KIT_IMAGE_SPECS_BY_KEY["esmfold2"]
PROTENIX_KIT = metadata.KIT_IMAGE_SPECS_BY_KEY["protenix"]
OPENDDE = metadata.get_model_image_spec("opendde")
KIT_SCRIPTS_DIR = "/opt/boileroom-kit"
SH = "/bin/sh"
SYSTEM_PATH = "/usr/bin:/bin"
#: The images whose last build step is the cuEquivariance kernel smoke (imports plus ``ldd`` of ``libcue_ops.so``).
CUE_SMOKE_SPECS = {"protenix-kit": PROTENIX_KIT, "opendde": OPENDDE}
#: Every Dockerfile that fetches the kit at ``KIT_COMMIT``.
KIT_FETCHING_SPECS = {"esmfold2-kit": ESMFOLD2_KIT, "protenix-kit": PROTENIX_KIT, "opendde": OPENDDE}
OTHER_COMMIT = "0123456789abcdef0123456789abcdef01234567"


def _dockerfile(spec: metadata.RuntimeImageSpec) -> Dockerfile:
    return Dockerfile(spec.dockerfile_path)


def _runtime_module(path: Path, name: str) -> ModuleType:
    """Load a worker runtime file standalone, as its worker interpreter does (it never imports boileroom)."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _words(instruction: Instruction) -> list[str]:
    """Split a RUN body into shell words (quotes removed, ``&&`` and ``;`` kept as words when they stand alone)."""
    return shlex.split(instruction.body.replace(";", " ; "), posix=True)


def _python_scripts(instruction: Instruction) -> list[tuple[str, list[str], str]]:
    """Return every ``<python> [flags] -c <script>`` of a RUN body as (interpreter word, flags, script)."""
    words = _words(instruction)
    scripts = []
    for index, word in enumerate(words):
        if word != "-c" or index + 1 >= len(words):
            continue
        start = index - 1
        while start >= 0 and words[start].startswith("-"):
            start -= 1
        interpreter = words[start]
        if Path(interpreter.strip('"')).name.startswith("python"):
            scripts.append((interpreter, words[start + 1 : index], words[index + 1]))
    return scripts


def _cue_smoke_run(dockerfile: Dockerfile) -> Instruction:
    """Return the one final-stage RUN that checks ``libcue_ops.so``."""
    runs = [run for run in dockerfile.of("RUN") if "libcue_ops.so" in run.body]
    assert len(runs) == 1, f"{dockerfile.path}: expected one RUN checking libcue_ops.so, found {len(runs)}"
    return runs[0]


def _imported_modules(script: str) -> list[str]:
    """Return the modules a ``-c`` script imports by statement, in order."""
    names: list[str] = []
    for node in ast.walk(ast.parse(script)):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def _write(path: Path, text: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return path


def _executable(path: Path, text: str) -> Path:
    _write(path, text)
    path.chmod(0o755)
    return path


@dataclass
class _Venv:
    """A stand-in for an image's Python install: a real interpreter with fake packages in its own site-packages.

    ``python -I`` ignores ``PYTHONPATH`` but keeps the interpreter's own site-packages, so the fake packages live there,
    exactly where the image's packages live.
    """

    root: Path
    site: Path

    @property
    def python(self) -> Path:
        return self.root / "bin" / "python"

    def package(self, name: str, text: str = "") -> None:
        """Write ``name`` (dotted) as a package with ``text`` as its ``__init__``, creating parent packages."""
        parts = name.split(".")
        for depth in range(1, len(parts)):
            init = self.site.joinpath(*parts[:depth], "__init__.py")
            if not init.exists():
                _write(init)
        _write(self.site.joinpath(*parts, "__init__.py"), text)

    def module(self, name: str, text: str = "") -> None:
        """Write ``name`` (dotted) as a plain module, creating parent packages."""
        parts = name.split(".")
        for depth in range(1, len(parts)):
            init = self.site.joinpath(*parts[:depth], "__init__.py")
            if not init.exists():
                _write(init)
        _write(self.site.joinpath(*parts[:-1], f"{parts[-1]}.py"), text)


def _make_venv(root: Path) -> _Venv:
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(root)], check=True, timeout=120)
    site = root / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    assert site.is_dir()
    return _Venv(root, site)


# --------------------------------------------------------------------------------------------------------------------
# The cuEquivariance kernel smoke of the Protenix kit image and the OpenDDE image, executed.
# --------------------------------------------------------------------------------------------------------------------

FAKE_LDD = """\
#!/bin/sh
# Stand-in for ldd: the driver's libcuda.so.1 is never found (a builder has no GPU); libcublas resolves only when the
# nvidia-* wheel directory that carries it is on LD_LIBRARY_PATH, as it does for the real loader.
printf '%s\\n' "$1" >> "$FAKE_LDD_LOG"
printf '\\tlinux-vdso.so.1 (0x00007ffd)\\n'
printf '\\tlibcuda.so.1 => not found\\n'
case ":$LD_LIBRARY_PATH:" in
  */nvidia/cublas/lib:*) printf '\\tlibcublas.so.12 => /wheels/nvidia/cublas/lib/libcublas.so.12 (0x00007f00)\\n' ;;
  *) printf '\\tlibcublas.so.12 => not found\\n' ;;
esac
if [ -n "${FAKE_LDD_EXTRA:-}" ]; then printf '\\t%s\\n' "$FAKE_LDD_EXTRA"; fi
exit "${FAKE_LDD_STATUS:-0}"
"""


@dataclass
class _CueImage:
    """The Protenix kit or OpenDDE image as the cuEquivariance smoke step sees it."""

    key: str
    dockerfile: Dockerfile
    venv: _Venv
    bin: Path
    ldd_log: Path

    def run_smoke(self, **extra_env: str) -> subprocess.CompletedProcess[str]:
        """Execute the smoke RUN with ``/bin/sh -c``, its ARG and ENV values in effect, against the stand-ins."""
        run = _cue_smoke_run(self.dockerfile)
        # OpenDDE installs its interpreter at ${OPENDDE_HOME}: point the build argument at the stand-in.
        build_args = {"OPENDDE_HOME": str(self.venv.root)} if self.key == "opendde" else None
        args, env = self.dockerfile.scope(until=run, build_args=build_args)
        environ = {
            **args,
            **env,
            "PATH": f"{self.bin}:{SYSTEM_PATH}",
            "FAKE_LDD_LOG": str(self.ldd_log),
            "HOME": str(self.venv.root),
            **extra_env,
        }
        return subprocess.run(
            [SH, "-c", run.body], env=environ, capture_output=True, text=True, timeout=120, cwd=self.venv.root
        )

    def ldd_calls(self) -> list[str]:
        return self.ldd_log.read_text().splitlines() if self.ldd_log.exists() else []


@pytest.fixture(params=sorted(CUE_SMOKE_SPECS))
def cue_image(request: pytest.FixtureRequest, tmp_path: Path) -> _CueImage:
    """A healthy stand-in image: every smoke import works and libcue_ops.so links all but libcuda.so.1."""
    key = request.param
    venv = _make_venv(tmp_path / "python")
    for name in ("cuequivariance_torch", "protenix", "protenix_opt", "torch", "opendde_opt", "runner", "numpy"):
        venv.package(name)
    venv.module("protenix.model.layer_norm")
    venv.module("opendde_opt.lnstream")
    venv.module("opendde_opt.lncensus")
    venv.module("runner.batch_inference")
    _write(venv.site / "torch" / "lib" / "libtorch_cuda.so", "elf")
    # nvidia-* wheels: a namespace package whose wheels each carry a lib directory.
    _write(venv.site / "nvidia" / "cublas" / "lib" / "libcublas.so.12", "elf")
    # cuEquivariance's kernel packages load libcue_ops.so (and so need libcuda.so.1) when imported: the smoke must not.
    for name in ("cuequivariance_ops", "cuequivariance_ops_torch"):
        venv.package(name, "raise ImportError('libcuda.so.1: cannot open shared object file')\n")
    _write(venv.site / "cuequivariance_ops" / "lib" / "libcue_ops.so", "elf")
    # GCC 13's libstdc++ next to OpenDDE's interpreter (a venv's lib directory stands in for ${OPENDDE_HOME}/lib).
    _write(venv.root / "lib" / "libstdc++.so.6", "GLIBCXX_3.4.30\0GLIBCXX_3.4.31\0GLIBCXX_3.4.32\0")
    bin_dir = tmp_path / "bin"
    _executable(bin_dir / "ldd", FAKE_LDD)
    # The Protenix kit's smoke (and OpenDDE's last numpy check) runs the image's `python` from PATH.
    _executable(bin_dir / "python", f'#!/bin/sh\nexec "{venv.python}" "$@"\n')
    return _CueImage(key, Dockerfile(CUE_SMOKE_SPECS[key].dockerfile_path), venv, bin_dir, tmp_path / "ldd.log")


def test_cue_smoke_passes_when_only_libcuda_is_unresolved(cue_image: _CueImage) -> None:
    result = cue_image.run_smoke()

    assert result.returncode == 0, result.stdout + result.stderr
    library = cue_image.venv.site / "cuequivariance_ops" / "lib" / "libcue_ops.so"
    assert cue_image.ldd_calls() == [str(library)]
    # No unresolved library and ldd's exit status 0: the report maps the library to a falsy value.
    assert f"libcue_ops.so: {{'{library}': 0}}" in result.stdout


def test_cue_smoke_fails_on_another_unresolved_library(cue_image: _CueImage) -> None:
    result = cue_image.run_smoke(FAKE_LDD_EXTRA="libcusparseLt.so.0 => not found")

    assert result.returncode != 0
    assert "libcusparseLt.so.0" in result.stdout


def test_cue_smoke_resolves_against_the_nvidia_wheel_libraries(cue_image: _CueImage) -> None:
    for item in sorted((cue_image.venv.site / "nvidia").rglob("*"), reverse=True):
        item.unlink() if item.is_file() else item.rmdir()

    result = cue_image.run_smoke()

    assert result.returncode != 0
    assert "libcublas.so.12" in result.stdout


def test_cue_smoke_fails_when_ldd_fails(cue_image: _CueImage) -> None:
    result = cue_image.run_smoke(FAKE_LDD_STATUS="1")

    assert result.returncode != 0
    assert cue_image.ldd_calls()


def test_cue_smoke_fails_without_the_kernel_library(cue_image: _CueImage) -> None:
    (cue_image.venv.site / "cuequivariance_ops" / "lib" / "libcue_ops.so").unlink()

    result = cue_image.run_smoke()

    assert result.returncode != 0
    assert "libcue_ops.so: missing" in result.stdout
    assert cue_image.ldd_calls() == []


def _smoke_imports() -> list[tuple[str, str, bool]]:
    """Return (image, top-level module, checked before the library) for every module a cuEquivariance smoke imports."""
    found: dict[tuple[str, str], bool] = {}
    for key, spec in sorted(CUE_SMOKE_SPECS.items()):
        before_ldd = True
        for _, _, script in _python_scripts(_cue_smoke_run(Dockerfile(spec.dockerfile_path))):
            if "ldd" in script:
                before_ldd = False
                continue
            for name in _imported_modules(script):
                found.setdefault((key, name.split(".")[0]), before_ldd)
    return sorted((key, module, before_ldd) for (key, module), before_ldd in found.items())


@pytest.mark.parametrize(("cue_image", "module", "before_ldd"), _smoke_imports(), indirect=["cue_image"])
def test_cue_smoke_fails_when_a_kit_import_fails(cue_image: _CueImage, module: str, before_ldd: bool) -> None:
    cue_image.venv.package(module, "raise ImportError('undefined symbol: _ZN3c104cuda')\n")

    result = cue_image.run_smoke()

    assert result.returncode != 0
    assert "undefined symbol" in result.stderr
    # The kit imports run first: a broken kit stops the step before the library check.
    assert (cue_image.ldd_calls() == []) is before_ldd


@pytest.mark.parametrize("contents", ["GLIBCXX_3.4.30\0GLIBCXX_3.4.31\0", None], ids=["gcc12", "absent"])
def test_opendde_smoke_fails_without_gcc13_libstdcxx(tmp_path: Path, contents: str | None) -> None:
    venv = _make_venv(tmp_path / "python")
    for name in ("cuequivariance_torch", "torch", "opendde_opt", "runner", "numpy"):
        venv.package(name)
    for name in ("opendde_opt.lnstream", "opendde_opt.lncensus", "runner.batch_inference"):
        venv.module(name)
    if contents is not None:
        _write(venv.root / "lib" / "libstdc++.so.6", contents)
    bin_dir = tmp_path / "bin"
    _executable(bin_dir / "ldd", FAKE_LDD)
    _executable(bin_dir / "python", f'#!/bin/sh\nexec "{venv.python}" "$@"\n')
    image = _CueImage("opendde", Dockerfile(OPENDDE.dockerfile_path), venv, bin_dir, tmp_path / "ldd.log")

    result = image.run_smoke()

    assert result.returncode != 0
    assert image.ldd_calls() == []


def _final_stage_python_scripts() -> list[tuple[str, Instruction, str, list[str], str]]:
    found: list[tuple[str, Instruction, str, list[str], str]] = []
    for key, spec in KIT_FETCHING_SPECS.items():
        for run in _dockerfile(spec).of("RUN"):
            found.extend((key, run, interpreter, flags, script) for interpreter, flags, script in _python_scripts(run))
    return found


def test_inline_build_scripts_are_found() -> None:
    keys = {key for key, *_ in _final_stage_python_scripts()}
    assert keys >= set(CUE_SMOKE_SPECS)


@pytest.mark.parametrize(
    ("key", "run", "interpreter", "flags", "script"),
    _final_stage_python_scripts(),
    ids=lambda value: value if isinstance(value, str) and len(value) < 30 else "",
)
def test_inline_build_scripts_parse_as_python_310_and_never_import_the_driver_linked_ops(
    key: str, run: Instruction, interpreter: str, flags: list[str], script: str
) -> None:
    # The kit interpreters are 3.11; keep the inline scripts parseable by the oldest worker Python as well.
    tree = ast.parse(script, feature_version=(3, 10))
    calls = [
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute | ast.Name)
        and getattr(node.func, "attr", getattr(node.func, "id", "")) in {"import_module", "__import__"}
        and node.args
        and isinstance(node.args[0], ast.Constant)
    ]
    imported = _imported_modules(script) + calls
    assert not [name for name in imported if name.split(".")[0] in {"cuequivariance_ops", "cuequivariance_ops_torch"}]


def test_both_images_check_libcue_ops_with_the_same_script() -> None:
    scripts = {
        key: [script for _, _, script in _python_scripts(_cue_smoke_run(Dockerfile(spec.dockerfile_path)))]
        for key, spec in CUE_SMOKE_SPECS.items()
    }
    ldd_scripts = {key: [script for script in found if "ldd" in script] for key, found in scripts.items()}
    assert all(len(found) == 1 for found in ldd_scripts.values()), ldd_scripts
    assert ldd_scripts["protenix-kit"] == ldd_scripts["opendde"]
    tree = ast.parse(ldd_scripts["opendde"][0])
    allowed = {
        element.value
        for node in ast.walk(tree)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub) and isinstance(node.right, ast.Set)
        for element in node.right.elts
        if isinstance(element, ast.Constant)
    }
    (check,) = OPENDDE.interpreter_smoke_checks
    assert allowed == set(check.allowed_unresolved) == {"libcuda.so.1"}


def test_opendde_build_smoke_matches_the_interpreter_check_of_its_spec() -> None:
    from boileroom.models.opendde.core import DEFAULT_OPENDDE_PYTHON

    dockerfile = _dockerfile(OPENDDE)
    run = _cue_smoke_run(dockerfile)
    args, env = dockerfile.scope(until=run)
    values = {**args, **env}
    (check,) = OPENDDE.interpreter_smoke_checks
    scripts = _python_scripts(run)
    kit_interpreters = {expand(interpreter, values) for interpreter, _, _ in scripts if "OPENDDE_HOME" in interpreter}
    assert kit_interpreters == {check.python} == {DEFAULT_OPENDDE_PYTHON}
    assert all(flags == ["-I"] for interpreter, flags, _ in scripts if "OPENDDE_HOME" in interpreter)
    imports = [name for interpreter, _, script in scripts if "ldd" not in script for name in _imported_modules(script)]
    imports = [name for name in imports if name != "numpy"]
    assert imports == list(check.imports)
    words = [expand(word, values) for word in _words(run)]
    assert f"LD_LIBRARY_PATH={check.library_path}" in words
    for library, version in check.required_symbol_versions:
        assert words[words.index(version) + 1] == library
    assert [(package, library) for package, library in check.driver_linked_libraries] == [
        ("cuequivariance_ops", "libcue_ops.so")
    ]


@pytest.mark.parametrize("key", sorted(CUE_SMOKE_SPECS))
def test_cue_smoke_is_the_last_build_step_after_the_kit_install(key: str) -> None:
    dockerfile = Dockerfile(CUE_SMOKE_SPECS[key].dockerfile_path)
    smoke = _cue_smoke_run(dockerfile)
    steps = dockerfile.stage()
    install = [step for step in steps if step.keyword == "RUN" and "run.sh install" in step.body]
    assert len(install) == 1
    assert steps.index(install[0]) < steps.index(smoke)
    assert not [step for step in steps[steps.index(smoke) + 1 :] if step.keyword in {"RUN", "COPY", "ADD"}]
    if key == "opendde":
        (copy,) = [step for step in steps if step.keyword == "COPY" and "--from=libstdcxx" in step.flags]
        assert steps.index(copy) < steps.index(smoke)


def test_protenix_kit_smoke_runs_the_image_python_311() -> None:
    dockerfile = _dockerfile(PROTENIX_KIT)
    smoke = _cue_smoke_run(dockerfile)
    scripts = _python_scripts(smoke)
    assert len(scripts) == 2
    assert {(interpreter, tuple(flags)) for interpreter, flags, _ in scripts} == {("python", ("-I",))}
    # The kernel package, stock's fast LayerNorm (the kit modes' LAYERNORM_TYPE) and the kit itself.
    imports = [name for _, _, script in scripts if "ldd" not in script for name in _imported_modules(script)]
    assert imports == ["cuequivariance_torch", "protenix.model.layer_norm", "protenix_opt"]
    steps = dockerfile.stage()
    (install,) = [
        step for step in steps if step.keyword == "RUN" and f"cpython-{PROTENIX_KIT.python_version}." in step.body
    ]
    words = _words(install)
    link = words.index("ln")
    assert words[link + 1 : link + 4] == ["-sf", "/usr/local/bin/python3", "/usr/local/bin/python"]
    assert steps.index(install) < steps.index(smoke)


def _fused_layernorm_arch_check() -> tuple[Instruction, str, list[str]]:
    """Return the Protenix kit's build RUN, flags and script that check the fused LayerNorm's SASS with cuobjdump."""
    found = [
        (run, script, flags)
        for run in _dockerfile(PROTENIX_KIT).of("RUN")
        for _, flags, script in _python_scripts(run)
        if "cuobjdump" in script
    ]
    assert len(found) == 1, found
    run, script, flags = found[0]
    return run, script, flags


def test_protenix_kit_checks_the_fused_layernorm_carries_a100_and_h100_code() -> None:
    """Upstream's -gencode list follows the devel base's nvcc; the build must fail if it ever drops sm_80 or sm_90."""
    run, script, _ = _fused_layernorm_arch_check()
    words = _words(run)
    # Right after the build-time import that compiles it, in the same layer.
    assert words.index("import protenix.model.layer_norm") < words.index(script)
    assert "'sm_80', 'sm_90'" in script and "--list-elf" in script


@pytest.mark.parametrize(
    ("elf", "passes"),
    [
        ("fast_layer_norm_cuda_v2.1.sm_80.cubin\nfast_layer_norm_cuda_v2.2.sm_90.cubin\nx.3.sm_100.cubin", True),
        ("fast_layer_norm_cuda_v2.1.sm_90.cubin", False),
        ("fast_layer_norm_cuda_v2.1.sm_80.cubin\nfast_layer_norm_cuda_v2.2.sm_86.cubin", False),
        ("", False),
    ],
    ids=["a100-h100-b200", "h100-only", "no-sm90", "ptx-only"],
)
def test_fused_layernorm_arch_check_executed(tmp_path: Path, elf: str, passes: bool) -> None:
    """The check run against a stand-in extension and a stand-in cuobjdump: it passes only with sm_80 and sm_90."""
    _, script, flags = _fused_layernorm_arch_check()
    venv = _make_venv(tmp_path / "venv")
    extension = _write(tmp_path / "fast_layer_norm_cuda_v2.so")
    venv.package("protenix.model.layer_norm")
    venv.module(
        "protenix.model.layer_norm.layer_norm",
        f"import types\nfast_layer_norm_cuda_v2 = types.SimpleNamespace(__file__={str(extension)!r})\n",
    )
    cuda_home = tmp_path / "cuda"
    elf_lines = "".join(f"ELF file {index}: {line}\\n" for index, line in enumerate(elf.splitlines(), 1))
    _executable(
        cuda_home / "bin" / "cuobjdump",
        f'#!/bin/sh\n[ "$1" = --list-elf ] && [ "$2" = {shlex.quote(str(extension))} ] || exit 9\nprintf "{elf_lines}"\n',
    )
    result = subprocess.run(
        [str(venv.python), *flags, "-c", script],
        env={"CUDA_HOME": str(cuda_home), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert (result.returncode == 0) is passes, result.stdout + result.stderr
    if passes:
        assert "['sm_100', 'sm_80', 'sm_90']" in result.stdout


def test_fused_layernorm_arch_check_fails_when_cuobjdump_fails(tmp_path: Path) -> None:
    _, script, flags = _fused_layernorm_arch_check()
    venv = _make_venv(tmp_path / "venv")
    venv.package("protenix.model.layer_norm")
    venv.module(
        "protenix.model.layer_norm.layer_norm",
        "import types\nfast_layer_norm_cuda_v2 = types.SimpleNamespace(__file__='/nowhere.so')\n",
    )
    _executable(tmp_path / "cuda" / "bin" / "cuobjdump", "#!/bin/sh\nexit 1\n")
    result = subprocess.run(
        [str(venv.python), *flags, "-c", script],
        env={"CUDA_HOME": str(tmp_path / "cuda"), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0


# --------------------------------------------------------------------------------------------------------------------
# The ESMFold2 kit: both build routes run kit_wheels.sh, then kit_finish.sh, which ends with kit_smoke.py.
# --------------------------------------------------------------------------------------------------------------------

LOGGING_SH = """\
#!/bin/sh
printf '%s\\n' "$*" >> "$SH_LOG"
"""


def _wheels_runs(dockerfile: Dockerfile) -> list[Instruction]:
    return [run for run in dockerfile.of("RUN") if "WHEELS_FROM" in run.body]


def _run_wheels_steps(tmp_path: Path, build_args: Mapping[str, str] | None) -> tuple[list[str], list[int]]:
    """Execute the ESMFold2 kit's WHEELS_FROM RUN steps with their resolved ARGs; return the scripts run and the codes."""
    dockerfile = _dockerfile(ESMFOLD2_KIT)
    bin_dir = tmp_path / "bin"
    _executable(bin_dir / "sh", LOGGING_SH)
    log = tmp_path / "sh.log"
    codes = []
    for run in _wheels_runs(dockerfile):
        args, env = dockerfile.scope(until=run, build_args=build_args)
        environ = {**args, **env, "PATH": f"{bin_dir}:{SYSTEM_PATH}", "SH_LOG": str(log)}
        codes.append(subprocess.run([SH, "-c", run.body], env=environ, capture_output=True, timeout=60).returncode)
    return (log.read_text().splitlines() if log.exists() else []), codes


def _modal_build_step_scripts(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Run the Modal build's steps of the ESMFold2 kit in order; return the scripts each step runs."""
    from boileroom.images import modal as modal_images

    calls: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: calls.append(list(command)))
    scripts = []
    for step in modal_images._kit_build(ESMFOLD2_KIT).steps:
        calls.clear()
        step.function()
        scripts.append([" ".join(command[1:]) for command in calls if command[0] == "sh"])
    return scripts


EXPECTED_SCRIPTS = [f"{KIT_SCRIPTS_DIR}/kit_wheels.sh", f"{KIT_SCRIPTS_DIR}/kit_finish.sh"]


def test_esmfold2_docker_build_runs_both_kit_scripts(tmp_path: Path) -> None:
    scripts, codes = _run_wheels_steps(tmp_path, None)

    assert codes == [0, 0]
    assert scripts == EXPECTED_SCRIPTS


def test_esmfold2_modal_build_runs_each_kit_script_in_its_own_build_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One script per Modal build step, as one per RUN in the Dockerfile: a failed finish never reruns the compile."""
    from boileroom.images.modal import ESMFOLD2_KIT_BUILD_ARGS

    image_scripts, codes = _run_wheels_steps(tmp_path, ESMFOLD2_KIT_BUILD_ARGS)
    step_scripts = _modal_build_step_scripts(monkeypatch)

    assert codes == [0, 0]
    assert image_scripts == []
    assert step_scripts == [[script] for script in EXPECTED_SCRIPTS]


@pytest.mark.parametrize("wheels_from", ["prebuilt", ""])
def test_esmfold2_kit_refuses_an_unknown_wheels_route(tmp_path: Path, wheels_from: str) -> None:
    scripts, codes = _run_wheels_steps(tmp_path, {"WHEELS_FROM": wheels_from})

    assert codes == [2, 2]
    assert scripts == []


def _copies_to_kit_dir(dockerfile: Dockerfile) -> list[tuple[Instruction, list[str]]]:
    copies = []
    for copy in dockerfile.of("COPY"):
        if any(flag.startswith("--from") for flag in copy.flags):
            continue
        *sources, destination = shlex.split(copy.body)
        if destination.rstrip("/") == KIT_SCRIPTS_DIR:
            copies.append((copy, sources))
    return copies


def test_esmfold2_kit_scripts_are_copied_into_the_image_before_they_run() -> None:
    dockerfile = _dockerfile(ESMFOLD2_KIT)
    steps = dockerfile.stage()
    copies = _copies_to_kit_dir(dockerfile)
    finish = (ESMFOLD2_KIT.context_path / "kit_finish.sh").read_text()
    smoke_line = f"python -I {KIT_SCRIPTS_DIR}/kit_smoke.py"
    assert smoke_line in finish.splitlines()
    for run in _wheels_runs(dockerfile):
        invoked = [word for word in _words(run) if word.startswith(f"{KIT_SCRIPTS_DIR}/")]
        assert len(invoked) == 1
        needed = {Path(invoked[0]).name} | ({"kit_smoke.py"} if invoked[0].endswith("kit_finish.sh") else set())
        copied = {source for copy, sources in copies if steps.index(copy) < steps.index(run) for source in sources}
        assert needed <= copied, f"{run.line}: {needed - copied} not copied before it runs"
    for _, sources in copies:
        for source in sources:
            assert (ESMFOLD2_KIT.context_path / source).is_file()


def test_esmfold2_kit_stack_default_is_the_modal_build_stack() -> None:
    from boileroom.images.modal import ESMFOLD2_KIT_BUILD_ARGS, ESMFOLD2_KIT_STACK

    args, _ = _dockerfile(ESMFOLD2_KIT).scope()
    assert args["STACK"] == ESMFOLD2_KIT_STACK == ESMFOLD2_KIT_BUILD_ARGS["STACK"]
    assert args["WHEELS_FROM"] == "build"
    assert ESMFOLD2_KIT_BUILD_ARGS["WHEELS_FROM"] == "defer"


# kit_finish.sh, executed: `cd` is overridden by a shell function, every other command by a PATH stand-in. The python
# stand-in logs its arguments and runs kit_smoke.py itself, with the interpreter flags it was given, on a stand-in kit.

PYTHON_SHIM = """\
import os, sys
with open(os.environ["SHIM_LOG"], "a") as log:
    log.write(" ".join(["python", *sys.argv[1:]]) + "\\n")
arguments = [argument.replace("/opt/boileroom-kit/", os.environ["SHIM_KIT_DIR"] + "/") for argument in sys.argv[1:]]
if any(argument.endswith("kit_smoke.py") for argument in arguments):
    os.execv(os.environ["SHIM_KIT_PYTHON"], [os.environ["SHIM_KIT_PYTHON"], *arguments])
"""

KIT_ATTN = """\
ENV_REQUIRE = "ESMFOLD2_OPT_REQUIRE_FAST_ENV"
COMMON_MODULE = "transformers.models.esmfold2.modeling_esmfold2_common"
ESMC_MODULE = "transformers.models.esmc.modeling_esmc"
FLAG = "FLASH_ATTN_AVAILABLE"
REQUIRED = (
    ("atom_attn", "flash_attn", "flash_attn"),
    ("esmc_mlp", "te", "transformer_engine.pytorch"),
    ("esmc_rope", "flash_attn_triton", "flash_attn.ops.triton.rotary"),
)


def require_refusal_metadata(environ=None, image=None, vers=None):
    return None


def metadata_words(vers=None):
    return "flash_attn=2.8.3 transformer_engine=2.15.0 xformers=0.0.35"
"""


@dataclass
class _EsmKit:
    venv: _Venv
    bin: Path
    log: Path

    def finish(self, **extra_env: str) -> subprocess.CompletedProcess[str]:
        script = ESMFOLD2_KIT.context_path / "kit_finish.sh"
        program = 'cd() { printf "cd %s\\n" "$*" >> "$SHIM_LOG"; }; . "$0"'
        environ = {
            "PATH": f"{self.bin}:{SYSTEM_PATH}",
            "HOME": str(self.venv.root),
            "SHIM_LOG": str(self.log),
            "SHIM_KIT_DIR": str(ESMFOLD2_KIT.context_path),
            "SHIM_KIT_PYTHON": str(self.venv.python),
            **extra_env,
        }
        return subprocess.run(
            [SH, "-c", program, str(script)], env=environ, capture_output=True, text=True, timeout=120
        )

    def calls(self) -> list[str]:
        return self.log.read_text().splitlines() if self.log.exists() else []


@pytest.fixture
def esm_kit(tmp_path: Path) -> _EsmKit:
    venv = _make_venv(tmp_path / "python")
    for name in ("flash_attn", "transformer_engine", "xformers", "esmfold2_opt"):
        venv.package(name)
    for name in ("flash_attn_2_cuda", "flash_attn.ops.triton.rotary", "transformer_engine.pytorch", "xformers.ops"):
        venv.module(name)
    venv.module("esmfold2_opt.attn", KIT_ATTN)
    venv.module("transformers.models.esmfold2.modeling_esmfold2_common", "FLASH_ATTN_AVAILABLE = True\n")
    venv.module("transformers.models.esmc.modeling_esmc", "_te_available = True\n_xformers_available = True\n")
    bin_dir = tmp_path / "bin"
    shim = _write(tmp_path / "python_shim.py", PYTHON_SHIM)
    _executable(bin_dir / "python", f'#!/bin/sh\nexec "{sys.executable}" -I "{shim}" "$@"\n')
    for command in ("bash", "mkdir", "chmod"):
        _executable(bin_dir / command, f'#!/bin/sh\nprintf "{command} %s\\n" "$*" >> "$SHIM_LOG"\n')
    return _EsmKit(venv, bin_dir, tmp_path / "calls.log")


def test_kit_finish_ends_with_the_kernel_smoke_under_isolated_mode(esm_kit: _EsmKit) -> None:
    result = esm_kit.finish()

    assert result.returncode == 0, result.stdout + result.stderr
    calls = esm_kit.calls()
    assert calls[0] == "cd /kit/esmfold2"
    assert "bash run.sh install" in calls
    assert calls[-1] == f"python -I {KIT_SCRIPTS_DIR}/kit_smoke.py"
    assert "esmfold2 kit smoke: flash_attn=2.8.3 transformer_engine=2.15.0 xformers=0.0.35" in result.stdout


@pytest.mark.parametrize(
    ("broken", "reason"),
    [
        ("xformers/ops.py", "import xformers.ops"),
        ("flash_attn_2_cuda.py", "import flash_attn_2_cuda"),
        ("esmfold2_opt/attn.py", "import esmfold2_opt.attn"),
    ],
)
def test_kit_finish_fails_when_a_compiled_kernel_does_not_import(esm_kit: _EsmKit, broken: str, reason: str) -> None:
    (esm_kit.venv.site / broken).write_text("raise ImportError('undefined symbol: _ZN3c104cuda')\n")

    result = esm_kit.finish()

    assert result.returncode != 0
    assert "esmfold2 kit smoke FAILED" in result.stderr
    assert reason in result.stderr


def test_kit_finish_fails_when_the_fork_would_fall_back(esm_kit: _EsmKit) -> None:
    esm_kit.venv.module("transformers.models.esmc.modeling_esmc", "_te_available = True\n_xformers_available = False\n")

    result = esm_kit.finish()

    assert result.returncode != 0
    assert "_xformers_available is false" in result.stderr


def test_kit_finish_stops_at_the_first_failing_step(esm_kit: _EsmKit) -> None:
    _executable(esm_kit.bin / "bash", '#!/bin/sh\nprintf "bash %s\\n" "$*" >> "$SHIM_LOG"\nexit 3\n')

    result = esm_kit.finish()

    assert result.returncode == 3
    assert not [call for call in esm_kit.calls() if "kit_smoke.py" in call]


# --------------------------------------------------------------------------------------------------------------------
# (b) What the images record: the kit commit, the stack, and the ESMFold2 kit's fail-loud switch.
# --------------------------------------------------------------------------------------------------------------------


def test_esmfold2_kit_image_refuses_slow_paths_by_default() -> None:
    from boileroom.models.esmfold2.core import KIT_REQUIRE_FAST_ENV

    assert _dockerfile(ESMFOLD2_KIT).env()[KIT_REQUIRE_FAST_ENV] == "1"


@pytest.mark.parametrize(
    ("build_args", "stack"),
    [
        (None, "img_esmfold2_a100"),
        ({"WHEELS_FROM": "defer", "STACK": "img_esmfold2_a100"}, "img_esmfold2_a100"),
        ({"STACK": "img_ef2_fa"}, "img_ef2_fa"),
    ],
    ids=["docker-default", "modal", "h100-only"],
)
def test_esmfold2_kit_image_records_the_stack_it_was_built_for(build_args: dict[str, str] | None, stack: str) -> None:
    from boileroom.images.modal import ESMFOLD2_KIT_BUILD_ARGS
    from boileroom.models.esmfold2.core import KIT_STACK_ENV

    if build_args is not None and "WHEELS_FROM" in build_args:
        assert build_args == dict(ESMFOLD2_KIT_BUILD_ARGS)
    dockerfile = _dockerfile(ESMFOLD2_KIT)
    assert dockerfile.env(build_args=build_args)[KIT_STACK_ENV] == stack
    assert dockerfile.labels(build_args=build_args)["io.boileroom.kit.stack"] == stack


@pytest.mark.parametrize("spec", list(KIT_FETCHING_SPECS.values()), ids=list(KIT_FETCHING_SPECS))
@pytest.mark.parametrize("commit", [None, OTHER_COMMIT], ids=["default", "build-arg"])
def test_kit_image_records_the_commit_it_fetched(spec: metadata.RuntimeImageSpec, commit: str | None) -> None:
    from boileroom.models.esmfold2.core import KIT_COMMIT_ENV

    runtime = _runtime_module(REPO / "boileroom/models/protenix/runtime.py", "_protenix_runtime_for_env")
    assert KIT_COMMIT_ENV == runtime.KIT_COMMIT_ENV
    build_args = None if commit is None else {"KIT_COMMIT": commit}
    expected = commit or metadata.KIT_COMMIT
    dockerfile = _dockerfile(spec)
    assert dockerfile.env(build_args=build_args)[KIT_COMMIT_ENV] == expected
    if spec is ESMFOLD2_KIT:
        assert dockerfile.labels(build_args=build_args)["org.opencontainers.image.revision"] == expected


def _fetch_run(dockerfile: Dockerfile) -> Instruction:
    runs = [run for run in dockerfile.of("RUN") if "fetch" in _words(run) and "KIT_COMMIT" in run.body]
    assert len(runs) == 1
    return runs[0]


@pytest.mark.parametrize("key", sorted(KIT_FETCHING_SPECS))
def test_dockerfile_fetches_and_verifies_the_commit_the_code_expects(key: str) -> None:
    dockerfile = _dockerfile(KIT_FETCHING_SPECS[key])
    run = _fetch_run(dockerfile)
    args, env = dockerfile.scope(until=run)
    assert args["KIT_COMMIT"] == metadata.KIT_COMMIT
    words = [expand(word, {**args, **env}) for word in _words(run)]
    # `git ... fetch [options] <repository> <commit>`: the pinned commit, from the pinned repository.
    fetch = words.index("fetch")
    commit = words.index(metadata.KIT_COMMIT, fetch)
    assert all(word.startswith("-") or word.isdigit() for word in words[fetch + 1 : commit - 1])
    assert words[commit - 1] == args["KIT_REPOSITORY"]
    # `test "$(git ... rev-parse HEAD)" = "$KIT_COMMIT"`: the checkout is refused unless it is the pinned commit.
    check = words.index("test")
    assert "rev-parse HEAD" in words[check + 1] and words[check + 2 : check + 4] == ["=", metadata.KIT_COMMIT]


def test_opendde_image_keeps_the_kit_checkout_for_provenance() -> None:
    dockerfile = _dockerfile(OPENDDE)
    steps = dockerfile.stage()
    fetch = _fetch_run(dockerfile)
    assert "init" in _words(fetch) and "/kit" in _words(fetch)
    removed = [
        word
        for step in steps[steps.index(fetch) :]
        if step.keyword == "RUN"
        for words in [_words(step)]
        for index, word in enumerate(words)
        if index and "rm" in words[max(0, index - 3) : index]
    ]
    assert not [word for word in removed if word.rstrip("/") in {"/kit", "/kit/.git"} or word.startswith("/kit/.git")]
