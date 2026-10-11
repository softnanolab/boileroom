"""Tests of the ESMFold2 kit image's build-time smoke check, with stand-in kernel modules (no GPU, no kit)."""

import importlib.util
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

pytestmark = pytest.mark.contract

SMOKE = Path(__file__).resolve().parents[2] / "boileroom/models/esmfold2/kit/kit_smoke.py"
COMMON = "transformers.models.esmfold2.modeling_esmfold2_common"
ESMC = "transformers.models.esmc.modeling_esmc"


def _smoke() -> Any:
    """Load kit_smoke.py the way the image runs it: as a standalone script, not part of boileroom."""
    spec = importlib.util.spec_from_file_location("kit_smoke", SMOKE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _attn(refusal: str | None = None) -> SimpleNamespace:
    calls: list[dict] = []

    def require_refusal_metadata(environ: dict | None = None, image: Any = None, vers: Any = None) -> str | None:
        calls.append(dict(environ or {}))
        return refusal

    return SimpleNamespace(
        ENV_REQUIRE="ESMFOLD2_OPT_REQUIRE_FAST_ENV",
        COMMON_MODULE=COMMON,
        ESMC_MODULE=ESMC,
        FLAG="FLASH_ATTN_AVAILABLE",
        REQUIRED=(
            ("atom_attn", "flash_attn", "flash_attn"),
            ("esmc_mlp", "te", "transformer_engine.pytorch"),
            ("esmc_rope", "flash_attn_triton", "flash_attn.ops.triton.rotary"),
        ),
        require_refusal_metadata=require_refusal_metadata,
        metadata_words=lambda vers=None: "flash_attn=2.8.3.post1 transformer_engine=2.15.0 xformers=0.0.35",
        calls=calls,
    )


def _importer(modules: dict[str, Any], missing: tuple[str, ...] = ()) -> Any:
    """An importer that serves ``modules``, raises for ``missing`` and returns an empty module for anything else."""
    imported: list[str] = []

    def import_module(name: str) -> Any:
        imported.append(name)
        if name in missing:
            raise ImportError(f"{name}: undefined symbol: _ZN3c104cuda")
        return modules.get(name, ModuleType(name))

    import_module.imported = imported  # type: ignore[attr-defined]
    return import_module


def _healthy(**overrides: Any) -> dict[str, Any]:
    modules: dict[str, Any] = {
        "esmfold2_opt.attn": _attn(),
        COMMON: SimpleNamespace(FLASH_ATTN_AVAILABLE=True),
        # _flash_attn_available reads the GPU at import, so it is False on a CPU builder and must not fail the build.
        ESMC: SimpleNamespace(_te_available=True, _xformers_available=True, _flash_attn_available=False),
    }
    modules.update(overrides)
    return modules


def test_kit_smoke_passes_on_a_healthy_cpu_image(capsys: pytest.CaptureFixture[str]) -> None:
    smoke = _smoke()
    modules = _healthy()
    import_module = _importer(modules)

    assert smoke.main(import_module) == 0

    assert capsys.readouterr().out.strip() == (
        "esmfold2 kit smoke: flash_attn=2.8.3.post1 transformer_engine=2.15.0 xformers=0.0.35"
    )
    for name in ("flash_attn_2_cuda", "flash_attn.ops.triton.rotary", "transformer_engine.pytorch", "xformers.ops"):
        assert name in import_module.imported
    assert modules["esmfold2_opt.attn"].calls == [{"ESMFOLD2_OPT_REQUIRE_FAST_ENV": "1"}]


@pytest.mark.parametrize(
    ("modules", "missing", "message"),
    [
        (_healthy(), ("flash_attn_2_cuda",), "import flash_attn_2_cuda: ImportError"),
        (_healthy(), ("transformer_engine.pytorch",), "import transformer_engine.pytorch: ImportError"),
        (_healthy(**{COMMON: SimpleNamespace(FLASH_ATTN_AVAILABLE=False)}), (), "FLASH_ATTN_AVAILABLE is false"),
        (
            _healthy(**{ESMC: SimpleNamespace(_te_available=False, _xformers_available=True)}),
            (),
            "_te_available is false",
        ),
        (
            _healthy(**{ESMC: SimpleNamespace(_te_available=True, _xformers_available=False)}),
            (),
            "_xformers_available is false",
        ),
        (
            _healthy(**{"esmfold2_opt.attn": _attn("the distribution metadata lacks flash_attn")}),
            (),
            "refuses this image with ESMFOLD2_OPT_REQUIRE_FAST_ENV=1: the distribution metadata lacks flash_attn",
        ),
    ],
    ids=["flash-attn-abi", "te-missing", "fork-flag-off", "te-flag-off", "xformers-flag-off", "metadata-refusal"],
)
def test_kit_smoke_fails_the_build_with_a_named_reason(
    capsys: pytest.CaptureFixture[str], modules: dict, missing: tuple[str, ...], message: str
) -> None:
    assert _smoke().main(_importer(modules, missing)) == 1

    err = capsys.readouterr().err
    assert "esmfold2 kit smoke FAILED" in err and message in err


def test_kit_smoke_without_the_kit_reports_unread_words(capsys: pytest.CaptureFixture[str]) -> None:
    assert _smoke().main(_importer({}, ("esmfold2_opt.attn",))) == 1

    captured = capsys.readouterr()
    assert captured.out.strip() == "esmfold2 kit smoke: unread"
    assert "import esmfold2_opt.attn: ImportError" in captured.err


# --------------------------------------------------------------------------------------------------------------------
# kit_sass.sh: every compiled wheel must carry machine code for each compute capability of the stack.
# --------------------------------------------------------------------------------------------------------------------

KIT_DIR = SMOKE.parent
SASS = KIT_DIR / "kit_sass.sh"
SH = shutil.which("sh") or "/bin/sh"
#: A stand-in ``cuobjdump --list-elf``: prints the object's own contents, and fails on an object without device code.
FAKE_CUOBJDUMP = """\
#!/bin/sh
[ -s "$2" ] || { echo "cuobjdump info: File '$2' does not contain device code" >&2; exit 255; }
cat "$2"
"""
BOTH = "ELF file    1: k.1.sm_80.cubin\nELF file    2: k.2.sm_90.cubin\n"


def _wheel(directory: Path, name: str, objects: dict[str, str]) -> None:
    """A wheel whose shared objects list ``objects`` (path in the wheel -> cuobjdump listing) as their device code."""
    with zipfile.ZipFile(directory / f"{name}-1.0-cp312-cp312-linux_x86_64.whl", "w") as wheel:
        wheel.writestr(f"{name}/__init__.py", "")
        for path, listing in objects.items():
            wheel.writestr(path, listing)


def _healthy_wheels(directory: Path) -> None:
    _wheel(directory, "flash_attn", {"flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so": BOTH})
    _wheel(
        directory,
        "transformer_engine",
        {"transformer_engine/libtransformer_engine.so": BOTH, "transformer_engine/wheel_lib/libcommon.so.1": ""},
    )
    # A kernel family built for one architecture only is fine while another object covers the rest.
    _wheel(directory, "xformers", {"xformers/_C.so": "ELF file 1: a.sm_80.cubin\n", "xformers/_C_fa3.so": "x.sm_90a\n"})


def _run_sass(tmp_path: Path, archs: str = "80;90") -> subprocess.CompletedProcess[str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    cuobjdump = bin_dir / "cuobjdump"
    cuobjdump.write_text(FAKE_CUOBJDUMP)
    cuobjdump.chmod(0o755)
    (bin_dir / "python").symlink_to(sys.executable)
    environ = {**os.environ, "PATH": f"{bin_dir}:{os.environ.get('PATH', '/usr/bin:/bin')}", "TMPDIR": str(tmp_path)}
    return subprocess.run(
        [SH, str(SASS), str(tmp_path / "wheels"), archs], env=environ, capture_output=True, text=True, timeout=60
    )


@pytest.fixture
def wheels(tmp_path: Path) -> Path:
    directory = tmp_path / "wheels"
    directory.mkdir()
    _healthy_wheels(directory)
    return directory


def test_kit_sass_passes_wheels_with_code_for_every_stack_arch(tmp_path: Path, wheels: Path) -> None:
    result = _run_sass(tmp_path)

    assert result.returncode == 0, result.stderr
    assert "xformers-1.0-cp312-cp312-linux_x86_64.whl: sm_80 sm_90" in result.stdout


def test_kit_sass_passes_an_sm90_only_stack(tmp_path: Path, wheels: Path) -> None:
    _wheel(wheels, "flash_attn", {"flash_attn_2_cuda.so": "k.sm_90.cubin\n"})

    assert _run_sass(tmp_path, "90").returncode == 0


def test_kit_sass_refuses_a_wheel_missing_an_arch(tmp_path: Path, wheels: Path) -> None:
    """flash-attn that silently compiled sm_90 only must fail the build, not the first A100 fold."""
    _wheel(wheels, "flash_attn", {"flash_attn_2_cuda.so": "k.sm_90.cubin\n", "flash_attn/other.so": ""})

    result = _run_sass(tmp_path)

    assert result.returncode == 2
    assert "flash_attn-1.0-cp312-cp312-linux_x86_64.whl has no machine code for sm_80 (found: sm_90)" in result.stderr


def test_kit_sass_refuses_a_wheel_without_device_code(tmp_path: Path, wheels: Path) -> None:
    _wheel(wheels, "transformer_engine", {"transformer_engine/libtransformer_engine.so": ""})

    result = _run_sass(tmp_path)

    assert result.returncode == 2
    assert "has no machine code for sm_80 sm_90 (found: none)" in result.stderr


def test_kit_sass_refuses_a_missing_wheel(tmp_path: Path, wheels: Path) -> None:
    next(wheels.glob("xformers-*.whl")).unlink()

    result = _run_sass(tmp_path)

    assert result.returncode == 2
    assert f"no xformers wheel in {wheels}" in result.stderr


def test_kit_wheels_checks_the_kernels_before_removing_the_toolkit() -> None:
    """cuobjdump goes with the toolkit, so the check must run after the last compile and before the purge."""
    lines = (KIT_DIR / "kit_wheels.sh").read_text().splitlines()
    check = next(index for index, line in enumerate(lines) if "kit_sass.sh" in line and not line.startswith("#"))

    assert lines[check] == 'sh "$(dirname "$0")/kit_sass.sh" "$W" "$SM_ARCHS"'
    assert check > next(index for index, line in enumerate(lines) if "flash_attn built in" in line)
    assert check < next(index for index, line in enumerate(lines) if "pip uninstall" in line)
    assert check < next(index for index, line in enumerate(lines) if "apt-get purge" in line)
    assert "cuda-cuobjdump-13-0" in "\n".join(lines[:check])


def test_kit_dockerfile_copies_the_kernel_check_with_the_wheels_script() -> None:
    """kit_wheels.sh finds kit_sass.sh beside itself, on the Docker route and on Modal's compile step alike."""
    copies = [line for line in (KIT_DIR / "Dockerfile").read_text().splitlines() if line.startswith("COPY ")]

    assert "COPY kit_wheels.sh kit_sass.sh /opt/boileroom-kit/" in copies
