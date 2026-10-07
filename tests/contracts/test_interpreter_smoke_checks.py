"""Contract tests for the second-interpreter image smoke check (OpenDDE's virtualenv), run locally without Docker."""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from boileroom.images.import_checks import interpreter_smoke_command
from boileroom.images.metadata import InterpreterSmokeCheck, get_model_image_spec

LDD_RESOLVED = "\tlibc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007f0000000000)\n"


@dataclass
class _Sandbox:
    """A fake package with a driver-linked library, a fake ``ldd`` and a fake libstdc++."""

    site: Path
    bin: Path
    library_dir: Path
    libstdcxx: Path


@pytest.fixture
def sandbox(tmp_path: Path) -> _Sandbox:
    site = tmp_path / "site"
    (site / "fakeops" / "lib").mkdir(parents=True)
    (site / "fakeops" / "__init__.py").write_text("raise ImportError('libcuda.so.1: cannot open shared object file')\n")
    (site / "fakeops" / "lib" / "libfake_ops.so").write_bytes(b"\x7fELF")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    ldd = bin_dir / "ldd"
    ldd.write_text('#!/bin/sh\nprintf "%b" "$FAKE_LDD_OUTPUT"\nexit "${FAKE_LDD_STATUS:-0}"\n')
    ldd.chmod(ldd.stat().st_mode | stat.S_IXUSR)
    library_dir = tmp_path / "lib"
    library_dir.mkdir()
    libstdcxx = library_dir / "libstdc++.so.6"
    libstdcxx.write_bytes(b"\x00GLIBCXX_3.4.31\x00GLIBCXX_3.4.32\x00")
    return _Sandbox(site=site, bin=bin_dir, library_dir=library_dir, libstdcxx=libstdcxx)


def _check(sandbox: _Sandbox, **overrides: Any) -> InterpreterSmokeCheck:
    fields: dict[str, Any] = {
        "python": sys.executable,
        "library_path": str(sandbox.library_dir),
        "imports": ("json", "email.message"),
        "driver_linked_libraries": (("fakeops", "libfake_ops.so"),),
        "required_symbol_versions": ((str(sandbox.libstdcxx), "GLIBCXX_3.4.32"),),
    }
    fields.update(overrides)
    return InterpreterSmokeCheck(**fields)


def _run(sandbox: _Sandbox, check: InterpreterSmokeCheck, ldd_output: str, ldd_status: int = 0) -> Any:
    """Run the command minus its ``docker run --rm <image>`` prefix: the part that runs inside the container."""
    command = interpreter_smoke_command("example/opendde:test", check)
    assert command[:4] == ["docker", "run", "--rm", "example/opendde:test"]
    env = {
        "PATH": f"{sandbox.bin}{os.pathsep}{os.environ['PATH']}",
        "PYTHONPATH": str(sandbox.site),
        "LD_LIBRARY_PATH": "/usr/local/nvidia/lib64",
        "FAKE_LDD_OUTPUT": ldd_output,
        "FAKE_LDD_STATUS": str(ldd_status),
    }
    return subprocess.run(command[4:], capture_output=True, text=True, env=env, timeout=60)


def test_smoke_check_passes_when_only_libcuda_is_unresolved(sandbox: _Sandbox) -> None:
    """``fakeops`` itself fails to import without a driver, so the check looks at its library with ldd instead."""
    result = _run(sandbox, _check(sandbox), LDD_RESOLVED + "\tlibcuda.so.1 => not found\n")
    assert result.returncode == 0, result.stderr
    assert "OK: json" in result.stdout
    assert "libfake_ops.so links (unresolved only: ['libcuda.so.1'])" in result.stdout
    assert "provides GLIBCXX_3.4.32" in result.stdout


def test_smoke_check_runs_the_named_interpreter_with_the_library_path_prepended(sandbox: _Sandbox) -> None:
    """The image's own LD_LIBRARY_PATH (driver paths) must survive, as it does in the OpenDDE worker environment."""
    probe = sandbox.site / "ldpath_probe.py"
    probe.write_text("import os, sys\nprint('LD=' + os.environ['LD_LIBRARY_PATH'])\nprint('PY=' + sys.executable)\n")
    result = _run(sandbox, _check(sandbox, imports=("ldpath_probe",), driver_linked_libraries=()), "")
    assert result.returncode == 0, result.stderr
    assert f"LD={sandbox.library_dir}:/usr/local/nvidia/lib64" in result.stdout
    assert f"PY={sys.executable}" in result.stdout


@pytest.mark.parametrize(
    ("ldd_output", "ldd_status", "message"),
    [
        (LDD_RESOLVED + "\tlibcuda.so.1 => not found\n\tlibcublas.so.12 => not found\n", 0, "libcublas.so.12"),
        ("\tnot a dynamic executable\n", 1, "unresolved libraries"),
    ],
    ids=["unexpected-unresolved-library", "ldd-fails"],
)
def test_smoke_check_fails_on_a_broken_driver_linked_library(
    sandbox: _Sandbox, ldd_output: str, ldd_status: int, message: str
) -> None:
    result = _run(sandbox, _check(sandbox), ldd_output, ldd_status)
    assert result.returncode == 1
    assert message in result.stderr


def _isolated_check(sandbox: _Sandbox) -> InterpreterSmokeCheck:
    """Return a check whose interpreter sees the sandbox only, not this venv's own ``torch`` and ``nvidia`` wheels.

    The fake ``ldd`` resolves libcublas only from an ``nvidia`` wheel directory, as on the OpenDDE image, where
    libcue_ops.so has no RUNPATH and NEEDs libcublas.so.12 that only the venv's ``nvidia`` wheel provides. It records
    the search path it was given in ``ldd_path.log``.
    """
    (sandbox.bin / "ldd").write_text(
        "#!/bin/sh\n"
        f'echo "$LD_LIBRARY_PATH" > "{sandbox.bin}/ldd_path.log"\n'
        'case ":$LD_LIBRARY_PATH:" in\n'
        '  *"/nvidia/cublas/lib:"*) printf "\\tlibcublas.so.12 => /site/nvidia/cublas/lib/libcublas.so.12\\n" ;;\n'
        '  *) printf "\\tlibcublas.so.12 => not found\\n" ;;\n'
        "esac\n"
        'printf "\\tlibcuda.so.1 => not found\\n"\n'
    )
    python = sandbox.bin / "python-isolated"
    python.write_text(f'#!/bin/sh\nexec "{sys.executable}" -S "$@"\n')  # -S: no site-packages, PYTHONPATH only
    python.chmod(python.stat().st_mode | stat.S_IXUSR)
    return _check(sandbox, python=str(python))


def test_ldd_searches_the_cuda_libraries_of_the_torch_and_nvidia_wheels(sandbox: _Sandbox) -> None:
    """``import torch`` preloads the wheels' cuBLAS and NVRTC at run time, but plain ldd would report them missing."""
    for directory in ("nvidia/cublas/lib", "nvidia/cuda_nvrtc/lib", "torch/lib"):
        (sandbox.site / directory).mkdir(parents=True)
    (sandbox.site / "torch" / "__init__.py").write_text("raise ImportError('the check must not import torch')\n")

    result = _run(sandbox, _isolated_check(sandbox), "")

    assert result.returncode == 0, result.stderr
    assert "libfake_ops.so links (unresolved only: ['libcuda.so.1'])" in result.stdout
    search_path = (sandbox.bin / "ldd_path.log").read_text().strip().split(":")
    assert search_path[0] == str(sandbox.site / "fakeops" / "lib")
    assert {str(sandbox.site / "nvidia/cublas/lib"), str(sandbox.site / "nvidia/cuda_nvrtc/lib")} <= set(search_path)
    assert str(sandbox.site / "torch/lib") in search_path
    # The interpreter's library path and the image's own driver paths stay after the wheel directories.
    assert search_path[-2:] == [str(sandbox.library_dir), "/usr/local/nvidia/lib64"]


def test_ldd_still_fails_when_no_wheel_provides_the_library(sandbox: _Sandbox) -> None:
    result = _run(sandbox, _isolated_check(sandbox), "")

    assert result.returncode == 1
    assert "unresolved libraries: libcublas.so.12" in result.stderr


def test_smoke_check_fails_on_a_missing_import(sandbox: _Sandbox) -> None:
    result = _run(sandbox, _check(sandbox, imports=("json", "boileroom_no_such_module")), LDD_RESOLVED)
    assert result.returncode == 1
    assert "FAILED: boileroom_no_such_module - ModuleNotFoundError" in result.stderr
    assert "OK: json" in result.stdout  # every failure is reported, not just the first


def test_smoke_check_fails_when_the_library_is_not_in_the_package(sandbox: _Sandbox) -> None:
    check = _check(sandbox, driver_linked_libraries=(("fakeops", "libcue_ops.so"), ("no_such_package", "libx.so")))
    result = _run(sandbox, check, LDD_RESOLVED)
    assert result.returncode == 1
    assert "libcue_ops.so not found in package fakeops" in result.stderr
    assert "libx.so not found in package no_such_package" in result.stderr


@pytest.mark.parametrize("contents", [b"\x00GLIBCXX_3.4.30\x00", None], ids=["old-libstdcxx", "missing-file"])
def test_smoke_check_fails_without_the_required_symbol_version(sandbox: _Sandbox, contents: bytes | None) -> None:
    if contents is None:
        sandbox.libstdcxx.unlink()
    else:
        sandbox.libstdcxx.write_bytes(contents)
    result = _run(sandbox, _check(sandbox), LDD_RESOLVED)
    assert result.returncode == 1
    assert "does not provide GLIBCXX_3.4.32" in result.stderr


def test_opendde_spec_checks_its_virtualenv() -> None:
    (check,) = get_model_image_spec("opendde").interpreter_smoke_checks
    assert check.python == "/opt/opendde/bin/python"
    assert check.library_path == "/opt/opendde/lib"
    assert {"cuequivariance_torch", "opendde_opt", "runner.batch_inference"} <= set(check.imports)
    assert check.driver_linked_libraries == (("cuequivariance_ops", "libcue_ops.so"),)
    assert check.allowed_unresolved == ("libcuda.so.1",)


@pytest.mark.parametrize(("image_key", "expects_venv_check"), [("opendde", True), ("esm", False)])
def test_image_import_check_runs_the_interpreter_checks(
    monkeypatch: pytest.MonkeyPatch, image_key: str, expects_venv_check: bool
) -> None:
    """The CI smoke check used to run the system interpreter only, so a broken OpenDDE virtualenv passed."""
    from scripts.images import check_model_imports

    commands: list[list[str]] = []
    monkeypatch.setattr(check_model_imports.subprocess, "run", lambda cmd, **kwargs: commands.append(list(cmd)))
    spec = get_model_image_spec(image_key)

    check_model_imports._check_image(
        image_key, "example/image:test", spec.context_path / "requirements.txt", spec.context_path / "core.py", False
    )

    venv_commands = [cmd for cmd in commands if "/opt/opendde/bin/python" in cmd]
    if expects_venv_check:
        (command,) = venv_commands
        assert command[:4] == ["docker", "run", "--rm", "example/image:test"]
        assert "/opt/opendde/lib" in command
    else:
        assert venv_commands == []
    assert len(commands) == 2 + int(expects_venv_check)
