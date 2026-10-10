"""`job-started.sh`: every way of not getting an explicit ALLOW must end in the kill.

The script kills every process of its user, so it is only ever run here with `kill` shadowed by a
shell function that records its arguments; a function takes precedence over the builtin.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

from infra.modal_ci import image

HOOK = image.SANDBOX_DIR / "job-started.sh"
BASH = shutil.which("bash") or "bash"
pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def run_hook(tmp_path: Path, guard_body: str | None, *, errexit: bool) -> subprocess.CompletedProcess[str]:
    guard = tmp_path / "guard.py"
    if guard_body is not None:
        guard.write_text(guard_body)
    script = tmp_path / "hook.sh"
    script.write_text(HOOK.read_text().replace("/opt/ci/guard.py", str(guard)))
    harness = f'kill() {{ echo "KILL $*"; }}; . {script}'
    flags = ["-e", "-o", "pipefail"] if errexit else []
    return subprocess.run(  # noqa: S603
        [BASH, "--noprofile", "--norc", *flags, "-c", harness],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def killed(result: subprocess.CompletedProcess[str]) -> bool:
    lines = result.stdout.splitlines()
    return any(line.startswith("KILL -s KILL -- -1") for line in lines) and any(
        line.startswith("KILL -s KILL ") and "--" not in line for line in lines
    )


def test_the_harness_really_shadows_kill(tmp_path) -> None:
    """Guard rail for the tests below: the builtin must not be the one that runs."""
    out = subprocess.run(  # noqa: S603
        [BASH, "--noprofile", "--norc", "-c", "kill() { echo shadowed; }; kill -s KILL -- -1"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "shadowed"


@pytest.mark.parametrize("errexit", [False, True], ids=["plain", "bash-e"])
def test_allow_lets_the_job_through_without_killing(tmp_path, errexit) -> None:
    result = run_hook(tmp_path, 'print("ALLOW")\n', errexit=errexit)
    assert result.returncode == 0
    assert "KILL" not in result.stdout


@pytest.mark.parametrize("errexit", [False, True], ids=["plain", "bash-e"])
@pytest.mark.parametrize(
    "guard_body",
    [
        'print("DENY repo_mismatch"); raise SystemExit(1)\n',  # a denial
        "raise RuntimeError('boom')\n",  # the guard crashes
        "import os; os._exit(139)\n",  # the guard dies on a signal-like status
        "",  # prints nothing
        'print("allow")\n',  # not exactly ALLOW
        'print("ALLOW "); raise SystemExit(3)\n',  # a trailing space and a failure status
        'print("ALLOW"); raise SystemExit(3)\n',  # ALLOW but a failing exit: not trusted
        None,  # the guard is missing altogether
    ],
    ids=["deny", "crash", "dies", "silent", "lowercase", "padded", "allow-then-fail", "missing"],
)
def test_anything_but_a_clean_allow_kills_the_runner(tmp_path, guard_body, errexit) -> None:
    result = run_hook(tmp_path, guard_body, errexit=errexit)
    assert result.returncode != 0
    assert killed(result), result.stdout


def test_the_kill_trap_is_armed_before_the_guard_runs_and_cleared_only_on_allow() -> None:
    lines = [line for line in HOOK.read_text().splitlines() if line.strip() and not line.startswith("#")]
    armed = next(i for i, line in enumerate(lines) if line.startswith("trap ") and "EXIT" in line)
    guard = next(i for i, line in enumerate(lines) if "guard.py" in line)
    cleared = [i for i, line in enumerate(lines) if line.strip() == "trap - EXIT"]
    assert armed < guard
    assert cleared and all(i > guard for i in cleared)
