"""Contract tests for the shared CLI subprocess helpers."""

import os
import sys

import pytest

from boileroom.models._cli import run_command


def test_run_command_labels_timeouts_with_output_tail() -> None:
    """A timed-out command raises a labeled error carrying its captured output."""
    script = "import sys, time; print('started', flush=True); time.sleep(30)"
    with pytest.raises(RuntimeError, match="Tool command timed out after 0.5 seconds") as excinfo:
        run_command([sys.executable, "-c", script], env=dict(os.environ), timeout=0.5, label="Tool")

    assert "started" in str(excinfo.value)
    assert excinfo.value.__cause__ is not None


def test_run_command_reports_nonzero_exit_with_output_tail() -> None:
    """A failing command raises with its exit code and stderr tail."""
    script = "import sys; print('boom', file=sys.stderr); sys.exit(3)"
    with pytest.raises(RuntimeError, match="exit code 3") as excinfo:
        run_command([sys.executable, "-c", script], env=dict(os.environ), timeout=10, label="Tool")

    assert "boom" in str(excinfo.value)
