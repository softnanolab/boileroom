"""The reference keeps original upstream math after adapter compatibility patches."""

import hashlib
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytestmark = pytest.mark.contract


@pytest.fixture
def stock_source(monkeypatch):
    import _minifold_reference as reference

    source = b"raise RuntimeError('module-level code must not run')\ndef triangular(x):\n    return x + 1\n"
    monkeypatch.setattr(reference, "MINIFORMER_SHA256", hashlib.sha256(source).hexdigest())
    return source


def test_reference_verifies_source_before_executing_only_original_function(stock_source):
    from _minifold_reference import _original_triangular

    original = _original_triangular(stock_source)
    assert original(3) == 4
    with pytest.raises(AssertionError, match="Unexpected stock"):
        _original_triangular(stock_source + b"\n")


@pytest.mark.parametrize("fails", [False, True])
def test_reference_constructs_with_original_and_restores_adapter_after_failure(monkeypatch, stock_source, fails):
    from _minifold_reference import _stock_triangular

    from boileroom.models.minifold.loading import UPSTREAM_COMMIT

    process = Mock(return_value=subprocess.CompletedProcess([], 0, stdout=stock_source))
    monkeypatch.setattr(subprocess, "run", process)
    installed = lambda x: x + 10  # noqa: E731
    module = SimpleNamespace(triangular=installed)
    retained = None
    try:
        with _stock_triangular(module):
            retained = module.triangular  # Mirrors TriangularUpdate.__init__ storing self.fn.
            assert retained(3) == 4
            if fails:
                raise RuntimeError("constructor failed")
    except RuntimeError as error:
        assert fails and str(error) == "constructor failed"
    assert retained is not None
    assert retained(3) == 4  # Later reference inference still uses the original function.
    assert module.triangular is installed and module.triangular(3) == 13
    process.assert_called_once_with(
        ["git", "-C", "/opt/minifold", "show", f"{UPSTREAM_COMMIT}:minifold/model/miniformer.py"],
        check=True,
        capture_output=True,
        timeout=10,
    )


def test_unknown_reference_source_never_rebinds_adapter(monkeypatch):
    from _minifold_reference import _stock_triangular

    monkeypatch.setattr(subprocess, "run", Mock(return_value=subprocess.CompletedProcess([], 0, stdout=b"unknown")))
    installed = object()
    module = SimpleNamespace(triangular=installed)
    with pytest.raises(AssertionError, match="Unexpected stock"), _stock_triangular(module):
        pytest.fail("must reject source before construction")
    assert module.triangular is installed
