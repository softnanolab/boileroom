"""The kits' refusal sets and provenance words: one definition in boileroom, literal copies where boileroom is absent.

``boileroom/models/protenix/runtime.py`` runs in the model image's interpreter and never imports boileroom, so it
carries its own copies of the refusal exit codes, the refusal class names and the provenance sentinels. They must
match :mod:`boileroom.optimization` and :mod:`boileroom.provenance`, or one family would refuse what another fails on,
and the same missing fact would read differently per model.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from boileroom import optimization, provenance

pytestmark = pytest.mark.contract

RUNTIME = Path(__file__).resolve().parents[2] / "boileroom/models/protenix/runtime.py"


def _runtime() -> ModuleType:
    """Load the worker runtime standalone, as the worker interpreter does."""
    spec = importlib.util.spec_from_file_location("_kit_refusal_sets_runtime", RUNTIME)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class KernelsRefused(SystemExit):
    """The shape of ``opendde_opt.lncensus.KernelsRefused``: a ``SystemExit`` subclass with the census's exit code 5."""

    def __init__(self, problems: list[str]) -> None:
        super().__init__(5)
        self.problems = problems


def test_runtime_refusal_sets_match_boileroom() -> None:
    from boileroom.models import _worker

    runtime = _runtime()
    assert runtime.KIT_REFUSAL_EXIT_CODES == optimization.KIT_REFUSAL_EXIT_CODES == {3, 5}
    assert _worker._KIT_REFUSAL_EXIT_CODES == optimization.KIT_REFUSAL_EXIT_CODES
    assert runtime.KIT_REFUSAL_CLASS_NAMES == optimization.KIT_REFUSAL_CLASS_NAMES


def test_refusal_class_names_are_exceptions_not_exit_codes() -> None:
    """``KernelsRefused`` is a ``SystemExit``: its name in the class set would suggest exit code 5 is redundant."""
    assert "KernelsRefused" not in optimization.KIT_REFUSAL_CLASS_NAMES
    assert "KernelsRefused" not in _runtime().KIT_REFUSAL_CLASS_NAMES


def test_kernels_refused_is_a_refusal_through_its_exit_code() -> None:
    """The real ``KernelsRefused`` is ``SystemExit(5)``; every boileroom path maps it by code, not by name."""
    from boileroom.models import _worker

    error = KernelsRefused(["triangle_attention: cuequivariance absent"])
    assert not optimization.is_kit_refusal_exception(error)
    assert optimization.is_refusal(error) and _worker._is_refusal(error)
    runtime = _runtime()

    def refuse() -> None:
        raise error

    with pytest.raises(runtime.OptimizationUnavailableError, match="exited with code 5.*cuequivariance absent"):
        runtime._guard("OpenDDE", "lncensus.arm()", refuse)


def test_runtime_provenance_sentinels_are_boilerooms() -> None:
    words = {provenance._NOT_LOADED, provenance._ABSENT, provenance._NONE, provenance._UNKNOWN}
    assert words == _runtime().PROVENANCE_SENTINELS


@pytest.mark.parametrize("name", sorted(optimization.KIT_REFUSAL_CLASS_NAMES))
def test_refusal_classes_and_their_subclasses_are_refusals(name: str) -> None:
    base = type(name, (RuntimeError,), {})
    subclass = type("BigRefusal", (base,), {})
    runtime = _runtime()
    for error in (base("refused"), subclass("refused")):
        assert optimization.is_kit_refusal_exception(error)
        assert runtime._is_kit_refusal(error)


@pytest.mark.parametrize("error", [RuntimeError("CUDA out of memory"), ValueError("bad"), KeyError("x")])
def test_other_exceptions_are_not_refusals(error: Exception) -> None:
    assert not optimization.is_kit_refusal_exception(error)
    assert not _runtime()._is_kit_refusal(error)
