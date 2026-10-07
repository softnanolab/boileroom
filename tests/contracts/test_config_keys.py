"""Contract tests for the config keys a core accepts and the shared ``msa`` / ``templates`` / ``optimization`` rules."""

from typing import Any, ClassVar

import pytest

from boileroom.base import SHARED_CONFIG_KEYS, Algorithm
from boileroom.optimization import OptimizationUnavailableError


class _Core(Algorithm):
    """A core without an optimization kit and without MSA or template support."""

    DEFAULT_CONFIG: ClassVar[dict[str, Any]] = {"device": "cuda:0", "seed": None}
    STATIC_CONFIG_KEYS: ClassVar[frozenset[str]] = frozenset({"device"})

    def _load(self) -> None:
        self.ready = True


class _MSACore(_Core):
    SUPPORTS_USER_MSA = True


class _KitCore(_Core):
    """A core with an optimization kit that does no ``optimization`` check of its own."""

    DEFAULT_CONFIG: ClassVar[dict[str, Any]] = {"device": "cuda:0", "seed": None, "optimization": "vanilla"}
    STATIC_CONFIG_KEYS: ClassVar[frozenset[str]] = frozenset({"device", "optimization"})


def test_shared_keys_are_accepted_by_a_core_that_does_not_list_them() -> None:
    """Every model takes ``msa`` / ``templates`` / ``optimization`` (unset or vanilla) without a DEFAULT_CONFIG entry."""
    assert not SHARED_CONFIG_KEYS & set(_Core.DEFAULT_CONFIG)
    unset = {"msa": None, "templates": None, "optimization": "vanilla"}
    assert set(unset) == SHARED_CONFIG_KEYS
    assert _Core(unset)._merge_options(dict.fromkeys(("msa", "templates")))["optimization"] == "vanilla"


def test_unknown_key_at_construction_names_it_and_the_allowed_keys() -> None:
    with pytest.raises(ValueError, match=r"does not accept config keys \['sede'\]; allowed keys: \[.*'seed'.*\]"):
        _Core({"sede": 1})


def test_unknown_key_per_call_is_refused() -> None:
    with pytest.raises(ValueError, match=r"config keys \['num_samples'\]"):
        _Core()._merge_options({"num_samples": 3})


def test_unknown_key_through_update_config_is_refused_and_leaves_config_unchanged() -> None:
    core = _Core()
    with pytest.raises(ValueError, match="does not accept config keys"):
        core.update_config({"bogus": 1, "seed": 4})
    assert core.config == {"device": "cuda:0", "seed": None}
    core.update_config({"seed": 4})
    assert core.config["seed"] == 4


def test_known_keys_merge_and_static_keys_stay_construction_only() -> None:
    core = _Core({"seed": 1})
    assert core._merge_options({"seed": 2}) == {"device": "cuda:0", "seed": 2}
    with pytest.raises(ValueError, match="only be set at initialization"):
        core._merge_options({"device": "cpu"})


@pytest.mark.parametrize(("key", "value"), [("msa", [">q\nAAAA\n"]), ("templates", {"t": "data_x"})])
def test_unsupported_shared_input_is_refused_at_construction_and_per_call(key: str, value: Any) -> None:
    """The refusal reads the merged config, so a value given through ``config=`` is not silently ignored."""
    with pytest.raises(ValueError, match=rf"_Core does not support user-supplied {key!r}"):
        _Core({key: value})
    with pytest.raises(ValueError, match=rf"does not support user-supplied {key!r}"):
        _Core()._merge_options({key: value})


def test_supported_shared_input_passes() -> None:
    core = _MSACore({"msa": [">q\nAAAA\n"]})
    assert core._merge_options({"msa": None})["msa"] is None
    with pytest.raises(ValueError, match="does not support user-supplied 'templates'"):
        core._merge_options({"templates": {"t": "data_x"}})


def test_non_kit_core_accepts_vanilla_and_refuses_kit_modes() -> None:
    assert _Core({"optimization": "vanilla"}).config["optimization"] == "vanilla"
    for mode in ("exact", "fast"):
        with pytest.raises(
            OptimizationUnavailableError, match=rf"_Core has no optimization kit, so optimization={mode!r}"
        ):
            _Core({"optimization": mode})
    with pytest.raises(ValueError, match="optimization must be one of"):
        _Core({"optimization": "turbo"})


def test_kit_core_takes_every_mode_and_refuses_an_unknown_one_without_its_own_check() -> None:
    """The base validates ``optimization`` for a family that lists it too, so a new kit family cannot forget to."""
    for mode in ("vanilla", "exact", "fast"):
        assert _KitCore({"optimization": mode}).config["optimization"] == mode
    with pytest.raises(ValueError, match=r"optimization must be one of \['vanilla', 'exact', 'fast'\], got 'turbo'"):
        _KitCore({"optimization": "turbo"})
    core = _KitCore({"optimization": "fast"})
    with pytest.raises(ValueError, match="optimization must be one of"):
        core.update_config({"optimization": "turbo"})
    assert core.config["optimization"] == "fast"


def test_optimization_is_never_a_per_call_option() -> None:
    with pytest.raises(ValueError, match=r"only be set at initialization.*\['optimization'\]"):
        _Core()._merge_options({"optimization": "vanilla"})


def test_esmfold2_refuses_templates_given_at_construction() -> None:
    from boileroom.models.esmfold2.core import ESMFold2Core

    with pytest.raises(ValueError, match="ESMFold2Core does not support user-supplied 'templates'"):
        ESMFold2Core({"templates": {"t": "data_x"}})
    with pytest.raises(ValueError, match=r"does not accept config keys \['bogus'\]"):
        ESMFold2Core({"bogus": True})


def test_chai_keeps_its_constraint_path_option() -> None:
    from boileroom.models.chai.core import Chai1Core

    core = Chai1Core({"device": "cpu"})
    assert core._merge_options({"constraint_path": "/tmp/restraints.csv"})["constraint_path"] == "/tmp/restraints.csv"
