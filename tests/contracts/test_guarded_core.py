"""Contract tests for building and loading cores in ``@modal.enter()`` without ever raising there."""

import json
import sys
from collections.abc import Callable
from types import ModuleType
from typing import Any, ClassVar

import pytest

from boileroom.base import Algorithm
from boileroom.models.registry import MODEL_SPECS, ModelSpec, resolve_object
from boileroom.optimization import GuardedCore, OptimizationUnavailableError


class _FakeCore(Algorithm):
    """A core with the real config validation and a load that can be told to fail."""

    DEFAULT_CONFIG: ClassVar[dict[str, Any]] = {"device": None}
    STATIC_CONFIG_KEYS: ClassVar[frozenset[str]] = frozenset({"device"})

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        self.loads = 0
        self.closed = False
        self.load_errors: list[BaseException] = []

    def _load(self) -> None:
        self.ready = True

    def _initialize(self) -> None:
        self.loads += 1
        if self.load_errors:
            raise self.load_errors.pop(0)
        self._load()

    def fold(self, sequences: Any, options: dict | None = None) -> tuple[str, Any]:
        self._merge_options(options)
        return ("predicted", sequences)

    embed = fold

    def close(self) -> None:
        self.closed = True


def _factory(config: dict[str, Any], *load_errors: BaseException) -> Callable[[], _FakeCore]:
    def build() -> _FakeCore:
        core = _FakeCore(config)
        core.load_errors = list(load_errors)
        return core

    return build


def test_a_loaded_core_is_returned_and_closed() -> None:
    guarded = GuardedCore(_factory({}))
    core = guarded.get()
    assert core.ready and core.loads == 1 and guarded.failure is None
    assert guarded.get() is core and core.loads == 1
    guarded.close()
    assert core.closed


def test_a_construction_error_stands_and_nothing_loads() -> None:
    """An unknown config key used to raise in ``@modal.enter()``, which Modal answers with a silent restart loop."""
    guarded = GuardedCore(_factory({"num_sample": 3}))
    assert guarded.core is None and isinstance(guarded.failure, ValueError)
    for _ in range(2):
        with pytest.raises(ValueError, match=r"does not accept config keys \['num_sample'\]"):
            guarded.get()
    guarded.close()


def test_a_construction_exit_is_mapped_like_a_load_exit() -> None:
    def refuse() -> _FakeCore:
        raise SystemExit(3)

    guarded = GuardedCore(refuse)
    with pytest.raises(OptimizationUnavailableError, match="core construction load refused"):
        guarded.get()


def test_a_transient_load_failure_is_retried_on_the_next_call() -> None:
    guarded = GuardedCore(_factory({}, OSError("download dropped")))
    assert isinstance(guarded.failure, OSError)
    core = guarded.get()
    assert core.ready and core.loads == 2 and guarded.failure is None


def test_a_refused_load_stands_without_a_retry() -> None:
    guarded = GuardedCore(_factory({}, SystemExit(3)))
    for _ in range(2):
        with pytest.raises(OptimizationUnavailableError, match="_FakeCore load refused"):
            guarded.get()
    assert guarded.core is not None and guarded.core.loads == 1


def test_a_keyboard_interrupt_is_not_kept() -> None:
    with pytest.raises(KeyboardInterrupt):
        GuardedCore(_factory({}, KeyboardInterrupt()))


_MODAL_CLASSES = [
    (spec, path) for spec in MODEL_SPECS for path in (spec.modal_class_path, spec.kit_modal_class_path) if path
]
#: Modal methods a class serves beside its family's task method; each must raise the standing failure too.
_EXTRA_MODAL_METHODS: dict[str, set[str]] = {"esm3": {"inverse_fold"}}


def _modal_hooks(user_cls: type, flags: int) -> dict[str, Callable[..., Any]]:
    """The raw functions Modal registers on ``user_cls`` for ``flags``, collected the way ``@app.cls`` collects them."""
    from modal._partial_function import _find_partial_methods_for_user_cls

    return {
        name: partial.raw_f
        for name, partial in _find_partial_methods_for_user_cls(user_cls, flags).items()
        if partial.raw_f is not None
    }


def _bound_hooks(instance: object, flags: int) -> dict[str, Callable[..., Any]]:
    """The hooks Modal's container runtime binds to ``instance`` for ``flags``."""
    from modal._partial_function import _find_callables_for_obj

    return _find_callables_for_obj(instance, flags)


def _install_fake_core(spec: ModelSpec, monkeypatch: pytest.MonkeyPatch) -> None:
    """Stand in for the family's core module, so ``_build_core``'s lazy import finds a :class:`_FakeCore`."""
    assert spec.apptainer_core_class_path is not None
    module_path, _, class_name = spec.apptainer_core_class_path.rpartition(".")
    fake_module = ModuleType(module_path)
    setattr(fake_module, class_name, type(class_name, (_FakeCore,), {}))
    monkeypatch.setitem(sys.modules, module_path, fake_module)


@pytest.mark.parametrize(("spec", "modal_class_path"), _MODAL_CLASSES, ids=lambda item: getattr(item, "key", ""))
def test_modal_registers_the_shared_hooks_for_every_class(spec: ModelSpec, modal_class_path: str) -> None:
    """Modal must see the mixin's enter, exit and forwarding method on each class, and its ``config`` parameter."""
    from modal._partial_function import _PartialFunctionFlags as Flags
    from modal.cls import _get_class_constructor_signature

    from boileroom.backend.modal_server import ModalCoreServer, ModalEmbedServer, ModalFoldServer

    user_cls = resolve_object(modal_class_path)._get_user_cls()
    task_method = spec.contract.task_method
    server = {"fold": ModalFoldServer, "embed": ModalEmbedServer}[task_method]
    assert issubclass(user_cls, server)

    assert _modal_hooks(user_cls, Flags.ENTER_POST_SNAPSHOT) == _modal_hooks(ModalCoreServer, Flags.ENTER_POST_SNAPSHOT)
    assert set(_modal_hooks(user_cls, Flags.ENTER_POST_SNAPSHOT)) == {"_initialize"}
    assert _modal_hooks(user_cls, Flags.ENTER_PRE_SNAPSHOT) == {}
    assert _modal_hooks(user_cls, Flags.EXIT) == _modal_hooks(ModalCoreServer, Flags.EXIT)
    assert set(_modal_hooks(user_cls, Flags.EXIT)) == {"_shutdown"}
    interface = _modal_hooks(user_cls, Flags.interface_flags())
    assert set(interface) == {task_method} | _EXTRA_MODAL_METHODS.get(spec.key, set())
    assert interface[task_method] == _modal_hooks(server, Flags.interface_flags())[task_method]

    # Modal reads modal.parameter() only from the decorated class's own __dict__; a class that left ``config`` to the
    # mixin would take no constructor arguments and every lookup with a config would fail.
    parameters = _get_class_constructor_signature(user_cls).parameters
    assert list(parameters) == ["config"] and parameters["config"].default == b"{}"
    # Each class builds its own core; the mixin's placeholder only raises.
    assert "_build_core" in user_cls.__dict__


@pytest.mark.parametrize(("spec", "modal_class_path"), _MODAL_CLASSES, ids=lambda item: getattr(item, "key", ""))
def test_modal_enter_never_raises_and_the_call_raises_the_failure(
    spec: ModelSpec, modal_class_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every Modal class builds its core in ``@modal.enter()``; a bad config must surface from the call instead."""
    from modal._partial_function import _PartialFunctionFlags as Flags

    _install_fake_core(spec, monkeypatch)
    user_cls = resolve_object(modal_class_path)._get_user_cls()
    task_method = spec.contract.task_method

    def enter(instance: object) -> None:
        for hook in _bound_hooks(instance, Flags.ENTER_POST_SNAPSHOT).values():
            hook()

    def shutdown(instance: object) -> None:
        for hook in _bound_hooks(instance, Flags.EXIT).values():
            hook()

    bad = object.__new__(user_cls)
    bad.config = json.dumps({"num_sample": 3}).encode()
    enter(bad)
    call = _bound_hooks(bad, Flags.interface_flags())[task_method]
    with pytest.raises(ValueError, match=r"does not accept config keys \['num_sample'\]"):
        call("MKV")
    for name in _EXTRA_MODAL_METHODS.get(spec.key, set()):
        with pytest.raises(ValueError, match=r"does not accept config keys \['num_sample'\]"):
            _bound_hooks(bad, Flags.interface_flags())[name]("MKV", None, [0])
    shutdown(bad)

    good = object.__new__(user_cls)
    good.config = b"{}"
    enter(good)
    assert _bound_hooks(good, Flags.interface_flags())[task_method]("MKV") == ("predicted", "MKV")
    core = good._core.core
    assert isinstance(core, _FakeCore) and core.loads == 1 and not core.closed
    shutdown(good)
    assert core.closed


def _local_fold_server(build: Callable[[dict[str, Any]], Any]) -> Any:
    """An undecorated :class:`ModalFoldServer` whose core comes from ``build``; its hooks are called directly."""
    from boileroom.backend.modal_server import ModalFoldServer

    class Server(ModalFoldServer):
        def _build_core(self, config: dict[str, Any]) -> Any:
            return build(config)

    return Server()


def test_modal_server_passes_the_decoded_config_to_the_core() -> None:
    seen: list[dict[str, Any]] = []

    def build(config: dict[str, Any]) -> _FakeCore:
        seen.append(config)
        return _FakeCore(config)

    server = _local_fold_server(build)
    server.config = json.dumps({"device": "cpu"}).encode()
    server._initialize()
    assert seen == [{"device": "cpu"}]
    assert server.fold("MKV", options=None) == ("predicted", "MKV")


def test_modal_server_keeps_a_malformed_config_for_the_first_call() -> None:
    """Decoding runs inside the guard too: a raise in ``@modal.enter()`` would loop the container silently."""
    server = _local_fold_server(_FakeCore)
    server.config = b"{not json"
    server._initialize()
    with pytest.raises(json.JSONDecodeError):
        server.fold("MKV")
    server._shutdown()


def test_modal_server_without_a_core_factory_fails_from_the_call() -> None:
    from boileroom.backend.modal_server import ModalEmbedServer

    server = ModalEmbedServer()
    server.config = b"{}"
    server._initialize()
    with pytest.raises(NotImplementedError, match="ModalEmbedServer must implement _build_core"):
        server.embed("MKV")


def test_modal_server_call_and_exit_before_enter() -> None:
    server = _local_fold_server(_FakeCore)
    with pytest.raises(RuntimeError, match="has not been initialized"):
        server.fold("MKV")
    server._shutdown()


def test_modal_server_refusal_stands_across_calls_and_exit_still_runs() -> None:
    builds: list[int] = []

    def refuse(config: dict[str, Any]) -> _FakeCore:
        builds.append(1)
        raise OptimizationUnavailableError("no kit on this card")

    server = _local_fold_server(refuse)
    server.config = b"{}"
    server._initialize()
    for _ in range(2):
        with pytest.raises(OptimizationUnavailableError, match="no kit on this card"):
            server.fold("MKV")
    assert builds == [1]
    server._shutdown()
