"""Freeing the kit TriMul geometry caches that the kit's LRU evicted but its weight finalizers still hold."""

import gc
import sys
import threading
import weakref
from collections import OrderedDict
from collections.abc import Iterator
from types import ModuleType, SimpleNamespace

import pytest

from boileroom.models._worker import release_evicted_kit_caches


class _Cache(dict):
    """A geometry cache: a plain dict in the kit, a subclass here so a test can hold a weak reference to it."""


class _Owner:
    """A model weight whose death would run the kit's eviction finalizer."""


class _RecordingLock:
    """The adapter's lock, recording whether the sweep held it."""

    def __init__(self) -> None:
        self.entered = 0
        self._lock = threading.Lock()

    def __enter__(self) -> None:
        self._lock.acquire()
        self.entered += 1

    def __exit__(self, *exc: object) -> None:
        self._lock.release()


def _evict_key(cache: dict, key: tuple, w_ptr: int, reason: str) -> None:
    """The kit's eviction callback; finalizers are matched to it by identity."""
    cache.pop(key, None)


@pytest.fixture
def native(monkeypatch: pytest.MonkeyPatch) -> Iterator[ModuleType]:
    """A stand-in for ``opt_core.kernels.trimul.native`` with an empty per-device LRU."""
    module = ModuleType("esmfold2_opt.opt_core.kernels.trimul.native")
    module._SHARED = OrderedDict()  # type: ignore[attr-defined]
    module._LOCK = _RecordingLock()  # type: ignore[attr-defined]
    module._evict_key = _evict_key  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    yield module
    for finalizer, info in list(weakref.finalize._registry.items()):  # type: ignore[attr-defined]
        if info.func is _evict_key:
            finalizer.detach()


def _geometry(native: ModuleType, owner: _Owner, key: tuple, *, live: bool) -> tuple[weakref.ref, weakref.finalize]:
    """Create one geometry cache the way ``payload_cache()`` / ``after_call()`` do."""
    cache = _Cache({("pack", key): object()})
    if live:
        native._SHARED[key] = cache
    finalizer = weakref.finalize(owner, native._evict_key, cache, ("pack", key), 1234, "gc")
    return weakref.ref(cache), finalizer


def test_evicted_caches_are_freed_and_live_ones_kept(native: ModuleType) -> None:
    owner = _Owner()
    evicted, evicted_finalizers = [], []
    for n in (160, 200, 240):
        cache, finalizer = _geometry(native, owner, ("cuda:0", n, 128, 128), live=False)
        # The kit registers one finalizer per weight; two weights pin the same evicted cache.
        evicted_finalizers += [finalizer, weakref.finalize(owner, native._evict_key, cache(), ("pack", n), 5678, "gc")]
        evicted.append(cache)
    live, live_finalizer = _geometry(native, owner, ("cuda:0", 280, 128, 128), live=True)
    gc.collect()
    assert all(cache() is not None for cache in evicted)

    assert release_evicted_kit_caches() == 3

    assert all(cache() is None for cache in evicted)
    assert not any(finalizer.alive for finalizer in evicted_finalizers)
    assert live() is not None and live_finalizer.alive
    assert native._LOCK.entered == 1
    assert release_evicted_kit_caches() == 0


def test_releasing_never_runs_the_eviction_callback(native: ModuleType) -> None:
    """Detaching drops the finalizer without calling it: an evicted cache must not be emptied under a running fold."""
    owner = _Owner()
    cache, _ = _geometry(native, owner, ("cuda:0", 160, 128, 128), live=False)
    held = cache()
    assert held is not None

    assert release_evicted_kit_caches() == 1

    assert ("pack", ("cuda:0", 160, 128, 128)) in held


def test_without_the_kit_nothing_is_released(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in [name for name in sys.modules if name.endswith(".kernels.trimul.native")]:
        monkeypatch.delitem(sys.modules, name)
    assert release_evicted_kit_caches() == 0


@pytest.mark.parametrize(
    "module",
    [
        SimpleNamespace(_SHARED=None, _evict_key=_evict_key),
        SimpleNamespace(_SHARED={}),
    ],
    ids=["no-lru", "no-eviction-callback"],
)
def test_an_adapter_of_another_shape_is_left_alone(monkeypatch: pytest.MonkeyPatch, module: SimpleNamespace) -> None:
    owner = _Owner()
    finalizer = weakref.finalize(owner, _evict_key, _Cache(), ("pack", 1), 1, "gc")
    monkeypatch.setitem(sys.modules, "other_kit.kernels.trimul.native", module)
    try:
        assert release_evicted_kit_caches() == 0
        assert finalizer.alive
    finally:
        finalizer.detach()


def test_weakly_held_and_foreign_finalizers_are_kept(native: ModuleType) -> None:
    """A kit that holds its cache weakly (the upstream fix) needs no help; other finalizers are not the kit's."""
    owner = _Owner()
    cache = _Cache()
    weak = weakref.finalize(owner, native._evict_key, weakref.ref(cache), ("pack", 1), 1, "gc")
    foreign = weakref.finalize(owner, dict.clear, _Cache())
    try:
        assert release_evicted_kit_caches() == 0
        assert weak.alive and foreign.alive
    finally:
        weak.detach()
        foreign.detach()
