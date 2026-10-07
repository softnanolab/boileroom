"""Shared body of the ``@app.cls`` Modal classes that serve a model core.

Each family's Modal class keeps its own ``@app.cls(...)`` decorator (image, GPU, volumes, timeouts), its
``config: bytes = modal.parameter(default=b"{}")`` declaration and a :meth:`ModalCoreServer._build_core` that imports
and constructs its core. Everything else lives here, once:

- ``@modal.enter()`` builds and loads the core inside a :class:`~boileroom.optimization.GuardedCore`, so a bad config,
  a missing kit or a failed load is raised from the first call instead of putting the container into Modal's silent
  restart loop;
- ``@modal.exit()`` closes the core;
- ``fold`` (:class:`ModalFoldServer`) or ``embed`` (:class:`ModalEmbedServer`) forwards to the loaded core.

Modal collects ``@modal.enter``/``@modal.exit``/``@modal.method`` through the class's MRO, so these hooks are registered
for every subclass. It reads ``modal.parameter()`` declarations only from the decorated class's own ``__dict__``, which is
why ``config`` is declared on each subclass rather than here.
"""

import json
from typing import Any

import modal

from ..optimization import GuardedCore


class ModalCoreServer:
    """Lifecycle shared by every model's Modal class: build the guarded core on enter, close it on exit.

    Subclasses implement :meth:`_build_core` and inherit from :class:`ModalFoldServer` or :class:`ModalEmbedServer`
    for the forwarding method.

    Attributes
    ----------
    config : bytes
        JSON-encoded core config, declared by each subclass as a ``modal.parameter``.
    """

    config: bytes
    _core: GuardedCore[Any] | None = None

    def _build_core(self, config: dict[str, Any]) -> Any:
        """Import and construct this class's core from its decoded config.

        Runs inside the guard, so the import stays lazy (the model libraries exist only in the runtime image) and any
        exception it raises is reported from the first call.

        Parameters
        ----------
        config : dict[str, Any]
            The decoded ``config`` parameter.

        Returns
        -------
        Any
            The unloaded core; the guard calls its ``_initialize()``.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement _build_core()")

    @modal.enter()
    def _initialize(self) -> None:
        """Decode the config and build and load the core, keeping any failure for the first call."""
        self._core = GuardedCore(lambda: self._build_core(json.loads(self.config.decode("utf-8"))))

    @modal.exit()
    def _shutdown(self) -> None:
        """Close the core if one was built."""
        if self._core is not None:
            self._core.close()

    def _loaded_core(self) -> Any:
        """Return the loaded core, raising the standing construction or load failure instead."""
        if self._core is None:
            raise RuntimeError(f"{type(self).__name__} has not been initialized")
        return self._core.get()


class ModalFoldServer(ModalCoreServer):
    """Modal class body for a structure-prediction core."""

    @modal.method()
    def fold(self, sequences: Any, options: dict[str, Any] | None = None) -> Any:
        """Run the core's ``fold``.

        Parameters
        ----------
        sequences : Any
            The family's fold input; the core validates it.
        options : dict[str, Any] | None, optional
            Per-call overrides of the non-static config keys.

        Returns
        -------
        Any
            The core's structure-prediction output.
        """
        return self._loaded_core().fold(sequences, options=options)


class ModalEmbedServer(ModalCoreServer):
    """Modal class body for an embedding core."""

    @modal.method()
    def embed(self, sequences: Any, options: dict[str, Any] | None = None) -> Any:
        """Run the core's ``embed``.

        Parameters
        ----------
        sequences : Any
            The family's embedding input; the core validates it.
        options : dict[str, Any] | None, optional
            Per-call overrides of the non-static config keys.

        Returns
        -------
        Any
            The core's embedding output.
        """
        return self._loaded_core().embed(sequences, options=options)
