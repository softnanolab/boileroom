"""Local backend for running models in the current Python process."""

from __future__ import annotations

from typing import Any, Type

from .base import Backend


class LocalBackend(Backend):
    """
    Backend that instantiates and runs model locally.
    This requires that the model's dependencies are installed in the current Python environment.
    None of the dependency conflicts are resolved programmatically, so use this at your own risk.

    Usage:
    ```python
    from boileroom import Chai1
    model = Chai1(backend="local")
    result = model.fold([sequence])
    ```

    """

    def __init__(
        self,
        model_cls: Type[Any],
        config: dict | None = None,
        *,
        device: str | None = None,
    ) -> None:
        super().__init__()
        self._model_cls = model_cls
        self._config = dict(config) if config is not None else {}
        if device is not None:
            # Mirror Modal backend interface by threading the requested device through config
            self._config.setdefault("device", device)
        self._model: Any | None = None

    def startup(self) -> None:
        if self._model is None:
            model = self._instantiate_model()
            self._initialize_model(model)
            self._model = model

    def shutdown(self) -> None:
        self._model = None

    def get_model(self) -> Any:
        if self._model is None:
            raise RuntimeError("Local backend model is not initialized. Call start() before use.")
        return self._model

    def _instantiate_model(self) -> Any:
        return self._model_cls(self._config)

    def _initialize_model(self, model: Any) -> None:
        initialize = getattr(model, "_initialize", None)
        if callable(initialize):
            initialize()
