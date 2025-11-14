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
        """
        Initialize the LocalBackend with the model class and optional configuration.
        
        Parameters:
            model_cls: The model class to instantiate when the backend starts.
            config: Optional configuration mapping; a shallow copy is stored for backend use.
            device: Optional device identifier; if provided, it is set into the stored config under the key "device" unless already present.
        
        Notes:
            Stores the provided values on the instance and initializes the active model reference to None.
        """
        super().__init__()
        self._model_cls = model_cls
        self._config = dict(config) if config is not None else {}
        if device is not None:
            # Mirror Modal backend interface by threading the requested device through config
            self._config.setdefault("device", device)
        self._model: Any | None = None

    def startup(self) -> None:
        """
        Ensure the local model is instantiated and initialized.
        
        If no model is currently cached, create a model instance, run its optional initialization, and store it for future use.
        """
        if self._model is None:
            model = self._instantiate_model()
            self._initialize_model(model)
            self._model = model

    def shutdown(self) -> None:
        """
        Release the currently instantiated local model and clear the backend's cached instance.
        
        This sets the internal model reference to None so the backend behaves as uninitialized.
        """
        self._model = None

    def get_model(self) -> Any:
        """
        Get the active model instance managed by the backend.
        
        Raises:
            RuntimeError: If no model has been started; call start() first.
        
        Returns:
            The initialized model instance.
        """
        if self._model is None:
            raise RuntimeError("Local backend model is not initialized. Call start() before use.")
        return self._model

    def _instantiate_model(self) -> Any:
        """
        Create and return a model instance using the backend's stored model class and configuration.
        
        Returns:
            Any: An instance of the configured model class constructed with the backend's config.
        """
        return self._model_cls(self._config)

    def _initialize_model(self, model: Any) -> None:
        """
        Call the model's optional `_initialize` hook if it exists.
        
        If `model` has a callable attribute named `_initialize`, this function invokes it; otherwise it does nothing.
        
        Parameters:
            model (Any): The model instance to initialize.
        """
        initialize = getattr(model, "_initialize", None)
        if callable(initialize):
            initialize()