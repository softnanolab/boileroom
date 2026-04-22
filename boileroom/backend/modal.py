import json
import os
import threading
from typing import Any

import modal
from modal.exception import InvalidError

from .base import Backend

_modal_app_managers_lock = threading.Lock()
_modal_app_managers: dict[modal.App, "ModalAppManager"] = {}


def get_modal_app(name: str) -> modal.App:
    """Create a Modal app for a specific model entrypoint."""
    return modal.App(f"boileroom-{name}")


def modal_app_of(model_cls: Any) -> modal.App:
    """Return the app that owns a Modal-decorated class.

    This centralizes the Modal SDK ``Cls._get_app()`` usage required by
    Modal >=1.1.0 so callers do not reach into the SDK directly.
    """
    try:
        return model_cls._get_app()
    except (AttributeError, AssertionError, InvalidError) as error:
        raise TypeError("ModalBackend requires a Modal-decorated class registered on a Modal app.") from error


def _get_modal_app_manager(modal_app: modal.App) -> "ModalAppManager":
    """Return the shared manager for a Modal app object."""
    with _modal_app_managers_lock:
        manager = _modal_app_managers.get(modal_app)
        if manager is None:
            manager = ModalAppManager(modal_app)
            _modal_app_managers[modal_app] = manager
        return manager


class _ModalAppToken:
    """Opaque token representing a Modal app context acquisition."""

    __slots__ = ("token_id",)

    def __init__(self, token_id: int) -> None:
        self.token_id = token_id


class ModalAppManager:
    """Manage shared access to a Modal app context."""

    def __init__(self, modal_app: modal.App) -> None:
        self._app = modal_app
        self._lock = threading.Lock()
        self._context: Any = None
        self._attached_external = False
        self._tokens: set[int] = set()
        self._next_token_id = 0

    def acquire(self) -> _ModalAppToken:
        """Ensure the Modal app is running and return an acquisition token."""
        with self._lock:
            self._ensure_running()
            token = _ModalAppToken(self._next_token_id)
            self._next_token_id += 1
            self._tokens.add(token.token_id)
            return token

    def release(self, token: _ModalAppToken) -> None:
        """Release a previously acquired token, stopping the app if appropriate."""
        with self._lock:
            if token.token_id not in self._tokens:
                return
            self._tokens.remove(token.token_id)

            if self._tokens:
                return

            if self._attached_external:
                self._attached_external = False
                return

            if self._context is not None:
                self._context.__exit__(None, None, None)
                self._context = None

    def _ensure_running(self) -> None:
        """Ensure the Modal app is running and acquire a local context if needed.

        If the app is already attached externally or a local context exists, this is a no-op. Otherwise, attempt to start the app with interactive mode controlled by the MODAL_INTERACTIVE environment variable ("true" enables interactive). On successful entry, store the entered context on self._context. If entering the context raises InvalidError whose message indicates the app is "already running", mark the manager as attached externally by setting self._attached_external and return; other InvalidError exceptions are re-raised.

        Raises
        ------
        InvalidError
            Re-raised when entering the Modal context fails for reasons other than the app already running.
        """
        if self._attached_external or self._context is not None:
            return

        context = self._app.run(interactive=os.environ.get("MODAL_INTERACTIVE", "false").lower() == "true")
        try:
            context.__enter__()
        except InvalidError as error:
            if "already running" in str(error).lower():
                self._attached_external = True
                return
            raise
        else:
            self._context = context


class ModalBackend(Backend):
    """Backend for Modal."""

    def __init__(self, model_cls, config: dict | None = None, device: str | None = None) -> None:
        super().__init__()
        self._app = modal_app_of(model_cls)
        self._app_manager = _get_modal_app_manager(self._app)
        self._config = dict(config) if config is not None else {}
        self._model_cls = model_cls
        self._device = device
        self._remote_cls = None
        self.model: Any | None = None
        self._context_token: _ModalAppToken | None = None

    def startup(self) -> None:
        if self._context_token is None:
            token = self._app_manager.acquire()
            try:
                self._ensure_model()
            except Exception:
                self._app_manager.release(token)
                raise
            self._context_token = token
            return

        self._ensure_model()

    def shutdown(self) -> None:
        if self._context_token is not None:
            self._app_manager.release(self._context_token)
            self._context_token = None
        self.model = None

    def get_model(self) -> Any:
        if self.model is None:
            raise RuntimeError("Modal backend model is not initialized. Call start() before use.")
        return self.model

    def _instantiate_remote_model(self):
        remote_cls = self._resolve_remote_cls()
        return remote_cls(config=json.dumps(self._config).encode("utf-8"))

    def _resolve_remote_cls(self):
        """Resolve and cache the remote model class, applying GPU options when a device is configured.

        If a device string was provided to the backend, the returned class will have GPU options applied via with_options(gpu=...). The resolved class is cached on the instance for subsequent calls.

        Returns
        -------
        Any
            The resolved remote class used to instantiate the remote model.
        """
        if self._remote_cls is None:
            remote_cls = self._model_cls
            if self._device is not None:
                # Note that the app will still show that's using T4,
                # but the actual method / function call will use the correct GPU,
                # and display accordingly in the Modal dashboard.
                remote_cls = remote_cls.with_options(gpu=self._device)
            self._remote_cls = remote_cls
        return self._remote_cls

    def _ensure_model(self) -> None:
        if self.model is None:
            self.model = self._instantiate_remote_model()
