
import json
import threading

import modal
from modal.exception import InvalidError

from ..images import esm_image
from ..images.volumes import model_weights
from ..utils import MODEL_DIR, MINUTES
from .base import Backend
from ..base import Algorithm


app = modal.App("boileroom")


class _ModalAppToken:
    """Opaque token representing a Modal app context acquisition."""

    __slots__ = ("token_id",)

    def __init__(self, token_id: int) -> None:
        self.token_id = token_id

class ModalAppManager:
    """Manage shared access to the global Modal app context."""

    def __init__(self, modal_app: modal.App) -> None:
        self._app = modal_app
        self._lock = threading.Lock()
        self._context = None
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
        if self._attached_external or self._context is not None:
            return

        context = self._app.run()
        try:
            context.__enter__()
        except InvalidError as error:
            if "already running" in str(error).lower():
                self._attached_external = True
                return
            raise
        else:
            self._context = context


modal_app_manager = ModalAppManager(app)


class ModalBackend(Backend):
    """Backend for Modal."""

    def __init__(self, model_cls, config: dict | None = None, device: str | None = None) -> None:
        super().__init__()
        self._app = app
        self._config = config or {}
        self._model_cls = model_cls
        self._device = device
        self._remote_cls = None
        self.model = None
        self._context_token: _ModalAppToken | None = None

    def startup(self) -> None:
        if self._context_token is None:
            token = modal_app_manager.acquire()
            try:
                if self.model is None:
                    self.model = self._instantiate_remote_model()
            except Exception:
                modal_app_manager.release(token)
                raise
            self._context_token = token
            return

        if self.model is None:
            self.model = self._instantiate_remote_model()

    def shutdown(self) -> None:
        if self._context_token is not None:
            modal_app_manager.release(self._context_token)
            self._context_token = None
        self.model = None

    def _instantiate_remote_model(self):
        remote_cls = self._resolve_remote_cls()
        return remote_cls(config=json.dumps(self._config).encode("utf-8"))

    def _resolve_remote_cls(self):
        if self._remote_cls is None:
            remote_cls = self._model_cls
            if self._device is not None:
                remote_cls = remote_cls.with_options(gpu=self._device)
            self._remote_cls = remote_cls
        return self._remote_cls