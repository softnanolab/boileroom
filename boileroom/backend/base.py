from abc import ABC, abstractmethod
import atexit
from collections.abc import Callable


class Backend(ABC):
    """Base class for all backends."""

    def __init__(self) -> None:
        self._is_running = False
        self._atexit_hook: Callable[[], None] | None = None

    def start(self) -> None:
        if self._is_running:
            return

        self.startup()
        self._is_running = True

        if self._atexit_hook is None:

            def _stop_backend() -> None:
                if self._is_running:
                    self.stop()

            self._atexit_hook = _stop_backend
            atexit.register(self._atexit_hook)

    def stop(self) -> None:
        if not self._is_running:
            return

        self.shutdown()
        self._is_running = False

        if self._atexit_hook is not None:
            try:
                atexit.unregister(self._atexit_hook)
            finally:
                self._atexit_hook = None

    @abstractmethod
    def startup(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def shutdown(self) -> None:
        raise NotImplementedError
