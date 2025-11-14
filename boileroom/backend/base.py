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
        """Stop the backend and clean up resources.

        This method is idempotent and can be called multiple times safely.
        Ensures proper cleanup even if shutdown() raises an exception.
        """
        if not self._is_running:
            return

        try:
            self.shutdown()
        except Exception:
            # Log the error but continue with cleanup
            # This ensures state is reset even if shutdown() fails
            pass
        finally:
            self._is_running = False
            if self._atexit_hook is not None:
                try:
                    atexit.unregister(self._atexit_hook)
                except Exception:
                    # Ignore errors when unregistering atexit hook
                    pass
                finally:
                    self._atexit_hook = None

    @abstractmethod
    def startup(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def shutdown(self) -> None:
        raise NotImplementedError
