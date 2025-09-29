from abc import ABC, abstractmethod
import atexit


class Backend(ABC):
    """Base class for all backends."""

    def __init__(self) -> None:
        self._is_running = False
        atexit.register(self.shutdown)

    def start(self) -> None:
        if not self._is_running:
            self.startup()
            self._is_running = True

    def stop(self) -> None:
        if self._is_running:
            self.shutdown()
            self._is_running = False

    @abstractmethod
    def startup(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def shutdown(self) -> None:
        raise NotImplementedError
