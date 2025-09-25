from abc import ABC, abstractmethod
import atexit


class Backend(ABC):
    """Base class for all backends."""

    def __init__(self) -> None:
        """Initialize the backend."""
        pass

    def __post_init__(self) -> None:
        """Post-initialize the backend."""
        self.startup()
        atexit.register(self.shutdown)

    @abstractmethod
    def startup(self) -> None:
        """Startup the backend."""
        raise NotImplementedError

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the backend."""
        raise NotImplementedError
