import importlib
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .conda import CondaBackend
    from .local import LocalBackend
    from .modal import ModalBackend
    from .apptainer import ApptainerBackend

__all__ = ["CondaBackend", "LocalBackend", "ModalBackend", "ApptainerBackend"]


def __getattr__(name: str):
    """Lazily import backend modules and their main classes.

    Parameters
    ----------
    name : str
        Name of the attribute requested (should be one of the public backend classes).

    Returns
    -------
    type
        The backend class corresponding to the requested name.

    Raises
    ------
    AttributeError
        If an unknown attribute is requested.
    """
    if name == "CondaBackend":
        module = importlib.import_module(".conda", __name__)
        value = module.CondaBackend
    elif name == "LocalBackend":
        module = importlib.import_module(".local", __name__)
        value = module.LocalBackend
    elif name == "ModalBackend":
        module = importlib.import_module(".modal", __name__)
        value = module.ModalBackend
    elif name == "ApptainerBackend":
        module = importlib.import_module(".apptainer", __name__)
        value = module.ApptainerBackend
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    setattr(sys.modules[__name__], name, value)
    return value
