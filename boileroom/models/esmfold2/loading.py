"""Checkpoint I/O compatibility for the pinned esm 3.4.1 runtime."""

from functools import partial
from threading import RLock
from typing import Any
from unittest.mock import patch

_LOAD_LOCK = RLock()


def load_pretrained(model_class: Any, model_name: str, **kwargs: Any) -> Any:
    """Use buffered Safetensors reads while retaining ESM's strict key checks.

    Memory-mapped tensor slicing can stall on mounted model volumes while holding
    the GIL, preventing Modal heartbeats. esm 3.4.1 does not expose Safetensors'
    backend argument, so scope this compatibility override to model loading and
    restore the upstream binding even on failure. Serialize our overrides to
    avoid overlapping restoration when multiple cores initialize in one process.
    """
    from esm.models import hub
    from safetensors.torch import load_file

    with _LOAD_LOCK, patch.object(hub, "load_file", partial(load_file, backend="pread")):
        return model_class.from_pretrained(model_name, **kwargs)
