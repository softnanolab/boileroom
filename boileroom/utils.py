"""Utility functions and constants for the BoilerRoom package."""

import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


# Time constants
SECONDS = 1
MINUTES = 60
HOURS = 60 * MINUTES

# Directory constants
MODAL_MODEL_DIR = "/mnt/models"
CACHE_DIR = os.path.expanduser("~/.cache/boileroom")

# Amino acid constants
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AMINO_ACID_DICT = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
VALID_AMINO_ACIDS = set(AMINO_ACIDS)  # For faster lookups

GPUS_AVAIL_ON_MODAL = ["T4", "L4", "A10G", "A100-40GB", "A100-80GB", "L40S", "H100"]


def validate_sequence(sequence: str) -> bool:
    """Validate that a sequence contains only valid amino acids.

    Parameters
    ----------
    sequence : str
        A string of amino acids in single-letter code.

    Returns
    -------
    bool
        True if sequence is valid.

    Raises
    ------
    ValueError
        If sequence contains invalid characters.
    """
    sequence = sequence.replace(":", "")  # remove any linkers first ":"
    invalid_chars = set(sequence) - VALID_AMINO_ACIDS
    # TODO: we should think whether there's not a cleaner way to throw an error on Modal
    # the traceback is otherwise quite messy and hard to debug
    if invalid_chars:
        raise ValueError(f"Invalid amino acid(s) in sequence: {sorted(invalid_chars)}")
    return True


def safe_mkdir(path: Path, parents: bool = False) -> None:
    """Safely create a directory, avoiding parent directory creation issues in containers.

    When `parents=False`, creates the directory and its immediate parent if needed.
    This is useful for bind-mounted paths where only the immediate parent should exist
    (e.g., MODEL_DIR is bind-mounted, so we can create MODEL_DIR/boltz but not ancestors).

    Parameters
    ----------
    path : Path
        Directory path to create.
    parents : bool
        If True, create all parent directories. If False, only create immediate parent if missing.
    """
    if path.exists():
        return
    if parents:
        path.mkdir(parents=True, exist_ok=True)
    else:
        # For bind-mounted paths: create immediate parent if missing, then create target
        if not path.parent.exists():
            # Only create immediate parent, not ancestors (bind mount should handle that)
            try:
                path.parent.mkdir(exist_ok=True)
            except OSError:
                raise OSError(
                    f"Cannot create {path}: parent directory {path.parent} does not exist "
                    f"and could not be created. Ensure the bind mount is set up correctly."
                )
        path.mkdir(exist_ok=True)


def ensure_cache_dir() -> Path:
    """Create the cache directory (including parent directories) if it does not exist and return its path.

    Returns
    -------
    Path
        Path to the cache directory.
    """
    cache_path = Path(CACHE_DIR)
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


def get_model_dir() -> Path:
    """Resolve and set the MODEL_DIR environment variable to a user-writable cache location following XDG conventions.

    Precedence:
    1) Respect existing MODEL_DIR if set (assumed to be bind-mounted in containers, don't resolve).
    2) Use XDG_CACHE_HOME if defined; otherwise default to ~/.cache.
    3) Append "boileroom/models" and create the directory if it does not exist.

    If the resolved directory cannot be created, falls back to a temporary directory.

    Returns
    -------
    Path
        Absolute path to the resolved model directory.
    """
    try:
        if os.environ.get("MODEL_DIR"):
            # If MODEL_DIR is set, assume it's already absolute and possibly bind-mounted
            # Don't use resolve() as it may fail in containers where parent dirs don't exist
            model_dir_str = os.environ["MODEL_DIR"]
            resolved = Path(model_dir_str).expanduser()
            # Only expand user, don't resolve (bind-mounted paths may not have parent dirs in container)
            if not resolved.is_absolute():
                resolved = resolved.resolve()
        else:
            xdg_cache_home = os.getenv("XDG_CACHE_HOME")
            if xdg_cache_home:
                base_cache_dir = Path(xdg_cache_home).expanduser().resolve()
            else:
                base_cache_dir = Path.home() / ".cache"

            resolved = (base_cache_dir / "boileroom" / "models").resolve()

        # Use safe_mkdir with parents=True for local paths
        safe_mkdir(resolved, parents=True)
        os.environ["MODEL_DIR"] = str(resolved)
        return resolved
    except (OSError, PermissionError) as exc:
        logger.warning(f"Falling back to a temporary model cache due to filesystem error: {str(exc)}")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Falling back to a temporary model cache due to unexpected error: {str(exc)}")

    # Fallback to a user-writable temporary directory; ensure directory exists
    fallback_base = Path(tempfile.gettempdir()) / "boileroom" / "models"
    fallback_base.mkdir(parents=True, exist_ok=True)
    os.environ["MODEL_DIR"] = str(fallback_base)
    return fallback_base


def get_model_cache_dir(model_name: str) -> Path:
    """Get model-specific cache directory under MODEL_DIR.

    Returns MODEL_DIR/{model_name}, creating it if needed.
    Falls back to temporary directory if MODEL_DIR is not set.

    Parameters
    ----------
    model_name : str
        Model name (e.g., 'boltz', 'chai', 'esm').

    Returns
    -------
    Path
        Path to model-specific cache directory.
    """
    model_dir = os.environ.get("MODEL_DIR")
    if model_dir:
        cache_dir = Path(model_dir).expanduser() / model_name
        # Use safe_mkdir with parents=False since MODEL_DIR should already exist
        safe_mkdir(cache_dir, parents=False)
        return cache_dir
    else:
        # Fallback to temporary directory
        fallback_dir = Path(tempfile.gettempdir()) / "boileroom" / "models" / model_name
        fallback_dir.mkdir(parents=True, exist_ok=True)
        return fallback_dir


def format_time(seconds: float) -> str:
    """Convert a duration in seconds to a compact human-readable string.

    The output includes hours and minutes only when their values are greater than zero. Seconds are included when no larger unit is present or when seconds are greater than zero; fractional seconds are discarded (floored).

    Parameters
    ----------
    seconds : float
        Duration in seconds.

    Returns
    -------
    str
        A string like "2h 30m 15s", omitting any zero-valued hour/minute components.
    """
    hours = int(seconds // HOURS)
    minutes = int((seconds % HOURS) // MINUTES)
    secs = int(seconds % MINUTES)

    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:
        parts.append(f"{secs}s")

    return " ".join(parts)


def get_gpu_memory_info() -> Optional[Dict[str, int]]:
    """Get GPU memory information if available.

    Returns
    -------
    Optional[Dict[str, int]]
        Dictionary with 'total' and 'free' memory in MB,
        or None if no GPU is available.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None

        device = torch.cuda.current_device()
        total = torch.cuda.get_device_properties(device).total_memory // (1024 * 1024)
        free = torch.cuda.memory_reserved(device) // (1024 * 1024)

        return {"total": total, "free": free, "used": total - free}
    except Exception as e:
        print(f"Error getting GPU memory info: {e}")
        return None


class Timer:
    """Context manager for timing operations."""

    def __init__(self, description: str):
        self.description = description
        self.duration = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.duration = time.perf_counter() - self.start_time
        logging.info(f"{self.description} completed in {self.duration:.2f} seconds")
