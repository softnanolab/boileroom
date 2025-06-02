"""Utility functions and constants for the BoilerRoom package."""

import os
import time
import logging
from pathlib import Path
from typing import Dict, Optional


# Time constants
SECONDS = 1
MINUTES = 60
HOURS = 60 * MINUTES

# Directory constants
MODEL_DIR = "/mnt/models"
CACHE_DIR = os.path.expanduser("~/.cache/boileroom")

# Amino acid constants
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AMINO_ACID_DICT = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
VALID_AMINO_ACIDS = set(AMINO_ACIDS)  # For faster lookups

GPUS_AVAIL_ON_MODAL = ["T4", "L4", "A10G", "A100-40GB", "A100-80GB", "L40S", "H100"]


def validate_sequence(sequence: str) -> bool:
    """Validate that a sequence contains only valid amino acids.

    Args:
        sequence: A string of amino acids in single-letter code

    Returns:
        bool: True if sequence is valid

    Raises:
        ValueError: If sequence contains invalid characters
    """
    sequence = sequence.replace(":", "")  # remove any linkers first ":"
    invalid_chars = set(sequence) - VALID_AMINO_ACIDS
    # TODO: we should think whether there's not a cleaner way to throw an error on Modal
    # the traceback is otherwise quite messy and hard to debug
    if invalid_chars:
        raise ValueError(f"Invalid amino acid(s) in sequence: {sorted(invalid_chars)}")
    return True


def ensure_cache_dir() -> Path:
    """Ensure the cache directory exists.

    Returns:
        Path: Path to cache directory
    """
    cache_path = Path(CACHE_DIR)
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


def format_time(seconds: float) -> str:
    """Format time in seconds to human readable string.

    Args:
        seconds: Time in seconds

    Returns:
        str: Formatted time string (e.g. "2h 30m 15s")
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

    Returns:
        Optional[Dict[str, int]]: Dictionary with 'total' and 'free' memory in MB,
                                 or None if no GPU is available
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
