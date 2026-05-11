"""Helpers for Modal images backed by published Docker images."""

from __future__ import annotations

from modal import Image

from ..utils import MODAL_MODEL_DIR
from .metadata import format_image_reference, get_image_tag, get_model_image_spec, render_modal_runtime_env


def get_modal_image(identifier: str) -> Image:
    """Return a Modal image sourced from the published Docker image for a model."""
    spec = get_model_image_spec(identifier)
    image = Image.from_registry(format_image_reference(spec.image_name, get_image_tag()))
    return image.env(render_modal_runtime_env(spec, MODAL_MODEL_DIR))
