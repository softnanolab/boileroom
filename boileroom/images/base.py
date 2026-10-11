"""Backward-compatible access to the published base image."""

from modal import Image

from .metadata import get_modal_base_image_reference

base_image = Image.from_registry(get_modal_base_image_reference())
