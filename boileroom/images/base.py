"""Backward-compatible access to the published base image."""

from modal import Image

from .metadata import BASE_IMAGE_SPEC, format_image_reference, get_modal_image_tag

base_image = Image.from_registry(format_image_reference(BASE_IMAGE_SPEC.image_name, get_modal_image_tag()))
