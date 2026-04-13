"""Backward-compatible access to the published base image."""

from modal import Image

from .metadata import BASE_IMAGE_SPEC, MODAL_IMAGE_TAG_ENV, format_image_reference, get_modal_image_tag

_base_image_tag = get_modal_image_tag()
base_image = Image.from_registry(format_image_reference(BASE_IMAGE_SPEC.image_name, _base_image_tag)).env(
    {MODAL_IMAGE_TAG_ENV: _base_image_tag}
)
