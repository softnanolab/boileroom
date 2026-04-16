"""Backward-compatible access to the published base image."""

from modal import Image

from .metadata import MODAL_IMAGE_TAG_ENV, get_modal_base_image_reference, get_modal_image_tag

_base_image_tag = get_modal_image_tag()
base_image = Image.from_registry(get_modal_base_image_reference()).env({MODAL_IMAGE_TAG_ENV: _base_image_tag})
