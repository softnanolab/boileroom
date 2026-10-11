"""Canonical MiniFold Docker image shared by Modal and Apptainer."""

from ...images.modal import get_modal_image

minifold_image = get_modal_image("minifold")
