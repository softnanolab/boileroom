"""Modal image definition for ESM family of models."""

from .base import base_image

# Define the base image with all dependencies
esm_image = (
    base_image.pip_install("torch>=2.5.1,<2.7.0", "torch-tensorrt")
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
)
