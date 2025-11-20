"""Modal image definition for ESM family of models."""

from ...images.base import base_image

esm_image = base_image.pip_install("torch>=2.5.1,<2.7.0", "transformers==4.49.0").env(
    {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
)
