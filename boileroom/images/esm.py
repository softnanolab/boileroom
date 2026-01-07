"""Modal image definition for ESM family of models."""

from modal import Image

# Define the base image with all dependencies
esm_image = (
    Image.debian_slim(python_version="3.12")
    .apt_install("wget", "git")
    .pip_install("torch>=2.5.1,<2.7.0", "torch-tensorrt", "biotite>=1.0.1", "transformers==4.49.0")
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
)