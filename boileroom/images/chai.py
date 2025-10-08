"""Modal image definition for Chai-1."""

from modal import Image

chai_image = (
    Image.debian_slim(python_version="3.12")
    .apt_install("wget", "git")
    .pip_install("chai_lab==0.6.1")
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}) # TODO: unclear whether needed
)
