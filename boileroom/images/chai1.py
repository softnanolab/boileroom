"""Modal image definition for Chai-1 model."""

from modal import Image

# Define the base image with all dependencies for Chai-1
chai1_image = (
    Image.debian_slim(python_version="3.12")
    .apt_install("wget", "git")
    .pip_install(
        "chai_lab==0.5.0",
        "hf_transfer==0.1.8",
        "biotite>=1.0.1",
    )
    .pip_install(
        "torch==2.7.1",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"
    })
)