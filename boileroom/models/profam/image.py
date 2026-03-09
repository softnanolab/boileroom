"""Modal image definition for ProFam."""

import modal

from ...images.base import base_image

hf_secret = modal.Secret.from_name("huggingface-secret")

profam_image = (
    base_image.pip_install(
        "torch>=2.0",
        "transformers>=4.49.0,<5.0.0",
        "tokenizers",
        "pytorch-lightning",
        "omegaconf",
        "rootutils",
        "datasets",
        "safetensors",
        "accelerate",
        "huggingface-hub",
        "biopython",
        "scipy",
        "scikit-learn",
        "numba",
        "hydra-core",
        "pyyaml",
        "rich",
    )
    # ProFam's setup.py uses find_packages() but several sub-packages
    # (src/sequence, src/evaluators, src/pipelines) are missing __init__.py.
    # Clone, patch, install.
    .run_commands(
        "git clone --depth 1 https://github.com/alex-hh/profam.git /tmp/profam"
        " && touch /tmp/profam/src/sequence/__init__.py"
        "         /tmp/profam/src/evaluators/__init__.py"
        "         /tmp/profam/src/pipelines/__init__.py"
        " && pip install /tmp/profam"
        " && rm -rf /tmp/profam",
    )
    # Download ProFam-1 checkpoint from HuggingFace (gated repo).
    .run_commands(
        "python -c \""
        "from huggingface_hub import snapshot_download; "
        "snapshot_download('judewells/ProFam-1', "
        "local_dir='/models/profam', "
        "local_dir_use_symlinks=False)"
        "\"",
        secrets=[hf_secret],
    )
)
