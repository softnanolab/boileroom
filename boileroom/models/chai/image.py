"""Modal image definition for Chai-1."""

from pathlib import Path

from ...images.base import base_image
from ...utils import MODAL_MODEL_DIR, get_model_cache_dir

# Derive CHAI_DOWNLOADS_DIR from MODEL_DIR/chai, fallback to MODAL_MODEL_DIR/chai
_chai_downloads_dir = Path(MODAL_MODEL_DIR) / "chai"

chai_image = (
    base_image.pip_install(
        "torch==2.5.1+cu118",
        extra_index_url="https://download.pytorch.org/whl/cu118",
    )
    .pip_install("hf_transfer==0.1.8")
    .pip_install("chai_lab==0.6.1")
    .env(
        {
            "CHAI_DOWNLOADS_DIR": str(_chai_downloads_dir),
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "DISABLE_PANDERA_IMPORT_WARNING": "True",
        }
    )
)
