from modal import Image

from ..utils import MODAL_MODEL_DIR

base_image = (
    Image.debian_slim(python_version="3.12")
    .apt_install("wget", "git")
    .pip_install("biotite>=1.0.1")
    .env({"MODEL_DIR": MODAL_MODEL_DIR})
    )