"""Modal image definition for MiniFold model."""

from ...images.base import base_image

minifold_image = base_image.pip_install(
    "torch>=2.5.1,<2.7.0",
    "fair-esm",
    "biopython==1.81",
    "ml_collections",
    "dm-tree",
    "einops",
    "modelcif",
    "httpx",  # required by boileroom.backend (CondaBackend import chain)
    "minifold @ git+https://github.com/jwohlwend/minifold.git",
)
