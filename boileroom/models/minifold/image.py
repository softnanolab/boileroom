"""Modal image definition for MiniFold model."""

from ...images.base import base_image

minifold_image = (
    base_image.pip_install(
        "torch>=2.5.1,<2.7.0",
        "fair-esm",
        "biopython==1.81",
        "ml_collections",
        "dm-tree",
        "einops",
        "modelcif",
        "httpx",  # required by boileroom.backend import chain
    )
    .run_commands(
        # minifold's pyproject.toml only declares packages=["minifold"], missing subpackages.
        # Clone, patch to use find_packages discovery, then install.
        "git clone https://github.com/jwohlwend/minifold.git /tmp/minifold",
        'cd /tmp/minifold && sed -i \'s/packages = \\["minifold"\\]/packages = {find = {}}/\' pyproject.toml && pip install .',
    )
)
