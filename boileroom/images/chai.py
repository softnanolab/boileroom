"""Modal image definition for Chai-1."""

from .base import base_image

chai_image = base_image.pip_install("chai_lab==0.6.1")
