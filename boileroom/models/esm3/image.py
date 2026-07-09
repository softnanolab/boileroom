"""Modal image for the ESM-C / ESM3 runtime.

ESM-C and ESM3 are served from the MIT-licensed 2026 Chan Zuckerberg Biohub
``esm`` fork, the same package that backs ESMFold2, so they share the single
``esmfold2`` Biohub runtime image instead of building a separate one.
"""

from ...images.modal import get_modal_image

esm3_image = get_modal_image("esmfold2")
