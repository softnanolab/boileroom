"""Modal image for the ESM-C sparse-autoencoder feature model.

The SAE model reuses ESM-C hidden states, so it shares the single Biohub ``esm``
runtime image (the ``esmfold2`` image, as ESM-C / ESM3 do). The SAE itself only
adds ``huggingface_hub`` + ``safetensors`` weight loading, both already present in
that image.
"""

from ...images.modal import get_modal_image

sae_image = get_modal_image("esmfold2")
