"""SAE features from the released Biohub checkpoints, checked against Biohub's own pipeline.

The references in ``tests/data/sae`` come from ``scripts/testing/sae_biohub_reference.py``: Biohub's native
``EsmcModel`` with the ``EsmcSaeLayer`` attached, run in float32 on the CPU. Both tests are opt-in:

- ``test_released_checkpoint_matches_biohub_reference`` downloads the pinned SAE layer from the Hub and encodes the
  stored layer states on the CPU. It needs ``huggingface_hub`` and ``safetensors``, which boileroom does not install::

      uv run --extra dev --with huggingface_hub --with safetensors python -m pytest -m integration \
          tests/sae/test_sae_reference.py -k released

- ``test_local_sae_matches_biohub_reference`` runs ``SAE(config={"feature_source": "local"})`` end to end on a
  backend, so it also covers which ESM-C hidden state the SAE reads.
"""

import json
import pathlib
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any

import numpy as np
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]

REFERENCE_DIR = pathlib.Path(__file__).parent.parent / "data" / "sae"
MANIFEST = json.loads((REFERENCE_DIR / "manifest.json").read_text())
MODELS = sorted(MANIFEST["references"])
#: Backends run ESM-C in bfloat16 on the GPU, while the reference runs it in float32, so a few features near each
#: residue's top-k cut swap in or out. bfloat16 ESM-C gives a per-residue Jaccard of about 0.97 and a pooled cosine of
#: about 0.999; reading the neighbouring layer's hidden state instead drops them to about 0.55 and 0.94.
MIN_RESIDUE_JACCARD = 0.9
MIN_POOLED_COSINE = 0.99


def _reference_features(reference: dict, sequence_name: str) -> np.ndarray:
    """Dense ``(residues, num_features)`` reference activations of one sequence."""
    arrays = np.load(REFERENCE_DIR / reference["features"])
    indices, values = arrays[f"{sequence_name}.indices"], arrays[f"{sequence_name}.values"]
    dense = np.zeros((indices.shape[0], reference["num_features"]), dtype=np.float32)
    np.put_along_axis(dense, indices.astype(np.int64), values, axis=-1)
    return dense


def _mean_support_jaccard(features: np.ndarray, reference: np.ndarray) -> float:
    ours, theirs = features != 0, reference != 0
    union = np.maximum((ours | theirs).sum(-1), 1)
    return float(((ours & theirs).sum(-1) / union).mean())


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


@pytest.mark.parametrize("esmc_model_name", MODELS)
def test_released_checkpoint_matches_biohub_reference(esmc_model_name: str) -> None:
    """The loader and encoder reproduce Biohub's features from the same layer states."""
    pytest.importorskip("huggingface_hub")
    pytest.importorskip("safetensors")
    torch = pytest.importorskip("torch")
    from boileroom.models.sae.sae_module import SparseAutoencoder

    reference = MANIFEST["references"][esmc_model_name]
    sae = SparseAutoencoder.from_pretrained(
        reference["sae_repo"], layer=reference["layer"], revision=reference["sae_revision"], device="cpu"
    )
    assert (sae.d_model, sae.num_features, sae.config.k) == (
        reference["d_model"],
        reference["num_features"],
        reference["k"],
    )
    # The all-layer releases carry placeholder normalization statistics.
    assert not sae.has_normalization_stats

    for sequence_name in reference["state_sequences"]:
        states = np.load(REFERENCE_DIR / reference["features"])[f"{sequence_name}.states"]
        with torch.no_grad():
            features = sae.encode(torch.from_numpy(states)).numpy()
        expected = _reference_features(reference, sequence_name)
        np.testing.assert_array_equal(features != 0, expected != 0)
        np.testing.assert_allclose(features, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.gpu
@pytest.mark.xdist_group("sae")
@pytest.mark.parametrize("esmc_model_name", MODELS)
def test_local_sae_matches_biohub_reference(
    esmc_model_name: str,
    backend_option: str,
    device_option: str | None,
    output_ctx: Callable[[], AbstractContextManager[Any]],
) -> None:
    """``feature_source="local"`` loads the released SAE and feeds it the layer Biohub trained it on."""
    from boileroom import SAE

    reference = MANIFEST["references"][esmc_model_name]
    sequences = MANIFEST["sequences"]
    config = {"feature_source": "local", "esmc_model_name": esmc_model_name}

    with output_ctx(), SAE(backend=backend_option, device=device_option, config=config) as model:
        result = model.embed(list(sequences.values()), options={"include_per_residue": True})

    assert result.layer == reference["layer"]
    assert result.sae_model == reference["sae_repo"]
    assert result.num_features == reference["num_features"]
    assert result.features is not None
    for row, (sequence_name, sequence) in enumerate(sequences.items()):
        expected = _reference_features(reference, sequence_name)
        features = result.features[row, : len(sequence)]
        jaccard = _mean_support_jaccard(features, expected)
        cosine = _cosine(result.pooled_features[row], expected.max(axis=0))
        assert jaccard >= MIN_RESIDUE_JACCARD, f"{sequence_name}: per-residue feature Jaccard {jaccard:.3f}"
        assert cosine >= MIN_POOLED_COSINE, f"{sequence_name}: pooled feature cosine {cosine:.4f}"
