"""Tests for SAECore using a fake ESM-C embedder (no Modal / esm / GPU)."""

from dataclasses import dataclass

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from boileroom.models.sae.core import SAECore  # noqa: E402
from boileroom.models.sae.sae_module import SAEModuleConfig, SparseAutoencoder  # noqa: E402


@dataclass
class _FakeEmbedding:
    hidden_states: np.ndarray
    chain_index: np.ndarray
    residue_index: np.ndarray


class _FakeESMC:
    """Returns canned hidden states shaped (layers, batch, residues, d_model)."""

    def __init__(self, hidden_states: np.ndarray, chain_index: np.ndarray, residue_index: np.ndarray) -> None:
        self._out = _FakeEmbedding(hidden_states, chain_index, residue_index)
        self.calls: list[dict | None] = []

    def embed(self, sequences, options=None):  # noqa: ANN001
        self.calls.append(options)
        return self._out


def _make_core(d_model=8, num_features=16, k=3, layer=1, batch=2, residues=4, seed=0):
    rng = np.random.default_rng(seed)
    n_layers = 3
    hidden = rng.standard_normal((n_layers, batch, residues, d_model)).astype(np.float32)
    chain_index = np.zeros((batch, residues), dtype=np.int64)
    chain_index[0, -1] = -1  # first sequence has one padded residue
    residue_index = np.tile(np.arange(residues), (batch, 1))
    residue_index[0, -1] = -1
    embedder = _FakeESMC(hidden, chain_index, residue_index)
    sae = SparseAutoencoder(SAEModuleConfig(d_model=d_model, num_features=num_features, k=k))
    with torch.no_grad():
        sae.W_enc.normal_(generator=torch.Generator().manual_seed(seed))
    core = SAECore(
        config={
            "device": "cpu",
            "feature_source": "local",
            "normalize_features": False,
            "sae_layer": layer,
            "num_features": num_features,
            "k": k,
        },
        embedder=embedder,
        sae=sae,
    )
    core._load()
    return core, embedder, sae, hidden, chain_index


def test_embed_requests_hidden_states_and_pools() -> None:
    core, embedder, sae, hidden, chain_index = _make_core()
    out = core.embed(["ACDE", "ACDE"])
    # The embedder must have been asked for hidden states.
    assert embedder.calls and embedder.calls[0] == {"include_fields": ["hidden_states"]}
    assert out.pooled_features.shape == (2, 16)
    assert out.layer == 1
    assert out.num_features == 16
    # Pooled features are non-negative (TopK/ReLU) and finite.
    assert (out.pooled_features >= 0).all()
    assert np.isfinite(out.pooled_features).all()


def test_pooled_matches_manual_max_over_valid_residues() -> None:
    core, embedder, sae, hidden, chain_index = _make_core()
    out = core.embed(["ACDE", "ACDE"])
    # Recompute expected pooled vector for sequence 0 (last residue is padding).
    with torch.no_grad():
        layer_states = torch.as_tensor(hidden[1, 0], dtype=torch.float32)
        acts = sae.encode(layer_states)
        valid = torch.as_tensor(chain_index[0] != -1)
        expected = acts[valid].max(dim=0).values.numpy()
    np.testing.assert_allclose(out.pooled_features[0], expected, rtol=1e-5, atol=1e-6)


def test_padding_residue_excluded_from_pool() -> None:
    layer = 1  # matches the default sae_layer used by _make_core
    core, _embedder, sae, hidden, chain_index = _make_core(layer=layer)
    # Make the padded residue of sequence 0 dominate the hidden states; its (large)
    # activation must not leak into the pooled vector because it is masked out.
    hidden[layer, 0, -1, :] = 1e4
    out_padded = core.embed(["ACD", "ACDE"])
    with torch.no_grad():
        layer_states = torch.as_tensor(hidden[layer, 0], dtype=torch.float32)
        acts = sae.encode(layer_states)
        valid = torch.as_tensor(chain_index[0] != -1)
        max_valid = float(acts[valid].max())
        all_max = float(acts.max())
    pooled_max = float(out_padded.pooled_features[0].max())
    assert pooled_max == pytest.approx(max_valid, rel=1e-5)
    # Guard that the test is meaningful: the masked padded residue really is the
    # dominant activation, so a broken mask would produce a strictly larger pool.
    assert all_max > pooled_max


def test_include_per_residue_returns_dense_activations() -> None:
    core, *_ = _make_core()
    out = core.embed(["ACDE", "ACDE"], options={"include_per_residue": True})
    assert out.features is not None
    assert out.features.shape == (2, 4, 16)


def test_per_residue_none_by_default() -> None:
    core, *_ = _make_core()
    out = core.embed(["ACDE", "ACDE"])
    assert out.features is None


def test_static_config_key_rejected_per_call() -> None:
    core, *_ = _make_core()
    with pytest.raises(ValueError):
        core.embed(["ACDE"], options={"num_features": 999})


def test_missing_hidden_states_raises() -> None:
    class _NoHidden:
        def embed(self, sequences, options=None):  # noqa: ANN001
            return _FakeEmbedding(None, np.zeros((1, 3), dtype=np.int64), np.zeros((1, 3), dtype=np.int64))

    sae = SparseAutoencoder(SAEModuleConfig(d_model=8, num_features=16, k=3))
    core = SAECore(config={"device": "cpu", "feature_source": "local"}, embedder=_NoHidden(), sae=sae)
    core._load()
    with pytest.raises(ValueError, match="hidden_states"):
        core.embed(["ACDE"])


def test_layer_out_of_range_raises() -> None:
    core, *_ = _make_core(layer=99)
    with pytest.raises(IndexError):
        core.embed(["ACDE", "ACDE"])


def test_local_defaults_resolve_per_model() -> None:
    # Default local model is esmc_600m -> its repo + representative layer 27.
    core = SAECore(config={"feature_source": "local"})
    assert core.config["esmc_model_name"] == "esmc_600m"
    assert core.config["sae_layer"] == 27
    assert core.config["sae_repo_id"] == "biohub/ESMC-600M-sae-k64-codebook16384"
    # 300m selects its own repo + layer.
    core_300 = SAECore(config={"feature_source": "local", "esmc_model_name": "esmc_300m"})
    assert core_300.config["sae_layer"] == 22
    assert core_300.config["sae_repo_id"] == "biohub/ESMC-300M-sae-k64-codebook16384"


def test_local_defaults_do_not_override_explicit_values() -> None:
    core = SAECore(
        config={
            "feature_source": "local",
            "esmc_model_name": "esmc_600m",
            "sae_layer": 12,
            "sae_repo_id": "acme/custom-sae",
        }
    )
    assert core.config["sae_layer"] == 12
    assert core.config["sae_repo_id"] == "acme/custom-sae"


def test_forge_default_layer_is_unaffected_by_local_resolution() -> None:
    core = SAECore()
    assert core.config["feature_source"] == "forge"
    assert core.config["sae_layer"] == 60


def test_infer_d_model_unknown_model() -> None:
    sae = SparseAutoencoder(SAEModuleConfig(d_model=8, num_features=16, k=3))
    core = SAECore(
        config={"feature_source": "local", "esmc_model_name": "esmc_999x"},
        embedder=_FakeESMC(
            np.zeros((3, 1, 2, 8), dtype=np.float32),
            np.zeros((1, 2), dtype=np.int64),
            np.zeros((1, 2), dtype=np.int64),
        ),
        sae=sae,
    )
    core._load()
    with pytest.raises(ValueError, match="Unknown ESM-C model"):
        core._infer_d_model()
