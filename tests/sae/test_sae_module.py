"""Unit tests for the sparse-autoencoder module (torch-only, no model deps)."""

from __future__ import annotations

import json
import sys
import types
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from torch import Tensor

torch = pytest.importorskip("torch")

from boileroom.models.sae.sae_module import (  # noqa: E402
    CHECKPOINT_KEYS,
    ZSCORE_EPS,
    SAEModuleConfig,
    SparseAutoencoder,
    max_pool_features,
    topk_activation,
)


def _released_state_dict(d_model: int = 8, num_features: int = 16, seed: int = 0) -> dict[str, Tensor]:
    """A state dict in the layout of a released Biohub layer checkpoint (placeholder idf / max)."""
    generator = torch.Generator().manual_seed(seed)
    return {
        "W_enc": torch.randn(d_model, num_features, generator=generator),
        "W_dec": torch.randn(num_features, d_model, generator=generator),
        "b_dec": torch.randn(d_model, generator=generator),
        "idf": torch.ones(num_features),
        "max": torch.ones(num_features),
    }


def _with_stats(sae: SparseAutoencoder) -> SparseAutoencoder:
    with torch.no_grad():
        sae.max.copy_(torch.linspace(0.5, 8.0, sae.num_features))
        sae.idf.copy_(torch.linspace(1.0, 3.0, sae.num_features))
    return sae


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        SAEModuleConfig(d_model=0, num_features=16)
    with pytest.raises(ValueError):
        SAEModuleConfig(d_model=8, num_features=16, activation="softmax")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        SAEModuleConfig(d_model=8, num_features=16, k=32)  # k > num_features


def test_topk_activation_keeps_exactly_k_positive() -> None:
    pre = torch.tensor([[3.0, -1.0, 2.0, 5.0, -4.0]])
    out = topk_activation(pre, k=2)
    # Only the two largest positive pre-activations (5, 3) survive.
    assert torch.count_nonzero(out) == 2
    assert out[0, 3] == pytest.approx(5.0)
    assert out[0, 0] == pytest.approx(3.0)
    assert (out >= 0).all()


def test_topk_keeps_fewer_than_k_when_fewer_are_positive() -> None:
    out = topk_activation(torch.tensor([[-1.0, -2.0, 3.0, -0.5]]), k=2)
    assert out.tolist() == [[0.0, 0.0, 3.0, 0.0]]


def test_topk_never_returns_negative_even_if_k_large() -> None:
    pre = torch.tensor([[-1.0, -2.0, 3.0]])
    out = topk_activation(pre, k=3)  # k == features -> plain relu
    assert out.tolist() == [[0.0, 0.0, 3.0]]


def test_standardize_matches_reference_zscore() -> None:
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    # The reference divides by the unbiased standard deviation plus 1e-5.
    expected = (x - 2.5) / (torch.tensor(5.0 / 3.0).sqrt() + ZSCORE_EPS)
    assert torch.allclose(SparseAutoencoder.standardize(x), expected)


def test_encode_matches_reference_formula() -> None:
    cfg = SAEModuleConfig(d_model=8, num_features=16, k=3)
    sae = SparseAutoencoder.from_state_dict(_released_state_dict(), cfg)
    x = torch.randn(5, 8, generator=torch.Generator().manual_seed(1))
    z = (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + 1e-5)
    pre = torch.relu((z - sae.b_dec) @ sae.W_enc)
    top = torch.topk(pre, 3, dim=-1)
    expected = torch.zeros_like(pre).scatter(-1, top.indices, top.values)
    with torch.no_grad():
        assert torch.equal(sae.encode(x), expected)


def test_encode_is_invariant_to_input_offset_and_scale() -> None:
    # The SAE z-scores each residue, so an affine rescale of the representation leaves its features unchanged.
    sae = SparseAutoencoder.from_state_dict(_released_state_dict(), SAEModuleConfig(d_model=8, num_features=16, k=3))
    x = torch.randn(5, 8, generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        assert torch.allclose(sae.encode(x), sae.encode(3.0 * x + 7.0), atol=1e-5)


def test_encode_respects_sparsity_budget() -> None:
    cfg = SAEModuleConfig(d_model=16, num_features=64, k=5, activation="topk")
    sae = SparseAutoencoder(cfg)
    torch.manual_seed(0)
    x = torch.randn(10, 16)
    acts = sae.encode(x)
    assert acts.shape == (10, 64)
    # At most k active features per residue.
    assert int((acts > 0).sum(dim=-1).max()) <= 5


def test_relu_variant_has_no_hard_budget() -> None:
    cfg = SAEModuleConfig(d_model=8, num_features=32, activation="relu")
    sae = SparseAutoencoder(cfg)
    with torch.no_grad():
        # (z - b_dec) is then positive in every coordinate, and so is every pre-activation.
        sae.b_dec.fill_(-10.0)
        sae.W_enc.abs_()
    acts = sae.encode(torch.randn(4, 8))
    assert (acts >= 0).all()
    assert int((acts > 0).sum()) > 4 * 5  # more than a k=5 budget would allow


def test_forward_returns_reconstruction_shape() -> None:
    cfg = SAEModuleConfig(d_model=12, num_features=48, k=8)
    sae = SparseAutoencoder(cfg)
    x = torch.randn(6, 12)
    acts, recon = sae(x)
    assert acts.shape == (6, 48)
    assert recon.shape == (6, 12)


def test_save_and_load_roundtrip(tmp_path) -> None:
    cfg = SAEModuleConfig(d_model=10, num_features=40, k=4)
    sae = _with_stats(SparseAutoencoder(cfg))
    with torch.no_grad():
        sae.W_enc.normal_()
        sae.b_dec.normal_()
    sae.save(tmp_path)
    reloaded = SparseAutoencoder.load(tmp_path)
    x = torch.randn(5, 10)
    assert torch.allclose(sae.encode(x), reloaded.encode(x))
    assert torch.equal(reloaded.max, sae.max)
    assert torch.equal(reloaded.idf, sae.idf)


def test_from_state_dict_loads_released_layout() -> None:
    state = _released_state_dict()
    sae = SparseAutoencoder.from_state_dict(state, SAEModuleConfig(d_model=8, num_features=16, k=3))
    assert set(sae.state_dict()) == set(CHECKPOINT_KEYS)
    for key, tensor in state.items():
        assert torch.equal(sae.state_dict()[key], tensor)
    assert not sae.training


def test_from_state_dict_rejects_encoder_bias() -> None:
    # The reference architecture has no encoder bias; a checkpoint carrying one is a different model.
    state = {**_released_state_dict(), "b_enc": torch.zeros(16)}
    with pytest.raises(KeyError, match=r"unexpected \['b_enc'\]"):
        SparseAutoencoder.from_state_dict(state, SAEModuleConfig(d_model=8, num_features=16, k=3))


@pytest.mark.parametrize("dropped", ["idf", "max", "b_dec"])
def test_from_state_dict_rejects_missing_tensor(dropped: str) -> None:
    state = _released_state_dict()
    del state[dropped]
    with pytest.raises(KeyError, match=rf"missing \['{dropped}'\]"):
        SparseAutoencoder.from_state_dict(state, SAEModuleConfig(d_model=8, num_features=16, k=3))


def test_from_state_dict_rejects_foreign_names() -> None:
    foreign = {
        "encoder.weight": torch.randn(16, 8),
        "decoder.weight": torch.randn(16, 8),
        "b_pre": torch.randn(8),
    }
    with pytest.raises(KeyError, match="EsmcSaeLayer layout"):
        SparseAutoencoder.from_state_dict(foreign, SAEModuleConfig(d_model=8, num_features=16, k=3))


def test_from_state_dict_does_not_transpose() -> None:
    state = _released_state_dict()
    state["W_enc"] = state["W_enc"].t().contiguous()
    with pytest.raises(ValueError, match="'W_enc' has shape"):
        SparseAutoencoder.from_state_dict(state, SAEModuleConfig(d_model=8, num_features=16, k=3))


def test_placeholder_stats_refuse_normalization() -> None:
    sae = SparseAutoencoder.from_state_dict(_released_state_dict(), SAEModuleConfig(d_model=8, num_features=16, k=3))
    assert not sae.has_normalization_stats
    with pytest.raises(ValueError, match="placeholder normalization statistics"):
        sae.normalize_features(torch.ones(2, 16))


def test_normalize_features_divides_by_max_and_scales_by_idf() -> None:
    sae = _with_stats(SparseAutoencoder(SAEModuleConfig(d_model=8, num_features=16, k=3)))
    acts = torch.rand(4, 16)
    assert sae.has_normalization_stats
    assert torch.allclose(sae.normalize_features(acts), acts / sae.max * sae.idf)


@pytest.mark.parametrize(
    ("buffer", "value", "match"),
    [("max", 0.0, "'max' must be finite and positive"), ("idf", float("nan"), "'idf' must be finite")],
)
def test_invalid_stats_refuse_normalization(buffer: str, value: float, match: str) -> None:
    sae = _with_stats(SparseAutoencoder(SAEModuleConfig(d_model=8, num_features=16, k=3)))
    with torch.no_grad():
        getattr(sae, buffer)[3] = value
    with pytest.raises(ValueError, match=match):
        sae.normalize_features(torch.ones(2, 16))


# ---------------------------------------------------------------------------
# from_pretrained against a fake Hugging Face Hub
# ---------------------------------------------------------------------------
_REPO_CONFIG = {
    "d_model": 8,
    "codebook_dim": 16,
    "k": 3,
    "available_layers": [0, 1, 2, 3],
    "use_residual_update_instead_of_states": False,
}


@pytest.fixture
def fake_hub(tmp_path, monkeypatch):
    """Serve ``config.json`` and ``layer_<N>.safetensors`` from ``tmp_path`` via stand-in Hub modules."""
    downloads: list[str] = []

    def publish(config: dict, state: dict[str, Tensor], layer: int = 2) -> None:
        (tmp_path / "config.json").write_text(json.dumps(config))
        torch.save(state, tmp_path / f"layer_{layer}.safetensors")

    def hf_hub_download(repo_id: str, filename: str, revision=None, cache_dir=None) -> str:  # noqa: ANN001
        downloads.append(filename)
        return str(tmp_path / filename)

    def load_file(path: str, device: str = "cpu") -> dict[str, Tensor]:
        return torch.load(path, map_location=device)

    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = hf_hub_download  # type: ignore[attr-defined]
    safetensors = types.ModuleType("safetensors")
    safetensors_torch = types.ModuleType("safetensors.torch")
    safetensors_torch.load_file = load_file  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "safetensors", safetensors)
    monkeypatch.setitem(sys.modules, "safetensors.torch", safetensors_torch)
    return types.SimpleNamespace(publish=publish, downloads=downloads)


def test_from_pretrained_loads_released_checkpoint(fake_hub) -> None:
    # Regression for softnanolab/boileroom#115: the released layout (no b_enc, plus idf / max) must load.
    state = _released_state_dict()
    fake_hub.publish(_REPO_CONFIG, state)
    sae = SparseAutoencoder.from_pretrained("biohub/fake-sae", layer=2, d_model=8, num_features=16, k=3)
    assert fake_hub.downloads == ["config.json", "layer_2.safetensors"]
    assert sae.config == SAEModuleConfig(d_model=8, num_features=16, k=3)
    assert torch.equal(sae.W_enc, state["W_enc"])


def test_from_pretrained_takes_architecture_from_repo_config(fake_hub) -> None:
    fake_hub.publish(_REPO_CONFIG, _released_state_dict())
    sae = SparseAutoencoder.from_pretrained("biohub/fake-sae", layer=2)
    assert (sae.d_model, sae.num_features, sae.config.k) == (8, 16, 3)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"d_model": 9}, r"d_model=9 \(repo has 8\)"),
        ({"num_features": 32}, r"num_features=32 \(repo has 16\)"),
        ({"k": 64}, r"k=64 \(repo has 3\)"),
    ],
)
def test_from_pretrained_rejects_architecture_mismatch(fake_hub, overrides: dict, match: str) -> None:
    fake_hub.publish(_REPO_CONFIG, _released_state_dict())
    with pytest.raises(ValueError, match=match):
        SparseAutoencoder.from_pretrained("biohub/fake-sae", layer=2, **overrides)


def test_from_pretrained_rejects_unavailable_layer(fake_hub) -> None:
    fake_hub.publish(_REPO_CONFIG, _released_state_dict())
    with pytest.raises(ValueError, match="no SAE for layer 7"):
        SparseAutoencoder.from_pretrained("biohub/fake-sae", layer=7)
    assert fake_hub.downloads == ["config.json"]


def test_from_pretrained_rejects_residual_update_saes(fake_hub) -> None:
    fake_hub.publish({**_REPO_CONFIG, "use_residual_update_instead_of_states": True}, _released_state_dict())
    with pytest.raises(ValueError, match="residual updates"):
        SparseAutoencoder.from_pretrained("biohub/fake-sae", layer=2)


def test_from_pretrained_rejects_checkpoint_with_encoder_bias(fake_hub) -> None:
    fake_hub.publish(_REPO_CONFIG, {**_released_state_dict(), "b_enc": torch.zeros(16)})
    with pytest.raises(KeyError, match="b_enc"):
        SparseAutoencoder.from_pretrained("biohub/fake-sae", layer=2)


def test_max_pool_features_masks_padding() -> None:
    acts = torch.tensor([[1.0, 0.0], [0.0, 9.0], [5.0, 5.0]])
    mask = torch.tensor([True, True, False])  # drop the last residue
    pooled = max_pool_features(acts, mask)
    assert pooled.tolist() == [1.0, 9.0]


def test_max_pool_all_masked_returns_zeros() -> None:
    acts = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    pooled = max_pool_features(acts, torch.tensor([False, False]))
    assert pooled.tolist() == [0.0, 0.0]


def test_max_pool_requires_2d() -> None:
    with pytest.raises(ValueError):
        max_pool_features(torch.zeros(2, 3, 4))
