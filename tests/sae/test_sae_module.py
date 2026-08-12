"""Unit tests for the sparse-autoencoder module (torch-only, no model deps)."""

import pytest

torch = pytest.importorskip("torch")

from boileroom.models.sae.sae_module import (  # noqa: E402
    SAEModuleConfig,
    SparseAutoencoder,
    max_pool_features,
    topk_activation,
)


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


def test_topk_never_returns_negative_even_if_k_large() -> None:
    pre = torch.tensor([[-1.0, -2.0, 3.0]])
    out = topk_activation(pre, k=3)  # k == features -> plain relu
    assert out.tolist() == [[0.0, 0.0, 3.0]]


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
        sae.b_enc.fill_(1.0)  # push activations positive
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
    sae = SparseAutoencoder(cfg)
    with torch.no_grad():
        sae.W_enc.normal_()
        sae.b_enc.normal_()
    sae.save(tmp_path)
    reloaded = SparseAutoencoder.load(tmp_path)
    x = torch.randn(5, 10)
    assert torch.allclose(sae.encode(x), reloaded.encode(x))


def test_from_state_dict_remaps_and_transposes() -> None:
    cfg = SAEModuleConfig(d_model=8, num_features=16, k=3)
    # Foreign names + transposed encoder orientation (num_features, d_model).
    foreign = {
        "encoder.weight": torch.randn(16, 8),
        "encoder.bias": torch.randn(16),
        "decoder.weight": torch.randn(16, 8),
        "pre_bias": torch.randn(8),
    }
    sae = SparseAutoencoder.from_state_dict(foreign, cfg)
    assert sae.W_enc.shape == (8, 16)
    # Encoder weight should be the transpose of the foreign (16, 8) matrix.
    assert torch.allclose(sae.W_enc, foreign["encoder.weight"].t())
    assert torch.allclose(sae.b_pre, foreign["pre_bias"])


def test_from_state_dict_strict_raises_on_missing() -> None:
    cfg = SAEModuleConfig(d_model=8, num_features=16, k=3)
    with pytest.raises(KeyError):
        SparseAutoencoder.from_state_dict({"encoder.weight": torch.randn(8, 16)}, cfg, strict=True)


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
