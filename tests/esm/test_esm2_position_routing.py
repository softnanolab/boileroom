"""Deterministic tests for ESM-2 rotary position-id routing.

These tests address the request to make the ``position_ids_skip`` coverage
*exact* rather than asserting only that embeddings change by some non-zero
amount. ESM-2 uses rotary position embeddings (RoPE), so position information
is not added to the token embeddings; it is injected inside attention by
rotating the query/key tensors. The amount of rotation is a closed-form
function of ``position_ids``, so the routing layer that ``position_ids_skip``
relies on can be checked against analytical expectations without loading any
model weights.

The tests below are CPU-only and do not require network access or a GPU; they
exercise :mod:`boileroom.models.esm.position_routing` directly together with
the real Transformers rotary kernels.
"""

import pytest


def _rotary_embedding(dim: int):
    """Build a real Transformers ESM rotary embedding module.

    Parameters
    ----------
    dim : int
        Per-head rotary dimension. Must be even; ``inv_freq`` will hold
        ``dim // 2`` frequencies and the cos/sin tables span ``dim`` columns.

    Returns
    -------
    transformers.models.esm.modeling_esm.RotaryEmbedding
        Initialized rotary embedding module with a populated ``inv_freq`` buffer.
    """
    modeling_esm = pytest.importorskip(
        "transformers.models.esm.modeling_esm", reason="requires transformers"
    )
    rotary_cls = getattr(modeling_esm, "RotaryEmbedding", None) or modeling_esm.EsmRotaryEmbedding
    return rotary_cls(dim)


def test_position_ids_rotary_cache_matches_closed_form():
    """Custom cos/sin tables must equal the analytical RoPE rotation for the ids.

    The routing helper builds its cache from ``arange`` and gathers by
    ``position_ids``; this must be identical to evaluating
    ``freqs = outer(position_ids, inv_freq)`` directly, which is the definition
    of RoPE. A multimer-style skip (``[0, 1, 514, 515]``) is used so the path
    that ``position_ids_skip`` triggers is the one under test.
    """
    torch = pytest.importorskip("torch", reason="requires torch")
    from boileroom.models.esm import position_routing

    dim = 8
    rotary = _rotary_embedding(dim)
    position_ids = torch.tensor([[0, 1, 514, 515]], dtype=torch.long)
    batch, seq = position_ids.shape
    key = torch.zeros(batch, 1, seq, dim, dtype=torch.float32)

    cache = position_routing._maybe_build_position_ids_rotary_cache(rotary, key, position_ids)
    assert cache is not None, "non-arange position ids must build a custom RoPE cache"
    cos, sin = cache

    freqs = torch.outer(position_ids.reshape(-1).to(torch.float32), rotary.inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    expected_cos = emb.cos().reshape(batch, seq, dim).unsqueeze(1)
    expected_sin = emb.sin().reshape(batch, seq, dim).unsqueeze(1)

    assert cos.shape == (batch, 1, seq, dim)
    assert sin.shape == (batch, 1, seq, dim)
    torch.testing.assert_close(cos, expected_cos)
    torch.testing.assert_close(sin, expected_sin)


def test_rotary_cache_matches_native_for_contiguous_positions():
    """Contiguous ids reproduce the stock Transformers cache (monomers untouched).

    ``position_ids_skip`` must only change cross-chain geometry. For contiguous
    ``[0, 1, ..., L-1]`` positions the custom cache has to equal the native
    ``_update_cos_sin_tables`` output bit-for-bit, guaranteeing monomers and
    arange-equivalent inputs behave exactly as upstream ESM-2.
    """
    torch = pytest.importorskip("torch", reason="requires torch")
    from boileroom.models.esm import position_routing

    dim = 8
    seq = 6
    rotary = _rotary_embedding(dim)
    key = torch.zeros(1, 2, seq, dim, dtype=torch.float32)

    native_cos, native_sin = rotary._update_cos_sin_tables(key, seq_dimension=-2)

    # arange ids are routed to the native path, so build the custom cache from the
    # closed form to prove the two definitions coincide for contiguous positions.
    position_ids = torch.arange(seq, dtype=torch.long).unsqueeze(0)
    freqs = torch.outer(position_ids.reshape(-1).to(torch.float32), rotary.inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    custom_cos = emb.cos().reshape(1, seq, dim).unsqueeze(1)
    custom_sin = emb.sin().reshape(1, seq, dim).unsqueeze(1)

    torch.testing.assert_close(custom_cos, native_cos)
    torch.testing.assert_close(custom_sin, native_sin)


def test_arange_position_ids_use_native_path():
    """Default contiguous ids must opt out of custom routing.

    ``_is_default_arange_position_ids`` is the guard that keeps monomers on the
    upstream kernel; contiguous (optionally zero-padded) rows are arange-like and
    return ``None`` from the cache builder, while a multimer skip is not.
    """
    torch = pytest.importorskip("torch", reason="requires torch")
    from boileroom.models.esm import position_routing

    assert position_routing._is_default_arange_position_ids(torch.tensor([[0, 1, 2, 3]]))
    # arange with right padding (special-token padding) still counts as arange.
    assert position_routing._is_default_arange_position_ids(torch.tensor([[0, 1, 2, 0]]))
    # a multimer skip is genuinely non-contiguous.
    assert not position_routing._is_default_arange_position_ids(torch.tensor([[0, 1, 514, 515]]))

    rotary = _rotary_embedding(8)
    key = torch.zeros(1, 1, 4, 8, dtype=torch.float32)
    assert (
        position_routing._maybe_build_position_ids_rotary_cache(
            rotary, key, torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
        )
        is None
    )


def test_rotary_routing_is_relative_and_skip_changes_cross_chain_scores():
    """RoPE attention is relative, so a skip only moves cross-chain logits.

    This is the exact statement behind ``position_ids_skip``: rotated
    ``q_i . k_j`` depends solely on ``p_i - p_j``. Therefore

    * a global shift of every position id leaves all attention logits unchanged,
      and
    * increasing the skip leaves the within-chain logit identical while changing
      the cross-chain logit.

    Both are checked exactly here (no model weights), which is far stronger than
    asserting the embedding difference is merely positive.
    """
    torch = pytest.importorskip("torch", reason="requires torch")
    from transformers.models.esm.modeling_esm import apply_rotary_pos_emb

    from boileroom.models.esm import position_routing

    dim = 8
    rotary = _rotary_embedding(dim)
    torch.manual_seed(0)
    # Two-token "complex": index 0 is chain A, index 1 is chain B.
    query = torch.randn(1, 1, 2, dim, dtype=torch.float32)
    key = torch.randn(1, 1, 2, dim, dtype=torch.float32)

    def attention_logits(positions: list[int]) -> torch.Tensor:
        position_ids = torch.tensor([positions], dtype=torch.long)
        cache = position_routing._maybe_build_position_ids_rotary_cache(rotary, key, position_ids)
        assert cache is not None
        cos, sin = cache
        rotated_q = apply_rotary_pos_emb(query, cos, sin)
        rotated_k = apply_rotary_pos_emb(key, cos, sin)
        return torch.matmul(rotated_q, rotated_k.transpose(-1, -2))

    # Same inter-token gap (1) at different absolute offsets -> identical logits.
    base = attention_logits([5, 6])
    shifted = attention_logits([105, 106])
    torch.testing.assert_close(base, shifted)

    # A larger skip preserves the diagonal (gap 0) but changes the off-diagonal.
    skipped = attention_logits([5, 506])
    torch.testing.assert_close(torch.diagonal(skipped, dim1=-2, dim2=-1), torch.diagonal(base, dim1=-2, dim2=-1))
    assert not torch.allclose(skipped[..., 0, 1], base[..., 0, 1], atol=1e-3), (
        "increasing the skip must change the cross-chain attention logit"
    )
