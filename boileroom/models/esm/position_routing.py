"""Route Boileroom's ESM-2 multimer ``position_ids`` into rotary embeddings.

HuggingFace's ESM-2 model accepts ``position_ids`` at the model-call boundary,
but older Transformer releases do not pass those ids down into the rotary
embedding helper. The helper instead rebuilds contiguous token positions from
the query/key tensor shape. That is fine for ordinary monomers, but it erases
Boileroom's deliberate multimer position gaps, so ``position_ids_skip`` changes
the bookkeeping tensors while the attention layers still see vanilla contiguous
rotary positions.

This module installs a narrow ESM-2 patch that keeps the public model call the
same while routing the current call's explicit ``position_ids`` into RoPE cache
construction. Default arange-style ids fall back to the native Transformer path;
non-contiguous multimer ids use the patched cache by default.
"""

import dataclasses
import inspect
import logging
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, cast

import torch

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class _PositionIdsRoutingState:
    """Per-call state for explicit ESM-2 position-id routing."""

    position_ids: torch.Tensor


_POSITION_IDS_ROUTING_STATE: ContextVar[_PositionIdsRoutingState | None] = ContextVar(
    "boileroom_esm_position_ids_routing", default=None
)
_POSITION_IDS_ROUTING_PATCHES_INSTALLED = False
_POSITION_IDS_ROUTING_WARNING_EMITTED = False


def _position_ids_routing_state() -> _PositionIdsRoutingState | None:
    """Return the current explicit position-id routing state, if active."""

    return _POSITION_IDS_ROUTING_STATE.get()


def _is_default_arange_position_ids(position_ids: torch.Tensor) -> bool:
    """Return ``True`` when ``position_ids`` is equivalent to ``arange`` per row."""

    if position_ids.ndim != 2:
        return False
    position_ids = position_ids.to(dtype=torch.long)

    for row in position_ids:
        non_zero_idx = torch.nonzero(row, as_tuple=False).view(-1)
        if non_zero_idx.numel() == 0:
            continue

        last_non_zero = int(non_zero_idx[-1].item()) + 1
        prefix = row[:last_non_zero]
        if not torch.all(prefix == torch.arange(last_non_zero, device=row.device, dtype=row.dtype)):
            return False

        if last_non_zero < row.shape[0] and torch.any(row[last_non_zero:] != 0):
            return False

    return True


@contextmanager
def esm_position_ids_context(position_ids: torch.Tensor | None):
    """Route explicit ESM-2 position ids for a single inference call."""

    if position_ids is None:
        yield
        return

    token = _POSITION_IDS_ROUTING_STATE.set(_PositionIdsRoutingState(position_ids=position_ids))
    try:
        yield
    finally:
        _POSITION_IDS_ROUTING_STATE.reset(token)


def _maybe_build_position_ids_rotary_cache(
    rotary_embedding: Any,
    k: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Build custom RoPE caches from ``position_ids`` when available."""

    if _is_default_arange_position_ids(position_ids):
        return None

    if position_ids.shape[0] != k.shape[0] or position_ids.shape[1] != k.shape[-2]:
        global _POSITION_IDS_ROUTING_WARNING_EMITTED
        if not _POSITION_IDS_ROUTING_WARNING_EMITTED:
            logger.warning(
                "ESM position-id routing disabled for rotary embedding due to shape mismatch: "
                "position_ids=%s, key_shape=%s",
                tuple(position_ids.shape),
                tuple(k.shape),
            )
            _POSITION_IDS_ROUTING_WARNING_EMITTED = True
        return None

    if position_ids.device != k.device or position_ids.dtype != torch.long:
        position_ids = position_ids.to(device=k.device, dtype=torch.long)

    seq_len = k.shape[-2]
    if seq_len <= 0:
        return None

    max_position = int(torch.max(position_ids).item()) + 1
    max_position = max(max_position, seq_len)
    t = torch.arange(max_position, device=k.device, dtype=rotary_embedding.inv_freq.dtype)
    freqs = torch.outer(t, rotary_embedding.inv_freq.to(device=k.device))
    emb = torch.cat((freqs, freqs), dim=-1)

    cos = emb.cos()[None, :, :]
    sin = emb.sin()[None, :, :]

    indices = position_ids.reshape(-1).clamp(min=0, max=cos.shape[1] - 1).to(torch.long)
    batch, _seq_len = position_ids.shape

    cos = cos.index_select(1, indices).reshape(batch, _seq_len, -1).unsqueeze(1)
    sin = sin.index_select(1, indices).reshape(batch, _seq_len, -1).unsqueeze(1)
    return cos, sin


def install_esm_position_ids_routing() -> None:
    """Install default ESM-2 RoPE routing for explicit multimer position ids.

    The patch is intentionally scoped to Transformer ESM rotary embedding
    classes whose ``forward`` method still takes only ``q`` and ``k``. Newer
    Transformer versions that natively accept ``position_ids`` are left
    untouched. This makes the corrected multimer behavior the default while
    avoiding a compatibility flag or a fork of the surrounding ESM-2 model code.
    """

    global _POSITION_IDS_ROUTING_PATCHES_INSTALLED
    if _POSITION_IDS_ROUTING_PATCHES_INSTALLED:
        return

    try:
        from transformers.models.esm import modeling_esm
    except (ImportError, ModuleNotFoundError):
        logger.debug("Skipping ESM position-id routing patch: transformers not importable at import-time.")
        return

    rotary_embedding_cls = getattr(modeling_esm, "RotaryEmbedding", None) or getattr(
        modeling_esm, "EsmRotaryEmbedding", None
    )
    if rotary_embedding_cls is None:
        logger.warning("Skipping ESM rotary position-id routing patch: rotary embedding class is unavailable.")
        return

    rotary_forward_sig = inspect.signature(rotary_embedding_cls.forward)
    if len(rotary_forward_sig.parameters) == 3:
        original_forward = rotary_embedding_cls.forward

        if not hasattr(original_forward, "__wrapped__"):

            def position_ids_rotary_forward(self: Any, q: torch.Tensor, k: torch.Tensor):
                state = _position_ids_routing_state()
                if state is None:
                    return original_forward(self, q, k)

                cache = _maybe_build_position_ids_rotary_cache(self, k, state.position_ids.to(device=k.device))
                if cache is None:
                    return original_forward(self, q, k)

                cos, sin = cache
                from transformers.models.esm.modeling_esm import apply_rotary_pos_emb

                return (
                    apply_rotary_pos_emb(q, cos, sin).to(dtype=q.dtype),
                    apply_rotary_pos_emb(k, cos, sin).to(dtype=k.dtype),
                )

            position_ids_rotary_forward.__name__ = "position_ids_routing_forward"
            cast(Any, position_ids_rotary_forward).__wrapped__ = original_forward
            cast(Any, rotary_embedding_cls)._position_ids_routing_original_forward = original_forward
            rotary_embedding_cls.forward = position_ids_rotary_forward
    elif "position_ids" in rotary_forward_sig.parameters:
        logger.debug("Skipping ESM rotary position-id routing patch: native rotary embedding accepts position_ids.")
    else:
        logger.warning(
            "Skipping ESM rotary position-id routing patch: signature mismatch for %s.forward: %s",
            rotary_embedding_cls.__name__,
            rotary_forward_sig,
        )

    _POSITION_IDS_ROUTING_PATCHES_INSTALLED = True
