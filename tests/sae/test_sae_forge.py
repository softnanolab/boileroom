"""Tests for the Forge SAE backend and the forge feature_source path.

A fake Forge backend returns canned per-token activations so the whole path runs
without the ``esm`` SDK, a token, or the network.
"""

import numpy as np
import pytest

# ``core`` imports torch at module scope; skip the whole module when torch is absent
# instead of failing collection (the forge backend itself stays torch-free).
pytest.importorskip("torch")

from boileroom.models.sae.core import SAECore  # noqa: E402
from boileroom.models.sae.forge import ForgeSAEBackend  # noqa: E402


class _FakeForge:
    """Returns (tokens, num_features) activations: BOS + residues + EOS."""

    def __init__(self, num_features: int = 8) -> None:
        self.num_features = num_features
        self.calls: list[str] = []

    def features(self, sdk_sequence: str) -> np.ndarray:
        self.calls.append(sdk_sequence)
        n_tokens = len(sdk_sequence) + 2  # BOS + tokens (incl '|') + EOS
        arr = np.zeros((n_tokens, self.num_features), dtype=np.float32)
        # Give each residue token a distinctive activation so pooling is checkable.
        for pos, ch in enumerate(sdk_sequence):
            arr[pos + 1, ord(ch) % self.num_features] = float(pos + 1)
        # Put a huge value on BOS/EOS to ensure they are excluded from the pool.
        arr[0, :] = 999.0
        arr[-1, :] = 999.0
        return arr


def _forge_core(num_features: int = 8) -> tuple[SAECore, _FakeForge]:
    forge = _FakeForge(num_features)
    core = SAECore(config={"feature_source": "forge", "num_features": num_features}, forge_backend=forge)
    core._load()
    return core, forge


def test_default_source_is_forge_with_6b_layer60():
    core = SAECore(forge_backend=_FakeForge())
    assert core.config["feature_source"] == "forge"
    assert core.config["sae_layer"] == 60
    assert core.config["forge_sae_model"] == "esmc-6b-2024-12-sae-layer60-k64-codebook16384"


def test_forge_embed_pools_and_excludes_bos_eos():
    core, forge = _forge_core()
    out = core.embed(["ACDE"])
    assert out.pooled_features.shape == (1, 8)
    # BOS/EOS activations (999) must not appear in the pooled vector.
    assert out.pooled_features.max() < 999.0
    assert out.sae_model == "esmc-6b-2024-12-sae-layer60-k64-codebook16384"
    assert forge.calls == ["ACDE"]


def test_forge_drops_chain_break_tokens():
    core, forge = _forge_core()
    out = core.embed(["AC:DE"])
    # Two chains, 4 residues total; residue/chain indices exclude the break token.
    assert out.chain_index.shape == (1, 4)
    assert out.chain_index.tolist() == [[0, 0, 1, 1]]
    # The SDK sequence sent to forge uses '|' as the chain break.
    assert forge.calls == ["AC|DE"]


def test_forge_include_per_residue():
    core, _ = _forge_core()
    out = core.embed(["ACDE"], options={"include_per_residue": True})
    assert out.features is not None
    assert out.features.shape == (1, 4, 8)


def test_forge_per_residue_none_by_default():
    core, _ = _forge_core()
    out = core.embed(["ACDE"])
    assert out.features is None


def test_forge_batch_padding():
    core, _ = _forge_core()
    out = core.embed(["ACDE", "AC"])
    assert out.pooled_features.shape == (2, 8)
    assert out.chain_index.shape == (2, 4)  # padded to the longest
    assert out.chain_index[1].tolist() == [0, 0, -1, -1]


def test_forge_wrong_token_row_count_raises():
    class _BadForge(_FakeForge):
        def features(self, sdk_sequence: str) -> np.ndarray:
            # Return one fewer row than BOS + tokens + EOS to simulate a
            # tokenization change; the core must reject it rather than misalign.
            return np.zeros((len(sdk_sequence) + 1, self.num_features), dtype=np.float32)

    core = SAECore(config={"feature_source": "forge", "num_features": 8}, forge_backend=_BadForge())
    core._load()
    with pytest.raises(ValueError, match="token rows"):
        core.embed(["ACDE"])


def test_invalid_feature_source_raises():
    with pytest.raises(ValueError, match="feature_source"):
        SAECore(config={"feature_source": "banana"})


def test_forge_backend_requires_token(monkeypatch):
    monkeypatch.delenv("ESM_API_KEY", raising=False)
    backend = ForgeSAEBackend(model="esmc-6b-2024-12", sae_model="x", token=None)
    with pytest.raises(ValueError, match="token"):
        backend._ensure_client()
