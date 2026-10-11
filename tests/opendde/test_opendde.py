"""OpenDDE integration test against a real backend (GPU, weights and MSA server required)."""

import time

import numpy as np
import pytest

from boileroom import OpenDDE

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.gpu,
    pytest.mark.xdist_group("opendde"),
]


def test_opendde_modal_fold_basic(backend_option: str, device_option: str | None, output_ctx) -> None:
    """Fold a small two-chain complex twice on one loaded model and validate the structure outputs."""
    chain = "GSHMKQLEDKVEELLSKNYHLENEVARLKKLVGER"
    config = {"sample": 1, "cycle": 4, "step": 50}
    options = {"include_fields": ["plddt", "ptm", "iptm", "pae", "cif", "token_chain_ids", "confidence"]}

    with output_ctx(), OpenDDE(backend=backend_option, device=device_option, config=config) as model:
        result = model.fold(f"{chain}:{chain}", options=options)
        started = time.monotonic()
        warm = model.fold(f"{chain}:{chain}", options={**options, "seeds": "102"})
        print(f"OpenDDE warm={time.monotonic() - started:.2f}s")

    assert result.seeds == [101] and warm.seeds == [102]
    assert result.metadata.sequence_lengths == [2 * len(chain)]
    assert result.atom_array is not None and sorted(set(result.atom_array[0].chain_id)) == ["A", "B"]
    assert result.cif is not None and result.cif[0].startswith("data_")
    assert result.pae is not None and result.pae[0].shape == (2 * len(chain),) * 2
    assert np.isfinite(result.pae[0]).all() and np.isfinite(warm.pae[0]).all()
    assert result.ptm is not None and np.isfinite(result.ptm[0]).all()
    assert result.iptm is not None and np.isfinite(result.iptm[0]).all()
