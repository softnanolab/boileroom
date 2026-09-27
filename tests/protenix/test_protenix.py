"""Protenix integration tests against a real backend."""

import numpy as np
import pytest

from boileroom import Protenix

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.gpu,
    pytest.mark.xdist_group("protenix"),
]


def test_protenix_modal_fold_basic(backend_option: str, device_option: str | None, output_ctx) -> None:
    """Fold a small two-chain complex and validate real Protenix structure outputs."""
    # A short homodimer with a reduced sampling budget keeps the MSA search and diffusion fast.
    chain = "GSHMKQLEDKVEELLSKNYHLENEVARLKKLVGER"
    sequence = f"{chain}:{chain}"
    config = {"sample": 1, "cycle": 4, "step": 50}
    options = {"include_fields": ["plddt", "ptm", "iptm", "pae", "cif", "token_chain_ids", "confidence"]}

    with output_ctx(), Protenix(backend=backend_option, device=device_option, config=config) as model:
        result = model.fold(sequence, options=options)

    expected_length = 2 * len(chain)
    assert result.metadata.sequence_lengths == [expected_length]

    assert result.atom_array is not None and len(result.atom_array) == 1
    assert len(result.atom_array[0]) > 0
    assert sorted(set(result.atom_array[0].chain_id)) == ["A", "B"]

    assert result.cif is not None and result.cif[0].startswith("data_")
    assert result.confidence is not None and result.confidence[0] is not None

    assert result.ptm is not None and result.ptm[0] is not None and np.isfinite(result.ptm[0]).all()
    assert result.iptm is not None and result.iptm[0] is not None and np.isfinite(result.iptm[0]).all()

    assert result.pae is not None
    assert result.pae[0].shape == (expected_length, expected_length)
    assert result.token_chain_ids is not None and len(result.token_chain_ids[0]) == expected_length
