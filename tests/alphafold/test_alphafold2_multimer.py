"""AlphaFold2-Multimer (ColabFold) integration tests against a real backend."""

import time

import numpy as np
import pytest

from boileroom import AlphaFold2Multimer

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.gpu,
    pytest.mark.xdist_group("alphafold2_multimer"),
]


def test_alphafold2_multimer_modal_fold_basic(backend_option: str, device_option: str | None, output_ctx) -> None:
    """Fold a small two-chain complex and validate real ColabFold structure outputs."""
    # A short homodimer keeps the MMseqs2 server query and folding fast.
    chain = "GSHMKQLEDKVEELLSKNYHLENEVARLKKLVGER"
    sequence = f"{chain}:{chain}"
    config = {"num_models": 1, "num_recycle": 1, "num_seeds": 1}
    options = {"include_fields": ["plddt", "ptm", "iptm", "pae", "cif", "ranking"]}

    with output_ctx(), AlphaFold2Multimer(backend=backend_option, device=device_option, config=config) as model:
        started = time.monotonic()
        result = model.fold(sequence, options=options)
        cold_seconds = time.monotonic() - started
        started = time.monotonic()
        warm = model.fold(sequence, options={**options, "random_seed": 1})
        warm_seconds = time.monotonic() - started
        monomer = model.fold(chain, options={**options, "random_seed": 2})
        print(f"AlphaFold2-Multimer cold={cold_seconds:.2f}s warm={warm_seconds:.2f}s")

    assert warm.ranking is not None and all("seed_001" in name for name in warm.ranking["order"])
    assert monomer.ranking is not None and all("seed_002" in name for name in monomer.ranking["order"])
    assert warm.pae is not None and warm.pae[0].shape == (2 * len(chain), 2 * len(chain))
    assert monomer.pae is not None and monomer.pae[0].shape == (len(chain), len(chain))
    assert np.isfinite(warm.pae[0]).all() and np.isfinite(monomer.pae[0]).all()

    expected_length = 2 * len(chain)
    assert result.metadata.sequence_lengths == [expected_length]

    assert result.atom_array is not None and len(result.atom_array) == 1
    assert len(result.atom_array[0]) > 0

    assert result.cif is not None and result.cif[0].startswith("data_")

    assert result.ranking is not None and "order" in result.ranking

    assert result.plddt is not None and result.plddt[0] is not None
    assert result.plddt[0].shape == (expected_length,)
    assert (result.plddt[0] >= 0).all() and (result.plddt[0] <= 1).all()

    assert result.ptm is not None and result.ptm[0] is not None and np.isfinite(result.ptm[0]).all()
    assert result.iptm is not None and result.iptm[0] is not None and np.isfinite(result.iptm[0]).all()

    assert result.pae is not None and result.pae[0] is not None
    assert result.pae[0].shape == (expected_length, expected_length)
