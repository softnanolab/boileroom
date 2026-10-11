"""Real-runtime MiniFold gates using external experiments and independent stock references.

Generate reference artifacts first with scripts/testing/minifold_stock_reference.py and
set BOILEROOM_MINIFOLD_REFERENCE_DIR. Missing references fail: an unrun parity gate is not a pass.
All tests are explicit integration/GPU tests and excluded from offline CI.
"""

import json

import numpy as np
import pytest
from _minifold_cases import CASES, _assert_stock_parity, _reference, experimental_ca, quality_metrics

from boileroom import MiniFold

pytestmark = [pytest.mark.integration, pytest.mark.gpu, pytest.mark.slow, pytest.mark.xdist_group("minifold")]


@pytest.mark.parametrize("model_size", ["12L", "48L"])
def test_minifold_external_monomers_stock_parity_and_lifecycle(
    backend_option, device_option, output_ctx, model_size, tmp_path
):
    """Verify real preprocessing/loading/output fidelity, public transport and resident repeat behavior."""
    references = {case["id"]: _reference(model_size, case, 3) for case in CASES}
    recycling_reference = _reference(model_size, CASES[0], 0)
    sequences = [case["sequence"] for case in CASES]
    quality = []
    with (
        output_ctx(),
        MiniFold(
            backend=backend_option, device=device_option, config={"model_size": model_size, "max_sequence_length": 128}
        ) as model,
    ):
        singles = []
        for case in CASES:
            result = model.fold(case["sequence"], {"include_fields": ["*"]})
            manifest, reference = references[case["id"]]
            _assert_stock_parity(result, 0, case, reference, manifest)
            singles.append(result)
            ids, observed = experimental_ca(case)
            ca = result.atom_array[0][result.atom_array[0].atom_name == "CA"]
            assert np.array_equal(ca.res_id, np.arange(1, len(case["sequence"]) + 1))
            quality.append(
                {"entry": case["id"], "model_size": model_size, **quality_metrics(ca.coord[ids - 1], observed)}
            )
        snapshot = singles[0].atom_array[0].coord.copy()
        metadata_snapshot = json.dumps(singles[0].metadata.runtime, sort_keys=True)
        for order in (list(range(len(CASES))), list(reversed(range(len(CASES))))):
            batch = model.fold([sequences[i] for i in order], {"include_fields": ["*"]})
            assert batch.metadata.sequence_lengths == [len(sequences[i]) for i in order]
            for index, i in enumerate(order):
                manifest, reference = references[CASES[i]["id"]]
                _assert_stock_parity(batch, index, CASES[i], reference, manifest)
                np.testing.assert_array_equal(
                    batch.residue_index[index, : len(sequences[i])], np.arange(len(sequences[i]))
                )
                assert (batch.residue_index[index, len(sequences[i]) :] == -1).all()
            assert batch.pae is None
        changed = model.fold(sequences[0], {"num_recycling": 0, "include_fields": ["*"]})
        _assert_stock_parity(changed, 0, CASES[0], recycling_reference[1], recycling_reference[0])
        minimal = model.fold(sequences[0])
        selected = model.fold(sequences[0], {"include_fields": ["plddt"]})
        assert minimal.plddt is minimal.pdb is minimal.cif is minimal.residue_index is None
        assert selected.plddt is not None and selected.pdb is selected.cif is None
        assert minimal.metadata.runtime["minifold.num_recycling"] == "3"
        assert changed.metadata.runtime["minifold.num_recycling"] == "0"
        for invalid in ("AC:DE", "A" * 129, [sequences[0], "AC?"]):
            with pytest.raises((ValueError, RuntimeError), match="single-chain|max_sequence_length|amino|Invalid"):
                model.fold(invalid)
        # Refusals must not poison the resident worker or mutate earlier CPU outputs.
        recovered = model.fold(sequences[0], {"include_fields": ["plddt"]})
        np.testing.assert_allclose(recovered.plddt[0], singles[0].plddt[0], atol=1e-5, rtol=1e-4)
        np.testing.assert_array_equal(singles[0].atom_array[0].coord, snapshot)
        assert json.dumps(singles[0].metadata.runtime, sort_keys=True) == metadata_snapshot
    report = tmp_path / f"{model_size}-experimental-quality.json"
    report.write_text(json.dumps(quality, indent=2) + "\n")
    print("Experimental quality (descriptive only, no held-out accuracy claim):", json.dumps(quality))
