"""Real public-wrapper/backend checks; requires an explicitly bounded caller.

The caller supplies the actual MiniFold wrapper (or a backend-selecting factory)
and owns the source image, resources, app/SIF and cleanup evidence. No deployment,
credential setup, image publication or app creation is performed by this helper.
"""

from __future__ import annotations

import base64
import io
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any


def run_public_modal_validation(wrapper_class, model_size: str = "12L", reference_report: dict | None = None) -> dict:
    """Exercise the real supplied public wrapper; report parity only with matching references."""
    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from _minifold_cases import (
        CASES,
        _assert_serialized_structure,
        _assert_stock_parity,
        _validate_reference_manifest,
        experimental_ca,
        quality_metrics,
    )
    from biotite.sequence import ProteinSequence

    started = time.monotonic()
    report: dict[str, Any] = {
        "passed": False,
        "model_size": model_size,
        "phases": {},
        "upstream_parity": "unrun; independent source-image/core gate reported separately",
        "coverage_limits": [
            "Only the supplied backend/device and selected checkpoint run.",
            "This gate does not prove compiled mode or other hardware.",
            "Experimental metrics are descriptive, not held-out accuracy guarantees.",
            "Public-wrapper shutdown is checked locally; caller must separately verify remote app/process termination.",
        ],
    }
    model = None
    phase = "public_backend_start"
    try:
        assert model_size in ("12L", "48L")
        references = {}
        if reference_report is not None:
            import hashlib

            manifest = reference_report["reference_manifest"]
            _validate_reference_manifest(manifest, model_size)
            assert set(reference_report["reference_artifacts_base64"]) == set(manifest["files"])
            for name, encoded in reference_report["reference_artifacts_base64"].items():
                content = base64.b64decode(encoded)
                assert hashlib.sha256(content).hexdigest() == manifest["files"][name]
                with np.load(io.BytesIO(content), allow_pickle=False) as arrays:
                    references[name] = {key: arrays[key].copy() for key in arrays.files}
        model = wrapper_class(config={"model_size": model_size, "max_sequence_length": 128})
        backend = model._backend
        remote_handle = backend.get_model()
        process = getattr(backend, "_process", None)
        client = getattr(backend, "_client", None)
        if hasattr(backend, "_process"):
            assert process is not None and process.poll() is None and client is not None
        assert backend._is_running
        report["backend_type"] = type(backend).__name__
        report["phases"][phase] = {"passed": True}
        with model:
            phase = "single_outputs_and_experimental_quality"
            saved = []
            quality = []
            for case in CASES:
                output = model.fold(case["sequence"], {"include_fields": ["*"]})
                atoms = output.atom_array[0]
                assert np.isfinite(atoms.coord).all()
                assert isinstance(output.plddt[0], np.ndarray)
                assert np.isfinite(output.plddt[0]).all() and ((output.plddt[0] >= 0) & (output.plddt[0] <= 1)).all()
                np.testing.assert_allclose(atoms.b_factor, output.plddt[0][atoms.res_id - 1] * 100, atol=1e-5)
                assert output.pae is None and output.pdb and output.cif
                np.testing.assert_array_equal(atoms.chain_id, "A")
                for index, residue in zip(atoms.res_id, atoms.res_name, strict=True):
                    assert ProteinSequence.convert_letter_3to1(residue) == case["sequence"][index - 1]
                _assert_serialized_structure(output, 0)
                assert output.metadata.sequence_lengths == [len(case["sequence"])]
                if references:
                    _assert_stock_parity(output, 0, case, references[f"{case['id']}-r3.npz"], manifest)
                ca = atoms[atoms.atom_name == "CA"]
                np.testing.assert_array_equal(ca.res_id, np.arange(1, len(case["sequence"]) + 1))
                ids, observed = experimental_ca(case)
                quality.append({"entry": case["id"], **quality_metrics(ca.coord[ids - 1], observed)})
                saved.append((output, atoms.coord.copy(), json.dumps(output.metadata.runtime, sort_keys=True)))
            report["experimental_quality"] = quality
            report["provenance"] = dict(saved[0][0].metadata.runtime)
            report["phases"][phase] = {"passed": True}
            phase = "reversed_mixed_length_and_repeat"
            for order in (list(range(len(CASES))), list(reversed(range(len(CASES))))):
                output = model.fold([CASES[index]["sequence"] for index in order], {"include_fields": ["*"]})
                assert output.metadata.sequence_lengths == [len(CASES[index]["sequence"]) for index in order]
                for index, original in enumerate(order):
                    np.testing.assert_allclose(output.atom_array[index].coord, saved[original][1], atol=1e-4, rtol=1e-4)
                    np.testing.assert_allclose(output.plddt[index], saved[original][0].plddt[0], atol=1e-5, rtol=1e-4)
                    length = len(CASES[original]["sequence"])
                    np.testing.assert_array_equal(output.residue_index[index, :length], np.arange(length))
                    assert (output.residue_index[index, length:] == -1).all()
                if hasattr(backend, "_process"):
                    assert backend._process is process and process.poll() is None
                    assert backend._client is client and backend._is_running
                else:
                    assert backend.get_model() is remote_handle
            report["phases"][phase] = {"passed": True}
            phase = "recycling_filtering_and_refusal_recovery"
            changed = model.fold(CASES[0]["sequence"], {"num_recycling": 0, "include_fields": ["*"]})
            assert changed.metadata.runtime["minifold.num_recycling"] == "0"
            if references:
                _assert_stock_parity(changed, 0, CASES[0], references[f"{CASES[0]['id']}-r0.npz"], manifest)
                report["upstream_parity"] = "passed; exact matching image/GPU/precision/raw reference"
            minimal = model.fold(CASES[0]["sequence"])
            assert minimal.plddt is minimal.pdb is minimal.cif is minimal.residue_index is None
            assert minimal.metadata.runtime["minifold.num_recycling"] == "3"
            for sequence in ("AC:DE", "A" * 129, [CASES[0]["sequence"], "AC?"]):
                try:
                    model.fold(sequence)
                except (ValueError, RuntimeError) as error:
                    assert any(
                        word in str(error) for word in ("single-chain", "max_sequence_length", "amino", "Invalid")
                    ), str(error)
                else:
                    raise AssertionError("Invalid input was accepted")
            selected = model.fold(CASES[0]["sequence"], {"include_fields": ["plddt"]})
            assert selected.plddt is not None and selected.pdb is selected.cif is None
            np.testing.assert_allclose(selected.plddt[0], saved[0][0].plddt[0], atol=1e-5, rtol=1e-4)
            for output, coordinates, metadata in saved:
                np.testing.assert_array_equal(output.atom_array[0].coord, coordinates)
                assert json.dumps(output.metadata.runtime, sort_keys=True) == metadata
            # Mutate a returned CPU object positively: later calls must be independent.
            saved[0][0].atom_array[0].coord[:] = -999.0
            saved[0][0].plddt[0][:] = 0.0
            independent = model.fold(CASES[0]["sequence"], {"include_fields": ["plddt"]})
            np.testing.assert_allclose(independent.atom_array[0].coord, saved[0][1], atol=1e-4, rtol=1e-4)
            np.testing.assert_allclose(independent.plddt[0], selected.plddt[0], atol=1e-5, rtol=1e-4)
            assert (saved[0][0].atom_array[0].coord == -999.0).all()
            assert (saved[0][0].plddt[0] == 0.0).all()
            report["phases"][phase] = {"passed": True}
        phase = "public_backend_shutdown"
        assert not backend._is_running and backend._atexit_hook is None
        if hasattr(backend, "_context_token"):
            assert backend._context_token is None and backend.model is None
        if hasattr(backend, "_process"):
            assert backend._process is None and backend._client is None
            assert process is not None and client is not None
            assert process.poll() is not None and client.is_closed
        report["phases"][phase] = {"passed": True}
        report["passed"] = True
    except Exception as error:
        report["phases"][phase] = {
            "passed": False,
            "error_type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
    finally:
        if model is not None:
            model._backend.stop()
        report["seconds"] = time.monotonic() - started
    return report


def run_public_validation(wrapper_class, backend="apptainer", device="cpu", model_size="12L", reference_report=None):
    """Use the same public API checks through an explicitly selected Apptainer backend."""
    from functools import partial

    return run_public_modal_validation(
        partial(wrapper_class, backend=backend, device=device), model_size, reference_report
    )
