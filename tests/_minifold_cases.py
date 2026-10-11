"""Source-grounded experimental cases and descriptive structural metrics (no GPU)."""

import gzip
import hashlib
import json
import os
from io import StringIO
from pathlib import Path

import numpy as np
from biotite.sequence import ProteinSequence
from biotite.structure.io import pdb, pdbx

from boileroom.models.minifold.loading import CHECKPOINT_REVISION, CHECKPOINT_SHA256, ESM_ARTIFACTS, UPSTREAM_COMMIT

DATA_DIR = Path(__file__).resolve().parent / "data" / "minifold"
MANIFEST = json.loads((DATA_DIR / "manifest.json").read_text())
CASES = MANIFEST["entries"]


def experimental_ca(case: dict) -> tuple[np.ndarray, np.ndarray]:
    """Map observed CA atoms explicitly through CIF label_seq_id to canonical sequence."""
    path = DATA_DIR / f"{case['id']}.cif.gz"
    content = gzip.decompress(path.read_bytes())
    assert hashlib.sha256(content).hexdigest() == case["sha256"]
    file = pdbx.CIFFile.read(StringIO(content.decode()))
    sequence = str(file.block["entity_poly"]["pdbx_seq_one_letter_code_can"].as_array()[0])
    assert "".join(sequence.split()) == case["sequence"]
    atoms = pdbx.get_structure(file, model=case["model"], use_author_fields=False)
    ca = atoms[(atoms.chain_id == case["label_asym_id"]) & (atoms.atom_name == "CA") & ~atoms.hetero]
    ids = ca.res_id.astype(int)
    assert ids.tolist() == case["resolved_label_seq_ids"]
    assert len(set(ids)) == len(ids) and np.all((ids >= 1) & (ids <= len(case["sequence"])))
    for index, residue in zip(ids, ca.res_name, strict=True):
        assert ProteinSequence.convert_letter_3to1(residue) == case["sequence"][index - 1]
    return ids, ca.coord.astype(float)


def quality_metrics(predicted: np.ndarray, observed: np.ndarray) -> dict[str, float]:
    """Return aligned CA RMSD and local CA distance fidelity; these are descriptive, not accuracy gates."""
    assert predicted.shape == observed.shape and predicted.ndim == 2 and predicted.shape[1] == 3
    assert np.isfinite(predicted).all() and np.isfinite(observed).all()
    a, b = predicted - predicted.mean(0), observed - observed.mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    correction = np.diag([1.0, 1.0, np.linalg.det(u @ vt)])
    aligned = a @ u @ correction @ vt
    pdist = np.linalg.norm(predicted[:, None] - predicted[None, :], axis=-1)
    odist = np.linalg.norm(observed[:, None] - observed[None, :], axis=-1)
    pairs = np.triu((odist < 15.0) & (odist > 0), k=1)
    error = np.abs(pdist[pairs] - odist[pairs])
    assert error.size
    return {
        "ca_rmsd_angstrom": float(np.sqrt(np.mean(np.sum((aligned - b) ** 2, axis=-1)))),
        "ca_local_distance_agreement": float(np.mean([np.mean(error < bound) for bound in (0.5, 1.0, 2.0, 4.0)])),
        "ca_distance_mae_angstrom": float(error.mean()),
    }


def _reference(model_size, case, recycling):
    location = os.environ.get("BOILEROOM_MINIFOLD_REFERENCE_DIR")
    assert location, "Generate independent stock references and set BOILEROOM_MINIFOLD_REFERENCE_DIR first"
    root = Path(location) / model_size
    manifest = json.loads((root / "manifest.json").read_text())
    _validate_reference_manifest(manifest, model_size)
    path = root / f"{case['id']}-r{recycling}.npz"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["files"][path.name]
    with np.load(path, allow_pickle=False) as file:
        return manifest, {key: file[key].copy() for key in file.files}


def _assert_stock_parity(result, index, case, reference, manifest):
    atoms = result.atom_array[index]
    positions, mask = reference["final_atom_positions"], reference["final_atom_mask"]
    residues, names = np.where(mask.astype(bool))
    np.testing.assert_array_equal(atoms.res_id, residues + 1)
    np.testing.assert_array_equal(atoms.atom_name, reference["atom_types"][names])
    np.testing.assert_array_equal(atoms.chain_id, "A")
    for residue, name in zip(atoms.res_id, atoms.res_name, strict=True):
        assert ProteinSequence.convert_letter_3to1(name) == case["sequence"][residue - 1]
    np.testing.assert_allclose(atoms.coord, positions[residues, names], atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(atoms.b_factor, reference["plddt"][residues], atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(result.plddt[index], reference["plddt"] / 100, atol=1e-5, rtol=1e-4)
    runtime = result.metadata.runtime
    assert runtime["minifold.commit"] == manifest["upstream_commit"]
    assert runtime["minifold.checkpoint_revision"] == manifest["checkpoint_revision"]
    assert runtime["minifold.precision"] == manifest["precision"]
    assert runtime["minifold.matmul_precision"] == manifest["matmul_precision"]
    assert runtime["minifold.allow_tf32"] == str(manifest["allow_tf32"])
    assert runtime["image_ref"] == manifest["runtime"]["image_ref"]
    assert runtime["torch"] == manifest["runtime"]["torch"]
    assert runtime["cuda"] == manifest["runtime"]["cuda"]
    assert runtime["gpu"] == manifest["gpu"]
    _assert_serialized_structure(result, index)


def _assert_serialized_structure(result, index):
    """Compare CPU structures and confidence after real PDB/mmCIF round trips."""
    atoms = result.atom_array[index]
    for rendered, kind in ((result.pdb[index], "pdb"), (result.cif[index], "cif")):
        parsed = (
            pdb.get_structure(pdb.PDBFile.read(StringIO(rendered)), model=1, extra_fields=["b_factor"])
            if kind == "pdb"
            else pdbx.get_structure(pdbx.CIFFile.read(StringIO(rendered)), model=1, extra_fields=["b_factor"])
        )
        np.testing.assert_array_equal(parsed.res_id, atoms.res_id)
        np.testing.assert_array_equal(parsed.atom_name, atoms.atom_name)
        np.testing.assert_array_equal(parsed.res_name, atoms.res_name)
        np.testing.assert_array_equal(parsed.chain_id, atoms.chain_id)
        np.testing.assert_allclose(parsed.coord, atoms.coord, atol=0.00051, rtol=0)
        np.testing.assert_allclose(parsed.b_factor, atoms.b_factor, atol=0.0051, rtol=0)


def _validate_reference_manifest(manifest, model_size):
    """Refuse missing, stale, optimized or untraceable references before parity is claimed."""
    from _minifold_reference import MINIFORMER_SHA256, PREDICT_SHA256

    assert model_size in ("12L", "48L") and manifest["model_size"] == model_size
    assert manifest["upstream_commit"] == UPSTREAM_COMMIT
    assert manifest["predict_sha256"] == PREDICT_SHA256
    assert manifest["miniformer_sha256"] == MINIFORMER_SHA256
    assert manifest["triangular_implementation"] == "original-pinned-upstream"
    assert manifest["checkpoint_revision"] == CHECKPOINT_REVISION
    assert manifest["checkpoint_sha256"] == CHECKPOINT_SHA256[model_size]
    assert manifest["esm_artifacts"] == ESM_ARTIFACTS
    assert manifest["runtime"]["image_ref"] and manifest["runtime"]["image_ref"] not in ("unknown", "none", "absent")
    assert manifest["compile"] is False and manifest["kernels"] is False
    assert manifest["batch_mode"] == "one-unpadded-sequence"
    assert manifest["matmul_precision"] == "highest"
    assert manifest["gpu"] and manifest["gpu"] not in ("unknown", "none", "absent")
    assert manifest["cases"] == [
        {"id": case["id"], "sequence": case["sequence"], "experimental_sha256": case["sha256"]} for case in CASES
    ]
    expected_files = {f"{case['id']}-r3.npz" for case in CASES} | {f"{CASES[0]['id']}-r0.npz"}
    assert expected_files <= set(manifest["files"]), "Missing expected raw reference artifacts"
