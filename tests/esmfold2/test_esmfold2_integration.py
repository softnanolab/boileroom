"""ESMFold2 integration tests against a real backend."""

import json
import os
import pathlib
from io import StringIO

import numpy as np
import pytest
from biotite.structure import AtomArray, filter_amino_acids, rmsd, superimpose
from biotite.structure.io.pdbx import CIFFile, get_structure

from boileroom import ESMFold2

pytestmark = [pytest.mark.integration, pytest.mark.slow, pytest.mark.gpu, pytest.mark.xdist_group("esmfold2")]

REFERENCE_DIR = pathlib.Path(__file__).parent.parent / "data" / "esmfold2"
#: Kit modes need the kit image and an A100/H100; they run only when the caller has pointed boileroom at one.
KIT_IMAGE_SOURCE_ENV = "BOILEROOM_KIT_IMAGE_SOURCE"
#: Modal GPU class for kit-vs-vanilla comparisons when ``--gpu`` is not given (the kit refuses the T4 default).
KIT_DEFAULT_DEVICE = "A100-80GB"


def test_esmfold2_modal_fold_basic(backend_option: str, device_option: str | None, output_ctx) -> None:
    """Fold a short protein with ESMFold2 and validate real structure outputs."""
    sequence = "MLKNVHVLVLGAGDVGSVVVRLLEK"
    config = {"model_name": "biohub/ESMFold2-Fast"}
    options = {
        "include_fields": ["plddt", "ptm", "cif"],
        "num_loops": 1,
        "num_sampling_steps": 5,
        "num_diffusion_samples": 1,
        "seed": 0,
    }

    with output_ctx(), ESMFold2(backend=backend_option, device=device_option, config=config) as model:
        result = model.fold(sequence, options=options)

    assert result.metadata.sequence_lengths == [len(sequence)]
    assert result.cif is not None
    assert len(result.cif) == 1
    assert result.cif[0].startswith("data_")
    assert result.atom_array is not None
    assert len(result.atom_array) == 1
    assert len(result.atom_array[0]) > 0
    assert result.plddt is not None
    assert result.plddt[0] is not None
    assert result.plddt[0].shape == (len(sequence),)
    assert result.ptm is not None
    assert result.ptm[0] is not None
    assert np.isfinite(result.ptm[0]).all()


def _load_manifest() -> dict:
    """Read the Biohub reference manifest written by ``scripts/testing/esmfold2_biohub_reference.py``."""
    return json.loads((REFERENCE_DIR / "manifest.json").read_text())


def _reference_ids() -> list[str]:
    """Reference ids, so a refreshed manifest re-parametrizes the test without edits."""
    return sorted(_load_manifest()["references"])


def _ca_atoms(atoms: AtomArray) -> AtomArray:
    """Protein C-alpha atoms in file order."""
    return atoms[(atoms.atom_name == "CA") & filter_amino_acids(atoms)]


def _cif_atoms(cif_text: str) -> AtomArray:
    """Parse one mmCIF string into an atom array."""
    return get_structure(CIFFile.read(StringIO(cif_text)), model=1)


def _chain_blocks(lengths: list[int]) -> list[tuple[slice, slice]]:
    """Residue-index slices of every off-diagonal (inter-chain) block of a PAE matrix."""
    bounds = np.cumsum([0, *lengths])
    slices = [slice(int(start), int(stop)) for start, stop in zip(bounds[:-1], bounds[1:], strict=True)]
    return [(rows, cols) for i, rows in enumerate(slices) for j, cols in enumerate(slices) if i != j]


def _reference_case(reference_id: str) -> tuple[dict, list[int], str, AtomArray, dict]:
    """Manifest entry, chain lengths, colon-joined sequence, reference atoms and confidence arrays of one reference."""
    manifest = _load_manifest()
    reference = manifest["references"][reference_id]
    lengths = [len(chain["sequence"]) for chain in reference["chains"]]
    sequence = ":".join(chain["sequence"] for chain in reference["chains"])
    atoms = _cif_atoms((REFERENCE_DIR / reference["cif"]).read_text())
    return reference, lengths, sequence, atoms, dict(np.load(REFERENCE_DIR / reference["confidence"]))


def _fold_options(manifest: dict) -> dict:
    """The fold options shared by every reference comparison: the reference's sampler and model settings, one seeded sample.

    ``lm_mask_pct`` is passed explicitly so the local fold and the Platform reference agree on it by construction
    (the Platform's own default differs between checkpoints). The Platform's ``lm_dropout`` has no local override:
    the checkpoints apply their configured per-loop dropout, which the reference generator mirrors.
    """
    return {
        **manifest["sampler"],
        "lm_mask_pct": manifest["platform_model_settings"]["lm_mask_pct"],
        "include_fields": ["plddt", "ptm", "iptm", "pae", "cif"],
        "num_diffusion_samples": 1,
        "seed": 0,
    }


def _ca_rmsd(reference: AtomArray, predicted: AtomArray, mask: np.ndarray | None = None) -> float:
    """C-alpha RMSD after superposition, optionally over a residue mask."""
    if mask is not None:
        reference, predicted = reference[mask], predicted[mask]
    fitted, _ = superimpose(reference, predicted)
    return float(rmsd(reference, fitted))


def _assert_matches_biohub_reference(result, lengths: list[int], expected_atoms: AtomArray, expected: dict) -> None:
    """Assert a boileroom fold reproduces the Biohub Platform prediction up to sampler noise.

    The Platform exposes no seed, so the tolerances cover sampler noise (on ubiquitin, two local seeds differ by
    up to 3.2 A of all-residue C-alpha RMSD, driven by the disordered C-terminal tail, and by 0.34 A of mean PAE),
    but they are far below the gap a wrong checkpoint, a broken confidence head or a mis-ordered PAE matrix would
    open. The structure is compared by C-alpha RMSD after superposition, over all residues loosely and over the
    residues both predictions call confident (pLDDT above 0.7) tightly; pLDDT, pTM, ipTM and the PAE matrix
    (including its inter-chain blocks) by absolute agreement.
    """
    total = sum(lengths)
    assert result.metadata.sequence_lengths == [total]
    assert result.cif is not None and result.plddt is not None and result.ptm is not None and result.pae is not None

    # Structure: same residues, same fold.
    predicted_ca = _ca_atoms(_cif_atoms(result.cif[0]))
    expected_ca = _ca_atoms(expected_atoms)
    assert len(predicted_ca) == len(expected_ca) == total
    assert np.array_equal(predicted_ca.res_name, expected_ca.res_name)
    # Loose: on 1UBQ vanilla's own seeds differ by up to 3.2 A here (SEED_NOISE), almost all of it in the tail.
    ca_rmsd = _ca_rmsd(expected_ca, predicted_ca)
    assert ca_rmsd < 3.5, f"C-alpha RMSD to the Biohub reference is {ca_rmsd:.2f} A"

    # Per-residue confidence: unit scale, same residues.
    plddt = result.plddt[0]
    assert plddt is not None and plddt.shape == expected["plddt"].shape
    assert float(plddt.min()) >= 0.0 and float(plddt.max()) <= 1.0
    confident = (plddt > 0.7) & (expected["plddt"] > 0.7)
    assert confident.sum() >= 0.5 * total, "fewer than half the residues are confident in both predictions"
    confident_rmsd = _ca_rmsd(expected_ca, predicted_ca, confident)
    assert confident_rmsd < 0.75, f"C-alpha RMSD over confident residues is {confident_rmsd:.2f} A"
    plddt_gap = float(np.abs(plddt - expected["plddt"]).mean())
    assert plddt_gap < 0.03, f"mean |pLDDT - reference| is {plddt_gap:.3f}"

    ptm = result.ptm[0]
    assert ptm is not None and np.isfinite(ptm).all()
    assert abs(float(ptm[0]) - float(expected["ptm"])) < 0.05

    # PAE: residue x residue in Angstrom, same layout, same inter-chain error.
    pae = result.pae[0]
    assert pae is not None and pae.shape == expected["pae"].shape == (total, total)
    assert np.isfinite(pae).all() and float(pae.min()) >= 0.0
    pae_gap = float(np.abs(pae - expected["pae"]).mean())
    assert pae_gap < 0.5, f"mean |PAE - reference| is {pae_gap:.2f} A"
    assert float(np.corrcoef(pae.ravel(), expected["pae"].ravel())[0, 1]) > 0.95

    if len(lengths) > 1:
        assert result.iptm is not None and result.iptm[0] is not None
        assert abs(float(result.iptm[0][0]) - float(expected["iptm"])) < 0.05
        for rows, cols in _chain_blocks(lengths):
            block_gap = abs(float(pae[rows, cols].mean()) - float(expected["pae"][rows, cols].mean()))
            assert block_gap < 0.5, f"inter-chain PAE block {rows}x{cols} differs by {block_gap:.2f} A"
    elif result.iptm is not None and result.iptm[0] is not None:
        # A single chain has no interface: Biohub reports no ipTM, the local model reports 0.
        assert np.isnan(expected["iptm"]) and float(np.nan_to_num(result.iptm[0][0])) == 0.0


@pytest.mark.parametrize("reference_id", _reference_ids())
def test_esmfold2_matches_biohub_reference(
    reference_id: str, backend_option: str, device_option: str | None, output_ctx
) -> None:
    """A vanilla boileroom fold of a PDB entry's sequence must reproduce Biohub's own ESMFold2 prediction.

    The references under ``tests/data/esmfold2`` were produced by the Biohub Platform (the hosted reference
    implementation) for the sequences of PDB entries, with the same checkpoint and sampler settings; see
    ``scripts/testing/esmfold2_biohub_reference.py``.
    """
    reference, lengths, sequence, expected_atoms, expected = _reference_case(reference_id)
    config = {"model_name": reference["model_name"]}

    with output_ctx(), ESMFold2(backend=backend_option, device=device_option, config=config) as model:
        result = model.fold(sequence, options=_fold_options(_load_manifest()))

    assert result.metadata.optimization is not None and result.metadata.optimization["active"] == "vanilla"
    _assert_matches_biohub_reference(result, lengths, expected_atoms, expected)


#: Vanilla seed-to-seed noise for 1UBQ on an A100-80GB (seeds 0-3, worst pair of the six), per checkpoint, measured with
#: the reference sampler settings on 2026-10-06. The kit modes must land within KIT_NOISE_FACTOR times this of the
#: Biohub reference: a kit kernel may perturb the sampler trajectory, but not beyond what a different seed does.
SEED_NOISE: dict[str, dict[str, float]] = {
    "biohub/ESMFold2": {"all_rmsd": 3.167, "conf_rmsd": 0.319, "pae_mae": 0.341, "pae_max": 10.243},
    "biohub/ESMFold2-Fast": {"all_rmsd": 2.892, "conf_rmsd": 0.389, "pae_mae": 0.340, "pae_max": 12.337},
}
KIT_NOISE_FACTOR = 1.1
KIT_REFERENCE_ENTRY = "1UBQ"


def _kit_reference_ids() -> list[str]:
    """The 1UBQ references, one per checkpoint."""
    return [ref for ref in _reference_ids() if ref.startswith(f"{KIT_REFERENCE_ENTRY}.")]


@pytest.mark.parametrize("reference_id", _kit_reference_ids())
@pytest.mark.parametrize("optimization", ["exact", "fast"])
def test_esmfold2_kit_mode_matches_biohub_reference_within_seed_noise(
    reference_id: str, optimization: str, backend_option: str, device_option: str | None, output_ctx
) -> None:
    """A kit mode's 1UBQ fold must sit within vanilla's seed-change noise of the Biohub reference.

    The kit's fused bf16 kernels do not reproduce vanilla bit for bit (same seed, same GPU: about 1.3 A of
    all-residue C-alpha RMSD on ubiquitin, mostly its disordered tail), so an elementwise comparison is not
    meaningful. Instead each of four metrics against the Biohub reference (all-residue and confident-residue
    C-alpha RMSD, mean and max PAE entry gap) is bounded by ``KIT_NOISE_FACTOR`` times the worst vanilla
    seed-to-seed value in ``SEED_NOISE``, measured on the GPU class the kit runs on. The general Biohub checks
    of ``test_esmfold2_matches_biohub_reference`` apply as well.
    """
    if not os.environ.get(KIT_IMAGE_SOURCE_ENV):
        pytest.skip(
            f"kit mode {optimization!r} needs a kit image: set {KIT_IMAGE_SOURCE_ENV} (see docs/optimization.md)"
        )
    reference, lengths, sequence, expected_atoms, expected = _reference_case(reference_id)
    device = device_option if device_option is not None else KIT_DEFAULT_DEVICE
    config = {"model_name": reference["model_name"], "optimization": optimization}

    with output_ctx(), ESMFold2(backend=backend_option, device=device, config=config) as model:
        result = model.fold(sequence, options=_fold_options(_load_manifest()))

    assert result.metadata.optimization is not None and result.metadata.optimization["active"] == optimization
    _assert_matches_biohub_reference(result, lengths, expected_atoms, expected)

    assert result.cif is not None and result.plddt is not None and result.pae is not None
    predicted_ca, expected_ca = _ca_atoms(_cif_atoms(result.cif[0])), _ca_atoms(expected_atoms)
    plddt, pae = result.plddt[0], result.pae[0]
    assert plddt is not None and pae is not None
    confident = (plddt > 0.7) & (expected["plddt"] > 0.7)
    measured = {
        "all_rmsd": _ca_rmsd(expected_ca, predicted_ca),
        "conf_rmsd": _ca_rmsd(expected_ca, predicted_ca, confident),
        "pae_mae": float(np.abs(pae - expected["pae"]).mean()),
        "pae_max": float(np.abs(pae - expected["pae"]).max()),
    }
    bounds = {metric: KIT_NOISE_FACTOR * noise for metric, noise in SEED_NOISE[reference["model_name"]].items()}
    violations = {metric: (measured[metric], bounds[metric]) for metric in bounds if measured[metric] > bounds[metric]}
    assert not violations, (
        f"kit {optimization!r} on {reference_id} exceeds vanilla seed noise vs the Biohub reference "
        f"(metric: measured > bound): {violations}; all measured: {measured}"
    )
