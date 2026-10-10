"""RF3 integration tests against a real backend."""

import json
import os
import pathlib
from collections.abc import Callable, Generator
from contextlib import AbstractContextManager
from io import StringIO
from typing import Any

import numpy as np
import pytest
from biotite.structure import AtomArray, filter_amino_acids, rmsd, superimpose
from biotite.structure.io.pdbx import CIFFile, get_structure

from boileroom import RF3

pytestmark = [pytest.mark.integration, pytest.mark.slow, pytest.mark.gpu, pytest.mark.xdist_group("rf3")]

REFERENCE_DIR = pathlib.Path(__file__).parent.parent / "data" / "rf3"
#: Kit modes need the kit image and an A100/H100; they run only when the caller has pointed boileroom at one.
KIT_IMAGE_SOURCE_ENV = "BOILEROOM_KIT_IMAGE_SOURCE"
#: Modal GPU class for kit runs when ``--gpu`` is not given (the kit refuses the wrapper's L4/T4-class defaults).
KIT_DEFAULT_DEVICE = "A100-80GB"
#: T4 lysozyme, a 163-residue protein used to build inputs of a chosen size by repeating it as chains.
LARGE_CHAIN = "MNIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLLTKSPSLNAAKSELDKAIGRNTNGVITKDEAEKLFNQDVDAAVRGILRNAKLKPVYDSLDAVRRAAINMVFQMGETGVAGFTNSLRMLQQKRWDEAAVNLAKSRWYNQTPNRAKRVITTFRTGTWDAYKNL"
#: Sixteen chains (2,608 residues): more than ``fast`` fits on an A100-80GB, and within what ``big`` completes.
LARGE_CHAIN_COUNT = 16
#: Every field the comparisons below read.
ALL_FIELDS = ["plddt", "ptm", "iptm", "pae", "cif", "confidence", "token_chain_ids", "token_res_ids", "atom_plddt"]


def _manifest() -> dict:
    """Read the stock-RF3 reference manifest written by ``scripts/testing/rf3_stock_reference.py``."""
    return json.loads((REFERENCE_DIR / "manifest.json").read_text())


def _reference_ids() -> list[str]:
    """Reference ids, so a refreshed manifest re-parametrizes the tests without edits."""
    return sorted(_manifest()["references"])


def _stock_options() -> dict[str, Any]:
    """Per-call options matching the settings the references were generated with (``seed`` and the pLDDT stop)."""
    settings = _manifest()["settings"]
    return {
        "seed": settings["seed"],
        "early_stopping_plddt_threshold": settings["early_stopping_plddt_threshold"],
        "include_fields": ALL_FIELDS,
    }


def _stock_config() -> dict[str, Any]:
    """Initialization config matching the sampler settings the references were generated with."""
    settings = _manifest()["settings"]
    return {key: settings[key] for key in ("n_recycles", "num_steps", "diffusion_batch_size")}


def _assert_ran_on(result: Any, device: str) -> None:
    """Check the card the kit reported is the one ``device`` asked for, so a claim about a fold names the right GPU.

    Modal can serve a request from a container started for another card when several wrappers with different devices
    live in one process, so the requested ``device`` alone does not say where a fold ran. An H100 request may be served
    by an H200, as Modal documents. An A100 request names its memory size (``A100-80GB``), which the card name carries
    too (``NVIDIA A100-SXM4-80GB``, ``NVIDIA A100 80GB PCIe``).
    """
    card = str((result.metadata.optimization or {}).get("gpu_name")).upper().replace(" ", "")
    model, _, memory = device.upper().partition("-")
    if model.startswith("H"):
        assert any(name in card for name in ("H100", "H200")), f"asked for {device}, ran on {card}"
    else:
        assert model in card and memory in card, f"asked for {device}, ran on {card}"


def _kit_device(mode: str, device_option: str | None) -> str:
    """The Modal GPU class a kit test runs on, skipping the test when no kit image has been selected."""
    if not os.environ.get(KIT_IMAGE_SOURCE_ENV):
        pytest.skip(f"kit mode {mode!r} needs a kit image: set {KIT_IMAGE_SOURCE_ENV} (see docs/optimization.md)")
    return device_option if device_option is not None else KIT_DEFAULT_DEVICE


def _ca_atoms(atoms: AtomArray) -> AtomArray:
    """Protein C-alpha atoms in file order."""
    return atoms[(atoms.atom_name == "CA") & filter_amino_acids(atoms)]


def _ca_rmsd(reference: AtomArray, predicted: AtomArray) -> float:
    """C-alpha RMSD after superposition."""
    fitted, _ = superimpose(reference, predicted)
    return float(rmsd(reference, fitted))


def _reference(reference_id: str) -> tuple[str, list[int], AtomArray, dict, dict]:
    """Colon-joined sequence, chain lengths, atoms, full confidences and summary of one stock-RF3 reference."""
    chains = _manifest()["references"][reference_id]["chains"]
    sequence = ":".join(chain["sequence"] for chain in chains)
    atoms = get_structure(
        CIFFile.read(StringIO((REFERENCE_DIR / f"{reference_id}_model.cif").read_text())),
        model=1,
        use_author_fields=False,
    )
    confidences = json.loads((REFERENCE_DIR / f"{reference_id}_confidences.json").read_text())
    summary = json.loads((REFERENCE_DIR / f"{reference_id}_summary_confidences.json").read_text())
    return sequence, [len(chain["sequence"]) for chain in chains], atoms, confidences, summary


def _assert_matches_stock_reference(
    result: Any, lengths: list[int], expected_atoms: AtomArray, confidences: dict, summary: dict, *, max_ca_rmsd: float
) -> None:
    """Assert a boileroom fold reproduces stock ``rf3 fold`` output for the same sequence, settings and seed.

    The reference was made by the same code and checkpoint. Measured on A100-40GB and A100-80GB, the vanilla seed-0
    fold lands within 0.35 A of C-alpha RMSD, 6e-4 of every score and 0.02 A of mean PAE of it, with the same
    differences on both cards. Other seeds are further away (up to 0.73 A on ubiquitin; on the GCN4 dimer seven of
    eight seeds are within 0.5 A and one packs the helices differently, 2.7 A), so the test fixes the seed. The bounds are several times the seed-0 differences, yet far below the gap a wrong checkpoint, a
    mis-ordered PAE matrix or a broken confidence head would open.
    """
    total = sum(lengths)
    assert result.metadata.sequence_lengths == [total]
    assert result.cif is not None and result.plddt is not None and result.pae is not None
    assert result.ptm is not None and result.iptm is not None and result.confidence is not None

    # Structure: same atoms, same fold.
    predicted_atoms = result.atom_array[0]
    assert len(predicted_atoms) == len(expected_atoms)
    assert np.array_equal(predicted_atoms.atom_name, expected_atoms.atom_name)
    assert np.array_equal(predicted_atoms.res_name, expected_atoms.res_name)
    predicted_ca, expected_ca = _ca_atoms(predicted_atoms), _ca_atoms(expected_atoms)
    assert len(predicted_ca) == len(expected_ca) == total
    ca_rmsd = _ca_rmsd(expected_ca, predicted_ca)
    assert ca_rmsd < max_ca_rmsd, f"C-alpha RMSD to the stock RF3 reference is {ca_rmsd:.2f} A"

    # Scores: RF3's own summary, lifted into the output fields.
    scores = result.confidence[0]
    for key in ("ptm", "iptm", "ranking_score", "overall_plddt"):
        assert abs(scores[key] - summary[key]) < 0.01, f"{key}: {scores[key]} vs reference {summary[key]}"
    assert result.ptm[0] is not None and abs(float(result.ptm[0][0]) - summary["ptm"]) < 0.01
    assert result.iptm[0] is not None and abs(float(result.iptm[0][0]) - summary["iptm"]) < 0.01
    assert scores["has_clash"] is False

    # PAE: residue x residue in Angstrom, with the same layout and inter-chain error.
    pae = result.pae[0]
    expected_pae = np.asarray(confidences["pae"], dtype=np.float32)
    assert pae.shape == expected_pae.shape == (total, total)
    pae_gap = float(np.abs(pae - expected_pae).mean())
    assert pae_gap < 0.1, f"mean |PAE - reference| is {pae_gap:.3f} A"
    bounds = np.cumsum([0, *lengths])
    for i in range(len(lengths)):
        for j in range(len(lengths)):
            if i != j:
                rows, cols = slice(int(bounds[i]), int(bounds[i + 1])), slice(int(bounds[j]), int(bounds[j + 1]))
                block_gap = abs(float(pae[rows, cols].mean()) - float(expected_pae[rows, cols].mean()))
                assert block_gap < 0.2, f"inter-chain PAE block {i}->{j} differs by {block_gap:.2f} A"

    # pLDDT: per residue (the mean of the atom values) and per atom, both on the unit scale.
    atom_plddt = result.atom_plddt[0]
    expected_atom_plddt = np.asarray(confidences["atom_plddts"], dtype=np.float32)
    assert atom_plddt.shape == expected_atom_plddt.shape == (len(expected_atoms),)
    assert float(np.abs(atom_plddt - expected_atom_plddt).mean()) < 0.01
    plddt = result.plddt[0]
    assert plddt.shape == (total,) and float(plddt.min()) >= 0.0 and float(plddt.max()) <= 1.0
    assert float(plddt.mean()) == pytest.approx(summary["overall_plddt"], abs=0.02)

    # Token layout: the chain letters and chain-local residue numbers of the CIF.
    expected_chains = np.concatenate([[chr(65 + i)] * length for i, length in enumerate(lengths)])
    assert np.array_equal(result.token_chain_ids[0], expected_chains)
    assert np.array_equal(result.token_res_ids[0], np.concatenate([np.arange(1, length + 1) for length in lengths]))


@pytest.fixture(scope="module")
def rf3_model(
    backend_option: str, device_option: str | None, output_ctx: Callable[[], AbstractContextManager[Any]]
) -> Generator[RF3, None, None]:
    """One RF3 model on the selected backend with the stock sampler settings, shared by the tests that fold with them."""
    with output_ctx(), RF3(backend=backend_option, device=device_option, config=_stock_config()) as model:
        yield model


def test_rf3_modal_fold_basic(rf3_model: RF3) -> None:
    """Fold a short protein and validate every output field against the structure it describes."""
    sequence = "MLKNVHVLVLGAGDVGSVVVRLLEK"
    result = rf3_model.fold(sequence, options={"include_fields": ALL_FIELDS, "early_stopping_plddt_threshold": 0.0})

    assert result.metadata.model_name == "RF3"
    assert result.metadata.sequence_lengths == [len(sequence)]
    assert result.metadata.optimization is not None and result.metadata.optimization["active"] == "vanilla"
    assert result.atom_array is not None and len(result.atom_array) == 1
    atoms = result.atom_array[0]
    assert len(atoms) > len(sequence)
    assert result.cif is not None and len(result.cif) == 1 and result.cif[0].startswith("data_")
    assert result.sample_ranks == [0] and result.sample_indices == [0] and result.seeds == [1]

    assert result.plddt is not None and result.plddt[0] is not None
    assert result.plddt[0].shape == (len(sequence),)
    assert float(result.plddt[0].min()) >= 0.0 and float(result.plddt[0].max()) <= 1.0
    assert result.atom_plddt is not None and result.atom_plddt[0].shape == (len(atoms),)
    assert result.pae is not None and result.pae[0].shape == (len(sequence), len(sequence))
    assert np.isfinite(result.pae[0]).all() and float(result.pae[0].min()) >= 0.0
    assert result.token_chain_ids is not None and set(result.token_chain_ids[0]) == {"A"}
    assert result.token_res_ids is not None
    assert np.array_equal(result.token_res_ids[0], np.arange(1, len(sequence) + 1))

    # One chain has no interface: RF3 reports ipTM 0, so the ranking score is 0.2 * pTM.
    assert result.ptm is not None and result.ptm[0] is not None and 0.0 < float(result.ptm[0][0]) <= 1.0
    assert result.iptm is not None and result.iptm[0] is not None and float(result.iptm[0][0]) == 0.0
    assert result.confidence is not None and result.confidence[0] is not None
    assert result.confidence[0]["ranking_score"] == pytest.approx(0.2 * float(result.ptm[0][0]), abs=1e-3)


def test_rf3_default_output_is_minimal(rf3_model: RF3) -> None:
    """Without ``include_fields`` only the metadata and the structure come back, as for the other families."""
    result = rf3_model.fold("MLKNVHVLVLGAGDVGSVVVRLLEK", options={"early_stopping_plddt_threshold": 0.0})

    assert result.atom_array is not None and len(result.atom_array[0]) > 0
    for field in ("plddt", "pae", "cif", "pdb", "confidence", "token_chain_ids", "atom_plddt"):
        assert getattr(result, field) is None, field


@pytest.mark.parametrize("reference_id", _reference_ids())
def test_rf3_matches_stock_rf3_reference(rf3_model: RF3, reference_id: str) -> None:
    """A vanilla boileroom fold must reproduce what the stock ``rf3 fold`` command wrote for the same inputs.

    The references under ``tests/data/rf3`` come from the RF3 command line at the pinned foundry commit with the
    same checkpoint and settings; see ``scripts/testing/rf3_stock_reference.py``.
    """
    sequence, lengths, expected_atoms, confidences, summary = _reference(reference_id)

    result = rf3_model.fold(sequence, options=_stock_options())

    assert result.metadata.optimization is not None and result.metadata.optimization["active"] == "vanilla"
    _assert_matches_stock_reference(result, lengths, expected_atoms, confidences, summary, max_ca_rmsd=1.0)


def test_rf3_reproduces_upstream_two_chain_baseline(
    backend_option: str, device_option: str | None, output_ctx: Callable[[], AbstractContextManager[Any]]
) -> None:
    """GLKE:GLKE with the quick settings must land on the baseline the foundry repository commits for its own CI.

    ``models/rf3/tests/data/integration_baselines/two_protein_chains`` (foundry at the pinned commit) holds the scores
    of this fold; its own test compares at +-0.02, and so does this one.
    """
    config = {"n_recycles": 1, "num_steps": 20, "diffusion_batch_size": 1}
    with output_ctx(), RF3(backend=backend_option, device=device_option, config=config) as model:
        result = model.fold(
            "GLKE:GLKE", options={"seed": 1, "early_stopping_plddt_threshold": 0.0, "include_fields": ALL_FIELDS}
        )

    assert result.confidence is not None and result.confidence[0] is not None
    scores = result.confidence[0]
    assert scores["ptm"] == pytest.approx(0.0847, abs=0.02)
    assert scores["iptm"] == pytest.approx(0.0025, abs=0.02)
    assert scores["ranking_score"] == pytest.approx(0.0189, abs=0.02)
    assert scores["overall_plddt"] == pytest.approx(0.7095, abs=0.02)
    assert scores["has_clash"] is False
    assert result.metadata.sequence_lengths == [8]


def test_rf3_returns_ranked_samples(
    backend_option: str, device_option: str | None, output_ctx: Callable[[], AbstractContextManager[Any]]
) -> None:
    """One call returns every diffusion sample, best first, and a repeated call with the same seed repeats them."""
    config = {"n_recycles": 2, "num_steps": 20, "diffusion_batch_size": 3}
    sequence = "RMKQLEDKVEELLSKNYHLENEVARLKKLVGER:RMKQLEDKVEELLSKNYHLENEVARLKKLVGER"
    options = {"seed": 3, "early_stopping_plddt_threshold": 0.0, "include_fields": ["confidence", "cif", "plddt"]}

    with output_ctx(), RF3(backend=backend_option, device=device_option, config=config) as model:
        first = model.fold(sequence, options=options)
        second = model.fold(sequence, options=options)

    assert first.cif is not None and len(first.cif) == 3
    assert first.confidence is not None and first.atom_array is not None and first.plddt is not None
    ranking = [scores["ranking_score"] for scores in first.confidence if scores is not None]
    assert len(ranking) == 3 and ranking == sorted(ranking, reverse=True)
    assert first.sample_ranks == [0, 1, 2]
    assert sorted(first.sample_indices or []) == [0, 1, 2]
    assert first.seeds == [3, 3, 3]
    assert len(first.atom_array) == len(first.plddt) == 3

    # The best sample is the one RF3 ranks first; the same seed gives the same samples in the same order.
    assert second.sample_indices == first.sample_indices
    assert second.confidence is not None
    assert [scores["ranking_score"] for scores in second.confidence if scores is not None] == pytest.approx(
        ranking, abs=1e-3
    )


def test_rf3_accepts_a_user_msa(rf3_model: RF3) -> None:
    """A caller-supplied alignment is staged for RF3 and the fold completes with finite scores."""
    sequence, lengths, _, _, _ = _reference("1ubq")
    rows = [sequence] + [sequence[:i] + "A" + sequence[i + 1 :] for i in (5, 20, 40, 60)]
    a3m = "".join(f">hit{index}\n{row}\n" for index, row in enumerate(rows))

    result = rf3_model.fold(sequence, options={**_stock_options(), "msa": [a3m]})

    assert result.metadata.sequence_lengths == lengths
    assert result.ptm is not None and result.ptm[0] is not None and np.isfinite(result.ptm[0]).all()
    assert result.plddt is not None and result.plddt[0] is not None
    assert result.plddt[0].shape == (sum(lengths),)


def test_rf3_early_stopping_reports_why_no_structure_was_written(rf3_model: RF3) -> None:
    """A fold below ``early_stopping_plddt_threshold`` writes no structure; the error says so instead of hiding it."""
    with pytest.raises(RuntimeError, match="stopped early.*below early_stopping_plddt_threshold"):
        rf3_model.fold("GLKE", options={"seed": 0, "early_stopping_plddt_threshold": 0.99})


@pytest.mark.parametrize("mode", ["exact", "fast", "big"])
@pytest.mark.parametrize("reference_id", _reference_ids())
def test_rf3_kit_matches_stock_rf3_reference(
    reference_id: str,
    mode: str,
    backend_option: str,
    device_option: str | None,
    output_ctx: Callable[[], AbstractContextManager[Any]],
) -> None:
    """The kit's ``exact``, ``fast`` and ``big`` modes must reproduce the stock reference up to what their kernels change.

    The worker runs ``python -I``, which ignores ``PYTHONHASHSEED``, so ``exact`` is not bit for bit reproducible and
    every mode is compared by tolerance. The scores and PAE bounds are those of the vanilla comparison. The structure
    bound is wider: against the seed-0 reference ``exact`` lands 0.71 A (ubiquitin) and 0.39 A (GCN4 dimer) away on
    A100-80GB and ``fast`` 0.84 A and 0.40 A. Other seeds are compared with the same seed-0 reference, so they include
    the sampler's own spread: over seeds 0-7 ``exact`` stays within 0.9 A and ``fast`` within 0.9 A for ubiquitin and
    1.9 A for one GCN4 seed, where vanilla itself reaches 2.7 A at that seed. ``fast`` and ``big`` re-associate bf16
    arithmetic, which the kit documents as within stock's seed-to-seed variation.
    """
    device = _kit_device(mode, device_option)
    sequence, lengths, expected_atoms, confidences, summary = _reference(reference_id)
    config = {**_stock_config(), "optimization": mode}

    with output_ctx(), RF3(backend=backend_option, device=device, config=config) as model:
        result = model.fold(sequence, options=_stock_options())

    assert result.metadata.optimization is not None and result.metadata.optimization["active"] == mode
    _assert_ran_on(result, device)
    _assert_matches_stock_reference(result, lengths, expected_atoms, confidences, summary, max_ca_rmsd=1.5)


def test_rf3_kit_big_folds_an_input_too_large_for_fast(
    backend_option: str, device_option: str | None, output_ctx: Callable[[], AbstractContextManager[Any]]
) -> None:
    """``big`` completes 2,608 residues on an A100-80GB, where ``fast`` runs out of memory (about eleven minutes)."""
    device = _kit_device("big", device_option)
    if device != "A100-80GB":
        pytest.skip(f"the input size was chosen for A100-80GB, not {device}")
    sequence = ":".join([LARGE_CHAIN] * LARGE_CHAIN_COUNT)

    with output_ctx(), RF3(backend=backend_option, device=device, config={"optimization": "big"}) as model:
        result = model.fold(
            sequence, options={"seed": 0, "early_stopping_plddt_threshold": 0.0, "include_fields": ["plddt", "ptm"]}
        )

    assert result.metadata.optimization is not None and result.metadata.optimization["active"] == "big"
    _assert_ran_on(result, device)
    assert result.plddt[0].shape == (LARGE_CHAIN_COUNT * len(LARGE_CHAIN),)
    assert 0.5 < float(np.mean(result.plddt[0])) <= 1.0 and 0.0 < float(result.ptm[0][0]) <= 1.0


def test_rf3_kit_fast_out_of_memory_names_big(
    backend_option: str, device_option: str | None, output_ctx: Callable[[], AbstractContextManager[Any]]
) -> None:
    """The input ``big`` folds above makes ``fast`` run out of memory, and the error says to use ``big``."""
    device = _kit_device("fast", device_option)
    if device != "A100-80GB":
        pytest.skip(f"the input size was chosen for A100-80GB, not {device}")
    sequence = ":".join([LARGE_CHAIN] * LARGE_CHAIN_COUNT)

    with (
        output_ctx(),
        RF3(backend=backend_option, device=device, config={"optimization": "fast"}) as model,
        pytest.raises(RuntimeError, match="out of GPU memory under optimization='fast'.*optimization='big'"),
    ):
        model.fold(sequence, options={"seed": 0, "early_stopping_plddt_threshold": 0.0})
