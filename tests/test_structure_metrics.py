"""CPU unit tests of the interface metrics and helpers the kit integration tests score with."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
from _structure_metrics import (
    BINDERS,
    GOLDENS,
    OPTIMIZATION_MODES,
    SEPARATION,
    TARGET,
    TOLERANCE,
    asymmetric_ipsae,
    ca_coordinates,
    chain_index_from_lengths,
    chain_index_from_token_chain_ids,
    cif_protein_sequence,
    d0,
    drop_residues_in_cif,
    ipsae,
    mean_distance_difference,
    scalar,
)

DATA = pathlib.Path(__file__).parent / "data"

#: Two residues per chain. For fewer than 26 kept residues d0 is max(1, 1.24 * 11 ** (1/3) - 1.8) = 1, so every kept
#: pair scores 1 / (1 + PAE ** 2).
#:   target -> binder: row 0 keeps PAE 2 (12 >= 10 is dropped) -> 1/5; row 1 keeps 5, 5 -> 1/26. Max 1/5.
#:   binder -> target: row 2 keeps 4, 8 -> (1/17 + 1/65) / 2 = 41/1105; row 3 keeps nothing (20, 20). Max 41/1105.
SYNTHETIC_PAE = np.array(
    [
        [0.0, 0.0, 2.0, 12.0],
        [0.0, 0.0, 5.0, 5.0],
        [4.0, 8.0, 0.0, 0.0],
        [20.0, 20.0, 0.0, 0.0],
    ]
)
SYNTHETIC_CHAINS = np.array([0, 0, 1, 1])


def test_d0_matches_the_ipsae_formula() -> None:
    """d0 is floored at 1 and uses n >= 26."""
    assert d0(1) == pytest.approx(1.0)
    assert d0(26) == pytest.approx(1.0)  # 1.24 * 11 ** (1/3) - 1.8 = 0.958, floored
    assert d0(100) == pytest.approx(1.24 * 85 ** (1 / 3) - 1.8)
    np.testing.assert_allclose(d0(np.array([0, 200])), [1.0, 1.24 * 185 ** (1 / 3) - 1.8])


def test_ipsae_on_a_hand_computed_case() -> None:
    """Both directions and the headline min match the values worked out by hand above."""
    score = ipsae(SYNTHETIC_PAE, SYNTHETIC_CHAINS)
    assert score.target_to_binder == pytest.approx(1 / 5)
    assert score.binder_to_target == pytest.approx(41 / 1105)
    assert score.min == pytest.approx(41 / 1105)


def test_ipsae_uses_the_kept_pair_count_for_d0() -> None:
    """With 40 binder residues kept for a row, d0 is 1.24 * 25 ** (1/3) - 1.8, not 1."""
    n_target, n_binder = 3, 40
    pae = np.full((n_target + n_binder,) * 2, 3.0)
    chains = chain_index_from_lengths(n_target, n_binder)
    scale = 1.24 * (40 - 15) ** (1 / 3) - 1.8
    expected = 1.0 / (1.0 + (3.0 / scale) ** 2)
    assert ipsae(pae, chains).target_to_binder == pytest.approx(expected)


def test_ipsae_drops_pairs_at_the_cutoff() -> None:
    """``PAE == cutoff`` is dropped (strict ``<``), and a direction with no kept pair scores 0."""
    pae = np.array([[0.0, 10.0], [9.999, 0.0]])
    chains = np.array([0, 1])
    score = ipsae(pae, chains)
    assert score.target_to_binder == 0.0
    assert score.binder_to_target == pytest.approx(1 / (1 + 9.999**2))
    assert score.min == 0.0
    assert asymmetric_ipsae(pae, chains == 0, chains == 1, pae_cutoff=10.5) == pytest.approx(1 / 101)


def test_ipsae_is_order_independent_of_chain_labels() -> None:
    """Swapping the labels swaps the two directions."""
    swapped = ipsae(SYNTHETIC_PAE, 1 - SYNTHETIC_CHAINS)
    assert swapped.target_to_binder == pytest.approx(41 / 1105)
    assert swapped.binder_to_target == pytest.approx(1 / 5)


@pytest.mark.parametrize(
    ("pae", "chains", "match"),
    [
        (np.zeros((3, 4)), np.array([0, 0, 1]), "square"),
        (np.zeros(4), np.array([0, 0, 1, 1]), "square"),
        (np.zeros((4, 4)), np.array([0, 0, 1]), "chain_index has shape"),
        (np.zeros((4, 4)), np.array([0, 0, 0, 0]), "both chains"),
        (np.zeros((4, 4)), np.array([0, 0, 2, 2]), "both chains"),
        (np.full((2, 2), np.nan), np.array([0, 1]), "non-finite"),
    ],
)
def test_ipsae_refuses_malformed_input(pae: np.ndarray, chains: np.ndarray, match: str) -> None:
    """A malformed PAE or chain labelling is refused, not scored."""
    with pytest.raises(ValueError, match=match):
        ipsae(pae, chains)


def test_chain_index_helpers() -> None:
    """Labels from chain lengths and from per-token chain ids agree on target-then-binder."""
    by_length = chain_index_from_lengths(len(TARGET), len(BINDERS["p53"]))
    assert by_length.shape == (len(TARGET) + len(BINDERS["p53"]),)
    assert by_length[: len(TARGET)].sum() == 0 and by_length[len(TARGET) :].all()
    token_ids = ["A"] * len(TARGET) + ["B"] * len(BINDERS["p53"])
    np.testing.assert_array_equal(chain_index_from_token_chain_ids(token_ids), by_length)
    with pytest.raises(ValueError, match="at least one residue"):
        chain_index_from_lengths(0, 3)
    with pytest.raises(ValueError, match="two chains"):
        chain_index_from_token_chain_ids(["A", "A"])
    with pytest.raises(ValueError, match="two chains"):
        chain_index_from_token_chain_ids(["A", "B", "C"])
    with pytest.raises(ValueError, match="non-empty"):
        chain_index_from_token_chain_ids([])


def test_scalar() -> None:
    """ipTM fields arrive as numbers or one-element arrays."""
    assert scalar(np.array([[0.5]])) == 0.5
    assert scalar(0.25) == 0.25
    with pytest.raises(ValueError, match="one value"):
        scalar([0.1, 0.2])


def test_goldens_are_complete_and_consistent() -> None:
    """Every family has every mode; measured kit goldens sit within the tolerance of vanilla and clear separation."""
    assert set(GOLDENS) == set(SEPARATION) == {"esmfold2", "protenix", "opendde"}
    for family, by_mode in GOLDENS.items():
        assert set(by_mode) == set(OPTIMIZATION_MODES), family
        vanilla = by_mode["vanilla"]["p53"]
        separation = SEPARATION[family]
        for mode, golden in by_mode.items():
            assert set(golden) == {"p53", "decoy"}, (family, mode)
            p53, decoy = golden["p53"], golden["decoy"]
            if p53 is None or decoy is None or vanilla is None:
                continue
            assert abs(p53 - vanilla) <= TOLERANCE, (family, mode)
            assert p53 - decoy >= separation.margin, (family, mode)
            assert separation.p53_min is None or p53 >= separation.p53_min, (family, mode)
            assert separation.decoy_max is None or decoy <= separation.decoy_max, (family, mode)


@pytest.fixture
def cif_text() -> str:
    return (DATA / "chai" / "pred.rank_0.cif").read_text()


def test_drop_residues_in_cif_removes_only_the_selected_residues(cif_text: str) -> None:
    """The atoms from residue 207 on are gone; the rest, and SEQRES, are unchanged."""
    import io

    from biotite.structure.io import pdbx

    half = drop_residues_in_cif(cif_text, 207)
    assert cif_protein_sequence(half) == cif_protein_sequence(cif_text)

    def atoms(text: str):
        return pdbx.get_structure(pdbx.CIFFile.read(io.StringIO(text)), model=1)

    before, after = atoms(cif_text), atoms(half)
    kept = np.asarray(before.res_id) < 207
    assert kept.any() and (~kept).any()
    assert len(after) == int(kept.sum())
    assert np.asarray(after.res_id).max() == 206
    np.testing.assert_allclose(after.coord, before.coord[kept], atol=1e-3)
    ca_before, ca_after = ca_coordinates(before), ca_coordinates(after)
    assert ca_after.shape == (206, 3)
    assert mean_distance_difference(ca_before[:206], ca_after) == pytest.approx(0.0, abs=1e-3)


def test_drop_residues_in_cif_refuses_a_no_op_or_an_empty_result(cif_text: str) -> None:
    """Dropping residues that do not exist, from text without atoms, or every residue is an error, not a silent copy."""
    with pytest.raises(ValueError, match="no atom"):
        drop_residues_in_cif(cif_text, 10_000)
    with pytest.raises(ValueError, match="no atom"):
        drop_residues_in_cif("data_x\n#\n", 1)
    with pytest.raises(ValueError, match="every atom"):
        drop_residues_in_cif(cif_text, 1)


def _mini_cif(rows: str, *, tail: str = "") -> str:
    """A two-category CIF whose ``_atom_site`` loop holds ``rows`` (group, id, seq id, x)."""
    return (
        "data_mini\n#\nloop_\n_entity_poly_seq.num\n_entity_poly_seq.mon_id\n1 GLY\n2 ALA\n#\n"
        "loop_\n_atom_site.group_PDB\n_atom_site.id\n_atom_site.label_seq_id\n_atom_site.Cartn_x\n"
        f"{rows}#\n_cell.length_a 1.0\n{tail}"
    )


def test_drop_residues_in_cif_keeps_atoms_without_a_residue_number_and_reads_past_blank_lines() -> None:
    """Waters (seq id ``.``) are kept; a blank line or a comment inside the loop does not end it."""
    text = _mini_cif("ATOM 1 1 0.0\n\nATOM 2 2 1.0\n# note\nATOM 3 2 2.0\nHETATM 4 . 3.0\n")
    out = drop_residues_in_cif(text, 2)
    assert "ATOM 1 1 0.0\n" in out and "HETATM 4 . 3.0\n" in out
    assert "ATOM 2 " not in out and "ATOM 3 " not in out
    # Every other category is byte-identical.
    assert out.startswith(text[: text.index("loop_\n_atom_site")]) and out.endswith("#\n_cell.length_a 1.0\n")


@pytest.mark.parametrize(
    ("rows", "tail", "message"),
    [
        ("ATOM 1 1 0.0\nATOM 2 2 'a b'\n", "", "does not split"),
        ("ATOM 1 1 0.0\nATOM 2 2\n;multi\nline\n;\n", "", "does not split"),
        ("ATOM 1 1 0.0\nATOM 2 2 1.0\n", "data_second\nloop_\n_atom_site.label_seq_id\n2\n", "more than one"),
        ("ATOM 1 1 0.0\nATOM 2 2 1.0\n", "_atom_site.label_seq_id 2\n", "outside a loop"),
    ],
)
def test_drop_residues_in_cif_refuses_rows_it_cannot_read(rows: str, tail: str, message: str) -> None:
    """A row it cannot split, a second atom loop or a non-loop atom record is an error, not a partial edit."""
    with pytest.raises(ValueError, match=message):
        drop_residues_in_cif(_mini_cif(rows, tail=tail), 2)


def test_mean_distance_difference_ignores_rigid_motion() -> None:
    """A translated copy compares as equal; moving one point does not."""
    coords = np.array([[0.0, 0.0, 0.0], [3.8, 0.0, 0.0], [3.8, 3.8, 0.0]])
    assert mean_distance_difference(coords, coords + 7.0) == pytest.approx(0.0, abs=1e-9)
    moved = coords.copy()
    moved[2] += [0.0, 3.0, 0.0]
    assert mean_distance_difference(coords, moved) > 0.5


def test_cif_protein_sequence_reads_seqres(cif_text: str) -> None:
    """The fixture's SEQRES is the 413-residue query the OpenDDE template test folds."""
    sequence = cif_protein_sequence(cif_text)
    assert len(sequence) == 413
    assert sequence.startswith("ICLQKTSNQILKPKLISYTLGQSGTCITDP")


def test_mean_distance_difference_refuses_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="differ in shape"):
        mean_distance_difference(np.zeros((3, 3)), np.zeros((4, 3)))
