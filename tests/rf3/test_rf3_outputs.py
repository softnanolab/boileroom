"""Parsing of RF3 output files, independent of the model and of the GPU."""

import json
from pathlib import Path

import numpy as np
import pytest

from boileroom.models.rf3.outputs import early_stop_message, read_json, read_token_confidence, sample_identity


def _atoms():
    """Two chains of two residues with two atoms each."""
    from biotite.structure import AtomArray

    atoms = AtomArray(8)
    atoms.chain_id = np.array(["A"] * 4 + ["B"] * 4)
    atoms.res_id = np.array([1, 1, 2, 2, 1, 1, 2, 2])
    atoms.res_name[:] = "ALA"
    return atoms


def _confidences(**overrides):
    values = {
        "atom_plddts": [0.8, 0.6, 0.9, 0.7, 0.6, 0.4, 0.5, 0.3],
        "pae": np.arange(16, dtype=float).reshape(4, 4).tolist(),
        "token_chain_ids": ["A_1", "A_1", "B_1", "B_1"],
    }
    values.update(overrides)
    return values


def test_sample_identity_reads_seed_and_sample_from_the_directory(tmp_path: Path) -> None:
    path = tmp_path / "seed-12_sample-3" / "boileroom_target_seed-12_sample-3_model.cif"

    assert sample_identity(path) == (12, 3)


def test_sample_identity_rejects_other_directories(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="Unrecognized RF3 sample path"):
        sample_identity(tmp_path / "somewhere" / "model.cif")


def test_read_json_returns_an_object(tmp_path: Path) -> None:
    path = tmp_path / "scores.json"
    path.write_text(json.dumps({"ptm": 0.5}), encoding="utf-8")

    assert read_json(path) == {"ptm": 0.5}


def test_read_json_names_the_missing_or_malformed_file(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="incomplete output: missing scores.json"):
        read_json(tmp_path / "scores.json")
    (tmp_path / "list.json").write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(RuntimeError, match="must be a JSON object: list.json"):
        read_json(tmp_path / "list.json")


def test_early_stop_message_reports_the_recorded_plddt(tmp_path: Path) -> None:
    (tmp_path / "target_ranking_scores.csv").write_text("mean_plddt,early_stopped\n0.31,True\n", encoding="utf-8")

    message = early_stop_message(tmp_path, "target")

    assert message is not None and "0.31" in message and "early_stopping_plddt_threshold" in message


def test_early_stop_message_is_none_without_a_stop(tmp_path: Path) -> None:
    assert early_stop_message(tmp_path, "target") is None
    (tmp_path / "target_ranking_scores.csv").write_text("sample,ranking_score\n0,0.5\n", encoding="utf-8")
    assert early_stop_message(tmp_path, "target") is None
    (tmp_path / "target_ranking_scores.csv").write_text("mean_plddt,early_stopped\n", encoding="utf-8")
    assert early_stop_message(tmp_path, "target") is None


def test_token_confidence_maps_residues_chains_and_plddt() -> None:
    result = read_token_confidence(_confidences(), _atoms())

    np.testing.assert_allclose(result["plddt"], [0.7, 0.8, 0.5, 0.4], atol=1e-6)
    assert result["pae"].shape == (4, 4)
    assert result["token_chain_ids"].tolist() == ["A", "A", "B", "B"]
    assert result["token_res_ids"].tolist() == [1, 2, 1, 2]
    assert result["atom_plddt"].shape == (8,)


@pytest.mark.parametrize("missing", ["atom_plddts", "pae", "token_chain_ids"])
def test_token_confidence_requires_every_field(missing: str) -> None:
    confidences = _confidences()
    del confidences[missing]

    with pytest.raises(RuntimeError, match=f"missing.*{missing}"):
        read_token_confidence(confidences, _atoms())


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"pae": [[0.0, 1.0, 2.0]]}, "square matrix"),
        ({"pae": []}, "square matrix"),
        ({"pae": [[0.0, -1.0], [1.0, 0.0]]}, "PAE contains invalid values"),
        ({"pae": np.full((4, 4), np.nan).tolist()}, "PAE contains invalid values"),
        ({"atom_plddts": [0.5] * 7}, "7 atom pLDDT values for 8 CIF atoms"),
        ({"atom_plddts": [0.5] * 7 + [np.nan]}, "atom pLDDT contains invalid values"),
        ({"pae": np.zeros((3, 3)).tolist(), "token_chain_ids": ["A_1", "A_1", "B_1"]}, "3 tokens but the CIF has 4"),
        ({"token_chain_ids": ["A_1", "A_1", "B_1"]}, "token chain IDs do not match"),
        ({"token_chain_ids": ["A_1", "B_1", "B_1", "B_1"]}, "disagree with the CIF chains"),
        ({"token_chain_ids": ["A_1", "A_1", "A_1", "A_1"]}, "disagree with the CIF chains"),
    ],
)
def test_token_confidence_rejects_inconsistent_files(overrides: dict, message: str) -> None:
    with pytest.raises(RuntimeError, match=message):
        read_token_confidence(_confidences(**overrides), _atoms())
