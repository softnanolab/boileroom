import json
from pathlib import Path

import numpy as np
import pytest

from boileroom.base import PredictionMetadata
from boileroom.models.protenix.types import ProtenixOutput


@pytest.fixture
def core_class():
    """Import the runtime adapter only inside tests."""
    from boileroom.models.protenix.core import ProtenixCore

    return ProtenixCore


def test_protenix_writes_protein_chain_json(tmp_path: Path, core_class) -> None:
    """Protenix input JSON should encode colon-joined chains as separate proteinChain records."""
    core = core_class()

    input_json = core._write_input_json("AAAA:CCCC", tmp_path)

    payload = json.loads(input_json.read_text(encoding="utf-8"))
    assert payload[0]["name"] == "boileroom_target"
    chains = payload[0]["sequences"]
    assert [item["proteinChain"]["sequence"] for item in chains] == ["AAAA", "CCCC"]
    assert [item["proteinChain"]["id"] for item in chains] == [["A"], ["B"]]


def test_protenix_rejects_obsolete_command_override(core_class) -> None:
    """The removed CLI escape hatch must not be silently ignored."""
    with pytest.raises(ValueError, match="no longer supported"):
        core_class({"protenix_command": "protenix-bin"})


@pytest.fixture
def sample_outputs(tmp_path: Path) -> Path:
    """Write a miniature, mapped Protenix 2.0 output with seeds 2 and 10."""
    from biotite.structure import AtomArray
    from biotite.structure.io.pdbx import CIFFile, set_structure

    atoms = AtomArray(8)
    atoms.chain_id = np.array(["B"] * 4 + ["A"] * 4)
    atoms.res_id = np.array([7, 7, 8, 8, 1, 1, 2, 2])
    atoms.res_name[:] = "ALA"
    atoms.atom_name = np.tile(["N", "CA"], 4)
    atoms.element = np.tile(["N", "C"], 4)
    atoms.coord = np.arange(24).reshape(8, 3)
    cif = CIFFile()
    set_structure(cif, atoms)
    for seed, plddt_score in [(10, 77.0), (2, 87.0)]:
        prediction_dir = tmp_path / "dataset" / "target" / f"seed_{seed}" / "predictions"
        prediction_dir.mkdir(parents=True)
        cif.write(prediction_dir / "boileroom_target_sample_0.cif")
        (prediction_dir / "boileroom_target_summary_confidence_sample_0.json").write_text(
            json.dumps({"plddt": plddt_score, "ptm": 0.71, "iptm": 0.62, "ranking_score": 0.65}),
            encoding="utf-8",
        )
        (prediction_dir / "boileroom_target_full_data_sample_0.json").write_text(
            json.dumps(
                {
                    "token_pair_pae": [[1, 2, 3, 4], [2, 1, 4, 3], [8, 9, 1, 2], [9, 8, 2, 1]],
                    "atom_to_token_idx": [0, 0, 1, 1, 2, 2, 3, 3],
                    "token_asym_id": [0, 0, 1, 1],
                    "atom_plddt": [0.8, 0.6, 0.9, 0.7, 0.6, 0.4, 0.5, 0.3],
                }
            ),
            encoding="utf-8",
        )
    return tmp_path


def test_protenix_collects_ranked_cifs_and_confidence(sample_outputs: Path, core_class) -> None:
    """PAE alignment and atom averages follow explicit mapping, with numeric seed order."""
    core = core_class()
    output = core._collect_outputs(
        sample_outputs,
        PredictionMetadata("Protenix", "test", [4]),
        {**core.config, "seeds": "10,2", "sample": 1, "include_fields": ["*"]},
    )

    assert isinstance(output, ProtenixOutput)
    assert output.atom_array is not None and len(output.atom_array) == 2
    assert output.cif is not None and output.cif[0].startswith("data_")
    assert output.confidence is not None
    confidence = output.confidence[0]
    assert confidence is not None
    assert confidence["ranking_score"] == 0.65
    assert confidence["plddt"] == 87.0
    assert output.seeds == [2, 10]
    assert output.sample_ranks == [0, 0]
    assert output.pae is not None and output.pae[0].shape == (4, 4)
    assert output.pae[0][0, 2] == 3
    assert output.pae[0][2, 0] == 8
    assert output.token_chain_ids is not None
    assert output.token_chain_ids[0].tolist() == ["B", "B", "A", "A"]
    assert output.token_res_ids is not None
    assert output.token_res_ids[0].tolist() == [7, 8, 1, 2]
    assert output.plddt is not None
    np.testing.assert_allclose(output.plddt[0], [0.7, 0.8, 0.5, 0.4])
    assert output.ptm is not None
    ptm = output.ptm[0]
    assert ptm is not None
    assert ptm[0] == 0.71
    assert output.iptm is not None
    iptm = output.iptm[0]
    assert iptm is not None
    assert iptm[0] == 0.62


def test_protenix_command_env_preserves_backend_device_by_default(monkeypatch, core_class, tmp_path) -> None:
    """Backend-selected CUDA visibility should not be overwritten by core defaults."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    monkeypatch.delenv("PROTENIX_ROOT_DIR", raising=False)
    from boileroom.models.protenix.core import _command_env

    assert _command_env(core_class().config)["CUDA_VISIBLE_DEVICES"] == "7"
    assert _command_env(core_class().config)["PROTENIX_ROOT_DIR"] == str(tmp_path / "protenix")
    assert _command_env({**core_class().config, "device": "cuda:1"})["CUDA_VISIBLE_DEVICES"] == "1"
    assert _command_env({**core_class().config, "device": "cpu"})["CUDA_VISIBLE_DEVICES"] == ""


def test_protenix_command_env_routes_msa_search_to_configured_server(monkeypatch, core_class) -> None:
    """The Protenix MSA client must use the configured ColabFold-compatible server, not its own default."""
    from boileroom.models.protenix.core import _command_env

    monkeypatch.setenv("MMSEQS_SERVICE_HOST_URL", "https://stale.example")
    assert _command_env(core_class().config)["MMSEQS_SERVICE_HOST_URL"] == "https://api.colabfold.com"
    custom = {**core_class().config, "msa_server_url": "https://msa.example"}
    assert _command_env(custom)["MMSEQS_SERVICE_HOST_URL"] == "https://msa.example"


def test_target_only_msa_suppresses_binder_search(tmp_path, core_class) -> None:
    """Target alignment is transported as text, binder gets a query-only alignment."""
    target = ">query\nAAAA\n>homolog\nAAcAA\n"
    path = core_class()._write_input_json("AAAA:CCCC", tmp_path, [target, None])
    records = json.loads(path.read_text())[0]["sequences"]
    msa_files = [Path(record["proteinChain"]["unpairedMsaPath"]) for record in records]
    assert msa_files[0].read_text() == target
    assert msa_files[1].read_text() == ">query\nCCCC\n"


@pytest.mark.parametrize("msa", [[">query\nCCCC\n", None], [">query\nAAAA\n>bad\nAA\n", None], [None]])
def test_invalid_msa_is_rejected_before_inference(tmp_path, core_class, msa) -> None:
    """Wrong query/row length or chain count must not reach paid inference."""
    with pytest.raises(ValueError):
        core_class()._write_input_json("AAAA:CCCC", tmp_path, msa)


def test_missing_seed_fails_even_when_cli_exit_was_zero(sample_outputs, core_class) -> None:
    """Upstream can skip failed inputs and still exit zero; reject partial results."""
    core = core_class()
    with pytest.raises(RuntimeError, match="incomplete or duplicate"):
        core._collect_outputs(
            sample_outputs,
            PredictionMetadata("Protenix", "test", [4]),
            {**core.config, "seeds": "2,10,11", "sample": 1},
        )


def test_missing_pae_fails_closed(sample_outputs, core_class) -> None:
    """A confidence file without PAE cannot masquerade as a usable oracle result."""
    path = next(sample_outputs.glob("**/*_full_data_*.json"))
    full = json.loads(path.read_text())
    del full["token_pair_pae"]
    path.write_text(json.dumps(full))
    core = core_class()
    with pytest.raises(RuntimeError, match="missing"):
        core._collect_outputs(
            sample_outputs, PredictionMetadata("Protenix", "test", [4]), {**core.config, "seeds": "2,10", "sample": 1}
        )


@pytest.mark.parametrize("seeds", ["1,a", "", "1,1", "-1", 7])
def test_parse_seeds_rejects_malformed_values(seeds) -> None:
    """Malformed seed lists fail with the documented validation message."""
    from boileroom.models.protenix.core import _parse_seeds

    with pytest.raises(ValueError, match="comma-separated unique nonnegative integers"):
        _parse_seeds(seeds)
