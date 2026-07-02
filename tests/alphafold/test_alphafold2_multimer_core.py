import json
import pickle
from pathlib import Path

import numpy as np

from boileroom.base import PredictionMetadata
from boileroom.models.alphafold.core import AlphaFold2MultimerCore, _command_env
from boileroom.models.alphafold.types import AlphaFold2MultimerOutput


def test_alphafold2_multimer_writes_multichain_fasta(tmp_path: Path) -> None:
    """AlphaFold2-Multimer should encode colon-joined chains as a multi-record FASTA."""
    core = AlphaFold2MultimerCore()

    fasta_path = core._write_fasta("AAAA:CCCC", tmp_path)

    assert fasta_path.read_text(encoding="utf-8").splitlines() == [">chain_0", "AAAA", ">chain_1", "CCCC"]


def test_alphafold2_multimer_builds_official_runner_command(tmp_path: Path) -> None:
    """The core should map boileroom config to DeepMind's `run_alphafold.py` flags."""
    core = AlphaFold2MultimerCore({"alphafold_command": "/bin/run_af2", "data_dir": "/data/af2"})

    command = core._build_command(
        tmp_path / "target.fasta",
        tmp_path / "out",
        {
            **core.config,
            "max_template_date": "2021-09-30",
            "db_preset": "reduced_dbs",
            "num_multimer_predictions_per_model": 1,
            "models_to_relax": "best",
            "use_gpu_relax": True,
            "use_precomputed_msas": True,
            "benchmark": True,
            "random_seed": 7,
        },
    )

    assert command[0] == "/bin/run_af2"
    assert "--model_preset=multimer" in command
    assert "--data_dir=/data/af2" in command
    assert "--max_template_date=2021-09-30" in command
    assert "--db_preset=reduced_dbs" in command
    assert "--num_multimer_predictions_per_model=1" in command
    assert "--models_to_relax=best" in command
    assert "--use_gpu_relax=true" in command
    assert "--use_precomputed_msas=true" in command
    assert "--benchmark=true" in command
    assert "--small_bfd_database_path=/data/af2/small_bfd/bfd-first_non_consensus_sequences.fasta" in command
    assert "--uniref90_database_path=/data/af2/uniref90/uniref90.fasta" in command
    assert "--uniprot_database_path=/data/af2/uniprot/uniprot.fasta" in command
    assert "--pdb_seqres_database_path=/data/af2/pdb_seqres/pdb_seqres.txt" in command
    assert "--random_seed=7" in command


def test_alphafold2_multimer_collects_ranked_outputs(tmp_path: Path) -> None:
    """Postprocessing should collect ranked PDB outputs and model pickle confidence."""
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    pdb_text = (Path(__file__).resolve().parents[1] / "data" / "multimer-check.pdb").read_text(encoding="utf-8")
    (target_dir / "ranked_0.pdb").write_text(pdb_text, encoding="utf-8")
    (target_dir / "ranking_debug.json").write_text(
        json.dumps({"iptm+ptm": {"model_1_multimer_v3_pred_0": 0.9}, "order": ["model_1_multimer_v3_pred_0"]}),
        encoding="utf-8",
    )
    with (target_dir / "result_model_1_multimer_v3_pred_0.pkl").open("wb") as handle:
        pickle.dump(
            {
                "plddt": np.asarray([90.0, 80.0], dtype=np.float32),
                "ptm": np.asarray(0.7, dtype=np.float32),
                "iptm": np.asarray(0.8, dtype=np.float32),
                "predicted_aligned_error": np.eye(2, dtype=np.float32),
            },
            handle,
        )

    core = AlphaFold2MultimerCore()
    output = core._collect_outputs(
        target_dir,
        PredictionMetadata("AlphaFold2-Multimer", "test", [8]),
        {"include_fields": ["*"]},
    )

    assert isinstance(output, AlphaFold2MultimerOutput)
    assert output.atom_array is not None and len(output.atom_array) == 1
    assert output.pdb is not None and output.pdb[0].startswith("ATOM")
    assert output.ranking is not None and output.ranking["order"] == ["model_1_multimer_v3_pred_0"]
    assert output.plddt is not None
    plddt = output.plddt[0]
    assert plddt is not None
    assert np.allclose(plddt, [0.9, 0.8])
    assert output.ptm is not None
    ptm = output.ptm[0]
    assert ptm is not None
    assert ptm[0] == 0.7
    assert output.iptm is not None
    iptm = output.iptm[0]
    assert iptm is not None
    assert iptm[0] == 0.8
    assert output.pae is not None
    pae = output.pae[0]
    assert pae is not None
    assert pae.shape == (2, 2)


def test_alphafold2_multimer_command_env_preserves_backend_device_by_default(monkeypatch) -> None:
    """Backend-selected CUDA visibility should not be overwritten by core defaults."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")

    assert _command_env(AlphaFold2MultimerCore().config)["CUDA_VISIBLE_DEVICES"] == "7"
    assert _command_env({**AlphaFold2MultimerCore().config, "device": "cuda:1"})["CUDA_VISIBLE_DEVICES"] == "1"
    assert _command_env({**AlphaFold2MultimerCore().config, "device": "cpu"})["CUDA_VISIBLE_DEVICES"] == ""
