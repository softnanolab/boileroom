"""Unit tests for the ColabFold-backed AlphaFold2-Multimer core."""

import json
from pathlib import Path

import numpy as np
import pytest

from boileroom.base import PredictionMetadata
from boileroom.inputs import MSAInput
from boileroom.models._cli import command_env
from boileroom.msa_cache import MSACache


@pytest.fixture
def core_class():
    """Import the runtime adapter only inside tests."""
    from boileroom.models.alphafold.core import AlphaFold2MultimerCore

    return AlphaFold2MultimerCore


def _write_colabfold_job(job_dir: Path, *, relaxed: bool = False) -> None:
    """Materialize a miniature ColabFold output tree for one ranked prediction."""
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "query.done.txt").write_text("", encoding="utf-8")
    (job_dir / "query.a3m").write_text(">query\nAAAA:CCCC\n", encoding="utf-8")
    pdb_text = (Path(__file__).resolve().parents[1] / "data" / "multimer-check.pdb").read_text(encoding="utf-8")
    tag = "relaxed" if relaxed else "unrelaxed"
    suffix = "alphafold2_multimer_v3_model_1_seed_000"
    (job_dir / f"query_{tag}_rank_001_{suffix}.pdb").write_text(pdb_text, encoding="utf-8")
    (job_dir / f"query_scores_rank_001_{suffix}.json").write_text(
        json.dumps(
            {
                "plddt": [90.0, 80.0],
                "pae": [[0.0, 1.0], [1.0, 0.0]],
                "ptm": 0.7,
                "iptm": 0.8,
            }
        ),
        encoding="utf-8",
    )


def test_write_fasta_joins_chains_into_single_record(tmp_path: Path, core_class) -> None:
    """ColabFold consumes one record whose chains are colon-joined."""
    fasta_path = core_class()._write_fasta("AAAA:CCCC", tmp_path)

    assert fasta_path.read_text(encoding="utf-8").splitlines() == [">query", "AAAA:CCCC"]


def test_build_command_maps_config_to_colabfold_flags(tmp_path: Path, core_class) -> None:
    """The core should drive `colabfold_batch` with the configured options."""
    core = core_class({"colabfold_command": "/bin/colabfold_batch", "data_dir": "/data/af2"})

    command = core._build_command(
        tmp_path / "target.fasta",
        tmp_path / "out",
        "mmseqs2_uniref_env",
        {**core.config, "use_templates": True, "use_amber": True, "use_gpu_relax": True},
    )

    assert command[0] == "/bin/colabfold_batch"
    assert command[command.index("--model-type") + 1] == "alphafold2_multimer_v3"
    assert command[command.index("--msa-mode") + 1] == "mmseqs2_uniref_env"
    assert command[command.index("--pair-mode") + 1] == "unpaired_paired"
    assert command[command.index("--data") + 1] == "/data/af2"
    assert command[command.index("--host-url") + 1] == "https://api.colabfold.com"
    assert "--templates" in command
    assert "--amber" in command
    assert "--use-gpu-relax" in command


def test_build_command_omits_msa_mode_for_a3m_input(tmp_path: Path, core_class) -> None:
    """When an a3m is supplied, the MSA mode flag is omitted so ColabFold reads it."""
    core = core_class({"data_dir": "/data/af2"})

    command = core._build_command(tmp_path / "provided.a3m", tmp_path / "out", None, core.config)

    assert "--msa-mode" not in command


def test_resolve_msa_input_uses_server_and_returns_cache_key(tmp_path: Path, core_class) -> None:
    """A cold cache with the server enabled writes a FASTA and yields a cache key."""
    core = core_class({"data_dir": str(tmp_path)})

    input_path, msa_mode, cache_key = core._resolve_msa_input(
        "AAAA:CCCC", ["AAAA", "CCCC"], None, tmp_path, core.config
    )

    assert input_path.suffix == ".fasta"
    assert msa_mode == "mmseqs2_uniref_env"
    assert cache_key == MSACache.hash_key("AAAA:CCCC|mmseqs2_uniref_env|unpaired_paired|alphafold2_multimer_v3")


def test_resolve_msa_input_single_sequence_when_server_disabled(tmp_path: Path, core_class) -> None:
    """Disabling the server falls back to a single-sequence run with no caching."""
    core = core_class({"data_dir": str(tmp_path)})

    input_path, msa_mode, cache_key = core._resolve_msa_input(
        "AAAA", ["AAAA"], None, tmp_path, {**core.config, "use_msa_server": False}
    )

    assert input_path.suffix == ".fasta"
    assert msa_mode == "single_sequence"
    assert cache_key is None


def test_resolve_msa_input_hits_cache(tmp_path: Path, core_class) -> None:
    """A warm cache returns the stored a3m and skips the server."""
    core = core_class({"data_dir": str(tmp_path)})
    cache_key = core._cache_key("AAAA:CCCC", core.config)
    source = tmp_path / "src.a3m"
    source.write_text(">query\nAAAA:CCCC\n", encoding="utf-8")
    core._msa_cache().put(cache_key, source)

    input_path, msa_mode, returned_key = core._resolve_msa_input(
        "AAAA:CCCC", ["AAAA", "CCCC"], None, tmp_path, core.config
    )

    assert input_path.suffix == ".a3m"
    assert msa_mode is None
    assert returned_key is None
    assert input_path.read_text(encoding="utf-8") == ">query\nAAAA:CCCC\n"


def test_resolve_msa_input_accepts_provided_msa(tmp_path: Path, core_class) -> None:
    """A provided MSA is materialized and used directly, bypassing the server."""
    core = core_class({"data_dir": str(tmp_path)})
    provided = MSAInput(path=str(tmp_path / "given.a3m"))
    (tmp_path / "given.a3m").write_text(">query\nAAAA:CCCC\n>hit\nAAAA:CCCC\n", encoding="utf-8")

    input_path, msa_mode, cache_key = core._resolve_msa_input(
        "AAAA:CCCC", ["AAAA", "CCCC"], provided, tmp_path, core.config
    )

    assert input_path.suffix == ".a3m"
    assert msa_mode is None
    assert cache_key is None
    assert ">hit" in input_path.read_text(encoding="utf-8")


def test_materialize_msa_strips_insertions(tmp_path: Path, core_class) -> None:
    """`remove_insertions` drops lowercase/dot alignment columns."""
    src = tmp_path / "ins.a3m"
    src.write_text(">query\nAADE\n>hit\nAAdeDE\n", encoding="utf-8")
    text = core_class()._materialize_msa(MSAInput(path=str(src), remove_insertions=True))

    assert "de" not in text
    assert ">hit\nAADE" in text


def test_collect_outputs_maps_scores_and_structure(tmp_path: Path, core_class) -> None:
    """Postprocessing maps ColabFold scores and structures onto the output contract."""
    from boileroom.models.alphafold.types import AlphaFold2MultimerOutput

    job_dir = tmp_path / "outputs"
    _write_colabfold_job(job_dir)

    core = core_class()
    output = core._collect_outputs(
        job_dir,
        PredictionMetadata("AlphaFold2-Multimer", "v3", [8]),
        {**core.config, "include_fields": ["*"]},
    )

    assert isinstance(output, AlphaFold2MultimerOutput)
    assert output.atom_array is not None and len(output.atom_array) == 1
    assert output.pdb is not None and output.pdb[0].startswith("ATOM")
    assert output.cif is not None and output.cif[0].startswith("data_")
    assert output.ranking is not None
    assert output.ranking["order"] == ["rank_001_alphafold2_multimer_v3_model_1_seed_000"]
    assert output.plddt is not None and output.plddt[0] is not None
    assert np.allclose(output.plddt[0], [0.9, 0.8])
    assert output.ptm is not None and output.ptm[0] is not None
    assert output.ptm[0][0] == pytest.approx(0.7)
    assert output.iptm is not None and output.iptm[0] is not None
    assert output.iptm[0][0] == pytest.approx(0.8)
    assert output.pae is not None and output.pae[0] is not None
    assert output.pae[0].shape == (2, 2)


def test_collect_outputs_requires_completion_marker(tmp_path: Path, core_class) -> None:
    """A missing done.txt marker fails closed rather than returning empty results."""
    job_dir = tmp_path / "outputs"
    job_dir.mkdir()
    with pytest.raises(RuntimeError, match="did not complete"):
        core_class()._collect_outputs(
            job_dir, PredictionMetadata("AlphaFold2-Multimer", "v3", [8]), core_class().config
        )


def test_cache_generated_msa_stores_alignment(tmp_path: Path, core_class) -> None:
    """The a3m produced by a server run is written into the shared cache."""
    core = core_class({"data_dir": str(tmp_path)})
    job_dir = tmp_path / "outputs"
    _write_colabfold_job(job_dir)
    cache_key = core._cache_key("AAAA:CCCC", core.config)

    core._cache_generated_msa(job_dir, cache_key, core.config)

    cached = core._msa_cache().get(cache_key)
    assert cached is not None and cached.read_text(encoding="utf-8").startswith(">query")


@pytest.mark.parametrize("num_models", [0, 6])
def test_invalid_num_models_is_rejected(core_class, num_models) -> None:
    """num_models outside 1..5 must not reach paid inference."""
    from boileroom.models.alphafold.core import _validate_config

    with pytest.raises(ValueError):
        _validate_config({**core_class().config, "num_models": num_models})


def test_command_env_preserves_backend_device_by_default(monkeypatch, core_class) -> None:
    """Backend-selected CUDA visibility should not be overwritten by core defaults."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")

    assert command_env(core_class().config)["CUDA_VISIBLE_DEVICES"] == "7"
    assert command_env({**core_class().config, "device": "cuda:1"})["CUDA_VISIBLE_DEVICES"] == "1"
    assert command_env({**core_class().config, "device": "cpu"})["CUDA_VISIBLE_DEVICES"] == ""


def test_command_env_disables_colabfold_unified_memory(monkeypatch, core_class) -> None:
    """ColabFold's unified-memory default stalls on Modal; the core must opt out unless the user overrides."""
    from boileroom.models.alphafold.core import _command_env

    monkeypatch.delenv("TF_FORCE_UNIFIED_MEMORY", raising=False)
    monkeypatch.delenv("XLA_PYTHON_CLIENT_MEM_FRACTION", raising=False)
    env = _command_env(core_class().config)
    assert env["TF_FORCE_UNIFIED_MEMORY"] == "0"
    assert env["XLA_PYTHON_CLIENT_MEM_FRACTION"] == "0.9"

    monkeypatch.setenv("TF_FORCE_UNIFIED_MEMORY", "1")
    assert _command_env(core_class().config)["TF_FORCE_UNIFIED_MEMORY"] == "1"
