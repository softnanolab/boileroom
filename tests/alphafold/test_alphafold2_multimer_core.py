"""Unit tests for the ColabFold-backed AlphaFold2-Multimer core."""

import json
from pathlib import Path

import numpy as np
import pytest

from boileroom.base import PredictionMetadata
from boileroom.inputs import MSAInput
from boileroom.models._runtime_utils import command_env
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


def test_obsolete_cli_override_is_rejected(core_class) -> None:
    """Do not silently ignore an old setting that could bypass model residency."""
    with pytest.raises(ValueError, match="no longer supported"):
        core_class({"colabfold_command": "/bin/colabfold_batch"})


def test_plddt_ranking_uses_mean_residue_confidence(tmp_path, core_class) -> None:
    """The advertised ranking metric must not silently contain pTM instead."""
    _write_colabfold_job(tmp_path)
    core = core_class({"rank_by": "plddt", "include_fields": ["ranking"]})
    output = core._collect_outputs(tmp_path, PredictionMetadata("AlphaFold2-Multimer", "test", [8]), core.config)
    assert list(output.ranking["plddt"].values()) == [85.0]


def test_resolve_msa_input_uses_server_and_returns_cache_key(tmp_path: Path, core_class) -> None:
    """A cold cache with the server enabled writes a FASTA and yields a cache key."""
    core = core_class({"data_dir": str(tmp_path)})

    input_path, msa_mode, cache_key = core._resolve_msa_input(
        "AAAA:CCCC", ["AAAA", "CCCC"], None, tmp_path, core.config
    )

    assert input_path.suffix == ".fasta"
    assert msa_mode == "mmseqs2_uniref_env"
    assert cache_key is not None


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


@pytest.mark.parametrize("legacy_key", [False, True])
def test_resolve_msa_input_does_not_reuse_other_providers(tmp_path: Path, core_class, legacy_key: bool) -> None:
    """Changing providers or encountering an unscoped legacy entry is a miss."""
    core = core_class({"data_dir": str(tmp_path)})
    cache_key = (
        MSACache.hash_key("AAAA:CCCC|mmseqs2_uniref_env|unpaired_paired|alphafold2_multimer_v3")
        if legacy_key
        else core._cache_key("AAAA:CCCC", core.config)
    )
    source = tmp_path / "src.a3m"
    source.write_text(">query\nAAAACCCC\n", encoding="utf-8")
    core._msa_cache().put(cache_key, source)
    config = {**core.config, "msa_server_url": "https://msa.example.org"}
    input_path, msa_mode, returned_key = core._resolve_msa_input("AAAA:CCCC", ["AAAA", "CCCC"], None, tmp_path, config)
    assert input_path.suffix == ".fasta"
    assert msa_mode == "mmseqs2_uniref_env"
    assert returned_key is not None and returned_key != cache_key


def test_resolve_msa_input_accepts_provided_msa(tmp_path: Path, core_class) -> None:
    """A provided MSA is materialized and used directly, bypassing the server."""
    core = core_class({"data_dir": str(tmp_path)})
    provided = MSAInput(path=str(tmp_path / "given.a3m"))
    (tmp_path / "given.a3m").write_text("#4,4\t1,1\n>101\t102\nAAAACCCC\n>hit\nAAAACCCC\n", encoding="utf-8")

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
    text = core_class()._materialize_msa(MSAInput(path=str(src), remove_insertions=True), ["AADE"])

    assert "de" not in text
    assert ">hit\nAADE" in text


def test_materialize_msa_accepts_colabfold_complex_header(tmp_path: Path, core_class) -> None:
    """ColabFold complex a3m files start with a ``#lengths\tcardinalities`` header."""
    src = tmp_path / "complex.a3m"
    src.write_text("#4,4\t1,1\n>101\t102\nAAAACCCC\n>hit\nAAaAACCCC\n", encoding="utf-8")
    text = core_class()._materialize_msa(MSAInput(path=str(src), remove_insertions=True), ["AAAA", "CCCC"])

    assert text.startswith("#4,4\t1,1\n")
    assert ">hit\nAAAACCCC" in text


def test_materialize_msa_serializes_multichain_rows_with_complex_header(core_class) -> None:
    """``:``-joined rows become ColabFold complex a3m over the unique chains."""
    msa = MSAInput(sequences=["AAAA:AAAA:CCC", "AAAG:AAAG:CCD"])
    text = core_class()._materialize_msa(msa, ["AAAA", "AAAA", "CCC"])

    assert text == "#4,3\t2,1\n>101\t102\nAAAACCC\n>seq_0\nAAAACCC\n>seq_1\nAAAGCCD\n"


def test_materialize_msa_rejects_rows_with_wrong_chain_count(core_class) -> None:
    """Multichain MSA rows must carry one ``:`` segment per chain."""
    with pytest.raises(ValueError, match="expected one per chain"):
        core_class()._materialize_msa(MSAInput(sequences=["AAAACCCC"]), ["AAAA", "CCCC"])


@pytest.mark.parametrize("entry", ["AAAA::CCCC", "AAAA:", ":AAAA"])
def test_split_chains_rejects_empty_segments(entry: str) -> None:
    """Stray ``:`` separators fail instead of silently changing the stoichiometry."""
    from boileroom.models.alphafold.msa import split_chains as _split_chains

    with pytest.raises(ValueError, match="empty chains"):
        _split_chains(entry)


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
