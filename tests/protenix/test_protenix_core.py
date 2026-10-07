import json
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest

from boileroom.base import PredictionMetadata
from boileroom.models.protenix.types import ProtenixOutput
from boileroom.optimization import OptimizationUnavailableError


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
    core = core_class()

    assert core._worker_env(core.config, None)["CUDA_VISIBLE_DEVICES"] == "7"
    assert core._worker_env(core.config, None)["PROTENIX_ROOT_DIR"] == str(tmp_path / "protenix")
    assert core._worker_env({**core.config, "device": "cuda:1"}, None)["CUDA_VISIBLE_DEVICES"] == "1"
    assert core._worker_env({**core.config, "device": "cpu"}, None)["CUDA_VISIBLE_DEVICES"] == ""


def test_protenix_command_env_routes_msa_search_to_configured_server(monkeypatch, core_class) -> None:
    """The Protenix MSA client must use the configured ColabFold-compatible server, not its own default."""
    core = core_class()
    monkeypatch.setenv("MMSEQS_SERVICE_HOST_URL", "https://stale.example")
    assert core._worker_env(core.config, None)["MMSEQS_SERVICE_HOST_URL"] == "https://api.colabfold.com"
    custom = {**core.config, "msa_server_url": "https://msa.example"}
    assert core._worker_env(custom, None)["MMSEQS_SERVICE_HOST_URL"] == "https://msa.example"


def test_target_only_msa_suppresses_binder_search(tmp_path, core_class) -> None:
    """Target alignment is transported as text, binder gets a query-only alignment."""
    target = ">query\nAAAAA\n>homolog\nAAcAAA\n"
    path = core_class()._write_input_json("AAAAA:CCCC", tmp_path, [target, None])
    records = json.loads(path.read_text())[0]["sequences"]
    msa_files = [Path(record["proteinChain"]["unpairedMsaPath"]) for record in records]
    assert msa_files[0].read_text() == target
    assert msa_files[1].read_text() == ">query\nCCCC\n"


@pytest.mark.parametrize("msa", [[">query\nCCCC\n", None], [">query\nAAAA\n>bad\nAA\n", None], [None]])
def test_invalid_msa_is_rejected_before_inference(tmp_path, core_class, msa) -> None:
    """Wrong query/row length or chain count must not reach paid inference."""
    with pytest.raises(ValueError):
        core_class()._write_input_json("AAAA:CCCC", tmp_path, msa)


def _upstream_features(text: str) -> list[tuple[str, list[int]]]:
    """Mirror Protenix's ``parse_fasta`` and ``MSACore.sequences_to_array`` for protein rows (pinned 2.0 source).

    Lines are stripped; blank and ``#`` lines are skipped. Uppercase letters and ``-`` are aligned; any other
    character counts as an inserted residue before the next aligned column. Returns ``(aligned row, deletions)``.
    """
    rows: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(">"):
            rows.append("")
        elif rows:
            rows[-1] += line
    features = []
    for row in rows:
        aligned, deletions, pending = "", [], 0
        for char in row:
            if char.isascii() and (char.isupper() or char == "-"):
                aligned += char
                deletions.append(pending)
                pending = 0
            else:
                pending += 1
        features.append((aligned, deletions))
    return features


def test_caller_a3m_is_rewritten_as_the_rows_upstream_parses(tmp_path, core_class) -> None:
    """Wrapped rows, blank lines and ``.`` insert-state gaps reach Protenix as the validated rows.

    Protenix counts every ``.`` as a deleted residue; the written file drops them, so upstream's deletion counts are
    the lowercase insertions alone, and its aligned rows are the ones ``a3m_rows`` validated.
    """
    from boileroom.inputs import a3m_rows

    caller = "\n>query\nMKTAY\n\n>hit one\n  MK..tA\nY-.\n>hit one\nM.Kaa-AY\n"
    path = core_class()._write_input_json("MKTAY:CCCC", tmp_path, [caller, None])
    written = Path(json.loads(path.read_text())[0]["sequences"][0]["proteinChain"]["unpairedMsaPath"]).read_text()

    assert written == ">query\nMKTAY\n>hit one\nMKtAY-\n>hit one\nMKaa-AY\n"
    features = _upstream_features(written)
    assert [aligned for aligned, _ in features] == a3m_rows(caller, "MKTAY")
    assert [deletions for _, deletions in features] == [[0] * 5, [0, 0, 1, 0, 0], [0, 0, 2, 0, 0]]


@pytest.mark.parametrize(
    "row",
    [
        pytest.param("MKT A", id="space"),
        pytest.param("MKT\tA", id="tab"),
        pytest.param("MKTA*", id="terminator"),
        pytest.param("MK1AY", id="digit"),
        pytest.param("MKTA\n#", id="comment-line"),
        pytest.param("MK\u00c4AY", id="non-ascii-upper"),
        pytest.param("MK\u00e9TAY", id="non-ascii-lower"),
    ],
)
def test_caller_a3m_rows_upstream_would_misread_are_refused(tmp_path, core_class, row) -> None:
    """Rows the shared validator accepts but Protenix would read as other rows (or crash on) are refused."""
    from boileroom.inputs import a3m_rows

    caller = f">query\nMKTAY\n>hit\n{row}\n"
    a3m_rows(caller, "MKTAY")  # the shared validator alone lets these through
    with pytest.raises(ValueError, match="Protenix msa entry 0: A3M row 1 contains"):
        core_class()._write_input_json("MKTAY:CCCC", tmp_path, [caller, None])
    assert not (tmp_path / "chain_0.a3m").exists()


def test_homolog_rows_for_a_chain_upstream_featurizes_without_msa_are_refused(tmp_path, core_class) -> None:
    """Protenix drops the MSA of a chain of 4 or fewer residues, so its homolog rows would be silently unused."""
    with pytest.raises(ValueError, match="Protenix msa entry 1: the chain has 4 residues"):
        core_class()._write_input_json("MKTAY:CCCC", tmp_path, [None, ">query\nCCCC\n>hit\nCCaCC\n"])
    path = core_class()._write_input_json("MKTAY:CCCC", tmp_path, [None, ">query\nCCCC\n"])
    records = json.loads(path.read_text())[0]["sequences"]
    assert Path(records[1]["proteinChain"]["unpairedMsaPath"]).read_text() == ">query\nCCCC\n"


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


class FakeWorker:
    """CPU stand-in for ``ModelWorker``: records calls, reports a ``describe()`` and a predict payload."""

    def __init__(self, config: dict[str, Any], env: dict[str, str], **kwargs: Any) -> None:
        self.env = env
        self.info: dict[str, Any] = {}
        self.calls: list[str] = []
        self.predict_error: Exception | None = None

    def start(self) -> dict[str, Any]:
        self.calls.append("start")
        self.info = {"python": "3.11.5", "layernorm": "openfold"}
        return self.info

    def predict(self, input_json: str, output_dir: str, config: dict[str, Any]) -> dict[str, Any]:
        self.calls.append("predict")
        if self.predict_error is not None:
            raise self.predict_error
        return {"kernel.resolved.trimul": "cuequivariance", "templates.staged": 0}

    def close(self) -> None:
        self.calls.append("close")
        self.info = {}


@pytest.fixture
def fake_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[FakeWorker]:
    """Patch the worker and the GPU probes; return the workers the core creates."""
    import boileroom.models.protenix.core as core_module
    from boileroom.optimization import GpuInfo

    workers: list[FakeWorker] = []

    def factory(*args: Any, **kwargs: Any) -> FakeWorker:
        workers.append(FakeWorker(*args, **kwargs))
        return workers[-1]

    gpu = GpuInfo("NVIDIA H100 80GB HBM3", (9, 0))
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(core_module, "ModelWorker", factory)
    monkeypatch.setattr(core_module, "describe_gpu", lambda device=None: gpu)
    monkeypatch.setattr(core_module, "detect_gpu", lambda device=None: gpu)
    return workers


def _stub_outputs(core: Any) -> None:
    core._collect_outputs = lambda output_dir, metadata, config: Mock(metadata=metadata)


@pytest.mark.parametrize(
    ("mode", "layernorm"), [("vanilla", "openfold"), ("exact", "fast_layernorm"), ("fast", "fast_layernorm")]
)
def test_layernorm_is_set_per_mode_over_the_inherited_value(
    monkeypatch: pytest.MonkeyPatch, core_class, tmp_path: Path, mode: str, layernorm: str
) -> None:
    """The image ENV (or a caller's environment) cannot move a mode off its LayerNorm."""
    from boileroom.models.protenix.core import PROTENIX_LAYERNORM

    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    assert PROTENIX_LAYERNORM[mode] == layernorm
    core = core_class()
    for inherited in ("fast_layernorm", "openfold", "torch"):
        monkeypatch.setenv("LAYERNORM_TYPE", inherited)
        assert core._worker_env({**core.config, "optimization": mode}, None)["LAYERNORM_TYPE"] == layernorm


@pytest.mark.parametrize(
    ("flat", "levers"),
    [("none", []), (None, []), ("lnstream", ["lnstream"]), ("lnstream,blk2", ["lnstream", "blk2"])],
)
def test_split_levers_reads_the_none_sentinel_as_no_levers(flat: str | None, levers: list[str]) -> None:
    """The worker reports an empty lever list as ``"none"``; it must not come back as a lever named "none"."""
    from boileroom.models.protenix.core import _split_levers

    assert _split_levers(flat) == levers


def test_protenix_worker_env_adds_nothing_of_opendde(monkeypatch: pytest.MonkeyPatch, core_class, tmp_path) -> None:
    """The family hook is OpenDDE's alone: a Protenix worker gets no JIT root, kit process env or venv paths."""
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("PATH", "/usr/bin")
    for name in ("MODEL_OPT_JIT_ROOT", "PYTHONHASHSEED", "LD_LIBRARY_PATH"):
        monkeypatch.delenv(name, raising=False)
    core = core_class()
    env = core._worker_env(core.config, None)
    assert env["PATH"] == "/usr/bin"
    assert not {"MODEL_OPT_JIT_ROOT", "PYTHONHASHSEED", "LD_LIBRARY_PATH"} & env.keys()


def test_metadata_runtime_merges_provenance_worker_and_request(core_class, fake_runtime) -> None:
    """One flat record: the core's provenance and GPU, ``worker.*`` from describe(), ``predict.*`` from the run."""
    import platform

    core = core_class()
    _stub_outputs(core)
    runtime = core.fold("AAAA").metadata.runtime
    assert runtime["python"] == platform.python_version() and "boileroom" in runtime and "protenix" in runtime
    assert runtime["gpu"] == "NVIDIA H100 80GB HBM3" and runtime["gpu_capability"] == "sm90"
    assert runtime["worker.python"] == "3.11.5" and runtime["worker.layernorm"] == "openfold"
    assert runtime["predict.kernel.resolved.trimul"] == "cuequivariance" and runtime["predict.templates.staged"] == "0"


def test_vanilla_uses_the_lenient_gpu_probe_and_runs_without_a_card(
    monkeypatch: pytest.MonkeyPatch, core_class, fake_runtime
) -> None:
    """Vanilla never calls the strict probe: without a readable GPU it still runs and records none."""
    import boileroom.models.protenix.core as core_module

    def strict(device: str | None = None) -> Any:
        raise OptimizationUnavailableError("optimization exact/fast needs a CUDA GPU; none is visible")

    monkeypatch.setattr(core_module, "detect_gpu", strict)
    monkeypatch.setattr(core_module, "describe_gpu", lambda device=None: None)
    core = core_class()
    _stub_outputs(core)
    runtime = core.fold("AAAA").metadata.runtime
    assert runtime["gpu"] == "none" and runtime["gpu_capability"] == "none"
    assert core._refusal is None and len(fake_runtime) == 1


@pytest.mark.parametrize("mode", ["exact", "fast"])
def test_kit_modes_use_the_strict_gpu_probe(monkeypatch: pytest.MonkeyPatch, core_class, fake_runtime, mode) -> None:
    """A kit mode never falls back to the lenient probe: an unreadable GPU refuses before any worker starts."""
    import boileroom.models.protenix.core as core_module

    def strict(device: str | None = None) -> Any:
        raise OptimizationUnavailableError("optimization exact/fast needs a CUDA GPU; none is visible")

    monkeypatch.setattr(core_module, "describe_gpu", Mock(side_effect=AssertionError("lenient probe in a kit mode")))
    core = core_class({"optimization": mode})
    _stub_outputs(core)
    assert core.fold("AAAA").metadata.runtime["gpu"] == "NVIDIA H100 80GB HBM3"
    monkeypatch.setattr(core_module, "detect_gpu", strict)
    with pytest.raises(OptimizationUnavailableError, match="none is visible"):
        core_class({"optimization": mode}).fold("AAAA")
    assert len(fake_runtime) == 1


def test_refused_predict_closes_the_worker_and_the_refusal_stands(core_class, fake_runtime) -> None:
    """A run the kit refused is not retried on a restarted worker: every later call raises the refusal."""
    core = core_class({"optimization": "fast"})
    _stub_outputs(core)
    core._initialize()
    worker = fake_runtime[0]
    worker.predict_error = OptimizationUnavailableError("Protenix refused: the kit went inactive")
    with pytest.raises(OptimizationUnavailableError, match="the kit went inactive"):
        core.fold("AAAA")
    assert worker.calls == ["start", "predict", "close"] and not core.ready
    worker.predict_error = None
    for call in (lambda: core.fold("AAAA"), core._initialize):
        with pytest.raises(OptimizationUnavailableError, match="refused earlier in this runtime"):
            call()
    assert len(fake_runtime) == 1 and worker.calls == ["start", "predict", "close"]


def test_failed_predict_that_is_not_a_refusal_does_not_stand(core_class, fake_runtime) -> None:
    """Only a refusal stands; an ordinary worker failure leaves the next call free to restart it."""
    core = core_class()
    _stub_outputs(core)
    core._initialize()
    fake_runtime[0].predict_error = RuntimeError("worker crashed")
    with pytest.raises(RuntimeError, match="worker crashed"):
        core.fold("AAAA")
    fake_runtime[0].predict_error = None
    assert core.fold("AAAA").metadata.runtime["predict.templates.staged"] == "0"


@pytest.mark.parametrize("options", [{"use_template": True}, {"templates": {"t": "data_t\n"}}])
def test_templates_are_refused_on_a_checkpoint_without_a_template_embedder(core_class, fake_runtime, options) -> None:
    """The mini checkpoints have no template embedder: a template would be ignored, so the request is refused."""
    core = core_class({"model_name": "protenix_mini_default_v0.5.0", **options})
    with pytest.raises(ValueError, match="has no template embedder"):
        core._initialize()
    with pytest.raises(ValueError, match="has no template embedder"):
        core.fold("AAAA")
    assert fake_runtime == []


def test_templates_are_accepted_on_template_capable_checkpoints(core_class) -> None:
    """Every checkpoint upstream lets search or take templates passes the check."""
    from boileroom.models.protenix.core import PROTENIX_TEMPLATE_MODELS

    assert core_class().config["model_name"] in PROTENIX_TEMPLATE_MODELS
    for model_name in PROTENIX_TEMPLATE_MODELS:
        core = core_class({"model_name": model_name})
        core._validate_templates({**core.config, "use_template": True})
        core._validate_templates({**core.config, "templates": {"t": "data_t\n"}})


def test_use_template_with_caller_templates_is_refused_before_staging(core_class, fake_runtime) -> None:
    """Caller templates replace the search; asking for both is refused before anything is staged or loaded."""
    core = core_class({"use_template": True})
    core._stage_templates = Mock(side_effect=AssertionError("staged"))
    with pytest.raises(ValueError, match="cannot be combined with use_template=True"):
        core.fold("AAAA", options={"templates": {"t": "data_t\n"}})
    assert fake_runtime == []


def test_other_chains_get_a_query_only_templates_path(core_class, tmp_path: Path) -> None:
    """Only ``templates_chain`` reads the staged hits; the other chains read a query-only file and are not searched."""
    from boileroom.models.protenix.templates import StagedTemplates

    staged = StagedTemplates(
        templates_path=str(tmp_path / "templates" / "hits.a3m"),
        mmcif_dir=str(tmp_path / "templates" / "mmcif"),
        release_dates_path=str(tmp_path / "templates" / "release_dates.json"),
        obsolete_pdbs_path=str(tmp_path / "templates" / "obsolete.dat"),
        cache_dir=str(tmp_path / "templates" / "cache"),
        query="AAAA",
        count=1,
        names=("t",),
    )
    path = core_class()._write_input_json("AAAA:CCCC", tmp_path, [None, None], staged, 0)
    records = [r["proteinChain"] for r in json.loads(path.read_text())[0]["sequences"]]
    assert records[0]["templatesPath"] == staged.templates_path
    assert Path(records[1]["templatesPath"]).read_text() == ">query\nCCCC\n"
    assert records[1]["templatesPath"] != records[1]["unpairedMsaPath"]


def test_unpaired_msa_is_an_unknown_key(core_class, fake_runtime) -> None:
    """The unreleased ``unpaired_msa`` alias is gone: it is refused as an unknown key, before any load."""
    with pytest.raises(ValueError, match=r"does not accept config keys \['unpaired_msa'\]"):
        core_class().fold("AAAA", options={"unpaired_msa": [">q\nAAAA\n"]})
    assert "unpaired_msa" not in core_class().config
    assert fake_runtime == []


def test_static_key_per_call_is_refused_by_the_wrapper_and_the_core(
    monkeypatch: pytest.MonkeyPatch, core_class
) -> None:
    """The wrapper refuses before dispatch (no remote load); the core refuses the same key for direct callers."""
    from boileroom.models.protenix.protenix import Protenix

    backend = Mock()
    wrapper = Protenix.__new__(Protenix)
    wrapper._backend = backend
    with pytest.raises(ValueError, match="can only be set at initialization.*model_name"):
        wrapper.fold("AAAA", options={"model_name": "protenix_mini_default_v0.5.0"})
    assert backend.mock_calls == []

    core = core_class()
    started = Mock()
    monkeypatch.setattr(core, "_load", started)
    with pytest.raises(ValueError, match="can only be set at initialization.*model_name"):
        core.fold("AAAA", options={"model_name": "protenix_mini_default_v0.5.0"})
    started.assert_not_called()
