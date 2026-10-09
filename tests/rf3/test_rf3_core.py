"""RF3 core tests: input staging, output parsing, configuration and worker wiring, without a GPU or heavy imports.

The sample layout is the one RF3 writes, and the structure is a real RF3 prediction (PDB 5VHT, two protein chains)
vendored from the upstream regression baseline, so parsing is checked against genuine model output.
"""

import json
import shutil
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest

from boileroom.base import PredictionMetadata
from boileroom.optimization import GpuInfo, OptimizationUnavailableError, resolve_optimization

DATA = Path(__file__).parents[1] / "data" / "rf3"
REAL_CIF = DATA / "5vht_model.cif"
REAL_SUMMARY = json.loads((DATA / "5vht_summary_confidences.json").read_text())
CHAIN_RESIDUES = 99
A100 = GpuInfo("NVIDIA A100-SXM4-40GB", (8, 0))


@pytest.fixture
def module() -> ModuleType:
    """Import the core inside tests, never at module scope."""
    from boileroom.models.rf3 import core

    return core


@pytest.fixture(autouse=True)
def weights_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point the weights root at a temporary directory and drop inherited RF3/kit variables."""
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "models"))
    for key in (
        "RF3_ROOT_DIR",
        "MODEL_OPT_TARGET_GPU",
        "MODEL_OPT_JIT_ROOT",
        "MODEL_OPT_STACK_KEY",
        "TRITON_CACHE_DIR",
        "TORCHINDUCTOR_CACHE_DIR",
        "ROSETTAFOLD3_OPT_DIGEST_DIR",
        "CUDA_VISIBLE_DEVICES",
    ):
        monkeypatch.delenv(key, raising=False)
    return tmp_path / "models"


@pytest.fixture
def real_atoms():
    """The vendored RF3 prediction as an atom array."""
    from biotite.structure.io.pdbx import CIFFile, get_structure

    return get_structure(CIFFile.read(str(REAL_CIF)), model=1, use_author_fields=False)


@pytest.fixture
def real_sequence(real_atoms) -> str:
    """The two chains of the vendored prediction as a colon-joined sequence.

    5VHT carries one genetically encoded non-canonical residue (``PBF``, residue 72 of each chain) that boileroom's
    protein-only input cannot express; it is represented by alanine. The fake worker never reads the sequence, so
    only the chain lengths matter here.
    """
    from biotite.sequence import ProteinSequence
    from biotite.structure import get_residues

    chains = []
    for chain_id in ("A", "B"):
        _, names = get_residues(real_atoms[real_atoms.chain_id == chain_id])
        chains.append("".join(_one_letter(ProteinSequence, name) for name in names))
    assert [len(chain) for chain in chains] == [CHAIN_RESIDUES, CHAIN_RESIDUES]
    return ":".join(chains)


def _one_letter(protein_sequence: Any, residue_name: str) -> str:
    try:
        return protein_sequence.convert_letter_3to1(residue_name)
    except KeyError:
        return "A"


def write_sample(
    output_dir: Path,
    sample: int,
    *,
    seed: int = 1,
    ranking_score: float | None = None,
    plddt_a: float = 0.9,
    plddt_b: float = 0.5,
) -> Path:
    """Write one RF3 sample (the real 5VHT structure with synthetic per-atom confidence) in RF3's layout."""
    from biotite.structure.io.pdbx import CIFFile, get_structure

    name = "boileroom_target"
    directory = output_dir / name / f"seed-{seed}_sample-{sample}"
    directory.mkdir(parents=True, exist_ok=True)
    prefix = directory / f"{name}_seed-{seed}_sample-{sample}"
    shutil.copy(REAL_CIF, f"{prefix}_model.cif")

    atoms = get_structure(CIFFile.read(str(REAL_CIF)), model=1, use_author_fields=False)
    summary = dict(REAL_SUMMARY)
    if ranking_score is not None:
        summary["ranking_score"] = ranking_score
    Path(f"{prefix}_summary_confidences.json").write_text(json.dumps(summary), encoding="utf-8")
    n_tokens = 2 * CHAIN_RESIDUES
    confidences = {
        "atom_plddts": np.where(atoms.chain_id == "A", plddt_a, plddt_b).tolist(),
        "pae": np.full((n_tokens, n_tokens), 3.0).tolist(),
        # Chain instance ids: every token of a chain carries the same one.
        "token_chain_ids": ["A_1"] * CHAIN_RESIDUES + ["B_1"] * CHAIN_RESIDUES,
        "token_res_ids": list(range(n_tokens)),
    }
    Path(f"{prefix}_confidences.json").write_text(json.dumps(confidences), encoding="utf-8")
    return directory


def metadata() -> PredictionMetadata:
    return PredictionMetadata("RF3", "rf3_foundry_01_24", [CHAIN_RESIDUES, CHAIN_RESIDUES])


# -- Configuration ---------------------------------------------------------------------------------------------------


def test_defaults_and_static_keys(module: ModuleType) -> None:
    """RF3 matches the upstream engine defaults, runs in its own interpreter, and fixes its pipeline at start-up."""
    core = module.RF3Core()

    assert core.config["n_recycles"] == 10 and core.config["diffusion_batch_size"] == 5
    assert core.config["num_steps"] == 50 and core.config["seed"] == 1
    assert core.config["optimization"] == "vanilla" and core.config["rf3_python"] == "/opt/rf3/bin/python"
    assert {"n_recycles", "diffusion_batch_size", "num_steps", "optimization", "rf3_python"} <= core.STATIC_CONFIG_KEYS
    assert "seed" not in core.STATIC_CONFIG_KEYS and "early_stopping_plddt_threshold" not in core.STATIC_CONFIG_KEYS
    assert core.SUPPORTS_USER_MSA and not core.SUPPORTS_USER_TEMPLATES


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"n_recycles": 0}, "n_recycles must be a positive integer"),
        ({"diffusion_batch_size": 1.5}, "diffusion_batch_size must be a positive integer"),
        ({"num_steps": True}, "num_steps must be a positive integer"),
        ({"seed": -1}, "seed must be a nonnegative integer"),
        ({"seed": "1"}, "seed must be a nonnegative integer"),
        ({"early_stopping_plddt_threshold": 1.5}, "early_stopping_plddt_threshold must be None or a number"),
        ({"early_stopping_plddt_threshold": "high"}, "early_stopping_plddt_threshold must be None or a number"),
        ({"timeout_seconds": 0}, "timeout_seconds must be a positive finite number or None"),
        ({"timeout_seconds": float("inf")}, "timeout_seconds must be a positive finite number or None"),
        ({"checkpoint_path": ""}, "checkpoint_path must be None or a nonempty path string"),
        ({"rf3_python": ""}, "rf3_python must be a nonempty path string"),
        ({"optimization": "turbo"}, "optimization"),
    ],
)
def test_invalid_configuration_is_rejected(module: ModuleType, overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        module._validate_config({**module.RF3Core.DEFAULT_CONFIG, **overrides})


@pytest.mark.parametrize("overrides", [{"timeout_seconds": None}, {"early_stopping_plddt_threshold": 0.0}, {"seed": 0}])
def test_boundary_configuration_is_accepted(module: ModuleType, overrides: dict[str, Any]) -> None:
    module._validate_config({**module.RF3Core.DEFAULT_CONFIG, **overrides})


def test_unknown_optimization_fails_at_construction(module: ModuleType) -> None:
    with pytest.raises(ValueError, match="optimization"):
        module.RF3Core({"optimization": "turbo"})


@pytest.mark.parametrize("mode", ["vanilla", "exact", "fast", "big"])
def test_every_listed_optimization_is_accepted(module: ModuleType, mode: str) -> None:
    """RF3 offers ``fast`` (the other families do not, see #125); the mode is checked again when the config changes."""
    core = module.RF3Core({"optimization": mode})
    module._validate_config(core.config)
    assert core.config["optimization"] == mode


def test_static_options_cannot_change_per_call(module: ModuleType) -> None:
    with pytest.raises(ValueError, match="can only be set at initialization"):
        module.RF3Core().fold("AAAA", options={"n_recycles": 3})


def test_templates_are_not_supported(module: ModuleType) -> None:
    with pytest.raises(ValueError, match="templates"):
        module.RF3Core().fold("AAAA", options={"templates": [None]})


def test_fold_takes_one_top_level_sequence(module: ModuleType) -> None:
    with pytest.raises(ValueError, match="exactly one top-level"):
        module.RF3Core().fold(["AAAA", "CCCC"])


# -- Input staging ---------------------------------------------------------------------------------------------------


def test_input_json_names_one_protein_per_chain(module: ModuleType, tmp_path: Path) -> None:
    """Chains become A, B, ... components under the fixed example name, and without MSAs carry no msa_path."""
    path = module.RF3Core()._write_input_json("MKTAYIAK:GSHM", tmp_path)

    (example,) = json.loads(path.read_text())
    assert example["name"] == "boileroom_target"
    assert example["components"] == [
        {"seq": "MKTAYIAK", "chain_id": "A"},
        {"seq": "GSHM", "chain_id": "B"},
    ]


def test_input_json_stages_supplied_alignments(module: ModuleType, tmp_path: Path) -> None:
    """Each chain's A3M is written beside the input and named by path; chains without one stay sequence-only."""
    target = ">query\nMKTA\n>hit\nMKcTA\n"

    path = module.RF3Core()._write_input_json("MKTA:GSHM", tmp_path, [target, None])

    first, second = json.loads(path.read_text())[0]["components"]
    assert Path(first["msa_path"]).read_text() == target
    assert Path(first["msa_path"]).parent == tmp_path
    assert "msa_path" not in second


@pytest.mark.parametrize(
    "sequence,msa,message",
    [
        ("MKTA::GSHM", None, "RF3 input requires 1 to 26 nonempty protein chains"),
        ("MKTA:", None, "RF3 input requires 1 to 26 nonempty protein chains"),
        (":".join(["MKTA"] * 27), None, "RF3 input requires 1 to 26 nonempty protein chains"),
        ("MKTA:GSHM", [None], "msa must have one A3M string or None per chain"),
        ("MKTA", [">query\nGGGG\n"], "first A3M row must match"),
        ("MKTA", [">query\nMKTA\n>hit\nMK\n"], "aligned length"),
        ("MKTA", ["not an alignment"], "A3M text or None"),
    ],
)
def test_invalid_input_names_rf3(
    module: ModuleType, tmp_path: Path, sequence: str, msa: list[str | None] | None, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        module.RF3Core()._write_input_json(sequence, tmp_path, msa)


def test_twenty_six_chains_are_allowed(module: ModuleType, tmp_path: Path) -> None:
    path = module.RF3Core()._write_input_json(":".join(["MKTA"] * 26), tmp_path)

    chain_ids = [component["chain_id"] for component in json.loads(path.read_text())[0]["components"]]
    assert chain_ids == [chr(65 + index) for index in range(26)]


# -- Output parsing against the real RF3 prediction ------------------------------------------------------------------


def test_collect_outputs_reads_real_rf3_prediction(module: ModuleType, tmp_path: Path) -> None:
    """The vendored RF3 5VHT prediction is parsed into structure, scores and per-residue confidence."""
    write_sample(tmp_path, 0)
    core = module.RF3Core({"diffusion_batch_size": 1})

    output = core._collect_outputs(tmp_path, metadata(), {**core.config, "include_fields": ["*"]})

    assert output.atom_array is not None and len(output.atom_array) == 1
    atoms = output.atom_array[0]
    assert len(atoms) == 1640 and sorted(set(atoms.chain_id)) == ["A", "B"]
    assert output.ptm is not None and output.ptm[0].item() == pytest.approx(0.9105080962181091, rel=1e-5)
    assert output.iptm is not None and output.iptm[0].item() == pytest.approx(0.9110991358757019, rel=1e-5)
    assert output.confidence is not None and output.confidence[0]["ranking_score"] == pytest.approx(0.911)
    assert output.plddt is not None and output.plddt[0].shape == (2 * CHAIN_RESIDUES,)
    np.testing.assert_allclose(output.plddt[0][:CHAIN_RESIDUES], 0.9, atol=1e-6)
    np.testing.assert_allclose(output.plddt[0][CHAIN_RESIDUES:], 0.5, atol=1e-6)
    assert output.atom_plddt is not None and output.atom_plddt[0].shape == (1640,)
    assert output.pae is not None and output.pae[0].shape == (2 * CHAIN_RESIDUES, 2 * CHAIN_RESIDUES)
    assert output.token_chain_ids is not None
    assert output.token_chain_ids[0].tolist() == ["A"] * CHAIN_RESIDUES + ["B"] * CHAIN_RESIDUES
    assert output.seeds == [1] and output.sample_ranks == [0] and output.sample_indices == [0]
    assert output.cif is not None and output.cif[0].startswith("data_")
    assert output.pdb is not None
    # The non-canonical PBF residues are written as HETATM records.
    assert sum(line.startswith(("ATOM", "HETATM")) for line in output.pdb[0].splitlines()) == 1640


def test_samples_are_ranked_by_ranking_score(module: ModuleType, tmp_path: Path) -> None:
    """Samples are returned best first with their original sample index; ties keep sample order."""
    for sample, score in enumerate([0.40, 0.911, 0.911]):
        write_sample(tmp_path, sample, ranking_score=score, plddt_a=0.1 * (sample + 1))
    core = module.RF3Core({"diffusion_batch_size": 3})

    output = core._collect_outputs(tmp_path, metadata(), {**core.config, "include_fields": ["*"]})

    assert output.sample_indices == [1, 2, 0] and output.sample_ranks == [0, 1, 2]
    assert output.confidence is not None
    assert [summary["ranking_score"] for summary in output.confidence] == pytest.approx([0.911, 0.911, 0.4])
    assert output.plddt is not None and output.plddt[0][0] == pytest.approx(0.2, abs=1e-6)
    assert output.plddt[2][0] == pytest.approx(0.1, abs=1e-6)


def test_default_include_fields_keep_structures_and_sample_identity(module: ModuleType, tmp_path: Path) -> None:
    """Without include_fields only the structure and sample identity are returned, like the other families."""
    write_sample(tmp_path, 0)
    core = module.RF3Core({"diffusion_batch_size": 1})

    output = core._collect_outputs(tmp_path, metadata(), {**core.config, "include_fields": None})

    assert output.atom_array is not None and len(output.atom_array[0]) == 1640
    assert output.seeds == [1] and output.sample_ranks == [0] and output.sample_indices == [0]
    assert output.plddt is None and output.pae is None and output.confidence is None
    assert output.cif is None and output.pdb is None


def test_named_include_fields_select_outputs(module: ModuleType, tmp_path: Path) -> None:
    write_sample(tmp_path, 0)
    core = module.RF3Core({"diffusion_batch_size": 1})

    output = core._collect_outputs(tmp_path, metadata(), {**core.config, "include_fields": ["cif", "ptm"]})

    assert output.cif is not None and output.pdb is None
    assert output.ptm is not None and output.pae is None


def test_missing_structure_is_an_error(module: ModuleType, tmp_path: Path) -> None:
    (tmp_path / "boileroom_target").mkdir()
    core = module.RF3Core({"diffusion_batch_size": 1})

    with pytest.raises(RuntimeError, match="produced no sample CIF files"):
        core._collect_outputs(tmp_path, metadata(), core.config)


def test_early_stop_is_reported_instead_of_a_missing_structure(module: ModuleType, tmp_path: Path) -> None:
    """A run that stopped on its pLDDT threshold names the cause."""
    directory = tmp_path / "boileroom_target"
    directory.mkdir()
    (directory / "boileroom_target_ranking_scores.csv").write_text("mean_plddt,early_stopped\n0.31,True\n")
    core = module.RF3Core({"diffusion_batch_size": 1})

    with pytest.raises(RuntimeError, match="stopped early and wrote no structure.*0.31"):
        core._collect_outputs(tmp_path, metadata(), core.config)


def test_sample_count_must_match_the_configured_batch(module: ModuleType, tmp_path: Path) -> None:
    write_sample(tmp_path, 0)
    write_sample(tmp_path, 1)
    core = module.RF3Core({"diffusion_batch_size": 5})

    with pytest.raises(RuntimeError, match="returned 2 samples; expected 5"):
        core._collect_outputs(tmp_path, metadata(), core.config)


def test_samples_must_belong_to_the_requested_seed(module: ModuleType, tmp_path: Path) -> None:
    write_sample(tmp_path, 0, seed=7)
    core = module.RF3Core({"diffusion_batch_size": 1})

    with pytest.raises(RuntimeError, match="for seed 1"):
        core._collect_outputs(tmp_path, metadata(), {**core.config, "seed": 1})


def test_summary_without_ranking_score_is_an_error(module: ModuleType, tmp_path: Path) -> None:
    directory = write_sample(tmp_path, 0)
    summary_path = next(directory.glob("*_summary_confidences.json"))
    summary_path.write_text(json.dumps({"ptm": 0.9}), encoding="utf-8")
    core = module.RF3Core({"diffusion_batch_size": 1})

    with pytest.raises(RuntimeError, match="no numeric ranking_score"):
        core._collect_outputs(tmp_path, metadata(), core.config)


def test_missing_confidence_file_names_the_file(module: ModuleType, tmp_path: Path) -> None:
    directory = write_sample(tmp_path, 0)
    (directory / "boileroom_target_seed-1_sample-0_confidences.json").unlink()
    core = module.RF3Core({"diffusion_batch_size": 1})

    with pytest.raises(RuntimeError, match="boileroom_target_seed-1_sample-0_confidences.json"):
        core._collect_outputs(tmp_path, metadata(), core.config)


@pytest.mark.parametrize("bad_score", [None, "0.9", True])
def test_ranking_score_must_be_numeric(module: ModuleType, bad_score: object) -> None:
    with pytest.raises(RuntimeError, match="no numeric ranking_score"):
        module._ranking_score({"ranking_score": bad_score})


# -- End to end through fold() with a fake worker --------------------------------------------------------------------


@pytest.fixture
def fake_worker(module: ModuleType, monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Replace the worker process with one that writes RF3-layout samples into the requested output directory."""
    worker = Mock()
    factory = Mock(return_value=worker)
    monkeypatch.setattr(module, "ModelWorker", factory)

    def predict(input_json: str, output_dir: str, config: dict[str, Any]) -> None:
        for sample in range(config["diffusion_batch_size"]):
            write_sample(Path(output_dir), sample, seed=config["seed"], ranking_score=0.5 + 0.1 * sample)

    worker.predict.side_effect = predict
    worker.factory = factory
    return worker


def test_fold_runs_the_worker_and_returns_ranked_samples(
    module: ModuleType, fake_worker: Mock, real_sequence: str
) -> None:
    """fold() stages the input, calls the worker once, and ranks what it wrote."""
    core = module.RF3Core({"diffusion_batch_size": 3})

    output = core.fold(real_sequence, options={"include_fields": ["*"]})

    fake_worker.predict.assert_called_once()
    input_json, _, config = fake_worker.predict.call_args.args
    assert Path(input_json).name == "input.json" and config["msa"] is None
    assert output.sample_indices == [2, 1, 0]
    # One entry per top-level sequence, chain separators not counted (the shared convention).
    assert output.metadata.model_name == "RF3" and output.metadata.sequence_lengths == [2 * CHAIN_RESIDUES]
    assert output.metadata.optimization == {
        "requested": "vanilla",
        "active": "vanilla",
        "kit_config": None,
        "gpu_name": None,
        "capability": None,
    }
    assert output.metadata.inference_time is not None and output.metadata.preprocessing_time is not None


def test_fold_loads_the_worker_once_across_calls(module: ModuleType, fake_worker: Mock, real_sequence: str) -> None:
    core = module.RF3Core({"diffusion_batch_size": 1})

    core.fold(real_sequence)
    core.fold(real_sequence)

    fake_worker.factory.assert_called_once()
    fake_worker.start.assert_called_once()
    assert fake_worker.predict.call_count == 2


def test_fold_applies_per_call_seed_and_alignments(module: ModuleType, fake_worker: Mock, real_sequence: str) -> None:
    """The seed is a per-call option; alignments are staged as files and not sent to the worker."""
    first, _ = real_sequence.split(":")
    a3m = f">query\n{first}\n"
    core = module.RF3Core({"diffusion_batch_size": 1})
    staged: dict[str, Any] = {}
    original = fake_worker.predict.side_effect

    def predict(input_json: str, output_dir: str, config: dict[str, Any]) -> None:
        staged["components"] = json.loads(Path(input_json).read_text())[0]["components"]
        staged["msa_text"] = Path(staged["components"][0]["msa_path"]).read_text()
        original(input_json, output_dir, config)

    fake_worker.predict.side_effect = predict

    output = core.fold(real_sequence, options={"seed": 7, "msa": [a3m, None]})

    assert output.seeds == [7]
    assert fake_worker.predict.call_args.args[2]["seed"] == 7 and fake_worker.predict.call_args.args[2]["msa"] is None
    assert staged["msa_text"] == a3m and "msa_path" not in staged["components"][1]


def test_invalid_per_call_option_fails_before_the_worker(
    module: ModuleType, fake_worker: Mock, real_sequence: str
) -> None:
    core = module.RF3Core({"diffusion_batch_size": 1})

    with pytest.raises(ValueError, match="seed must be a nonnegative integer"):
        core.fold(real_sequence, options={"seed": -3})

    fake_worker.predict.assert_not_called()


def test_close_releases_the_worker(module: ModuleType, fake_worker: Mock) -> None:
    core = module.RF3Core()
    core._initialize()
    assert core.ready

    core.close()

    fake_worker.close.assert_called_once()
    assert not core.ready


# -- Worker wiring and optimization ----------------------------------------------------------------------------------


def test_worker_uses_the_isolated_interpreter_and_rf3_runtime(module: ModuleType, fake_worker: Mock) -> None:
    core = module.RF3Core({"rf3_python": "/venv/bin/python"})

    core._initialize()
    core._initialize()

    fake_worker.factory.assert_called_once()
    kwargs = fake_worker.factory.call_args.kwargs
    assert kwargs["python_executable"] == "/venv/bin/python"
    assert kwargs["runtime_class"] == "RF3Runtime" and kwargs["label"] == "RF3"
    assert kwargs["runtime_path"].name == "runtime.py" and kwargs["runtime_path"].parent.name == "rf3"


def test_vanilla_never_probes_the_gpu(module: ModuleType, fake_worker: Mock, monkeypatch: pytest.MonkeyPatch) -> None:
    detect = Mock(side_effect=AssertionError("vanilla must not inspect the GPU"))
    monkeypatch.setattr(module, "detect_gpu", detect)

    core = module.RF3Core()
    core._initialize()

    detect.assert_not_called()
    assert core.optimization is not None and core.optimization.active == "vanilla"
    assert fake_worker.factory.call_args.kwargs["python_executable"] == module.DEFAULT_RF3_PYTHON


@pytest.mark.parametrize("mode", ["exact", "fast", "big"])
def test_kit_modes_run_in_the_kit_interpreter(
    module: ModuleType, fake_worker: Mock, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    monkeypatch.setattr(module, "detect_gpu", lambda device=None: A100)

    core = module.RF3Core({"optimization": mode})
    core._initialize()

    assert fake_worker.factory.call_args.kwargs["python_executable"] == module.KIT_RF3_PYTHON
    # The worker receives the mode itself: it is what the runtime hands to the kit.
    assert fake_worker.factory.call_args.args[0]["optimization"] == mode
    assert core.optimization is not None
    assert core.optimization.active == mode and core.optimization.kit_config == "a100"


@pytest.mark.parametrize("mode", ["exact", "fast", "big"])
def test_kit_modes_keep_an_explicit_interpreter(
    module: ModuleType, fake_worker: Mock, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    monkeypatch.setattr(module, "detect_gpu", lambda device=None: A100)

    module.RF3Core({"optimization": mode, "rf3_python": "/custom/python"})._initialize()

    assert fake_worker.factory.call_args.kwargs["python_executable"] == "/custom/python"


@pytest.mark.parametrize("mode", ["exact", "fast", "big"])
def test_kit_modes_are_refused_on_an_unserved_gpu_before_the_worker(
    module: ModuleType, fake_worker: Mock, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """An L4 cannot run the kit; the refusal names the card and the mode, and no worker starts."""
    monkeypatch.setattr(module, "detect_gpu", lambda device=None: GpuInfo("NVIDIA L4", (8, 9)))

    with pytest.raises(
        OptimizationUnavailableError, match=rf"optimization='{mode}' cannot run rf3 on NVIDIA L4 \(sm89\)"
    ):
        module.RF3Core({"optimization": mode})._initialize()

    fake_worker.factory.assert_not_called()


@pytest.mark.parametrize("mode", ["exact", "fast", "big"])
def test_fold_records_the_resolved_optimization(
    module: ModuleType, fake_worker: Mock, monkeypatch: pytest.MonkeyPatch, real_sequence: str, mode: str
) -> None:
    monkeypatch.setattr(module, "detect_gpu", lambda device=None: A100)

    output = module.RF3Core({"optimization": mode, "diffusion_batch_size": 1}).fold(real_sequence)

    assert output.metadata.optimization is not None
    assert output.metadata.optimization["requested"] == mode and output.metadata.optimization["active"] == mode
    assert output.metadata.optimization["kit_config"] == "a100"
    assert output.metadata.optimization["capability"] == "sm80"


@pytest.mark.parametrize(
    ("mode", "message", "hinted"),
    [
        ("fast", "RF3 worker failed:\ntorch.OutOfMemoryError: CUDA out of memory", True),
        ("fast", "RF3 worker failed:\nValueError: bad input", False),
        ("exact", "RF3 worker failed:\ntorch.OutOfMemoryError: CUDA out of memory", False),
    ],
)
def test_fast_out_of_memory_points_to_big(
    module: ModuleType,
    fake_worker: Mock,
    monkeypatch: pytest.MonkeyPatch,
    real_sequence: str,
    mode: str,
    message: str,
    hinted: bool,
) -> None:
    """Only a CUDA out-of-memory under ``fast`` is rewritten to name the lower-memory mode; the cause stays attached."""
    monkeypatch.setattr(module, "detect_gpu", lambda device=None: A100)
    fake_worker.predict.side_effect = RuntimeError(message)

    with pytest.raises(RuntimeError) as raised:
        module.RF3Core({"optimization": mode}).fold(real_sequence)

    assert ("optimization='big'" in str(raised.value)) is hinted
    assert message in str(raised.value)
    assert (raised.value.__cause__ is not None) is hinted


def test_vanilla_environment_has_no_kit_variables(module: ModuleType, weights_root: Path) -> None:
    env = module._command_env(module.RF3Core().config)

    assert env["RF3_ROOT_DIR"] == str(weights_root / "rf3")
    assert not any(key.startswith("MODEL_OPT_") for key in env)
    assert "TRITON_CACHE_DIR" not in env


@pytest.mark.parametrize("mode", ["exact", "fast", "big"])
@pytest.mark.parametrize(
    "gpu,config,capability",
    [(A100, "A100", "sm80"), (GpuInfo("NVIDIA H100 80GB HBM3", (9, 0)), "H100", "sm90")],
)
def test_kit_environment_names_gpu_and_caches(
    module: ModuleType, weights_root: Path, gpu: GpuInfo, config: str, capability: str, mode: str
) -> None:
    """The kit modes get their target GPU and per-stack JIT caches next to the weights."""
    resolution = resolve_optimization("rf3", mode, gpu)

    env = module._command_env(module.RF3Core().config, resolution)

    stack_key = f"torch2.13.0-{capability}"
    jit = weights_root / "rf3" / "jit"
    assert env["MODEL_OPT_TARGET_GPU"] == config
    assert env["MODEL_OPT_JIT_ROOT"] == str(jit) and env["MODEL_OPT_STACK_KEY"] == stack_key
    assert env["TRITON_CACHE_DIR"] == str(jit / stack_key / "triton")
    assert env["TORCHINDUCTOR_CACHE_DIR"] == str(jit / stack_key / "inductor")
    assert env["ROSETTAFOLD3_OPT_DIGEST_DIR"] == str(jit / "weights")
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"


def test_inherited_kit_environment_wins(module: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """An operator's cache location is never overwritten."""
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "mine"))

    env = module._command_env(module.RF3Core().config, resolve_optimization("rf3", "exact", A100))

    assert env["TRITON_CACHE_DIR"] == str(tmp_path / "mine")


def test_device_selects_the_visible_gpu(module: ModuleType) -> None:
    assert module._command_env({"device": "cuda:1"})["CUDA_VISIBLE_DEVICES"] == "1"
