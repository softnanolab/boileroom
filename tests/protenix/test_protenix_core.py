import json
from pathlib import Path

from boileroom.base import PredictionMetadata
from boileroom.models.protenix.core import ProtenixCore, _command_env
from boileroom.models.protenix.types import ProtenixOutput


def test_protenix_writes_protein_chain_json(tmp_path: Path) -> None:
    """Protenix input JSON should encode colon-joined chains as separate proteinChain records."""
    core = ProtenixCore()

    input_json = core._write_input_json("AAAA:CCCC", tmp_path)

    payload = json.loads(input_json.read_text(encoding="utf-8"))
    assert payload[0]["name"] == "boileroom_target"
    chains = payload[0]["sequences"]
    assert [item["proteinChain"]["sequence"] for item in chains] == ["AAAA", "CCCC"]
    assert [item["proteinChain"]["id"] for item in chains] == [["A"], ["B"]]


def test_protenix_builds_official_cli_command(tmp_path: Path) -> None:
    """The core should map boileroom config to the official `protenix pred` CLI."""
    core = ProtenixCore({"protenix_command": "protenix-bin"})

    command = core._build_command(
        tmp_path / "input.json",
        tmp_path / "out",
        {
            **core.config,
            "model_name": "protenix-v2",
            "seeds": "101,102",
            "cycle": 4,
            "step": 20,
            "sample": 1,
            "dtype": "fp32",
            "use_msa": False,
            "use_template": True,
            "use_default_params": False,
            "trimul_kernel": "torch",
            "triatt_kernel": "torch",
            "enable_cache": False,
            "enable_fusion": False,
            "enable_tf32": False,
        },
    )

    assert command[:2] == ["protenix-bin", "pred"]
    assert command[command.index("--model_name") + 1] == "protenix-v2"
    assert command[command.index("--use_msa") + 1] == "false"
    assert command[command.index("--use_template") + 1] == "true"
    assert command[command.index("--use_default_params") + 1] == "false"


def test_protenix_collects_ranked_cifs_and_confidence(tmp_path: Path) -> None:
    """Postprocessing should collect ranked CIF files and summary confidence JSON."""
    cif_text = (Path(__file__).resolve().parents[1] / "data" / "boltz" / "0_model_0.cif").read_text(encoding="utf-8")
    for seed, plddt_score in [(101, 87.0), (102, 77.0)]:
        prediction_dir = tmp_path / "dataset" / "target" / f"seed_{seed}" / "predictions"
        prediction_dir.mkdir(parents=True)
        (prediction_dir / "boileroom_target_sample_0.cif").write_text(cif_text, encoding="utf-8")
        (prediction_dir / "boileroom_target_summary_confidence_sample_0.json").write_text(
            json.dumps({"plddt": plddt_score, "ptm": 0.71, "iptm": 0.62, "ranking_score": 0.65}),
            encoding="utf-8",
        )

    core = ProtenixCore()
    output = core._collect_outputs(
        tmp_path,
        PredictionMetadata("Protenix", "test", [4]),
        {"include_fields": ["*"]},
    )

    assert isinstance(output, ProtenixOutput)
    assert output.atom_array is not None and len(output.atom_array) == 2
    assert output.cif is not None and output.cif[0].startswith("data_")
    assert output.confidence is not None
    confidence = output.confidence[0]
    assert confidence is not None
    assert confidence["ranking_score"] == 0.65
    assert confidence["plddt"] == 87.0
    assert output.plddt is None
    assert output.ptm is not None
    ptm = output.ptm[0]
    assert ptm is not None
    assert ptm[0] == 0.71
    assert output.iptm is not None
    iptm = output.iptm[0]
    assert iptm is not None
    assert iptm[0] == 0.62


def test_protenix_command_env_preserves_backend_device_by_default(monkeypatch) -> None:
    """Backend-selected CUDA visibility should not be overwritten by core defaults."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")

    assert _command_env(ProtenixCore().config)["CUDA_VISIBLE_DEVICES"] == "7"
    assert _command_env({**ProtenixCore().config, "device": "cuda:1"})["CUDA_VISIBLE_DEVICES"] == "1"
    assert _command_env({**ProtenixCore().config, "device": "cpu"})["CUDA_VISIBLE_DEVICES"] == ""
