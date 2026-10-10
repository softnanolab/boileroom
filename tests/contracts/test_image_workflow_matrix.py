"""Execute the publishing workflow's model selection without Docker or GitHub."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "models,promote", [("", "false"), ("protenix opendde alphafold", "false"), ("esm3", "false"), ("protenix", "true")]
)
def test_publishing_matrix_selects_models_and_supported_platforms(models: str, promote: str) -> None:
    """Selections preserve shared images, full promotion and AMD64-only models."""
    workflow = yaml.safe_load((ROOT / ".github/workflows/build-docker-images.yml").read_text())
    step = next(step for step in workflow["jobs"]["prepare-release"]["steps"] if step.get("id") == "image_metadata")
    script = step["run"].split("\n", 1)[1].rsplit("\nPY", 1)[0]
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env={**os.environ, "MODELS_FOR_RUN": models, "SHOULD_PROMOTE": promote},
        check=True,
        capture_output=True,
        text=True,
    )
    values = dict(line.split("=", 1) for line in result.stdout.splitlines())
    amd64 = json.loads(values["amd64_model_matrix"])["include"]
    arm64 = json.loads(values["arm64_model_matrix"])["include"]
    amd_models = {row["model"] for row in amd64}
    arm_models = {row["model"] for row in arm64}
    if not models or promote == "true":
        assert amd_models == {"alphafold", "protenix", "opendde", "boltz", "chai", "esm", "esmfold2", "rf3"}
        assert arm_models == {"boltz", "chai", "esm", "esmfold2"}
    elif models == "esm3":
        assert amd_models == arm_models == {"esmfold2"}
    else:
        assert amd_models == {"alphafold", "protenix", "opendde"}
        assert arm64 == []
    assert all(
        row["cuda_version"] == "12.6"
        for row in amd64
        if row["model"] in {"alphafold", "protenix", "opendde", "esmfold2", "rf3"}
    )
