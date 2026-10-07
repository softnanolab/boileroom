"""Execute the publishing workflow's model selection without Docker or GitHub."""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest
import yaml

from boileroom.images.metadata import BASE_IMAGE_SPEC, MODEL_IMAGE_SPECS, get_model_image_spec

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
        assert amd_models == {"alphafold", "protenix", "opendde", "boltz", "chai", "esm", "esmfold2"}
        assert arm_models == {"boltz", "chai", "esm", "esmfold2"}
    elif models == "esm3":
        assert amd_models == arm_models == {"esmfold2"}
    else:
        assert amd_models == {"alphafold", "protenix", "opendde"}
        assert arm64 == []
    assert all(
        row["cuda_version"] == "12.6"
        for row in amd64
        if row["model"] in {"alphafold", "protenix", "opendde", "esmfold2"}
    )


WORKFLOW_PATH = ROOT / ".github/workflows/build-docker-images.yml"
VERIFY_SCRIPTS = ("check_model_imports.py", "check_model_server_health.py")
PUBLISH_GUARD = "needs.prepare-release.outputs.image_tag != needs.prepare-release.outputs.candidate_tag"


def _job_steps(job: str) -> list[dict[str, Any]]:
    workflow = yaml.safe_load(WORKFLOW_PATH.read_text())
    return workflow["jobs"][job]["steps"]


def _heredoc(run: str) -> str:
    """Return the Python a ``python - <<'PY'`` step runs."""
    return textwrap.dedent(run.split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0])


@pytest.mark.parametrize("job", ["build-amd64-base", "build-amd64-models"])
def test_amd64_images_get_their_final_tags_only_after_the_candidate_was_verified(job: str) -> None:
    """A promoted run used to push the final alpha or stable tag first and smoke it afterwards."""
    steps = _job_steps(job)
    runs = [step.get("run", "") for step in steps]
    push = [index for index, run in enumerate(runs) if "--push" in run]
    verify = [index for index, run in enumerate(runs) if any(script in run for script in VERIFY_SCRIPTS)]
    final_writes = [index for index, run in enumerate(runs) if "IMAGE_TAG" in run.replace("CANDIDATE_TAG", "")]

    assert len(push) == 1
    assert '--tag="${CANDIDATE_TAG}"' in runs[push[0]]
    assert all('--tag="${CANDIDATE_TAG}"' in runs[index] for index in verify)
    assert len(final_writes) == 1
    publish = steps[final_writes[0]]
    assert publish["if"] == PUBLISH_GUARD
    assert "promote_one(" in publish["run"]
    assert final_writes[0] > max([push[0], *verify])
    if job == "build-amd64-models":
        assert len(verify) == 4


@pytest.mark.parametrize(
    "event,should_promote,sha_length,final",
    [
        ("push", "true", 12, "0.4.2-alpha.3"),
        ("release", "true", 40, "0.4.2"),
        ("workflow_dispatch", "false", 12, None),
    ],
)
def test_candidate_tag_is_a_cleanable_sha_tag_and_distinct_on_promoted_runs(
    tmp_path: Path, event: str, should_promote: str, sha_length: int, final: str | None
) -> None:
    from scripts.images.cleanup_dockerhub_tags import is_sha_tag

    step = next(step for step in _job_steps("prepare-release") if step.get("id") == "image_tag")
    output = tmp_path / "output"
    sha = "0123456789abcdef" * 2 + "01234567"
    subprocess.run(
        ["bash", "-c", step["run"]],
        env={
            **os.environ,
            "GITHUB_SHA": sha,
            "GITHUB_EVENT_NAME": event,
            "GITHUB_OUTPUT": str(output),
            "SHOULD_PROMOTE": should_promote,
            "DERIVED_DOCKER_TAG": final or "",
        },
        check=True,
    )
    values = dict(line.split("=", 1) for line in output.read_text().splitlines())
    assert values["candidate"] == f"sha-{sha[:sha_length]}"
    assert is_sha_tag(values["candidate"])
    # A validation-only run builds straight under the candidate, so the publish steps are skipped.
    assert values["value"] == (final or values["candidate"])


@pytest.mark.parametrize("model", [spec.key for spec in MODEL_IMAGE_SPECS] + [None])
def test_publish_steps_promote_the_candidate_of_their_own_image(
    monkeypatch: pytest.MonkeyPatch, model: str | None
) -> None:
    from scripts.images import promote_image_tags

    calls: list[tuple[Any, ...]] = []
    monkeypatch.setattr(promote_image_tags, "promote_one", lambda *args: calls.append(args))
    env = {
        "CUDA_VERSION": "12.6",
        "CANDIDATE_TAG": "sha-0123456789ab",
        "IMAGE_TAG": "0.4.2",
        "DOCKER_REPOSITORY_FOR_RUN": "docker.io/jakublala",
    }
    for name, value in {**env, "MODEL": model or ""}.items():
        monkeypatch.setenv(name, value)
    job = "build-amd64-models" if model else "build-amd64-base"
    step = next(step for step in _job_steps(job) if step["name"].startswith("Publish"))

    exec(compile(_heredoc(step["run"]), step["name"], "exec"), {})

    expected = get_model_image_spec(model) if model else BASE_IMAGE_SPEC
    assert calls == [(expected, "12.6", "sha-0123456789ab", "0.4.2", "docker.io/jakublala")]


def test_full_release_tags_the_pinned_kit_images() -> None:
    workflow = yaml.safe_load(WORKFLOW_PATH.read_text())
    job = workflow["jobs"]["publish-kit-image-tags"]
    assert job["if"] == "${{ github.event_name == 'release' && github.event.release.prerelease == false }}"
    assert job["env"]["IMAGE_TAG"] == "${{ needs.prepare-release.outputs.image_tag }}"
    run = job["steps"][-1]["run"]
    assert "promote_image_tags.py" in run
    assert "--kit-images-only" in run
    assert '--target-tag="${IMAGE_TAG}"' in run
    assert "--force-kit-tags" not in run
