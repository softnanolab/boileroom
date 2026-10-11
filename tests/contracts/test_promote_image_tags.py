"""Contract tests for promoting runtime image tags, the kit images included, without Docker."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from typing import Any

import pytest
from click.testing import CliRunner

from boileroom.images.metadata import (
    BASE_IMAGE_SPEC,
    KIT_IMAGE_DIGESTS,
    KIT_IMAGE_SPECS,
    KIT_IMAGE_TAG_ENV,
    MODEL_IMAGE_SPECS,
    get_supported_cuda,
)
from scripts.images import promote_image_tags

REPO = "docker.io/jakublala"
OTHER_DIGEST = "sha256:" + "e" * 64


def _stock_digest(reference: str) -> str:
    """A distinct, stable fake digest per stock source reference."""
    return "sha256:" + format(abs(hash(reference)) % 16**64, "064x")


@dataclass
class FakeRegistry:
    """Records Docker commands and answers ``imagetools`` like a registry would."""

    tags: dict[str, tuple[str, ...]] = field(default_factory=dict)
    commands: list[list[str]] = field(default_factory=list)
    # How ``imagetools create`` stores a target: copy the source as is, or (broken) store another digest.
    create_result: tuple[str, ...] | None = None

    def resolve(self, reference: str) -> tuple[str, ...] | None:
        if "@" in reference:
            return (reference.rsplit("@", 1)[1],)
        if reference in self.tags:
            return self.tags[reference]
        if ":cuda" in reference and "-kit" not in reference:
            # Every stock source tag exists; promotion targets are only created by ``imagetools create``.
            source_tag = reference.rsplit(":", 1)[1]
            if "sha-" in source_tag:
                return (_stock_digest(reference),)
        return None

    def run(self, cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.commands.append(list(cmd))
        if cmd[:4] == ["docker", "buildx", "imagetools", "inspect"]:
            resolved = self.resolve(cmd[-1])
            if resolved is None:
                return subprocess.CompletedProcess(cmd, 1, "", f"ERROR: {cmd[-1]}: not found")
            manifest: dict[str, Any] = {"digest": resolved[0]}
            if len(resolved) > 1:
                manifest["manifests"] = [{"digest": digest} for digest in resolved[1:]]
            return subprocess.CompletedProcess(cmd, 0, json.dumps(manifest), "")
        if cmd[:4] == ["docker", "buildx", "imagetools", "create"]:
            targets = [cmd[index + 1] for index, part in enumerate(cmd) if part == "-t"]
            source = self.resolve(cmd[-1])
            assert source is not None
            for target in targets:
                self.tags[target] = self.create_result or source
            return subprocess.CompletedProcess(cmd, 0, "", "")
        raise AssertionError(f"unexpected command {cmd}")

    @property
    def creates(self) -> list[list[str]]:
        return [cmd for cmd in self.commands if cmd[3] == "create"]


@pytest.fixture
def registry(monkeypatch: pytest.MonkeyPatch) -> FakeRegistry:
    """Replace Docker with a fake registry."""
    fake = FakeRegistry()
    monkeypatch.delenv(KIT_IMAGE_TAG_ENV, raising=False)
    monkeypatch.setattr(promote_image_tags, "ensure_buildx", lambda: None)
    monkeypatch.setattr(promote_image_tags.subprocess, "run", fake.run)
    return fake


def _promote(*extra: str) -> Any:
    return CliRunner().invoke(
        promote_image_tags.cli,
        ["--source-tag", "sha-abc1234", "--target-tag", "0.3.1", "--cuda-version", "12.6", *extra],
    )


def _kit_create(spec: Any, target_tag: str = "0.3.1") -> list[str]:
    return [
        "docker",
        "buildx",
        "imagetools",
        "create",
        "-t",
        f"{REPO}/{spec.image_name}:{target_tag}",
        f"{REPO}/{spec.image_name}@{KIT_IMAGE_DIGESTS[spec.image_name]}",
    ]


def test_promotion_tags_each_kit_image_digest(registry: FakeRegistry) -> None:
    result = _promote()

    assert result.exit_code == 0, result.output
    kit_creates = [cmd for cmd in registry.creates if "-kit" in cmd[-1]]
    assert kit_creates == [_kit_create(spec) for spec in KIT_IMAGE_SPECS]
    for spec in KIT_IMAGE_SPECS:
        assert registry.tags[f"{REPO}/{spec.image_name}:0.3.1"] == (KIT_IMAGE_DIGESTS[spec.image_name],)
    # The stock images are still promoted, from the source tag's digest, before the kit images.
    stock_creates = registry.creates[: -len(KIT_IMAGE_SPECS)]
    assert stock_creates
    assert all("-kit" not in cmd[-1] for cmd in stock_creates)


def test_stock_images_are_promoted_from_the_digest_the_source_tag_resolved_to(registry: FakeRegistry) -> None:
    result = _promote("--skip-kit-images")

    assert result.exit_code == 0, result.output
    expected_specs = [spec for spec in (BASE_IMAGE_SPEC, *MODEL_IMAGE_SPECS) if "12.6" in get_supported_cuda(spec)]
    assert len(registry.creates) == len(expected_specs)
    for spec, cmd in zip(expected_specs, registry.creates, strict=True):
        source_tag_reference = f"{REPO}/{spec.image_name}:cuda12.6-sha-abc1234"
        digest = _stock_digest(source_tag_reference)
        assert cmd[-1] == f"{REPO}/{spec.image_name}@{digest}"
        assert cmd[4:-1] == ["-t", f"{REPO}/{spec.image_name}:cuda12.6-0.3.1", "-t", f"{REPO}/{spec.image_name}:0.3.1"]
        assert registry.tags[f"{REPO}/{spec.image_name}:0.3.1"] == (digest,)


def test_stock_promotion_fails_when_a_target_does_not_serve_the_source(registry: FakeRegistry) -> None:
    registry.create_result = (OTHER_DIGEST,)

    result = _promote("--skip-kit-images")

    assert result.exit_code == 1
    assert "after the promotion" in result.output
    assert OTHER_DIGEST in result.output
    assert len(registry.creates) == 1  # stops at the first bad promotion


def test_stock_promotion_accepts_an_index_wrapping_the_source_manifest(
    monkeypatch: pytest.MonkeyPatch, registry: FakeRegistry
) -> None:
    """``imagetools create`` wraps a single-platform manifest in a new index; the source is then the index's child."""
    original_run = registry.run

    def wrapping_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if cmd[3] == "create":
            registry.create_result = ("sha256:" + "a" * 64, cmd[-1].rsplit("@", 1)[1])
        return original_run(cmd, **kwargs)

    monkeypatch.setattr(promote_image_tags.subprocess, "run", wrapping_run)
    result = _promote("--skip-kit-images")

    assert result.exit_code == 0, result.output


def test_missing_stock_source_is_refused(monkeypatch: pytest.MonkeyPatch, registry: FakeRegistry) -> None:
    monkeypatch.setattr(registry, "resolve", lambda reference: None)

    result = _promote("--skip-kit-images")

    assert result.exit_code == 1
    assert "does not exist" in result.output
    assert registry.creates == []


def test_skip_kit_images_promotes_only_the_stock_images(registry: FakeRegistry) -> None:
    result = _promote("--skip-kit-images")

    assert result.exit_code == 0, result.output
    assert registry.creates
    assert not any("-kit" in part for cmd in registry.commands for part in cmd)


def test_kit_images_only_tags_the_pinned_digests_without_a_source_tag(registry: FakeRegistry) -> None:
    """The full-release workflow job: only the kit images, no stock source tag or CUDA version needed."""
    result = CliRunner().invoke(promote_image_tags.cli, ["--target-tag", "0.4.3", "--kit-images-only"])

    assert result.exit_code == 0, result.output
    assert registry.creates == [_kit_create(spec, "0.4.3") for spec in KIT_IMAGE_SPECS]


@pytest.mark.parametrize(
    "args,message",
    [
        (["--target-tag", "0.4.3"], "--source-tag is required"),
        (["--target-tag", "0.4.3", "--kit-images-only", "--skip-kit-images"], "exclude each other"),
    ],
)
def test_promotion_option_errors(registry: FakeRegistry, args: list[str], message: str) -> None:
    result = CliRunner().invoke(promote_image_tags.cli, args)

    assert result.exit_code == 2
    assert message in result.output
    assert registry.commands == []


def test_existing_kit_tag_at_the_pinned_digest_is_left_alone(registry: FakeRegistry) -> None:
    for spec in KIT_IMAGE_SPECS:
        registry.tags[f"{REPO}/{spec.image_name}:0.3.1"] = ("sha256:" + "b" * 64, KIT_IMAGE_DIGESTS[spec.image_name])

    result = CliRunner().invoke(promote_image_tags.cli, ["--target-tag", "0.3.1", "--kit-images-only"])

    assert result.exit_code == 0, result.output
    assert registry.creates == []
    assert "already points at" in result.output


def test_kit_tag_at_another_digest_is_refused_before_anything_is_pushed(registry: FakeRegistry) -> None:
    """Rerunning a promotion for an existing release from a checkout with newer pins used to move its kit tag."""
    spec = KIT_IMAGE_SPECS[-1]
    target = f"{REPO}/{spec.image_name}:0.3.1"
    registry.tags[target] = (OTHER_DIGEST,)

    result = _promote()

    assert result.exit_code == 1
    assert f"{target} already exists at {OTHER_DIGEST}" in result.output
    assert "--force-kit-tags" in result.output
    assert registry.creates == []  # not even the stock images
    assert registry.tags[target] == (OTHER_DIGEST,)


def test_force_kit_tags_moves_a_kit_tag(registry: FakeRegistry) -> None:
    spec = KIT_IMAGE_SPECS[0]
    target = f"{REPO}/{spec.image_name}:0.3.1"
    registry.tags[target] = (OTHER_DIGEST,)

    result = CliRunner().invoke(
        promote_image_tags.cli, ["--target-tag", "0.3.1", "--kit-images-only", "--force-kit-tags"]
    )

    assert result.exit_code == 0, result.output
    assert registry.tags[target] == (KIT_IMAGE_DIGESTS[spec.image_name],)


def test_kit_promotion_fails_when_the_tag_does_not_serve_the_pinned_digest(registry: FakeRegistry) -> None:
    registry.create_result = (OTHER_DIGEST,)

    result = CliRunner().invoke(promote_image_tags.cli, ["--target-tag", "0.3.1", "--kit-images-only"])

    assert result.exit_code == 1
    assert f"not to {KIT_IMAGE_DIGESTS[KIT_IMAGE_SPECS[0].image_name]}" in result.output


def test_registry_errors_other_than_not_found_are_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    def failing_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 1, "", "ERROR: unauthorized: authentication required")

    monkeypatch.setattr(promote_image_tags.subprocess, "run", failing_run)
    with pytest.raises(promote_image_tags.PromotionError, match="unauthorized"):
        promote_image_tags.manifest_digests(f"{REPO}/boileroom-protenix-kit:0.3.1")


INSPECTED = f"{REPO}/boileroom-protenix-kit:0.3.1"


@pytest.mark.parametrize(
    "output",
    [
        f"ERROR: {INSPECTED}: not found",
        f"ERROR: failed to resolve {INSPECTED}: not found",
        f"ERROR: {INSPECTED}: manifest unknown: manifest unknown",
        f"#0 resolving\nERROR: {INSPECTED}: not found\n",
    ],
)
def test_a_missing_reference_reads_as_absent(monkeypatch: pytest.MonkeyPatch, output: str) -> None:
    monkeypatch.setattr(
        promote_image_tags.subprocess, "run", lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 1, "", output)
    )
    assert promote_image_tags.manifest_digests(INSPECTED) is None


@pytest.mark.parametrize(
    "output",
    [
        'ERROR: error getting credentials - err: exec: "docker-credential-desktop": executable file not found in $PATH',
        "ERROR: failed to authorize: failed to fetch oauth token: unexpected status: 404 Not Found",
        "ERROR: repository docker.io/jakublala/boileroom-protenix-kit not found: requested access is denied",
        f"ERROR: {REPO}/boileroom-esmfold2-kit:0.3.1: not found",
        f"ERROR: {INSPECTED}-rc: not found",
        "ERROR: manifest unknown",
    ],
    ids=["credential-helper", "token-404", "repository-auth", "other-image", "other-tag", "unnamed-manifest"],
)
def test_inspect_failures_that_only_mention_not_found_are_raised(monkeypatch: pytest.MonkeyPatch, output: str) -> None:
    """Any "not found" in the inspect output used to read as an absent tag, so promote pushed over an unread one."""
    monkeypatch.setattr(
        promote_image_tags.subprocess, "run", lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 1, "", output)
    )
    with pytest.raises(promote_image_tags.PromotionError, match="Could not inspect"):
        promote_image_tags.manifest_digests(INSPECTED)


def test_an_unreadable_kit_target_fails_the_promotion_before_any_push(
    monkeypatch: pytest.MonkeyPatch, registry: FakeRegistry
) -> None:
    """The --force-kit-tags guard must see the target; a credential failure no longer reads as "no tag, push"."""
    credential_error = 'error getting credentials - err: exec: "docker-credential-desktop": executable file not found'
    answer = registry.run

    def run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if cmd[3] == "inspect" and "-kit:" in cmd[-1]:
            registry.commands.append(list(cmd))
            return subprocess.CompletedProcess(cmd, 1, "", f"ERROR: {credential_error}")
        return answer(cmd, **kwargs)

    monkeypatch.setattr(promote_image_tags.subprocess, "run", run)
    result = CliRunner().invoke(promote_image_tags.cli, ["--target-tag", "0.3.1", "--kit-images-only"])

    assert result.exit_code == 1
    assert credential_error in result.output
    assert registry.creates == []


def test_kit_tag_override_is_refused_before_anything_is_pushed(
    monkeypatch: pytest.MonkeyPatch, registry: FakeRegistry
) -> None:
    """With BOILEROOM_KIT_IMAGE_TAG set the source would be a tag, not the digest runtimes pull."""
    monkeypatch.setenv(KIT_IMAGE_TAG_ENV, "sha-dc652b0")

    result = _promote()

    assert result.exit_code == 2
    assert "unset BOILEROOM_KIT_IMAGE_TAG" in result.output
    assert registry.commands == []


def test_kit_tag_override_does_not_matter_when_kit_images_are_skipped(
    monkeypatch: pytest.MonkeyPatch, registry: FakeRegistry
) -> None:
    monkeypatch.setenv(KIT_IMAGE_TAG_ENV, "sha-dc652b0")

    result = _promote("--skip-kit-images")

    assert result.exit_code == 0, result.output
