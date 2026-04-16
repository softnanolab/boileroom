"""Fast contract tests for Docker image build helpers."""

import subprocess

from scripts.images import build_model_images


def test_docker_manifest_exists_uses_docker_exit_status(monkeypatch) -> None:
    """Published-image checks should reflect the Docker manifest command status."""

    def fake_run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
        assert cmd == ["docker", "manifest", "inspect", "docker.io/example/image:tag"]
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        assert kwargs["check"] is False
        return subprocess.CompletedProcess(cmd, 0, stdout="{}", stderr="")

    monkeypatch.setattr(build_model_images.subprocess, "run", fake_run)

    assert build_model_images.docker_manifest_exists("docker.io/example/image:tag") is True


def test_docker_manifest_exists_warns_on_lookup_failure(monkeypatch, capsys) -> None:
    """Failed manifest checks should stay fail-open while leaving debugging context."""

    def fake_run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="network unavailable")

    monkeypatch.setattr(build_model_images.subprocess, "run", fake_run)

    assert build_model_images.docker_manifest_exists("docker.io/example/image:missing") is False

    captured = capsys.readouterr()
    assert "Could not confirm Docker manifest exists for docker.io/example/image:missing" in captured.err
    assert "network unavailable" in captured.err
