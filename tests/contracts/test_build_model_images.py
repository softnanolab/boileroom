"""Fast contract tests for Docker image build helpers."""

from pathlib import Path

from scripts.images import build_model_images


def test_build_base_push_uses_registry_cache(monkeypatch, tmp_path) -> None:
    """Pushed buildx builds should import and export stable registry cache layers."""
    calls: list[tuple[list[str], Path | None, bool]] = []

    def fake_run(cmd: list[str], log_file: Path | None = None, echo: bool = True) -> None:
        calls.append((cmd, log_file, echo))

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(build_model_images, "run", fake_run)

    build_model_images.build_base(
        "12.6",
        "sha-test",
        "linux/amd64",
        "--push",
        no_cache=False,
        use_local_docker_build=False,
    )

    cmd, _, _ = calls[0]
    assert "--cache-from" in cmd
    assert "type=registry,ref=docker.io/jakublala/boileroom-base:buildcache-cuda12.6" in cmd
    assert "--cache-to" in cmd
    assert "type=registry,ref=docker.io/jakublala/boileroom-base:buildcache-cuda12.6,mode=max" in cmd


def test_build_base_no_cache_disables_registry_cache(monkeypatch, tmp_path) -> None:
    """Explicit no-cache builds should not import or export BuildKit registry cache."""
    calls: list[tuple[list[str], Path | None, bool]] = []

    def fake_run(cmd: list[str], log_file: Path | None = None, echo: bool = True) -> None:
        calls.append((cmd, log_file, echo))

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(build_model_images, "run", fake_run)

    build_model_images.build_base(
        "12.6",
        "sha-test",
        "linux/amd64",
        "--push",
        no_cache=True,
        use_local_docker_build=False,
    )

    cmd, _, _ = calls[0]
    assert "--no-cache" in cmd
    assert "--cache-from" not in cmd
    assert "--cache-to" not in cmd


def test_build_model_push_uses_registry_cache(monkeypatch, tmp_path) -> None:
    """Pushed model buildx builds should import and export stable registry cache layers."""
    calls: list[tuple[list[str], Path | None, bool]] = []

    def fake_run(cmd: list[str], log_file: Path | None = None, echo: bool = True) -> None:
        calls.append((cmd, log_file, echo))

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(build_model_images, "run", fake_run)

    build_model_images.build_model(
        build_model_images.BuildTask(
            cuda_version="12.6",
            image_spec=build_model_images.MODEL_IMAGE_SPECS[0],
            base_image_reference="docker.io/jakublala/boileroom-base:cuda12.6-sha-test",
            tag="sha-test",
        ),
        "linux/amd64",
        "--push",
        no_cache=False,
        use_local_docker_build=False,
    )

    cmd, _, _ = calls[0]
    assert "--cache-from" in cmd
    assert "type=registry,ref=docker.io/jakublala/boileroom-boltz:buildcache-cuda12.6" in cmd
    assert "--cache-to" in cmd
    assert "type=registry,ref=docker.io/jakublala/boileroom-boltz:buildcache-cuda12.6,mode=max" in cmd


def test_build_model_no_cache_disables_registry_cache(monkeypatch, tmp_path) -> None:
    """Explicit no-cache model builds should not import or export BuildKit registry cache."""
    calls: list[tuple[list[str], Path | None, bool]] = []

    def fake_run(cmd: list[str], log_file: Path | None = None, echo: bool = True) -> None:
        calls.append((cmd, log_file, echo))

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(build_model_images, "run", fake_run)

    build_model_images.build_model(
        build_model_images.BuildTask(
            cuda_version="12.6",
            image_spec=build_model_images.MODEL_IMAGE_SPECS[0],
            base_image_reference="docker.io/jakublala/boileroom-base:cuda12.6-sha-test",
            tag="sha-test",
        ),
        "linux/amd64",
        "--push",
        no_cache=True,
        use_local_docker_build=False,
    )

    cmd, _, _ = calls[0]
    assert "--no-cache" in cmd
    assert "--cache-from" not in cmd
    assert "--cache-to" not in cmd
