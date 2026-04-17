"""Fast contract tests for Docker image build helpers."""

import sys
from argparse import Namespace
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


def test_build_cache_reference_uses_repository_env(monkeypatch) -> None:
    """Build cache tags should follow the same repository override as image tags."""
    monkeypatch.setenv("BOILEROOM_DOCKER_REPOSITORY", "docker.io/example")
    assert build_model_images.build_cache_reference("boileroom-base", "12.6") == (
        "docker.io/example/boileroom-base:buildcache-cuda12.6"
    )


def test_parse_args_exposes_skip_controls(monkeypatch) -> None:
    """The build helper should accept the documented skip flags."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_model_images.py",
            "--skip-existing",
            "--force-rebuild",
            "--verbose",
        ],
    )

    args = build_model_images.parse_args()
    assert args.skip_existing is True
    assert args.force_rebuild is True
    assert args.verbose is True


def test_build_base_verbose_echoes_plain_progress(monkeypatch, tmp_path) -> None:
    """Verbose builds should print command output and request plain BuildKit progress."""
    calls: list[tuple[list[str], Path | None, bool]] = []

    def fake_run(cmd: list[str], log_file: Path | None = None, echo: bool = True) -> None:
        calls.append((cmd, log_file, echo))

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(build_model_images, "run", fake_run)

    build_model_images.build_base(
        "12.6",
        "sha-test",
        "linux/amd64",
        "--load",
        no_cache=False,
        use_local_docker_build=False,
        verbose=True,
    )

    cmd, log_file, echo = calls[0]
    assert cmd[cmd.index("--progress") : cmd.index("--progress") + 2] == ["--progress", "plain"]
    assert log_file is not None
    assert echo is True


def test_main_skips_existing_base_and_model_tags(monkeypatch, tmp_path) -> None:
    """The main build flow should skip existing tags when skip-existing is set."""
    args = Namespace(
        tag="sha-test",
        cuda_versions=["12.6"],
        all_cuda=False,
        platform="linux/amd64",
        push=False,
        load=False,
        no_cache=False,
        verbose=False,
        skip_existing=True,
        force_rebuild=False,
        max_workers=1,
    )

    built_tasks: list[tuple[str, str]] = []
    checked_refs: list[str] = []
    built_bases: list[str] = []

    def fake_parse_args() -> Namespace:
        return args

    def fake_ensure_docker() -> None:
        return None

    def fake_image_reference_exists(image_reference: str) -> bool:
        checked_refs.append(image_reference)
        return image_reference.endswith("boileroom-base:cuda12.6-sha-test") or image_reference.endswith(
            "boileroom-boltz:cuda12.6-sha-test"
        )

    def fake_build_base(cuda_version: str, tag: str, *_args, **_kwargs) -> str:
        built_bases.append(f"{cuda_version}:{tag}")
        return f"docker.io/jakublala/boileroom-base:cuda{cuda_version}-{tag}"

    def fake_build_model(task, *_args, **_kwargs):
        built_tasks.append((task.image_spec.image_name, task.base_image_reference))
        return (f"{task.image_spec.image_name}:built",)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(build_model_images, "parse_args", fake_parse_args)
    monkeypatch.setattr(build_model_images, "ensure_docker", fake_ensure_docker)
    monkeypatch.setattr(build_model_images, "ensure_buildx_builder", lambda: None)
    monkeypatch.setattr(build_model_images, "image_reference_exists", fake_image_reference_exists)
    monkeypatch.setattr(build_model_images, "build_base", fake_build_base)
    monkeypatch.setattr(build_model_images, "build_model", fake_build_model)
    monkeypatch.setattr(build_model_images, "log_info", lambda *args, **kwargs: None)
    monkeypatch.setattr(build_model_images, "log_warn", lambda *args, **kwargs: None)
    monkeypatch.setattr(build_model_images, "log_success", lambda *args, **kwargs: None)

    build_model_images.main()

    assert built_bases == []
    assert built_tasks == [
        ("boileroom-chai1", "docker.io/jakublala/boileroom-base:cuda12.6-sha-test"),
        ("boileroom-esm", "docker.io/jakublala/boileroom-base:cuda12.6-sha-test"),
    ]
    assert checked_refs == [
        "docker.io/jakublala/boileroom-base:cuda12.6-sha-test",
        "docker.io/jakublala/boileroom-boltz:cuda12.6-sha-test",
        "docker.io/jakublala/boileroom-chai1:cuda12.6-sha-test",
        "docker.io/jakublala/boileroom-esm:cuda12.6-sha-test",
    ]
