"""Fast contract tests for Docker image build helpers."""

from pathlib import Path

from boileroom.images.metadata import DEFAULT_CUDA_VERSION
import pytest
from click.testing import CliRunner
from pytest import CaptureFixture, MonkeyPatch
from scripts.images import build_model_images


def test_build_base_push_uses_registry_cache(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """Pushed buildx builds should import and export stable registry cache layers."""
    calls: list[tuple[list[str], Path | None, bool]] = []

    def fake_run(cmd: list[str], log_file: Path | None = None, echo: bool = True) -> None:
        calls.append((cmd, log_file, echo))

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(build_model_images, "run", fake_run)

    build_model_images.build_base(
        DEFAULT_CUDA_VERSION,
        "sha-test",
        "docker.io/jakublala",
        "linux/amd64",
        "--push",
        no_cache=False,
        use_local_docker_build=False,
    )

    cmd, _, _ = calls[0]
    assert "--cache-from" in cmd
    assert f"type=registry,ref=docker.io/jakublala/boileroom-base:buildcache-cuda{DEFAULT_CUDA_VERSION}" in cmd
    assert "--cache-to" in cmd
    assert f"type=registry,ref=docker.io/jakublala/boileroom-base:buildcache-cuda{DEFAULT_CUDA_VERSION},mode=max" in cmd


def test_build_base_no_cache_disables_registry_cache(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """Explicit no-cache builds should not import or export BuildKit registry cache."""
    calls: list[tuple[list[str], Path | None, bool]] = []

    def fake_run(cmd: list[str], log_file: Path | None = None, echo: bool = True) -> None:
        calls.append((cmd, log_file, echo))

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(build_model_images, "run", fake_run)

    build_model_images.build_base(
        DEFAULT_CUDA_VERSION,
        "sha-test",
        "docker.io/jakublala",
        "linux/amd64",
        "--push",
        no_cache=True,
        use_local_docker_build=False,
    )

    cmd, _, _ = calls[0]
    assert "--no-cache" in cmd
    assert "--cache-from" not in cmd
    assert "--cache-to" not in cmd


def test_build_cache_reference_uses_explicit_repository() -> None:
    """Build cache tags should follow the requested repository."""
    assert build_model_images.build_cache_reference("example", "boileroom-base", DEFAULT_CUDA_VERSION) == (
        f"docker.io/example/boileroom-base:buildcache-cuda{DEFAULT_CUDA_VERSION}"
    )


def test_cli_exposes_documented_flags(monkeypatch: MonkeyPatch) -> None:
    """The build helper should accept the documented flags."""

    captured: list[build_model_images.BuildOptions] = []

    def fake_run_build(options: build_model_images.BuildOptions) -> None:
        captured.append(options)

    monkeypatch.setattr(build_model_images, "run_build", fake_run_build)

    result = CliRunner().invoke(
        build_model_images.cli,
        [
            "--cuda-version",
            "12.6",
            "--cuda-version",
            "11.8",
            "--skip-existing",
            "--force-rebuild",
            "--verbose",
            "--local-base",
            "--docker-user=example",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured == [
        build_model_images.BuildOptions(
            tag=None,
            docker_user="example",
            cuda_versions=["12.6", "11.8"],
            all_cuda=False,
            platform="linux/amd64",
            push=False,
            load=False,
            no_cache=False,
            verbose=True,
            skip_existing=True,
            force_rebuild=True,
            max_workers=1,
            local_base=True,
        )
    ]


def test_cli_rejects_unknown_flags(monkeypatch: MonkeyPatch) -> None:
    """Invalid CLI options should fail before invoking the build workflow."""

    captured: list[build_model_images.BuildOptions] = []
    monkeypatch.setattr(build_model_images, "run_build", captured.append)

    result = CliRunner().invoke(build_model_images.cli, ["--definitely-not-a-real-option"])

    assert result.exit_code != 0
    assert captured == []
    assert "No such option" in result.output


def test_run_build_validates_cuda_selection_before_docker(
    monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]
) -> None:
    """Semantic option errors should not be masked by missing Docker."""

    options = build_model_images.BuildOptions(
        tag="sha-test",
        docker_user="docker.io/jakublala",
        cuda_versions=None,
        all_cuda=False,
        platform="linux/amd64",
        push=False,
        load=False,
        no_cache=False,
        verbose=False,
        skip_existing=False,
        force_rebuild=False,
        max_workers=1,
        local_base=False,
    )
    docker_checked = False

    def fake_ensure_docker() -> None:
        nonlocal docker_checked
        docker_checked = True

    monkeypatch.setattr(build_model_images, "ensure_docker", fake_ensure_docker)

    with pytest.raises(SystemExit) as exc_info:
        build_model_images.run_build(options)

    assert exc_info.value.code == 1
    assert docker_checked is False
    assert "Specify at least one --cuda-version or use --all-cuda." in capsys.readouterr().err


def test_build_base_verbose_echoes_plain_progress(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """Verbose builds should print command output and request plain BuildKit progress."""
    calls: list[tuple[list[str], Path | None, bool]] = []

    def fake_run(cmd: list[str], log_file: Path | None = None, echo: bool = True) -> None:
        calls.append((cmd, log_file, echo))

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(build_model_images, "run", fake_run)

    build_model_images.build_base(
        DEFAULT_CUDA_VERSION,
        "sha-test",
        "docker.io/jakublala",
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


def test_build_base_local_base_uses_buildx_cache_load_then_push(monkeypatch, tmp_path) -> None:
    """Local-base builds should keep BuildKit cache while loading tags locally."""
    calls: list[tuple[list[str], Path | None, bool]] = []

    def fake_run(cmd: list[str], log_file: Path | None = None, echo: bool = True) -> None:
        calls.append((cmd, log_file, echo))

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(build_model_images, "run", fake_run)

    build_model_images.build_base(
        DEFAULT_CUDA_VERSION,
        "sha-test",
        "docker.io/jakublala",
        "linux/amd64",
        "--push",
        no_cache=False,
        use_local_docker_build=False,
        push_after_build=True,
    )

    build_cmd = calls[0][0]
    push_cmds = [call[0] for call in calls[1:]]
    assert build_cmd[:3] == ["docker", "buildx", "build"]
    assert "--load" in build_cmd
    assert "--push" not in build_cmd
    assert "--cache-from" in build_cmd
    assert "--cache-to" in build_cmd
    assert push_cmds == [
        ["docker", "push", f"docker.io/jakublala/boileroom-base:cuda{DEFAULT_CUDA_VERSION}-sha-test"],
        ["docker", "push", "docker.io/jakublala/boileroom-base:sha-test"],
    ]


def test_build_model_local_base_uses_buildx_cache_context_then_push(monkeypatch, tmp_path) -> None:
    """Local-base model builds should expose the loaded base image to buildx."""
    calls: list[tuple[list[str], Path | None, bool]] = []

    def fake_run(cmd: list[str], log_file: Path | None = None, echo: bool = True) -> None:
        calls.append((cmd, log_file, echo))

    model_spec = build_model_images.MODEL_IMAGE_SPECS[0]
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(build_model_images, "run", fake_run)

    base_image_reference = f"docker.io/jakublala/boileroom-base:cuda{DEFAULT_CUDA_VERSION}-sha-test"
    build_model_images.build_model(
        build_model_images.BuildTask(
            cuda_version=DEFAULT_CUDA_VERSION,
            image_spec=model_spec,
            base_image_reference=base_image_reference,
            docker_repository="docker.io/jakublala",
            tag="sha-test",
        ),
        "linux/amd64",
        "--push",
        no_cache=False,
        use_local_docker_build=False,
        push_after_build=True,
    )

    build_cmd = calls[0][0]
    push_cmds = [call[0] for call in calls[1:]]
    assert build_cmd[:3] == ["docker", "buildx", "build"]
    assert "--load" in build_cmd
    assert "--push" not in build_cmd
    assert "--cache-from" in build_cmd
    assert "--cache-to" in build_cmd
    assert "--build-context" in build_cmd
    assert f"{base_image_reference}=docker-image://{base_image_reference}" in build_cmd
    assert push_cmds == [
        ["docker", "push", f"docker.io/jakublala/{model_spec.image_name}:cuda{DEFAULT_CUDA_VERSION}-sha-test"],
        ["docker", "push", f"docker.io/jakublala/{model_spec.image_name}:sha-test"],
    ]


def test_main_skips_existing_base_and_model_tags(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """The main build flow should skip existing tags when skip-existing is set."""
    options = build_model_images.BuildOptions(
        tag="sha-test",
        docker_user="docker.io/jakublala",
        cuda_versions=[DEFAULT_CUDA_VERSION],
        all_cuda=False,
        platform="linux/amd64",
        push=False,
        load=False,
        no_cache=False,
        verbose=False,
        skip_existing=True,
        force_rebuild=False,
        max_workers=1,
        local_base=False,
    )

    built_tasks: list[tuple[str, str]] = []
    checked_refs: list[str] = []
    built_bases: list[str] = []

    def fake_ensure_docker() -> None:
        return None

    def fake_image_reference_exists(image_reference: str) -> bool:
        checked_refs.append(image_reference)
        existing_refs = (
            f"boileroom-base:cuda{DEFAULT_CUDA_VERSION}-sha-test",
            f"boileroom-boltz:cuda{DEFAULT_CUDA_VERSION}-sha-test",
        )
        return image_reference.endswith(existing_refs)

    def fake_build_base(cuda_version: str, tag: str, docker_repository: str, *_args, **_kwargs) -> str:
        built_bases.append(f"{cuda_version}:{tag}")
        return f"{docker_repository}/boileroom-base:cuda{cuda_version}-{tag}"

    def fake_build_model(task, *_args, **_kwargs):
        built_tasks.append((task.image_spec.image_name, task.base_image_reference))
        return (f"{task.image_spec.image_name}:built",)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(build_model_images, "ensure_docker", fake_ensure_docker)
    monkeypatch.setattr(build_model_images, "ensure_buildx_builder", lambda: None)
    monkeypatch.setattr(build_model_images, "image_reference_exists", fake_image_reference_exists)
    monkeypatch.setattr(build_model_images, "build_base", fake_build_base)
    monkeypatch.setattr(build_model_images, "build_model", fake_build_model)
    monkeypatch.setattr(build_model_images, "log_info", lambda *args, **kwargs: None)
    monkeypatch.setattr(build_model_images, "log_warn", lambda *args, **kwargs: None)
    monkeypatch.setattr(build_model_images, "log_success", lambda *args, **kwargs: None)

    build_model_images.run_build(options)

    assert built_bases == []
    assert built_tasks == [
        ("boileroom-chai1", f"docker.io/jakublala/boileroom-base:cuda{DEFAULT_CUDA_VERSION}-sha-test"),
        ("boileroom-esm", f"docker.io/jakublala/boileroom-base:cuda{DEFAULT_CUDA_VERSION}-sha-test"),
    ]
    assert checked_refs == [
        f"docker.io/jakublala/boileroom-base:cuda{DEFAULT_CUDA_VERSION}-sha-test",
        f"docker.io/jakublala/boileroom-boltz:cuda{DEFAULT_CUDA_VERSION}-sha-test",
        f"docker.io/jakublala/boileroom-chai1:cuda{DEFAULT_CUDA_VERSION}-sha-test",
        f"docker.io/jakublala/boileroom-esm:cuda{DEFAULT_CUDA_VERSION}-sha-test",
    ]


def test_local_base_push_builds_locally_then_pushes(monkeypatch, tmp_path) -> None:
    """Local-base push mode should keep FROM dependencies in the local Docker daemon."""
    options = build_model_images.BuildOptions(
        tag="sha-test",
        docker_user="docker.io/jakublala",
        cuda_versions=[DEFAULT_CUDA_VERSION],
        all_cuda=False,
        platform="linux/amd64",
        push=True,
        load=False,
        no_cache=False,
        verbose=False,
        skip_existing=False,
        force_rebuild=False,
        max_workers=1,
        local_base=True,
    )

    base_calls: list[tuple[bool, bool]] = []
    model_calls: list[tuple[bool, bool, str]] = []
    buildx_calls = 0

    def fake_ensure_buildx_builder() -> None:
        nonlocal buildx_calls
        buildx_calls += 1

    def fake_build_base(
        cuda_version: str,
        tag: str,
        docker_repository: str,
        _platform: str,
        _output_flag: str,
        _no_cache: bool,
        use_local_docker_build: bool,
        _verbose: bool,
        push_after_build: bool = False,
    ) -> str:
        base_calls.append((use_local_docker_build, push_after_build))
        return f"{docker_repository}/boileroom-base:cuda{cuda_version}-{tag}"

    def fake_build_model(
        task,
        _platform: str,
        _output_flag: str,
        _no_cache: bool,
        use_local_docker_build: bool,
        _verbose: bool,
        push_after_build: bool = False,
    ):
        model_calls.append((use_local_docker_build, push_after_build, task.base_image_reference))
        return (f"{task.image_spec.image_name}:built",)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(build_model_images, "ensure_docker", lambda: None)
    monkeypatch.setattr(build_model_images, "ensure_buildx_builder", fake_ensure_buildx_builder)
    monkeypatch.setattr(build_model_images, "build_base", fake_build_base)
    monkeypatch.setattr(build_model_images, "build_model", fake_build_model)
    monkeypatch.setattr(build_model_images, "log_info", lambda *args, **kwargs: None)
    monkeypatch.setattr(build_model_images, "log_warn", lambda *args, **kwargs: None)
    monkeypatch.setattr(build_model_images, "log_success", lambda *args, **kwargs: None)

    build_model_images.run_build(options)

    assert buildx_calls == 1
    assert base_calls == [(False, True)]
    assert model_calls == [
        (False, True, f"docker.io/jakublala/boileroom-base:cuda{DEFAULT_CUDA_VERSION}-sha-test"),
        (False, True, f"docker.io/jakublala/boileroom-base:cuda{DEFAULT_CUDA_VERSION}-sha-test"),
        (False, True, f"docker.io/jakublala/boileroom-base:cuda{DEFAULT_CUDA_VERSION}-sha-test"),
    ]
