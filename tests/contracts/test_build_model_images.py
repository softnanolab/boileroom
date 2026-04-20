"""Fast contract tests for Docker image build helpers."""

import sys
from argparse import Namespace
from pathlib import Path

from boileroom.images.metadata import DEFAULT_CUDA_VERSION
from scripts.images import build_model_images


def test_build_base_push_uses_registry_cache(monkeypatch, tmp_path) -> None:
    """Pushed buildx builds should import and export stable registry cache layers."""
    calls: list[tuple[list[str], Path | None, bool]] = []

    def fake_run(cmd: list[str], log_file: Path | None = None, echo: bool = True) -> None:
        calls.append((cmd, log_file, echo))

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(build_model_images, "run", fake_run)

    build_model_images.build_base(
        DEFAULT_CUDA_VERSION,
        "sha-test",
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


def test_build_base_no_cache_disables_registry_cache(monkeypatch, tmp_path) -> None:
    """Explicit no-cache builds should not import or export BuildKit registry cache."""
    calls: list[tuple[list[str], Path | None, bool]] = []

    def fake_run(cmd: list[str], log_file: Path | None = None, echo: bool = True) -> None:
        calls.append((cmd, log_file, echo))

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(build_model_images, "run", fake_run)

    build_model_images.build_base(
        DEFAULT_CUDA_VERSION,
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
    assert build_model_images.build_cache_reference("boileroom-base", DEFAULT_CUDA_VERSION) == (
        f"docker.io/example/boileroom-base:buildcache-cuda{DEFAULT_CUDA_VERSION}"
    )


def test_parse_args_exposes_documented_flags(monkeypatch) -> None:
    """The build helper should accept the documented flags."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_model_images.py",
            "--skip-existing",
            "--force-rebuild",
            "--verbose",
            "--local-base",
        ],
    )

    args = build_model_images.parse_args()
    assert args.skip_existing is True
    assert args.force_rebuild is True
    assert args.verbose is True
    assert args.local_base is True


def test_build_base_verbose_echoes_plain_progress(monkeypatch, tmp_path) -> None:
    """Verbose builds should print command output and request plain BuildKit progress."""
    calls: list[tuple[list[str], Path | None, bool]] = []

    def fake_run(cmd: list[str], log_file: Path | None = None, echo: bool = True) -> None:
        calls.append((cmd, log_file, echo))

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(build_model_images, "run", fake_run)

    build_model_images.build_base(
        DEFAULT_CUDA_VERSION,
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


def test_build_base_local_base_uses_daemon_builder_then_push(monkeypatch, tmp_path) -> None:
    """Local-base builds should publish tags from the daemon image store."""
    calls: list[tuple[list[str], Path | None, bool]] = []

    def fake_run(cmd: list[str], log_file: Path | None = None, echo: bool = True) -> None:
        calls.append((cmd, log_file, echo))

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(build_model_images, "run", fake_run)

    build_model_images.build_base(
        DEFAULT_CUDA_VERSION,
        "sha-test",
        "linux/amd64",
        "--push",
        no_cache=False,
        use_local_docker_build=False,
        push_after_build=True,
    )

    build_cmd = calls[0][0]
    push_cmds = [call[0] for call in calls[1:]]
    assert build_cmd[:2] == ["docker", "build"]
    assert "--load" not in build_cmd
    assert "--push" not in build_cmd
    assert "--cache-from" not in build_cmd
    assert "--cache-to" not in build_cmd
    assert push_cmds == [
        ["docker", "push", f"docker.io/jakublala/boileroom-base:cuda{DEFAULT_CUDA_VERSION}-sha-test"],
        ["docker", "push", "docker.io/jakublala/boileroom-base:sha-test"],
    ]


def test_build_model_local_base_uses_daemon_builder_then_push(monkeypatch, tmp_path) -> None:
    """Local-base model builds should resolve FROM tags from the daemon image store."""
    calls: list[tuple[list[str], Path | None, bool]] = []

    def fake_run(cmd: list[str], log_file: Path | None = None, echo: bool = True) -> None:
        calls.append((cmd, log_file, echo))

    model_spec = build_model_images.MODEL_IMAGE_SPECS[0]
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(build_model_images, "run", fake_run)

    build_model_images.build_model(
        build_model_images.BuildTask(
            cuda_version=DEFAULT_CUDA_VERSION,
            image_spec=model_spec,
            base_image_reference=f"docker.io/jakublala/boileroom-base:cuda{DEFAULT_CUDA_VERSION}-sha-test",
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
    assert build_cmd[:2] == ["docker", "build"]
    assert "--load" not in build_cmd
    assert "--push" not in build_cmd
    assert "--cache-from" not in build_cmd
    assert "--cache-to" not in build_cmd
    assert push_cmds == [
        ["docker", "push", f"docker.io/jakublala/{model_spec.image_name}:cuda{DEFAULT_CUDA_VERSION}-sha-test"],
        ["docker", "push", f"docker.io/jakublala/{model_spec.image_name}:sha-test"],
    ]


def test_main_skips_existing_base_and_model_tags(monkeypatch, tmp_path) -> None:
    """The main build flow should skip existing tags when skip-existing is set."""
    args = Namespace(
        tag="sha-test",
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

    def fake_parse_args() -> Namespace:
        return args

    def fake_ensure_docker() -> None:
        return None

    def fake_image_reference_exists(image_reference: str) -> bool:
        checked_refs.append(image_reference)
        existing_refs = (
            f"boileroom-base:cuda{DEFAULT_CUDA_VERSION}-sha-test",
            f"boileroom-boltz:cuda{DEFAULT_CUDA_VERSION}-sha-test",
        )
        return image_reference.endswith(existing_refs)

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
    args = Namespace(
        tag="sha-test",
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

    def fake_parse_args() -> Namespace:
        return args

    def fake_ensure_buildx_builder() -> None:
        nonlocal buildx_calls
        buildx_calls += 1

    def fake_build_base(
        cuda_version: str,
        tag: str,
        _platform: str,
        _output_flag: str,
        _no_cache: bool,
        use_local_docker_build: bool,
        _verbose: bool,
        push_after_build: bool = False,
    ) -> str:
        base_calls.append((use_local_docker_build, push_after_build))
        return f"docker.io/jakublala/boileroom-base:cuda{cuda_version}-{tag}"

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
    monkeypatch.setattr(build_model_images, "parse_args", fake_parse_args)
    monkeypatch.setattr(build_model_images, "ensure_docker", lambda: None)
    monkeypatch.setattr(build_model_images, "ensure_buildx_builder", fake_ensure_buildx_builder)
    monkeypatch.setattr(build_model_images, "build_base", fake_build_base)
    monkeypatch.setattr(build_model_images, "build_model", fake_build_model)
    monkeypatch.setattr(build_model_images, "log_info", lambda *args, **kwargs: None)
    monkeypatch.setattr(build_model_images, "log_warn", lambda *args, **kwargs: None)
    monkeypatch.setattr(build_model_images, "log_success", lambda *args, **kwargs: None)

    build_model_images.main()

    assert buildx_calls == 0
    assert base_calls == [(True, True)]
    assert model_calls == [
        (True, True, f"docker.io/jakublala/boileroom-base:cuda{DEFAULT_CUDA_VERSION}-sha-test"),
        (True, True, f"docker.io/jakublala/boileroom-base:cuda{DEFAULT_CUDA_VERSION}-sha-test"),
        (True, True, f"docker.io/jakublala/boileroom-base:cuda{DEFAULT_CUDA_VERSION}-sha-test"),
    ]
