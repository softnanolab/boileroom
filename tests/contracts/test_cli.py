"""Contract tests for repo-local Click command boundaries."""

from pathlib import Path

import pytest
from click.testing import CliRunner
from pytest import MonkeyPatch

from scripts.ci import derive_version
from scripts.harness import check_repo
from scripts.images import (
    build_model_images,
    check_model_imports,
    check_model_server_health,
    cleanup_dockerhub_tags,
    promote_image_tags,
)


def test_click_commands_support_help_aliases() -> None:
    """Converted maintenance commands should support both long and short help flags."""

    commands = (
        check_repo.cli,
        derive_version.cli,
        promote_image_tags.cli,
        cleanup_dockerhub_tags.cli,
        check_model_imports.cli,
        check_model_server_health.cli,
        build_model_images.cli,
    )
    runner = CliRunner()

    for command in commands:
        for help_flag in ("--help", "-h"):
            result = runner.invoke(command, [help_flag])
            assert result.exit_code == 0, result.output
            assert "Usage:" in result.output


def test_promote_cli_requires_source_and_target_tags() -> None:
    """Parser-level required options should still fail before Docker is touched."""

    result = CliRunner().invoke(promote_image_tags.cli, [])

    assert result.exit_code == 2
    assert "Missing option '--source-tag'" in result.output


def test_derive_version_cli_passes_path_options(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """Path-valued Click options should reach the plain runner as Path objects."""

    captured: list[derive_version.VersionOptions] = []

    def fake_run_derive_version(options: derive_version.VersionOptions) -> None:
        captured.append(options)

    monkeypatch.setattr(derive_version, "run_derive_version", fake_run_derive_version)

    base_pyproject = tmp_path / "pyproject.toml"
    write_pyproject = tmp_path / "release-pyproject.toml"
    github_output = tmp_path / "github-output.txt"
    result = CliRunner().invoke(
        derive_version.cli,
        [
            "--base-pyproject",
            str(base_pyproject),
            "--write-pyproject",
            str(write_pyproject),
            "--github-output",
            str(github_output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured == [
        derive_version.VersionOptions(
            head_ref="HEAD",
            release_tag=None,
            base_pyproject=base_pyproject,
            write_pyproject=write_pyproject,
            github_output=github_output,
        )
    ]
    assert isinstance(captured[0].base_pyproject, Path)
    assert isinstance(captured[0].write_pyproject, Path)
    assert isinstance(captured[0].github_output, Path)


def test_image_check_clis_forward_model_selection(monkeypatch: MonkeyPatch) -> None:
    """Per-model matrix jobs should select exactly one smoke-check target."""
    import_options: list[check_model_imports.ImportCheckOptions] = []
    health_options: list[check_model_server_health.HealthCheckOptions] = []
    monkeypatch.setattr(check_model_imports, "run_import_checks", import_options.append)
    monkeypatch.setattr(check_model_server_health, "run_server_health_checks", health_options.append)

    import_result = CliRunner().invoke(check_model_imports.cli, ["--model=esm3", "--pull"])
    health_result = CliRunner().invoke(check_model_server_health.cli, ["--model=chai", "--cleanup"])

    assert import_result.exit_code == 0, import_result.output
    assert health_result.exit_code == 0, health_result.output
    assert import_options[0].model_keys == ["esm3"]
    assert import_options[0].pull is True
    assert health_options[0].model_keys == ["chai"]
    assert health_options[0].cleanup is True


def test_image_import_cleanup_runs_after_check_failure(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """Import smoke cleanup should run even when the check body fails."""
    removed: list[str] = []

    def fail_check(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("failed")

    monkeypatch.setattr(check_model_imports, "_remove_image", removed.append)
    monkeypatch.setattr(check_model_imports, "_check_image", fail_check)

    with pytest.raises(RuntimeError, match="failed"):
        check_model_imports.check_image("esm", "example/esm:test", tmp_path, tmp_path, False, cleanup=True)

    assert removed == ["example/esm:test"]


def test_server_health_cleanup_runs_after_check_failure(monkeypatch: MonkeyPatch) -> None:
    """Server smoke cleanup should run even when the check body fails."""
    removed: list[str] = []

    def fail_check(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("failed")

    monkeypatch.setattr(check_model_server_health, "_remove_image", removed.append)
    monkeypatch.setattr(check_model_server_health, "_check_server_health", fail_check)

    with pytest.raises(RuntimeError, match="failed"):
        check_model_server_health.check_server_health("esm", "example/esm:test", False, 1.0, cleanup=True)

    assert removed == ["example/esm:test"]


def test_promote_cli_validates_cuda_selection_before_buildx(monkeypatch: MonkeyPatch) -> None:
    """Usage errors should be reported before external Docker checks."""

    buildx_checked = False

    def fake_ensure_buildx() -> None:
        nonlocal buildx_checked
        buildx_checked = True

    monkeypatch.setattr(promote_image_tags, "ensure_buildx", fake_ensure_buildx)

    result = CliRunner().invoke(
        promote_image_tags.cli,
        ["--source-tag", "sha-test", "--target-tag", "0.3.0"],
    )

    assert result.exit_code == 2
    assert "Specify at least one --cuda-version or use --all-cuda." in result.output
    assert buildx_checked is False
