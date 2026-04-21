"""Contract tests for repo-local Click command boundaries."""

from pathlib import Path

from click.testing import CliRunner

from scripts.ci import derive_version
from scripts.harness import check_repo
from scripts.images import build_model_images, check_model_imports, check_model_server_health, promote_image_tags


def test_click_commands_support_help_aliases() -> None:
    """Converted maintenance commands should support both long and short help flags."""

    commands = (
        check_repo.cli,
        derive_version.cli,
        promote_image_tags.cli,
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


def test_derive_version_cli_passes_path_options(monkeypatch, tmp_path) -> None:
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
