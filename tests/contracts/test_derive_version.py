"""Contract tests for CI/CD version derivation."""

from pathlib import Path

import pytest

from scripts.ci import derive_version


def test_main_version_uses_commit_count_after_baseline(monkeypatch) -> None:
    """Main releases should advance one patch version per commit after the baseline."""

    def fake_run_git(args: list[str]) -> str:
        assert args == ["rev-list", "--count", f"{derive_version.MAIN_VERSION_BASE_SHA}..HEAD"]
        return "7"

    monkeypatch.setattr(derive_version, "run_git", fake_run_git)

    assert derive_version.main_version() == "0.3.7"


def test_github_output_includes_pep440_version(tmp_path) -> None:
    """GitHub Actions should receive the package-compatible version."""
    output_path = tmp_path / "github-output.txt"

    derive_version.write_github_output(output_path, "0.3.2")

    assert output_path.read_text(encoding="utf-8") == "pep440=0.3.2\n"


def test_version_from_release_tag_accepts_plain_version_tag() -> None:
    """Release tags should become package-compatible versions."""
    assert derive_version.version_from_release_tag("0.3.9") == "0.3.9"
    assert derive_version.version_from_release_tag("refs/tags/0.3.10") == "0.3.10"


@pytest.mark.parametrize("tag", ["", "main", "refs/heads/main", "v0.3.1", "0.3", "0.3.x"])
def test_version_from_release_tag_rejects_invalid_inputs(tag: str) -> None:
    """Invalid release tags should fail fast."""
    with pytest.raises(ValueError, match="Expected a release tag like 0.3.x"):
        derive_version.version_from_release_tag(tag)


def test_write_pyproject_version_replaces_static_project_version(tmp_path) -> None:
    """PyPI builds should be able to inject the release tag version."""
    pyproject_path = tmp_path / "pyproject.toml"
    pyproject_path.write_text('[project]\nname = "boileroom"\nversion = "0.3.0"\n', encoding="utf-8")

    derive_version.write_pyproject_version(pyproject_path, "0.3.4")

    assert pyproject_path.read_text(encoding="utf-8") == '[project]\nname = "boileroom"\nversion = "0.3.4"\n'


def test_write_pyproject_version_rejects_missing_version_field(tmp_path) -> None:
    """Pyproject rewrites should fail when there is no version field to replace."""
    pyproject_path = tmp_path / "pyproject.toml"
    pyproject_path.write_text('[project]\nname = "boileroom"\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Expected exactly one project version line"):
        derive_version.write_pyproject_version(pyproject_path, "0.3.4")


def test_write_github_output_writes_pep440_value(tmp_path) -> None:
    """GitHub Actions output should receive the pep440 value verbatim."""
    output_path = tmp_path / "github-output.txt"

    derive_version.write_github_output(output_path, "not-a-version")

    assert output_path.read_text(encoding="utf-8") == "pep440=not-a-version\n"


def test_write_github_output_propagates_write_errors(monkeypatch, tmp_path) -> None:
    """GitHub output writes should surface filesystem errors."""
    output_path = tmp_path / "github-output.txt"

    def fake_open(*args: object, **kwargs: object) -> None:
        raise PermissionError("blocked")

    monkeypatch.setattr(Path, "open", fake_open)

    with pytest.raises(PermissionError, match="blocked"):
        derive_version.write_github_output(output_path, "0.3.2")
