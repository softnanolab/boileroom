"""Contract tests for CI/CD version derivation."""

from pathlib import Path

import pytest

from scripts.ci import derive_version


def test_main_version_uses_commit_count_after_baseline(monkeypatch) -> None:
    """Main builds should publish alpha prerelease tags for the target version."""

    def fake_run_git(args: list[str]) -> str:
        if args == ["tag", "--merged", "HEAD", "--list"]:
            return ""
        assert args == ["rev-list", "--count", f"{derive_version.MAIN_VERSION_BASE_SHA}..HEAD"]
        return "7"

    monkeypatch.setattr(derive_version, "run_git", fake_run_git)

    assert derive_version.main_version() == "0.3.0-alpha.7"


def test_main_version_counts_from_latest_stable_release_tag(monkeypatch) -> None:
    """Alpha numbers should restart after the previous stable release."""

    def fake_run_git(args: list[str]) -> str:
        if args == ["tag", "--merged", "HEAD", "--list"]:
            return "\n".join(["v0.2.2", "v0.3.0", "0.3.1", "notes"])
        assert args == ["rev-list", "--count", "v0.3.0..HEAD"]
        return "2"

    monkeypatch.setattr(derive_version, "run_git", fake_run_git)

    assert derive_version.main_version(base_version="0.3.1") == "0.3.1-alpha.2"


def test_main_version_rejects_already_released_base_version(monkeypatch) -> None:
    """Main prereleases should not continue after the target version has shipped."""

    def fake_run_git(args: list[str]) -> str:
        assert args == ["tag", "--merged", "HEAD", "--list"]
        return "\n".join(["v0.2.2", "v0.3.0", "v0.3.1"])

    monkeypatch.setattr(derive_version, "run_git", fake_run_git)

    with pytest.raises(ValueError, match=r"Bump pyproject\.version to the next release target"):
        derive_version.main_version(base_version="0.3.1")


def test_main_version_rejects_newer_reachable_stable_release(monkeypatch) -> None:
    """Main prereleases should not derive from behind a newer stable release."""

    def fake_run_git(args: list[str]) -> str:
        assert args == ["tag", "--merged", "HEAD", "--list"]
        return "\n".join(["v0.3.0", "v0.3.2"])

    monkeypatch.setattr(derive_version, "run_git", fake_run_git)

    with pytest.raises(ValueError, match=r"Bump pyproject\.version to the next release target"):
        derive_version.main_version(base_version="0.3.1")


def test_pyproject_version_reads_project_version(tmp_path) -> None:
    """The derivation script should read the static pyproject anchor."""
    pyproject_path = tmp_path / "pyproject.toml"
    pyproject_path.write_text('[project]\nname = "boileroom"\nversion = "0.4.2"\n', encoding="utf-8")

    assert derive_version.pyproject_version(pyproject_path) == "0.4.2"


def test_github_output_includes_pep440_version(tmp_path) -> None:
    """GitHub Actions should receive the package-compatible version."""
    output_path = tmp_path / "github-output.txt"

    derive_version.write_github_output(output_path, "0.3.1-alpha.2")

    assert output_path.read_text(encoding="utf-8") == "pep440=0.3.1a2\ndocker_tag=0.3.1-alpha.2\n"


def test_version_from_release_tag_accepts_plain_version_tag() -> None:
    """Release tags should become package-compatible versions."""
    assert derive_version.version_from_release_tag("v0.3.9", "0.3.0") == "0.3.9"
    assert derive_version.version_from_release_tag("refs/tags/v0.3.10", "0.3.0") == "0.3.10"


@pytest.mark.parametrize("tag", ["v0.2.1", "v0.4.1"])
def test_version_from_release_tag_rejects_invalid_inputs(tag: str) -> None:
    """Invalid release tags should fail fast."""
    with pytest.raises(ValueError, match=r"Expected a release tag like v0\.3\.x"):
        derive_version.version_from_release_tag(tag, "0.3.0")


@pytest.mark.parametrize(
    "tag",
    ["", "main", "refs/heads/main", "0.3.1", "refs/tags/0.3.1", "0.3", "0.3.x", "0.3.1-alpha.1"],
)
def test_version_from_release_tag_rejects_malformed_inputs(tag: str) -> None:
    """Malformed release tags should fail fast."""
    with pytest.raises(ValueError, match=r"Expected a release tag like v0\.3\.0"):
        derive_version.version_from_release_tag(tag, "0.3.0")


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

    derive_version.write_github_output(output_path, "0.3.2")

    assert output_path.read_text(encoding="utf-8") == "pep440=0.3.2\ndocker_tag=0.3.2\n"


def test_write_github_output_propagates_write_errors(monkeypatch, tmp_path) -> None:
    """GitHub output writes should surface filesystem errors."""
    output_path = tmp_path / "github-output.txt"

    def fake_open(*args: object, **kwargs: object) -> None:
        raise PermissionError("blocked")

    monkeypatch.setattr(Path, "open", fake_open)

    with pytest.raises(PermissionError, match="blocked"):
        derive_version.write_github_output(output_path, "0.3.2")
