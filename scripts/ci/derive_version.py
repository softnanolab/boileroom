"""Derive CI/CD release and prerelease versions."""

from __future__ import annotations

import argparse
import re
import subprocess
import tomllib
from pathlib import Path

MAIN_VERSION_BASE_SHA = "48e8a23fbde63e95a0e39fe0d5748c3f338b30b3"
DEFAULT_PYPROJECT_PATH = Path(__file__).resolve().parents[2] / "pyproject.toml"
VERSION_PATTERN = re.compile(r"^(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)$")
RELEASE_TAG_PATTERN = re.compile(
    r"^(?:refs/tags/)?v(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)$"
)


def run_git(args: list[str]) -> str:
    """Run a git command and return stripped stdout."""
    result = subprocess.run(["git", *args], check=True, capture_output=True, text=True)
    return result.stdout.strip()


def pyproject_version(path: Path = DEFAULT_PYPROJECT_PATH) -> str:
    """Return the static project version from ``pyproject.toml``."""
    return tomllib.loads(path.read_text(encoding="utf-8"))["project"]["version"].strip()


def parse_version(version: str) -> tuple[int, int, int]:
    """Return integer major, minor, and patch components for a simple version."""
    match = VERSION_PATTERN.fullmatch(version)
    if match is None:
        raise ValueError(f"Expected a version like 0.3.0, got {version!r}.")
    return int(match.group("major")), int(match.group("minor")), int(match.group("patch"))


def normalize_release_tag(tag: str) -> str:
    """Return a stable version from a release tag such as ``v0.3.7``."""
    match = RELEASE_TAG_PATTERN.fullmatch(tag)
    if match is None:
        raise ValueError(f"Expected a release tag like v0.3.0, got {tag!r}.")
    return f"{match.group('major')}.{match.group('minor')}.{match.group('patch')}"


def reachable_release_tags(head_ref: str = "HEAD") -> list[tuple[int, int, int, str]]:
    """Return reachable stable release tags."""
    raw_tags = run_git(["tag", "--merged", head_ref, "--list"])
    release_tags: list[tuple[int, int, int, str]] = []
    for tag in raw_tags.splitlines():
        try:
            normalized = normalize_release_tag(tag)
        except ValueError:
            continue
        version_parts = parse_version(normalized)
        release_tags.append((*version_parts, tag))
    return sorted(release_tags)


def prerelease_base_ref(head_ref: str = "HEAD", base_version: str | None = None) -> str:
    """Return the ref that starts the current prerelease sequence."""
    version = base_version or pyproject_version()
    version_parts = parse_version(version)
    stable_tags = reachable_release_tags(head_ref)
    matching_tags = [tag for *tag_version, tag in stable_tags if tuple(tag_version) == version_parts]
    if matching_tags:
        raise ValueError(
            f"Stable release tag {matching_tags[-1]!r} already matches pyproject.version {version!r}. "
            "Bump pyproject.version to the next release target or adjust the release process."
        )
    newer_tags = [tag for *tag_version, tag in stable_tags if tuple(tag_version) > version_parts]
    if newer_tags:
        raise ValueError(
            f"Stable release tag {newer_tags[-1]!r} is newer than pyproject.version {version!r}. "
            "Bump pyproject.version to the next release target or adjust the release process."
        )

    release_tags = [tag_parts for tag_parts in stable_tags if tag_parts[:3] < version_parts]
    if release_tags:
        return release_tags[-1][3]
    return MAIN_VERSION_BASE_SHA


def main_prerelease_number(head_ref: str = "HEAD", base_version: str | None = None) -> int:
    """Return the alpha prerelease number for ``head_ref``."""
    base_ref = prerelease_base_ref(head_ref, base_version)
    commit_count = int(run_git(["rev-list", "--count", f"{base_ref}..{head_ref}"]))
    return max(1, commit_count)


def main_version(head_ref: str = "HEAD", base_version: str | None = None) -> str:
    """Return the Docker tag for ``head_ref`` on the main prerelease line."""
    major, minor, patch = parse_version(base_version or pyproject_version())
    alpha = main_prerelease_number(head_ref, base_version)
    return f"{major}.{minor}.{patch}-alpha.{alpha}"


def version_from_release_tag(tag: str, base_version: str | None = None) -> str:
    """Return a stable version from a release tag such as ``v0.3.7``."""
    normalized = normalize_release_tag(tag)
    base_major, base_minor, _ = parse_version(base_version or pyproject_version())
    release_major, release_minor, _ = parse_version(normalized)
    if (release_major, release_minor) != (base_major, base_minor):
        raise ValueError(f"Expected a release tag like v{base_major}.{base_minor}.x, got {tag!r}.")
    return normalized


def pep440_version(version: str) -> str:
    """Return the PEP 440 spelling for a stable or alpha Docker tag."""
    if "-alpha." in version:
        return version.replace("-alpha.", "a", 1)
    return version


def write_pyproject_version(path: Path, version: str) -> None:
    """Replace the static project version in ``pyproject.toml``."""
    text = path.read_text(encoding="utf-8")
    updated, replacements = re.subn(r'(?m)^version = "[^"]+"$', f'version = "{version}"', text, count=1)
    if replacements != 1:
        raise ValueError(f"Expected exactly one project version line in {path}.")
    path.write_text(updated, encoding="utf-8")


def write_github_output(path: Path, version: str) -> None:
    """Append version outputs for GitHub Actions."""
    with path.open("a", encoding="utf-8") as output:
        output.write(f"pep440={pep440_version(version)}\n")
        output.write(f"docker_tag={version}\n")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head-ref", default="HEAD", help="Git ref to version. Defaults to HEAD.")
    parser.add_argument("--release-tag", help="Release tag to normalize, for example v0.3.7.")
    parser.add_argument(
        "--base-pyproject",
        type=Path,
        default=DEFAULT_PYPROJECT_PATH,
        help="Path to the pyproject.toml version anchor. Defaults to the repository pyproject.toml.",
    )
    parser.add_argument("--write-pyproject", type=Path, help="Optional pyproject.toml path to update.")
    parser.add_argument(
        "--github-output",
        type=Path,
        help="Optional GitHub Actions output file that receives pep440 and docker_tag values.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    base_version = pyproject_version(args.base_pyproject)
    version = (
        version_from_release_tag(args.release_tag, base_version)
        if args.release_tag
        else main_version(args.head_ref, base_version)
    )
    if args.write_pyproject is not None:
        write_pyproject_version(args.write_pyproject, version)
    if args.github_output is not None:
        write_github_output(args.github_output, version)
    print(version)


if __name__ == "__main__":
    main()
