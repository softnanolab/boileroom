"""Derive CI/CD release versions from the main-branch commit count."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

MAIN_VERSION_MAJOR = 0
MAIN_VERSION_MINOR = 3
MAIN_VERSION_BASE_PATCH = 0
MAIN_VERSION_BASE_SHA = "48e8a23fbde63e95a0e39fe0d5748c3f338b30b3"
RELEASE_TAG_PATTERN = re.compile(r"^(?P<version>0\.3\.\d+)$")


def run_git(args: list[str]) -> str:
    """Run a git command and return stripped stdout."""
    result = subprocess.run(["git", *args], check=True, capture_output=True, text=True)
    return result.stdout.strip()


def main_patch_for_head(head_ref: str = "HEAD") -> int:
    """Return the patch version offset for ``head_ref`` on the main release line."""
    commit_count = run_git(["rev-list", "--count", f"{MAIN_VERSION_BASE_SHA}..{head_ref}"])
    return MAIN_VERSION_BASE_PATCH + int(commit_count)


def main_version(head_ref: str = "HEAD") -> str:
    """Return the PEP 440 version for ``head_ref`` on the main release line."""
    patch = main_patch_for_head(head_ref)
    return f"{MAIN_VERSION_MAJOR}.{MAIN_VERSION_MINOR}.{patch}"


def version_from_release_tag(tag: str) -> str:
    """Return a PEP 440 version from a release tag such as ``0.3.7``."""
    normalized = tag.removeprefix("refs/tags/")
    match = RELEASE_TAG_PATTERN.fullmatch(normalized)
    if match is None:
        raise ValueError(f"Expected a release tag like 0.3.x, got {tag!r}.")
    return match.group("version")


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
        output.write(f"pep440={version}\n")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head-ref", default="HEAD", help="Git ref to version. Defaults to HEAD.")
    parser.add_argument("--release-tag", help="Release tag to normalize, for example 0.3.7.")
    parser.add_argument("--write-pyproject", type=Path, help="Optional pyproject.toml path to update.")
    parser.add_argument(
        "--github-output",
        type=Path,
        help="Optional GitHub Actions output file that receives the pep440 value.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    version = version_from_release_tag(args.release_tag) if args.release_tag else main_version(args.head_ref)
    if args.write_pyproject is not None:
        write_pyproject_version(args.write_pyproject, version)
    if args.github_output is not None:
        write_github_output(args.github_output, version)
    print(version)


if __name__ == "__main__":
    main()
