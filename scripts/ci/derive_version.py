"""Derive CI/CD release versions from the main-branch commit count."""

from __future__ import annotations

import argparse
import re
import subprocess
import tomllib
from pathlib import Path

MAIN_VERSION_BASE_SHA = "48e8a23fbde63e95a0e39fe0d5748c3f338b30b3"
DEFAULT_PYPROJECT_PATH = Path(__file__).resolve().parents[2] / "pyproject.toml"
VERSION_PATTERN = re.compile(r"^(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)$")


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


def main_patch_for_head(head_ref: str = "HEAD", base_patch: int | None = None) -> int:
    """Return the patch version offset for ``head_ref`` on the main release line."""
    commit_count = run_git(["rev-list", "--count", f"{MAIN_VERSION_BASE_SHA}..{head_ref}"])
    if base_patch is None:
        base_patch = parse_version(pyproject_version())[2]
    return base_patch + int(commit_count)


def main_version(head_ref: str = "HEAD", base_version: str | None = None) -> str:
    """Return the PEP 440 version for ``head_ref`` on the main release line."""
    major, minor, base_patch = parse_version(base_version or pyproject_version())
    patch = main_patch_for_head(head_ref, base_patch)
    return f"{major}.{minor}.{patch}"


def version_from_release_tag(tag: str, base_version: str | None = None) -> str:
    """Return a PEP 440 version from a release tag such as ``0.3.7``."""
    normalized = tag.removeprefix("refs/tags/")
    base_major, base_minor, _ = parse_version(base_version or pyproject_version())
    release_major, release_minor, _ = parse_version(normalized)
    if (release_major, release_minor) != (base_major, base_minor):
        raise ValueError(f"Expected a release tag like {base_major}.{base_minor}.x, got {tag!r}.")
    return normalized


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
        help="Optional GitHub Actions output file that receives the pep440 value.",
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
