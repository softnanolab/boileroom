#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from boileroom.images.metadata import (  # noqa: E402
    MODEL_IMAGE_SPECS,
    format_image_reference,
    get_supported_cuda,
    normalize_cuda_version,
    normalize_requested_tag,
    published_image_references,
)

_CUDA_TAG_PATTERN = re.compile(r"^cuda\d+\.\d+(?:-.+)?$")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run import smoke checks inside boileroom model images.")
    parser.add_argument("--tag", default="latest", help="Tag to check, for example latest, 0.3.0, or cuda12.6-latest.")
    parser.add_argument(
        "--cuda-version",
        action="append",
        dest="cuda_versions",
        help="CUDA version to validate canonically (repeatable).",
    )
    parser.add_argument("--all-cuda", action="store_true", help="Validate all supported CUDA variants canonically.")
    parser.add_argument("--pull", action="store_true", help="Pull images before running checks.")
    return parser.parse_args()


def compute_cuda_versions(requested: list[str] | None, all_cuda: bool) -> list[str]:
    """Resolve CUDA versions requested by the caller."""
    if all_cuda:
        return ["11.8", "12.6"]
    if not requested:
        return []
    return [normalize_cuda_version(cuda_version) for cuda_version in requested]


def iter_image_targets(tag: str, cuda_versions: list[str]) -> list[tuple[str, str, str, Path, Path]]:
    """Return image targets to validate.

    Returns tuples of ``(image_key, image_reference, display_tag, env_path, core_path)``.
    """
    normalized_tag = normalize_requested_tag(tag)
    if cuda_versions and _CUDA_TAG_PATTERN.fullmatch(normalized_tag):
        raise ValueError("Do not combine --all-cuda/--cuda-version with an already CUDA-qualified --tag.")

    targets: list[tuple[str, str, str, Path, Path]] = []
    for spec in MODEL_IMAGE_SPECS:
        env_path = spec.context_path / "environment.yml"
        core_path = spec.context_path / "core.py"
        if cuda_versions:
            for cuda_version in cuda_versions:
                if cuda_version not in get_supported_cuda(spec):
                    continue
                canonical_ref = published_image_references(spec.image_name, cuda_version, normalized_tag)[0]
                display_tag = canonical_ref.rsplit(":", 1)[1]
                targets.append((spec.key, canonical_ref, display_tag, env_path, core_path))
            continue

        image_reference = format_image_reference(spec.image_name, normalized_tag)
        targets.append((spec.key, image_reference, normalized_tag, env_path, core_path))
    return targets


def ensure_docker() -> None:
    """Ensure Docker is available."""
    try:
        subprocess.run(
            ["docker", "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("Docker is required but was not found on PATH.") from exc


def check_image(
    image_key: str,
    image_reference: str,
    env_path: Path,
    core_path: Path,
    pull: bool,
) -> None:
    """Run the import smoke test for one image."""
    print(f"Checking imports for {image_key} ({image_reference})")

    if pull:
        subprocess.run(["docker", "pull", image_reference], check=True)

    if not env_path.exists():
        raise FileNotFoundError(f"Missing environment file: {env_path}")
    if not core_path.exists():
        raise FileNotFoundError(f"Missing core module: {core_path}")

    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            image_reference,
            "/bin/sh",
            "-lc",
            "python --version && pip --version",
        ],
        check=True,
    )

    script = f"""
import ast
import importlib
import re
import sys
from pathlib import Path

env_yml = Path({str(env_path)!r})
core_file = Path({str(core_path)!r})

deps = []
in_pip_section = False
with env_yml.open(encoding="utf-8") as handle:
    for line in handle:
        stripped = line.strip()
        if stripped == '- pip:' or stripped == 'pip:':
            in_pip_section = True
            continue
        if in_pip_section:
            if not stripped or (not line.startswith(' ') and not line.startswith('\\t')):
                in_pip_section = False
                if not stripped:
                    continue
                break
            if stripped.startswith('#'):
                continue
            pkg_name = re.split(r'[>=<!=]', stripped.lstrip('- '))[0].strip()
            import_map = {{
                'pytorch-lightning': 'pytorch_lightning',
                'torch-tensorrt': None,
                'hf-transfer': None,
                'hf_transfer': None,
            }}
            import_name = import_map.get(pkg_name, pkg_name.replace('-', '_'))
            if import_name:
                deps.append(import_name)

deps.append('numpy')

try:
    ast.parse(core_file.read_text(encoding='utf-8'), filename=str(core_file))
    print(f'OK: {{core_file.name}} (syntax valid)')
except SyntaxError as exc:
    print(f'FAILED: {{core_file.name}} has syntax errors: {{exc}}', file=sys.stderr)
    sys.exit(1)

for dep in deps:
    try:
        importlib.import_module(dep)
        print(f'OK: {{dep}}')
    except Exception as exc:
        print(f'FAILED: {{dep}} - {{exc}}', file=sys.stderr)
        sys.exit(1)
"""

    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{REPO_ROOT}:{REPO_ROOT}:ro",
            "-e",
            f"PYTHONPATH={REPO_ROOT}",
            image_reference,
            "micromamba",
            "run",
            "-n",
            "base",
            "python",
            "-c",
            script,
        ],
        check=True,
    )


def main() -> None:
    """Run the import smoke workflow."""
    args = parse_args()
    ensure_docker()
    cuda_versions = compute_cuda_versions(args.cuda_versions, args.all_cuda)
    targets = iter_image_targets(args.tag, cuda_versions)
    if not targets:
        raise SystemExit("No image targets matched the requested CUDA selection.")

    for image_key, image_reference, _display_tag, env_path, core_path in targets:
        check_image(image_key, image_reference, env_path, core_path, args.pull)

    print(f"All module imports succeeded for tag selection: {normalize_requested_tag(args.tag)}")


if __name__ == "__main__":
    main()
