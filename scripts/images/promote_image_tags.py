#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from boileroom.images.metadata import (  # noqa: E402
    BASE_IMAGE_SPEC,
    MODEL_IMAGE_SPECS,
    RuntimeImageSpec,
    get_supported_cuda,
    normalize_cuda_version,
    normalize_requested_tag,
    published_image_references,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Promote validated boileroom image tags without rebuilding.")
    parser.add_argument("--source-tag", required=True, help="Validated source tag, for example sha-abcd1234.")
    parser.add_argument("--target-tag", required=True, help="Public target tag, for example latest or 0.3.0.")
    parser.add_argument(
        "--cuda-version",
        action="append",
        dest="cuda_versions",
        help="CUDA version to promote (repeatable). Supported values: 11.8, 12.6.",
    )
    parser.add_argument("--all-cuda", action="store_true", help="Promote all supported CUDA variants.")
    return parser.parse_args()


def ensure_buildx() -> None:
    """Ensure Docker buildx is available."""
    try:
        subprocess.run(
            ["docker", "buildx", "version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("Docker buildx is required but was not found.") from exc


def compute_cuda_versions(requested: list[str] | None, all_cuda: bool) -> list[str]:
    """Resolve requested CUDA versions."""
    if all_cuda:
        return sorted({cuda for spec in (BASE_IMAGE_SPEC, *MODEL_IMAGE_SPECS) for cuda in get_supported_cuda(spec)})
    if not requested:
        raise ValueError("Specify at least one --cuda-version or use --all-cuda.")
    return [normalize_cuda_version(cuda_version) for cuda_version in requested]


def promote_one(spec: RuntimeImageSpec, cuda_version: str, source_tag: str, target_tag: str) -> None:
    """Promote one canonical image manifest to its public target tags."""
    source_reference = published_image_references(spec.image_name, cuda_version, source_tag)[0]
    target_references = published_image_references(spec.image_name, cuda_version, target_tag)
    cmd = ["docker", "buildx", "imagetools", "create"]
    for target_reference in target_references:
        cmd.extend(["-t", target_reference])
    cmd.append(source_reference)
    print(f"Promoting {source_reference} -> {', '.join(target_references)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    """Promote validated runtime images to public tags."""
    args = parse_args()
    ensure_buildx()
    source_tag = normalize_requested_tag(args.source_tag)
    target_tag = normalize_requested_tag(args.target_tag)
    cuda_versions = compute_cuda_versions(args.cuda_versions, args.all_cuda)
    image_specs = (BASE_IMAGE_SPEC, *MODEL_IMAGE_SPECS)

    for cuda_version in cuda_versions:
        for spec in image_specs:
            if cuda_version not in get_supported_cuda(spec):
                continue
            promote_one(spec, cuda_version, source_tag, target_tag)


if __name__ == "__main__":
    main()
