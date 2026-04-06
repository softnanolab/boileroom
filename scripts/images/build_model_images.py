#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from boileroom.images.metadata import (  # noqa: E402
    BASE_IMAGE_SPEC,
    CUDA_MICROMAMBA_BASE,
    CUDA_TORCH_WHEEL_INDEX,
    MODEL_IMAGE_SPECS,
    RuntimeImageSpec,
    get_supported_cuda,
    normalize_cuda_version,
    normalize_requested_tag,
    published_image_references,
)

_CUDA_TAG_PATTERN = re.compile(r"^cuda\d+\.\d+(?:-.+)?$")


@dataclass(frozen=True)
class BuildTask:
    """Model-image build task."""

    cuda_version: str
    image_spec: RuntimeImageSpec
    base_image_reference: str
    tag: str


class Colors:
    reset = "\033[0m"
    bold = "\033[1m"
    red = "\033[31m"
    green = "\033[32m"
    yellow = "\033[33m"
    blue = "\033[34m"
    cyan = "\033[36m"
    magenta = "\033[35m"

    @staticmethod
    def wrap(text: str, color: str) -> str:
        if not sys.stdout.isatty():
            return text
        return f"{color}{text}{Colors.reset}"


def log_info(message: str) -> None:
    print(Colors.wrap(message, Colors.cyan))


def log_success(message: str) -> None:
    print(Colors.wrap(message, Colors.green))


def log_warn(message: str) -> None:
    print(Colors.wrap(message, Colors.yellow), file=sys.stderr)


def log_error(message: str) -> None:
    print(Colors.wrap(message, Colors.red), file=sys.stderr)


def run(cmd: list[str], log_file: Path | None = None, echo: bool = True) -> None:
    """Run a command and optionally capture the full output to a log file."""
    pretty = Colors.wrap("$ " + " ".join(shlex.quote(part) for part in cmd), Colors.blue)
    if echo:
        print(pretty)

    if log_file is None:
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"Command failed with exit code {result.returncode}")
        return

    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write(pretty + "\n")
        handle.flush()

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            if echo:
                print(line, end="")
            handle.write(line)
        return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(f"Command failed with exit code {return_code}")


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


def ensure_buildx_builder() -> None:
    """Ensure the dedicated buildx builder exists and is active."""
    try:
        subprocess.run(
            ["docker", "buildx", "version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("Docker buildx is required but was not found.") from exc

    inspect_result = subprocess.run(
        ["docker", "buildx", "inspect", "boileroom-builder"],
        capture_output=True,
        text=True,
        check=False,
    )
    builder_exists = inspect_result.returncode == 0
    has_container_driver = "driver: docker-container" in inspect_result.stdout

    if not builder_exists or not has_container_driver:
        if builder_exists:
            log_info("Removing existing 'boileroom-builder' to recreate it with docker-container driver...")
            subprocess.run(
                ["docker", "buildx", "rm", "boileroom-builder"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        log_info("Creating 'boileroom-builder' with docker-container driver...")
        subprocess.run(
            [
                "docker",
                "buildx",
                "create",
                "--name",
                "boileroom-builder",
                "--driver",
                "docker-container",
                "--driver-opt",
                "network=host",
                "--use",
            ],
            check=True,
        )
        log_success("Created and activated buildx builder 'boileroom-builder'.")
        return

    subprocess.run(["docker", "buildx", "use", "boileroom-builder"], check=True)
    log_info("Using existing buildx builder 'boileroom-builder'.")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Build base and per-model Docker images using shared boileroom image metadata."
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Unqualified tag to publish. Defaults to the current boileroom package version; explicit examples include 0.3.0 or sha-<commit>.",
    )
    parser.add_argument(
        "--cuda-version",
        action="append",
        dest="cuda_versions",
        help="CUDA version to build (repeatable). Supported values: 11.8, 12.6.",
    )
    parser.add_argument("--all-cuda", action="store_true", help="Build all supported CUDA variants.")
    parser.add_argument(
        "--platform",
        default="linux/amd64",
        help="Comma-separated buildx platforms. The default release path publishes linux/amd64 only.",
    )
    parser.add_argument("--push", action="store_true", help="Push built images to Docker Hub.")
    parser.add_argument(
        "--load",
        action="store_true",
        help="Load single-platform builds into the local Docker daemon. Incompatible with --push and multi-platform builds.",
    )
    parser.add_argument("--no-cache", action="store_true", help="Disable Docker build cache.")
    parser.add_argument("--max-workers", type=int, default=1, help="Maximum concurrent model-image builds.")
    return parser.parse_args()


def resolve_publish_tag(tag: str | None) -> str:
    """Validate and normalize an unqualified publish tag."""
    normalized = normalize_requested_tag(tag)
    if _CUDA_TAG_PATTERN.fullmatch(normalized):
        raise ValueError(
            f"Publish tag {tag!r} is already CUDA-qualified. "
            "Pass an unqualified tag such as 0.3.0 or sha-<commit> and select CUDA variants with --cuda-version."
        )
    return normalized


def compute_cuda_versions(requested: list[str] | None, all_cuda: bool) -> list[str]:
    """Resolve the CUDA versions to build."""
    if all_cuda:
        return sorted(CUDA_MICROMAMBA_BASE)
    if not requested:
        raise ValueError("Specify at least one --cuda-version or use --all-cuda.")
    return [normalize_cuda_version(cuda_version) for cuda_version in requested]


def resolve_output_flag(push: bool, load: bool, platform: str) -> str | None:
    """Return the buildx output flag for the requested mode."""
    multi_platform = "," in platform
    if push and load:
        raise ValueError("--push and --load cannot be used together.")
    if push:
        return "--push"
    if load:
        if multi_platform:
            raise ValueError("--load requires a single --platform value.")
        return "--load"
    if multi_platform:
        return None
    return "--load"


def should_use_local_docker_build(push: bool, platform: str) -> bool:
    """Return whether the build should use plain `docker build`."""
    return not push and "," not in platform


def build_base(
    cuda_version: str,
    tag: str,
    platform: str,
    output_flag: str | None,
    no_cache: bool,
    use_local_docker_build: bool,
) -> str:
    """Build the shared base image and return its canonical reference."""
    image_references = published_image_references(BASE_IMAGE_SPEC.image_name, cuda_version, tag)
    canonical_reference = image_references[0]
    micromamba_base = CUDA_MICROMAMBA_BASE[cuda_version]

    log_info("")
    log_info(Colors.wrap(f"=== Building base image for CUDA {cuda_version}: {canonical_reference}", Colors.bold))
    log_info(f"Using micromamba base: {micromamba_base}")
    log_info(f"Publishing tags: {', '.join(image_references)}")

    log_file = Path.cwd() / f"{canonical_reference.replace('/', '_').replace(':', '_')}.log"
    if use_local_docker_build:
        cmd = [
            "docker",
            "build",
            "--platform",
            platform,
            "--build-arg",
            f"MICROMAMBA_BASE={micromamba_base}",
        ]
    else:
        cmd = [
            "docker",
            "buildx",
            "build",
            "--platform",
            platform,
            "--build-arg",
            f"MICROMAMBA_BASE={micromamba_base}",
        ]
    for image_reference in image_references:
        cmd.extend(["-t", image_reference])
    cmd.extend(["-f", str(BASE_IMAGE_SPEC.dockerfile_path), str(BASE_IMAGE_SPEC.context_path)])
    if no_cache:
        cmd.append("--no-cache")
    if not use_local_docker_build and output_flag is not None:
        cmd.append(output_flag)

    run(cmd, log_file=log_file, echo=False)
    return canonical_reference


def build_model(
    task: BuildTask,
    platform: str,
    output_flag: str | None,
    no_cache: bool,
    use_local_docker_build: bool,
) -> tuple[str, ...]:
    """Build a single model image and return all published references."""
    image_references = published_image_references(task.image_spec.image_name, task.cuda_version, task.tag)
    canonical_reference = image_references[0]

    log_info(Colors.wrap(f"--- Building {task.image_spec.key} for CUDA {task.cuda_version}: {canonical_reference}", Colors.bold))
    log_info(f"Publishing tags: {', '.join(image_references)}")

    log_file = Path.cwd() / f"{canonical_reference.replace('/', '_').replace(':', '_')}.log"
    if use_local_docker_build:
        cmd = [
            "docker",
            "build",
            "--platform",
            platform,
            "--build-arg",
            f"BASE_IMAGE={task.base_image_reference}",
            "--build-arg",
            f"TORCH_WHEEL_INDEX={CUDA_TORCH_WHEEL_INDEX[task.cuda_version]}",
        ]
    else:
        cmd = [
            "docker",
            "buildx",
            "build",
            "--platform",
            platform,
            "--build-arg",
            f"BASE_IMAGE={task.base_image_reference}",
            "--build-arg",
            f"TORCH_WHEEL_INDEX={CUDA_TORCH_WHEEL_INDEX[task.cuda_version]}",
        ]
    for image_reference in image_references:
        cmd.extend(["-t", image_reference])
    cmd.extend(["-f", str(task.image_spec.dockerfile_path), str(task.image_spec.context_path)])
    if no_cache:
        cmd.append("--no-cache")
    if not use_local_docker_build and output_flag is not None:
        cmd.append(output_flag)

    run(cmd, log_file=log_file, echo=False)
    return image_references


def main() -> None:
    """Run the Docker image build workflow."""
    args = parse_args()

    try:
        ensure_docker()
        tag = resolve_publish_tag(args.tag)
        cuda_versions = compute_cuda_versions(args.cuda_versions, args.all_cuda)
        output_flag = resolve_output_flag(args.push, args.load, args.platform)
        use_local_docker_build = should_use_local_docker_build(args.push, args.platform)
        if not use_local_docker_build:
            ensure_buildx_builder()
    except Exception as exc:
        log_error(str(exc))
        raise SystemExit(1) from exc

    if use_local_docker_build:
        log_info("Using plain docker build for single-platform local images.")
    elif output_flag is None:
        log_warn(
            "Building multi-platform images without --push leaves them only in buildx cache. "
            "Use --push or choose a single --platform to auto-load locally."
        )

    log_info(Colors.wrap(f"Boileroom repo root: {REPO_ROOT}", Colors.magenta))
    log_info(f"Model images: {', '.join(spec.image_name for spec in MODEL_IMAGE_SPECS)}")
    log_info(f"CUDA versions: {', '.join(cuda_versions)}")
    log_info(f"Platforms: {args.platform}")

    built_references: list[str] = []
    tasks: list[BuildTask] = []

    for cuda_version in cuda_versions:
        try:
            base_reference = build_base(
                cuda_version,
                tag,
                args.platform,
                output_flag,
                args.no_cache,
                use_local_docker_build,
            )
        except Exception as exc:
            log_error(f"Failed to build base image for CUDA {cuda_version}: {exc}")
            raise SystemExit(1) from exc

        built_references.append(base_reference)

        for image_spec in MODEL_IMAGE_SPECS:
            supported_cuda = get_supported_cuda(image_spec)
            if cuda_version not in supported_cuda:
                log_warn(
                    f"Skipping {image_spec.image_name} for CUDA {cuda_version} "
                    f"(supported CUDA variants: {', '.join(supported_cuda)})"
                )
                continue
            tasks.append(
                BuildTask(
                    cuda_version=cuda_version,
                    image_spec=image_spec,
                    base_image_reference=base_reference,
                    tag=tag,
                )
            )

    if not tasks:
        log_warn("No model images matched the requested CUDA selection.")
    elif args.max_workers <= 1 or len(tasks) == 1:
        for task in tasks:
            try:
                built_references.extend(
                    build_model(task, args.platform, output_flag, args.no_cache, use_local_docker_build)
                )
            except Exception as exc:
                log_error(f"Failed to build {task.image_spec.image_name} for CUDA {task.cuda_version}: {exc}")
                raise SystemExit(1) from exc
    else:
        executor = ThreadPoolExecutor(max_workers=max(1, args.max_workers))
        try:
            future_map = {
                executor.submit(
                    build_model,
                    task,
                    args.platform,
                    output_flag,
                    args.no_cache,
                    use_local_docker_build,
                ): task
                for task in tasks
            }
            for future in as_completed(future_map):
                task = future_map[future]
                try:
                    built_references.extend(future.result())
                except Exception as exc:
                    log_error(f"Failed to build {task.image_spec.image_name} for CUDA {task.cuda_version}: {exc}")
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise SystemExit(1) from exc
        finally:
            executor.shutdown(wait=True)

    log_success("")
    log_success("=== Build complete ===")
    for image_reference in built_references:
        print(f"  {image_reference}")


if __name__ == "__main__":
    main()
