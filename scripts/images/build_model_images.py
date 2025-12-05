#!/usr/bin/env python
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import fire
import yaml


DOCKER_REGISTRY = "docker.io/jakublala"

CUDA_MICROMAMBA_BASE: dict[str, str] = {
    "11.8": "mambaorg/micromamba:2.4-cuda11.8.0-ubuntu22.04",
    "12.6": "mambaorg/micromamba:2.4-cuda12.6.3-ubuntu22.04",
}

CUDA_TORCH_WHEEL_INDEX: dict[str, str] = {
    "11.8": "https://download.pytorch.org/whl/cu118",
    "12.6": "https://download.pytorch.org/whl/cu126",
}


@dataclass
class ModelConfig:
    name: str
    dockerfile: Path
    tag_prefix: str
    supported_cuda: list[str] | None


@dataclass
class BuildTask:
    cuda_version: str
    model: ModelConfig
    base_image_tag: str
    torch_wheel_index: str
    no_cache: bool
    tag: str
    push: bool


class Colors:
    reset = "\033[0m"
    bold = "\033[1m"
    red = "\033[31m"
    green = "\033[32m"
    yellow = "\033[33m"
    blue = "\033[34m"
    magenta = "\033[35m"
    cyan = "\033[36m"

    @staticmethod
    def wrap(text: str, color: str) -> str:
        if not sys.stdout.isatty():
            return text
        return f"{color}{text}{Colors.reset}"


def log_info(msg: str) -> None:
    print(Colors.wrap(msg, Colors.cyan))


def log_success(msg: str) -> None:
    print(Colors.wrap(msg, Colors.green))


def log_warn(msg: str) -> None:
    print(Colors.wrap(msg, Colors.yellow), file=sys.stderr)


def log_error(msg: str) -> None:
    print(Colors.wrap(msg, Colors.red), file=sys.stderr)


def run(
    cmd: list[str],
    env: dict[str, str] | None = None,
    log_file: Path | None = None,
    echo: bool = True,
) -> None:
    """
    Run a command, optionally teeing full stdout/stderr into a log file.
    """
    pretty = Colors.wrap("$ " + " ".join(shlex.quote(c) for c in cmd), Colors.blue)
    if echo:
        print(pretty)

    if log_file is None:
        result = subprocess.run(cmd, env=env)
        if result.returncode != 0:
            raise RuntimeError(f"Command failed with exit code {result.returncode}")
        return

    log_file = log_file.resolve()
    log_file.parent.mkdir(parents=True, exist_ok=True)

    with log_file.open("a", encoding="utf-8") as handle:
        handle.write(pretty + "\n")
        handle.flush()

        process = subprocess.Popen(
            cmd,
            env=env,
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


def resolve_paths() -> tuple[Path, Path, Path]:
    script_path = Path(__file__).resolve()
    script_dir = script_path.parent
    repo_root = script_dir.parent.parent
    boileroom_dir = repo_root / "boileroom"
    images_dir = boileroom_dir / "images"
    return repo_root, boileroom_dir, images_dir


def ensure_docker() -> None:
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
    """
    Ensure a buildx builder with docker-container driver exists for multi-platform builds.
    Creates 'boileroom-builder' if none exists or if existing builder doesn't support multi-platform.
    """
    try:
        # Check if buildx is available
        subprocess.run(
            ["docker", "buildx", "version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("Docker buildx is required but was not found.") from exc

    # Check if boileroom-builder exists and has docker-container driver
    builder_exists = False
    has_docker_container_driver = False
    
    inspect_result = subprocess.run(
        ["docker", "buildx", "inspect", "boileroom-builder"],
        capture_output=True,
        text=True,
    )
    
    if inspect_result.returncode == 0:
        builder_exists = True
        # Check if it uses docker-container driver
        has_docker_container_driver = "driver: docker-container" in inspect_result.stdout
    
    if not builder_exists or not has_docker_container_driver:
        # Remove existing builder if it doesn't have the right driver
        if builder_exists:
            log_info("Removing existing 'boileroom-builder' to recreate with docker-container driver...")
            subprocess.run(
                ["docker", "buildx", "rm", "boileroom-builder"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        
        log_info("Creating 'boileroom-builder' with docker-container driver for multi-platform builds...")
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
        log_success("Created and activated buildx builder 'boileroom-builder' with docker-container driver")
    else:
        # Builder exists with correct driver, ensure it's in use
        use_result = subprocess.run(
            ["docker", "buildx", "use", "boileroom-builder"],
            capture_output=True,
            text=True,
        )
        if use_result.returncode == 0:
            log_info("Using existing buildx builder 'boileroom-builder' with docker-container driver")


def load_supported_cuda(model_dir: Path) -> list[str] | None:
    config_path = model_dir / "config.yaml"
    if not config_path.exists():
        return None
    text = config_path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"{config_path} is not valid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{config_path} must contain a mapping/object at top level.")
    value = data.get("supported_cuda")
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ValueError(f"{config_path}: supported_cuda must be a list of strings")
    return value


def discover_models(boileroom_dir: Path) -> list[ModelConfig]:
    models_dir = boileroom_dir / "models"
    boltz_dir = models_dir / "boltz"
    chai_dir = models_dir / "chai"
    esm_dir = models_dir / "esm"
    return [
        ModelConfig(
            name="boltz",
            dockerfile=boltz_dir / "Dockerfile",
            tag_prefix="boileroom-boltz",
            supported_cuda=load_supported_cuda(boltz_dir),
        ),
        ModelConfig(
            name="chai",
            dockerfile=chai_dir / "Dockerfile",
            tag_prefix="boileroom-chai1",
            supported_cuda=load_supported_cuda(chai_dir),
        ),
        ModelConfig(
            name="esm",
            dockerfile=esm_dir / "Dockerfile",
            tag_prefix="boileroom-esm",
            supported_cuda=load_supported_cuda(esm_dir),
        ),
    ]


def compute_cuda_versions(cuda_version: str | list[str] | None, all_cuda: bool) -> list[str]:
    if all_cuda:
        versions = sorted(CUDA_MICROMAMBA_BASE.keys())
    else:
        if cuda_version is None:
            raise ValueError("Specify at least one --cuda-version or use --all-cuda.")
        if isinstance(cuda_version, str):
            versions = [cuda_version]
        else:
            versions = list(cuda_version)
    for v in versions:
        if v not in CUDA_MICROMAMBA_BASE:
            raise ValueError(
                f"Unsupported CUDA version: {v}. "
                f"Supported: {', '.join(sorted(CUDA_MICROMAMBA_BASE))}"
            )
    return versions


def build_base(cuda_version: str, tag: str, no_cache: bool, images_dir: Path, push: bool) -> str:
    cuda_suffix = f"cuda{cuda_version}"
    if tag == "latest":
        base_tag = f"{DOCKER_REGISTRY}/boileroom-base:{cuda_suffix}"
    else:
        base_tag = f"{DOCKER_REGISTRY}/boileroom-base:{cuda_suffix}-{tag}"
    micromamba_base = CUDA_MICROMAMBA_BASE[cuda_version]

    log_info("")
    log_info(Colors.wrap(f"=== Building base for CUDA {cuda_version}: {base_tag}", Colors.bold))
    log_info(f"Using micromamba base: {micromamba_base}")
    log_info("Building for platforms: linux/amd64,linux/arm64")

    log_file = Path.cwd() / f"{base_tag.replace('/', '_').replace(':', '_')}.log"

    cmd: list[str] = [
        "docker",
        "buildx",
        "build",
        "--platform",
        "linux/amd64,linux/arm64",
        "--build-arg",
        f"MICROMAMBA_BASE={micromamba_base}",
        "-t",
        base_tag,
        "-f",
        str(images_dir / "Dockerfile"),
        str(images_dir),
    ]
    if no_cache:
        cmd.append("--no-cache")
    if push:
        cmd.append("--push")
    # When push=False, images are stored in buildx cache (not loadable to local daemon for multi-platform)
    
    run(cmd, log_file=log_file, echo=False)
    return base_tag


def build_model(task: BuildTask) -> str:
    cuda_suffix = f"cuda{task.cuda_version}"
    if task.tag == "latest":
        model_tag = f"{DOCKER_REGISTRY}/{task.model.tag_prefix}:{cuda_suffix}"
    else:
        model_tag = f"{DOCKER_REGISTRY}/{task.model.tag_prefix}:{cuda_suffix}-{task.tag}"

    log_info(Colors.wrap(f"--- Building {task.model.name} {task.cuda_version}: {model_tag}", Colors.bold))
    log_info("Building for platforms: linux/amd64,linux/arm64")

    log_file = Path.cwd() / f"{model_tag.replace('/', '_').replace(':', '_')}.log"

    cmd: list[str] = [
        "docker",
        "buildx",
        "build",
        "--platform",
        "linux/amd64,linux/arm64",
        "--build-arg",
        f"BASE_IMAGE={task.base_image_tag}",
        "--build-arg",
        f"TORCH_WHEEL_INDEX={task.torch_wheel_index}",
        "-t",
        model_tag,
        "-f",
        str(task.model.dockerfile),
        str(task.model.dockerfile.parent),
    ]
    if task.no_cache:
        cmd.append("--no-cache")
    if task.push:
        cmd.append("--push")
    # When push=False, images are stored in buildx cache (not loadable to local daemon for multi-platform)

    run(cmd, env=os.environ.copy(), log_file=log_file, echo=False)

    return model_tag


def main(
    no_cache: bool = False,
    tag: str = "dev",
    cuda_version: str | list[str] | None = None,
    all_cuda: bool = False,
    push: bool = False,
    max_workers: int = 1,
) -> None:
    """
    Build base and per-model Docker images using buildx with multi-platform support.

    Args:
        no_cache: Disable Docker build cache.
        tag: Tag suffix (e.g. dev, latest, myfeature).
        cuda_version: CUDA version(s) to build (11.8 or 12.6). Can be repeated.
        all_cuda: Build for all supported CUDA versions.
        push: Push images to Docker Hub. Required for multi-platform images to be accessible.
        max_workers: Max parallel image builds (models) across all CUDA versions.
    """
    try:
        ensure_docker()
        ensure_buildx_builder()
        repo_root, boileroom_dir, images_dir = resolve_paths()
        models = discover_models(boileroom_dir)
        cuda_versions = compute_cuda_versions(cuda_version, all_cuda)
    except Exception as exc:
        log_error(str(exc))
        raise SystemExit(1) from exc

    log_info(Colors.wrap(f"Boileroom repo root: {repo_root}", Colors.magenta))
    log_info(
        f"Models: {', '.join(m.name for m in models)}; "
        f"CUDA: {', '.join(cuda_versions)}"
    )
    log_info("Building for platforms: linux/amd64,linux/arm64")

    all_images: list[str] = []
    tasks: list[BuildTask] = []

    # 1) Build all base images serially (per CUDA), possibly pushing them.
    base_tags: dict[str, str] = {}
    for v in cuda_versions:
        torch_index = CUDA_TORCH_WHEEL_INDEX[v]
        try:
            base_tag = build_base(v, tag, no_cache, images_dir, push)
        except Exception as exc:
            log_error(f"Failed to build base for CUDA {v}: {exc}")
            raise SystemExit(1) from exc

        all_images.append(base_tag)
        base_tags[v] = base_tag

        # Queue per-model builds for this CUDA version
        for model in models:
            supported = model.supported_cuda
            if supported and v not in supported:
                log_warn(
                    f"Skipping {model.name} for CUDA {v} "
                    f"(unsupported by config.yaml: {supported})"
                )
                continue
            tasks.append(
                BuildTask(
                    cuda_version=v,
                    model=model,
                    base_image_tag=base_tag,
                    torch_wheel_index=torch_index,
                    platform="linux/amd64,linux/arm64",  # Not used but kept for compatibility
                    no_cache=no_cache,
                    tag=tag,
                    push=push,
                )
            )

    if not tasks:
        log_warn("No model images to build for the requested CUDA versions.")
    else:
        workers = max(1, max_workers)
        log_info(
            f"Building {len(tasks)} model images across CUDA "
            f"{', '.join(cuda_versions)} with up to {workers} workers."
        )
        if workers == 1 or len(tasks) == 1:
            for t in tasks:
                try:
                    img = build_model(t)
                    all_images.append(img)
                except Exception as exc:
                    log_error(
                        f"Failed to build {t.model.name} CUDA {t.cuda_version}: {exc}"
                    )
                    raise SystemExit(1) from exc
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                future_map = {ex.submit(build_model, t): t for t in tasks}
                for fut in as_completed(future_map):
                    t = future_map[fut]
                    try:
                        img = fut.result()
                        all_images.append(img)
                    except Exception as exc:
                        log_error(
                            f"Failed to build {t.model.name} CUDA {t.cuda_version}: {exc}"
                        )
                        raise SystemExit(1) from exc

    log_success("")
    log_success("=== Build complete ===")
    for img in all_images:
        print(f"  {img}")



if __name__ == "__main__":
    fire.Fire(main)