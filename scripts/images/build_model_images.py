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
    platform: str
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


def run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    pretty = Colors.wrap("$ " + " ".join(shlex.quote(c) for c in cmd), Colors.blue)
    print(pretty)
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}")


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


def load_supported_cuda(model_dir: Path) -> list[str] | None:
    config_path = model_dir / "config.yaml"
    if not config_path.exists():
        return None
    text = config_path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{config_path} is not valid JSON/YAML: {exc}") from exc
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


def build_base(cuda_version: str, tag: str, platform: str, no_cache: bool, images_dir: Path) -> str:
    cuda_suffix = f"cuda{cuda_version}"
    if tag == "latest":
        base_tag = f"{DOCKER_REGISTRY}/boileroom-base:{cuda_suffix}"
    else:
        base_tag = f"{DOCKER_REGISTRY}/boileroom-base:{cuda_suffix}-{tag}"
    micromamba_base = CUDA_MICROMAMBA_BASE[cuda_version]

    log_info("")
    log_info(Colors.wrap(f"=== Building base for CUDA {cuda_version}: {base_tag}", Colors.bold))
    log_info(f"Using micromamba base: {micromamba_base}")

    cmd: list[str] = [
        "docker",
        "build",
        "--platform",
        platform,
        "--build-arg",
        f"MICROMAMBA_BASE={micromamba_base}",
        "-t",
        base_tag,
        "-f",
        str(images_dir / "Dockerfile"),
        str(images_dir),
    ]
    if no_cache:
        cmd.insert(3, "--no-cache")
    run(cmd)
    return base_tag


def build_model(task: BuildTask) -> str:
    cuda_suffix = f"cuda{task.cuda_version}"
    if task.tag == "latest":
        model_tag = f"{DOCKER_REGISTRY}/{task.model.tag_prefix}:{cuda_suffix}"
    else:
        model_tag = f"{DOCKER_REGISTRY}/{task.model.tag_prefix}:{cuda_suffix}-{task.tag}"

    log_info(Colors.wrap(f"--- Building {task.model.name} {task.cuda_version}: {model_tag}", Colors.bold))

    cmd: list[str] = [
        "docker",
        "build",
        "--platform",
        task.platform,
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
        cmd.insert(3, "--no-cache")

    run(cmd, env=os.environ.copy())

    if task.push:
        log_info(f"Pushing {model_tag}")
        run(["docker", "push", model_tag])

    return model_tag


class ImageCLI:
    def build(
        self,
        platform: str = "linux/amd64",
        no_cache: bool = False,
        tag: str = "dev",
        cuda_version: str | list[str] | None = None,
        all_cuda: bool = False,
        push: bool = False,
        max_workers: int = 1,
    ) -> None:
        """
        Build base and per-model Docker images.

        Args:
            platform: Docker build platform.
            no_cache: Disable Docker build cache.
            tag: Tag suffix (e.g. dev, latest, myfeature).
            cuda_version: CUDA version(s) to build (11.8 or 12.6). Can be repeated.
            all_cuda: Build for all supported CUDA versions.
            push: Push images to Docker Hub.
            max_workers: Max parallel model builds per CUDA (1 = serial).
        """
        try:
            ensure_docker()
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

        all_images: list[str] = []

        for v in cuda_versions:
            torch_index = CUDA_TORCH_WHEEL_INDEX[v]
            try:
                base_tag = build_base(v, tag, platform, no_cache, images_dir)
            except Exception as exc:
                log_error(f"Failed to build base for CUDA {v}: {exc}")
                raise SystemExit(1) from exc

            if push:
                try:
                    log_info(f"Pushing {base_tag}")
                    run(["docker", "push", base_tag])
                except Exception as exc:
                    log_error(f"Failed to push base {base_tag}: {exc}")
                    raise SystemExit(1) from exc

            all_images.append(base_tag)

            tasks: list[BuildTask] = []
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
                        platform=platform,
                        no_cache=no_cache,
                        tag=tag,
                        push=push,
                    )
                )

            if not tasks:
                log_warn(f"No models to build for CUDA {v}")
                continue

            workers = max(1, max_workers)
            if workers == 1 or len(tasks) == 1:
                for t in tasks:
                    try:
                        img = build_model(t)
                        all_images.append(img)
                    except Exception as exc:
                        log_error(f"Failed to build {t.model.name} CUDA {t.cuda_version}: {exc}")
                        raise SystemExit(1) from exc
            else:
                log_info(
                    f"Building {len(tasks)} model images for CUDA {v} "
                    f"with up to {workers} workers."
                )
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


def main(argv: Iterable[str] | None = None) -> None:
    fire.Fire(ImageCLI, command=argv)


if __name__ == "__main__":
    main()