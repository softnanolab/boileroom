"""Helpers shared by image import smoke checks."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from .metadata import (
    DEFAULT_DOCKER_REPOSITORY,
    MODEL_IMAGE_SPECS,
    SUPPORTED_CUDA_VERSIONS,
    InterpreterSmokeCheck,
    RuntimeImageSpec,
    format_image_reference,
    get_supported_cuda,
    get_supported_platforms,
    normalize_cuda_version,
    normalize_requested_tag,
    published_image_references,
    split_platforms,
)

_CUDA_TAG_PATTERN = re.compile(r"^cuda\d+\.\d+(?:-.+)?$")

IMPORT_NAME_OVERRIDES: Final[dict[str, str | None]] = {
    "absl-py": "absl",
    "biopython": "Bio",
    "dm-haiku": "haiku",
    "ml-collections": "ml_collections",
    "tensorflow-cpu": "tensorflow",
    "pytorch-lightning": "pytorch_lightning",
    "torch-tensorrt": None,
    "hf-transfer": None,
    "hf_transfer": None,
}
"""Map package names to importable module names for smoke tests.

Add entries here when a package name does not map cleanly to
``package_name.replace("-", "_")`` or should be skipped entirely.
"""


def compute_cuda_versions(requested: list[str] | None, all_cuda: bool) -> list[str]:
    """Resolve CUDA versions requested by the caller."""
    if all_cuda:
        return sorted(normalize_cuda_version(cuda_version) for cuda_version in SUPPORTED_CUDA_VERSIONS)
    if not requested:
        return []
    return [normalize_cuda_version(cuda_version) for cuda_version in requested]


def package_name_to_import_name(package_name: str) -> str | None:
    """Return the importable module name for a dependency string."""
    return IMPORT_NAME_OVERRIDES.get(package_name, package_name.replace("-", "_"))


def requirement_line_to_package_name(line: str) -> str | None:
    """Return the package name from one requirements.txt line, or None for pip options.

    Extras such as ``colabfold[alphafold-minus-jax]`` are stripped so the bare
    distribution name is returned.
    """
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or stripped.startswith("-"):
        return None
    if "#egg=" in stripped:
        name = stripped.rsplit("#egg=", 1)[1].split("&", 1)[0].strip()
    elif " @ " in stripped:
        name = stripped.split(" @ ", 1)[0].strip()
    else:
        name = re.split(r"[>=<!=;\[]", stripped, maxsplit=1)[0].strip()
    name = name.split("[", 1)[0].strip()
    return name or None


def requirement_import_names(requirements_path: Path) -> list[str]:
    """Return import names represented by a requirements.txt file."""
    import_names = []
    with requirements_path.open(encoding="utf-8") as handle:
        for line in handle:
            package_name = requirement_line_to_package_name(line)
            if package_name is None:
                continue
            import_name = package_name_to_import_name(package_name)
            if import_name:
                import_names.append(import_name)
    return import_names


_INTERPRETER_SMOKE_TEMPLATE = """
import importlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

imports = {imports!r}
driver_linked_libraries = {driver_linked_libraries!r}
allowed_unresolved = set({allowed_unresolved!r})
required_symbol_versions = {required_symbol_versions!r}
failures = []

print(f'Interpreter: {{sys.executable}} (Python {{sys.version.split()[0]}})')
for name in imports:
    try:
        importlib.import_module(name)
        print(f'OK: {{name}}')
    except Exception as exc:
        print(f'FAILED: {{name}} - {{type(exc).__name__}}: {{exc}}', file=sys.stderr)
        failures.append(name)


def package_locations(package):
    try:
        spec = importlib.util.find_spec(package)
    except (ImportError, ValueError):
        return []
    return list(spec.submodule_search_locations or []) if spec is not None else []


# A loaded process finds the CUDA runtime libraries of the pip wheels (cuBLAS, NVRTC, ...) because ``import torch``
# preloads them; ldd only searches LD_LIBRARY_PATH, so give it the directories torch loads them from.
wheel_library_dirs = [
    str(path)
    for package, pattern in (('torch', 'lib'), ('nvidia', '*/lib'))
    for location in package_locations(package)
    for path in sorted(Path(location).glob(pattern))
    if path.is_dir()
]
for package, filename in driver_linked_libraries:
    locations = package_locations(package)
    libraries = sorted({{path for location in locations for path in Path(location).rglob(filename)}})
    if not libraries:
        print(f'FAILED: {{filename}} not found in package {{package}}', file=sys.stderr)
        failures.append(filename)
        continue
    for library in libraries:
        search_path = [str(library.parent), *wheel_library_dirs, os.environ.get('LD_LIBRARY_PATH', '')]
        env = dict(os.environ, LD_LIBRARY_PATH=':'.join(part for part in search_path if part))
        result = subprocess.run(['ldd', str(library)], capture_output=True, text=True, env=env)
        unresolved = {{line.split('=>')[0].strip() for line in result.stdout.splitlines() if 'not found' in line}}
        unexpected = sorted(unresolved - allowed_unresolved)
        if result.returncode != 0 or unexpected:
            detail = ', '.join(unexpected) or result.stderr.strip()
            print(f'FAILED: {{library}} has unresolved libraries: {{detail}}', file=sys.stderr)
            failures.append(str(library))
        else:
            print(f'OK: {{library}} links (unresolved only: {{sorted(unresolved) or "none"}})')

for library_file, version in required_symbol_versions:
    path = Path(library_file)
    if not path.is_file() or version.encode() not in path.read_bytes():
        print(f'FAILED: {{library_file}} does not provide {{version}}', file=sys.stderr)
        failures.append(version)
    else:
        print(f'OK: {{library_file}} provides {{version}}')

if failures:
    sys.exit(1)
"""


def interpreter_smoke_script(check: InterpreterSmokeCheck) -> str:
    """Return the Python source that runs one interpreter smoke check inside an image.

    The script uses the standard library only, so it runs under any interpreter in the image. It imports each module
    of ``check.imports``, finds each driver-linked library under its package without importing it and runs ``ldd`` on
    it (only ``check.allowed_unresolved`` may be missing on a host without the NVIDIA driver), and checks each required
    symbol version. ``ldd`` searches the library's own directory and the CUDA library directories of the ``torch`` and
    ``nvidia`` wheels first: ``import torch`` preloads those at run time, and nothing else puts them on the search
    path. It exits with status 1 after reporting every failure.

    Parameters
    ----------
    check : InterpreterSmokeCheck
        The check to run.

    Returns
    -------
    str
        Python source for ``<python> -c``.
    """
    return _INTERPRETER_SMOKE_TEMPLATE.format(
        imports=list(check.imports),
        driver_linked_libraries=[list(pair) for pair in check.driver_linked_libraries],
        allowed_unresolved=list(check.allowed_unresolved),
        required_symbol_versions=[list(pair) for pair in check.required_symbol_versions],
    )


def interpreter_smoke_command(image_reference: str, check: InterpreterSmokeCheck) -> list[str]:
    """Return the ``docker run`` command for one interpreter smoke check (CPU only: no ``--gpus``).

    Parameters
    ----------
    image_reference : str
        The image to check.
    check : InterpreterSmokeCheck
        The check to run.

    Returns
    -------
    list[str]
        The command, running ``check.python`` with ``check.library_path`` prepended to the image's ``LD_LIBRARY_PATH``
        when set, as the core does for its worker.
    """
    script = interpreter_smoke_script(check)
    if check.library_path is None:
        return ["docker", "run", "--rm", image_reference, check.python, "-c", script]
    # Prepend in a shell so the image's own LD_LIBRARY_PATH (the NVIDIA driver paths) is kept; the paths and the script
    # are positional arguments, so nothing in them is parsed by the shell.
    launcher = 'LD_LIBRARY_PATH="$1${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" exec "$2" -c "$3"'
    return [
        "docker",
        "run",
        "--rm",
        image_reference,
        "/bin/sh",
        "-c",
        launcher,
        "sh",
        check.library_path,
        check.python,
        script,
    ]


def iter_image_targets(
    tag: str | None,
    cuda_versions: list[str],
    *,
    docker_repository: str = DEFAULT_DOCKER_REPOSITORY,
    image_specs: Sequence[RuntimeImageSpec] | None = None,
    platform: str | None = None,
) -> list[tuple[str, str, str, Path, Path]]:
    """Return model-image targets for smoke checks.

    Returns tuples of ``(image_key, image_reference, display_tag, requirements_path, core_path)``.
    """
    normalized_tag = normalize_requested_tag(tag)
    if cuda_versions and _CUDA_TAG_PATTERN.fullmatch(normalized_tag):
        raise ValueError("Do not combine --all-cuda/--cuda-version with an already CUDA-qualified --tag.")

    specs = MODEL_IMAGE_SPECS if image_specs is None else image_specs
    requested_platforms = set(split_platforms(platform)) if platform is not None else None
    targets: list[tuple[str, str, str, Path, Path]] = []
    for spec in specs:
        if requested_platforms is not None and not requested_platforms.issubset(get_supported_platforms(spec)):
            continue

        requirements_path = spec.context_path / "requirements.txt"
        core_path = spec.context_path / "core.py"
        if cuda_versions:
            for cuda_version in cuda_versions:
                if cuda_version not in get_supported_cuda(spec):
                    continue
                canonical_ref = published_image_references(
                    spec.image_name, cuda_version, normalized_tag, docker_repository
                )[0]
                display_tag = canonical_ref.rsplit(":", 1)[1]
                targets.append((spec.key, canonical_ref, display_tag, requirements_path, core_path))
            continue

        image_reference = format_image_reference(spec.image_name, normalized_tag, docker_repository)
        targets.append((spec.key, image_reference, normalized_tag, requirements_path, core_path))
    return targets
