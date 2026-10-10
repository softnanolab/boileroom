"""Apptainer backend for running models in containers via HTTP microservice."""

import json
import logging
import os
import platform
import re
import secrets
import shutil
import socket
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from ..images.metadata import DEFAULT_PYTHON_VERSION
from ..optimization import REFUSAL_PROCESS_EXIT_CODE, OptimizationUnavailableError
from ..provenance import IMAGE_REF_ENV
from ..utils import ensure_cache_dir
from .base import Backend
from .transport import TRANSPORT_HMAC_KEY_ENV, deserialize_transport_payload

logger = logging.getLogger(__name__)

#: Environment variable that overrides how long :meth:`ApptainerBackend.startup` waits for the server to become ready.
STARTUP_TIMEOUT_ENV = "BOILEROOM_APPTAINER_STARTUP_TIMEOUT"
#: Default startup wait in seconds. The server answers /health only after the core has loaded its weights, and a first
#: run downloads them into MODEL_DIR: the kit images load about 27 GB (ESMFold2 plus its ESM-C trunk), which takes about
#: 23 minutes at a modest 20 MB/s, before any CUDA extension JIT. 30 minutes covers that; a server that exits is still
#: reported as soon as it dies, so the long wait only applies to a server that hangs.
DEFAULT_STARTUP_TIMEOUT = 1800.0
#: Lines of the server log quoted in a startup error.
_LOG_TAIL_LINES = 50
#: Docker Hub's registry API host; image references name it ``docker.io``.
_DOCKER_HUB_REGISTRY = "registry-1.docker.io"
#: Manifest types accepted when resolving a tag, so a multi-platform tag resolves to its index digest.
_MANIFEST_MEDIA_TYPES = (
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.v2+json",
)
_DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
_REGISTRY_TIMEOUT = httpx.Timeout(10.0)

ARCH_NORMALIZATION = {
    "x86_64": "amd64",
    "amd64": "amd64",
    "aarch64": "arm64",
    "arm64": "arm64",
    "armv7l": "arm",
    "armv6l": "arm",
}


def _normalize_arch(arch: str) -> str:
    """Normalize architecture labels to a small set."""
    return ARCH_NORMALIZATION.get(arch.lower(), arch.lower())


def _find_available_port(start_port: int = 8000, max_attempts: int = 100) -> int:
    """Find an available port starting from start_port.

    Parameters
    ----------
    start_port : int
        Starting port number to check.
    max_attempts : int
        Maximum number of ports to check.

    Returns
    -------
    int
        An available port number.

    Raises
    ------
    RuntimeError
        If no available port is found within max_attempts.
    """
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError(f"Could not find available port in range {start_port}-{start_port + max_attempts - 1}")


def _extract_device_number(device: str) -> str | None:
    """Extract device number from device string (e.g., 'cuda:0' -> '0').

    Parameters
    ----------
    device : str
        Device string in format 'cuda:N' or 'cpu'.

    Returns
    -------
    Optional[str]
        Device number as string, or None if device is 'cpu' or invalid.
    """
    if device.startswith("cuda:"):
        return device.split(":")[1]
    return None


def _is_tool_available(tool_name: str) -> bool:
    """Check if a tool is available in PATH.

    Parameters
    ----------
    tool_name : str
        Name of the tool to check (e.g., 'apptainer', 'singularity').

    Returns
    -------
    bool
        True if the tool is available and executable, False otherwise.
    """
    return shutil.which(tool_name) is not None


def _split_image_reference(image_uri: str) -> tuple[str, str, str | None, str | None]:
    """Split a Docker reference into its registry, repository, tag and digest.

    Parameters
    ----------
    image_uri : str
        Docker reference, with or without the ``docker://`` prefix.

    Returns
    -------
    tuple[str, str, str | None, str | None]
        The registry (``docker.io`` when the reference names none), the repository path, the tag and the digest; the
        tag and the digest are None when the reference has none.
    """
    reference = image_uri.removeprefix("docker://")
    name, _, digest = reference.partition("@")
    first_component, _, remainder = name.partition("/")
    if remainder and ("." in first_component or ":" in first_component or first_component == "localhost"):
        registry, path = first_component, remainder
    else:
        registry, path = "docker.io", name
    repository, tag = path, None
    if ":" in path.rsplit("/", 1)[-1]:
        repository, tag = path.rsplit(":", 1)
    return registry, repository, tag or None, digest or None


def _get_cached_sif_path(image_uri: str, cache_dir: Path) -> Path:
    """Get the cache path for a .sif file from an image URI.

    The file name carries the repository, the image name and the tag or digest, so two references that can resolve to
    different images never share a cache entry.

    Parameters
    ----------
    image_uri : str
        Docker URI, by tag (``docker://docker.io/jakublala/boileroom-chai:0.3.0``) or by digest
        (``docker://docker.io/jakublala/boileroom-esmfold2-kit@sha256:<hex>``).
    cache_dir : Path
        Base cache directory.

    Returns
    -------
    Path
        Path to the cached .sif file, for example ``<cache_dir>/images/jakublala-boileroom-chai_0.3.0.sif`` or
        ``<cache_dir>/images/jakublala-boileroom-esmfold2-kit_sha256-<hex>.sif``. A registry other than docker.io is
        prefixed to the name.
    """
    registry, repository, tag, digest = _split_image_reference(image_uri)
    stem = repository.replace("/", "-") + (f"_{tag}" if tag else "")
    if registry != "docker.io":
        stem = f"{registry.replace(':', '_')}-{stem}"
    if digest:
        algorithm, _, hex_digest = digest.partition(":")
        stem = f"{stem}_{algorithm}-{hex_digest}"
    return cache_dir / "images" / f"{stem}.sif"


def _registry_token(client: httpx.Client, challenge: str) -> str | None:
    """Fetch an anonymous pull token for a registry's ``WWW-Authenticate: Bearer`` challenge, or None."""
    scheme, _, parameters = challenge.partition(" ")
    if scheme.lower() != "bearer":
        return None
    fields = dict(re.findall(r'(\w+)="([^"]*)"', parameters))
    realm = fields.pop("realm", None)
    if not realm:
        return None
    response = client.get(realm, params=fields)
    response.raise_for_status()
    body = response.json()
    token = body.get("token") or body.get("access_token")
    return token if isinstance(token, str) and token else None


def _resolve_registry_digest(image_uri: str, client: httpx.Client | None = None) -> str | None:
    """Return the digest a tag reference points at in its registry now, or None when it cannot be resolved.

    Asks the registry's v2 API for the manifest of the tag (anonymously, as ``apptainer pull`` of a public image does).
    A multi-platform tag resolves to the digest of its index, which ``apptainer pull`` accepts like the tag. A digest
    reference is returned as is. Any failure (no network, a private repository, a registry without the v2 API) is
    logged and returns None, and the caller pulls by tag as before.

    Parameters
    ----------
    image_uri : str
        Docker reference, with or without the ``docker://`` prefix.
    client : httpx.Client | None
        HTTP client to use; None creates one for the call.

    Returns
    -------
    str | None
        The ``sha256:<hex>`` digest, or None.
    """
    registry, repository, tag, digest = _split_image_reference(image_uri)
    if digest is not None:
        return digest
    if tag is None:
        return None
    host = _DOCKER_HUB_REGISTRY if registry == "docker.io" else registry
    if registry == "docker.io" and "/" not in repository:
        repository = f"library/{repository}"
    url = f"https://{host}/v2/{repository}/manifests/{tag}"
    headers = {"Accept": ", ".join(_MANIFEST_MEDIA_TYPES)}
    own_client = client is None
    http = httpx.Client(timeout=_REGISTRY_TIMEOUT, follow_redirects=True) if client is None else client
    try:
        response = http.head(url, headers=headers)
        if response.status_code == 401:
            token = _registry_token(http, response.headers.get("www-authenticate", ""))
            if token is not None:
                response = http.head(url, headers={**headers, "Authorization": f"Bearer {token}"})
        response.raise_for_status()
        resolved = response.headers.get("docker-content-digest", "")
        if not _DIGEST_PATTERN.fullmatch(resolved):
            raise ValueError(f"the registry returned no sha256 digest (Docker-Content-Digest: {resolved!r})")
        return resolved
    except (httpx.HTTPError, ValueError) as error:
        logger.warning(f"Could not resolve the digest of {image_uri}; pulling it by tag: {error}")
        return None
    finally:
        if own_client:
            http.close()


def _digest_record_path(sif_path: Path) -> Path:
    """Return the file beside a cached .sif that records the digest it was pulled at."""
    return sif_path.with_name(f"{sif_path.name}.digest")


def _cached_image_reference(image_uri: str, sif_path: Path) -> str:
    """Return the reference of the image in ``sif_path``, for BOILEROOM_IMAGE_REF.

    A tag reference carries the digest the .sif was pulled at (``<name>:<tag>@sha256:<hex>``) when that was recorded,
    so the provenance names the exact image even after the tag moves on. Without a record (a .sif pulled before
    digests were recorded, or a pull whose digest could not be resolved), it is the reference as given.
    """
    reference = image_uri.removeprefix("docker://")
    if "@" in reference:
        return reference
    try:
        recorded = _digest_record_path(sif_path).read_text().strip()
    except OSError:
        return reference
    return f"{reference}@{recorded}" if _DIGEST_PATTERN.fullmatch(recorded) else reference


def _pull_pinned_image(image_uri: str, sif_path: Path, log_file: Path | None = None) -> None:
    """Pull an image into ``sif_path``, by digest when a tag reference resolves, and record that digest beside it.

    Resolving the tag first and pulling that digest means the recorded digest is exactly the content pulled, even if
    the tag moves during the pull. A digest reference is pulled as is.

    Raises
    ------
    RuntimeError
        If the pull fails (see :func:`_pull_image`).
    """
    record = _digest_record_path(sif_path)
    record.unlink(missing_ok=True)  # a failed pull must not leave a stale record beside whatever .sif remains
    reference = image_uri.removeprefix("docker://")
    # A tag reference only: _resolve_registry_digest returns a digest reference's own digest, which needs no record.
    resolved = None if "@" in reference else _resolve_registry_digest(image_uri)
    if resolved is None:
        _pull_image(image_uri, sif_path, log_file=log_file)
        return
    name = reference.rsplit(":", 1)[0]  # resolved, so the reference ends in ``:<tag>``
    logger.info(f"{image_uri} resolves to {resolved}; pulling that digest")
    _pull_image(f"docker://{name}@{resolved}", sif_path, log_file=log_file)
    record.write_text(f"{resolved}\n")


def _resolve_startup_timeout(startup_timeout: float | None) -> float:
    """Return the startup timeout: the argument, else BOILEROOM_APPTAINER_STARTUP_TIMEOUT, else the default.

    Raises
    ------
    ValueError
        If the chosen value is not a positive number of seconds.
    """
    if startup_timeout is None:
        raw = os.environ.get(STARTUP_TIMEOUT_ENV, "").strip()
        if not raw:
            return DEFAULT_STARTUP_TIMEOUT
        try:
            startup_timeout = float(raw)
        except ValueError:
            raise ValueError(f"{STARTUP_TIMEOUT_ENV} must be a number of seconds, got {raw!r}") from None
    if not startup_timeout > 0 or startup_timeout == float("inf"):
        raise ValueError(
            f"The Apptainer startup timeout must be a positive, finite number of seconds, got {startup_timeout!r}"
        )
    return float(startup_timeout)


def _read_log_tail(log_path: Path | None, lines: int = _LOG_TAIL_LINES) -> str:
    """Return the last ``lines`` lines of the server log, or a note saying why there are none."""
    if log_path is None or not log_path.exists():
        return "(no server log)"
    try:
        with open(log_path, errors="replace") as handle:
            return "".join(handle.readlines()[-lines:])
    except OSError as exc:
        logger.debug(f"Failed to read log file for error context: {exc}")
        return f"(Could not read log file: {exc})"


def _is_image_cached(sif_path: Path) -> bool:
    """Check if a .sif file exists and is valid.

    Parameters
    ----------
    sif_path : Path
        Path to .sif file.

    Returns
    -------
    bool
        True if file exists and is not empty, False otherwise.
    """
    return sif_path.exists() and sif_path.stat().st_size > 0


def _get_host_architecture() -> str:
    """Get the host system architecture.

    Returns
    -------
    str
        Architecture string (e.g., 'amd64', 'arm64', 'x86_64', 'aarch64').
    """
    return _normalize_arch(platform.machine())


def _get_image_architecture(sif_path: Path) -> str | None:
    """Get the architecture of a cached .sif file.

    Parameters
    ----------
    sif_path : Path
        Path to .sif file.

    Returns
    -------
    str | None
        Architecture string (e.g., 'amd64', 'arm64') if found, None otherwise.
    """
    result = subprocess.run(
        ["apptainer", "inspect", "--json", str(sif_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    try:
        data = json.loads(result.stdout)
        if isinstance(data, dict):
            arch = (
                data.get("data", {}).get("attributes", {}).get("arch") or data.get("arch") or data.get("architecture")
            )
            return arch.lower() if arch else None
    except (json.JSONDecodeError, KeyError):
        pass
    return None


def _check_architecture_compatibility(host_arch: str, image_arch: str) -> bool:
    """Check if image architecture matches host architecture.

    Parameters
    ----------
    host_arch : str
        Host architecture (e.g., 'amd64', 'arm64').
    image_arch : str
        Image architecture.

    Returns
    -------
    bool
        True if architectures match, False otherwise.
    """
    return _normalize_arch(host_arch) == _normalize_arch(image_arch)


def _build_ld_library_path(python_version: str = DEFAULT_PYTHON_VERSION) -> str:
    """Build LD_LIBRARY_PATH for CUDA libraries in the container.

    Includes paths for:
    - cuequivariance_ops (libcue_ops.so)
    - NVIDIA Python wheel packages (libcublas.so.12)
    - PyTorch CUDA libraries (libnvrtc.so.12)
    - System CUDA toolkit paths
    - NVIDIA driver libraries (added by --nv flag)

    Parameters
    ----------
    python_version : str
        Version of the interpreter whose ``site-packages`` holds those libraries.

    Returns
    -------
    str
        Colon-separated LD_LIBRARY_PATH string.
    """
    site_packages = f"/usr/local/lib/python{python_version}/site-packages"

    python_lib_paths = [
        f"{site_packages}/cuequivariance_ops/lib",  # libcue_ops.so
        f"{site_packages}/nvidia/cublas/lib",  # libcublas.so.12
        f"{site_packages}/torch/lib",  # libnvrtc.so.12 and other PyTorch CUDA libs
        f"{site_packages}/nvidia",  # Other NVIDIA Python wheel packages
    ]

    # System CUDA toolkit paths (fallback if provided by the container)
    cuda_toolkit_paths = [
        "/usr/local/cuda/lib64",
        "/usr/local/cuda/lib",
        "/usr/local/cuda-12/lib64",
        "/usr/local/cuda-12/lib",
    ]

    # NVIDIA driver paths (--nv flag adds these, but include explicitly for clarity)
    nvidia_driver_paths = [
        "/usr/local/nvidia/lib64",
        "/usr/local/nvidia/lib",
        "/.singularity.d/libs",
    ]

    # Combine all paths in priority order. Host LD_LIBRARY_PATH is intentionally
    # NOT appended: the container has its own torch wheel and `--nv` provides
    # driver libs, while host paths can drag in libs (including a host libpython)
    # that collide with the container's, producing
    # `Fatal Python error: _PyImport_Init: global import state already initialized`.
    ld_path_parts = python_lib_paths + cuda_toolkit_paths + nvidia_driver_paths

    return ":".join(ld_path_parts)


def _pull_image(image_uri: str, sif_path: Path, log_file: Path | None = None) -> None:
    """Pull Docker image and convert to .sif format.

    Parameters
    ----------
    image_uri : str
        Docker URI to pull (e.g., 'docker://docker.io/jakublala/boileroom-chai:0.3.0').
    sif_path : Path
        Path where .sif file should be saved.
    log_file : Path | None
        Optional log file to redirect stdout/stderr to.

    Raises
    ------
    RuntimeError
        If image pull fails or architecture is incompatible.
    """
    logger.info(f"Pulling image: {image_uri}")
    sif_path.parent.mkdir(parents=True, exist_ok=True)

    # Set APPTAINER_TMPDIR to a writable location for the pull command
    # This is needed because apptainer pull creates temporary build directories
    # Use existing APPTAINER_TMPDIR if set, otherwise use TMPDIR, or fall back to cache_dir/tmp
    apptainer_tmpdir = os.environ.get("APPTAINER_TMPDIR")
    if not apptainer_tmpdir:
        apptainer_tmpdir = os.environ.get("TMPDIR")
    if not apptainer_tmpdir:
        # Fall back to a tmp directory in the cache directory
        apptainer_tmpdir = str(sif_path.parent.parent / "tmp")

    apptainer_tmpdir_path = Path(apptainer_tmpdir)
    apptainer_tmpdir_path.mkdir(parents=True, exist_ok=True)

    # Prepare environment with APPTAINER_TMPDIR set
    env = os.environ.copy()
    env["APPTAINER_TMPDIR"] = str(apptainer_tmpdir_path)
    logger.debug(f"Setting APPTAINER_TMPDIR={env['APPTAINER_TMPDIR']} for image pull")

    cmd = ["apptainer", "pull", "--force", str(sif_path), image_uri]
    if log_file is not None:
        with open(log_file, "a") as f:
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True, check=False, env=env)
    else:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)

    if result.returncode != 0:
        error_msg = f"Failed to pull Apptainer image '{image_uri}':\n"
        error_msg += f"Command: {' '.join(cmd)}\nReturn code: {result.returncode}\n"
        if log_file is None:
            if result.stdout:
                error_msg += f"stdout:\n{result.stdout}\n"
            if result.stderr:
                error_msg += f"stderr:\n{result.stderr}\n"
        else:
            error_msg += f"See log file: {log_file}\n"
        raise RuntimeError(error_msg)

    if not _is_image_cached(sif_path):
        raise RuntimeError(f"Image pull completed but .sif file not found at {sif_path}")

    host_arch = _get_host_architecture()
    pulled_arch = _get_image_architecture(sif_path)
    if pulled_arch and not _check_architecture_compatibility(host_arch, pulled_arch):
        raise RuntimeError(
            f"Architecture mismatch: host is {host_arch}, image is {pulled_arch}. "
            f"Remove {sif_path} and pull a compatible version."
        )


class ApptainerBackend(Backend):
    """Backend that runs models in Apptainer containers via HTTP microservice.

    This backend ensures complete dependency independence between Boiler Room
    and model-specific environments by running containers from pre-built Docker images.
    Images are pulled from DockerHub and cached locally.
    """

    def __init__(
        self,
        core_class_path: str,
        image_uri: str,
        config: dict | None = None,
        device: str | None = None,
        cache_dir: Path | str | None = None,
        python_version: str = DEFAULT_PYTHON_VERSION,
        startup_timeout: float | None = None,
    ) -> None:
        """Initialize the ApptainerBackend with a Core class path and Docker image.

        Parameters
        ----------
        core_class_path : str
            Full module path to the Core class (e.g., 'boileroom.models.esm.esm2.ESM2Core').
            This is passed as a string to avoid importing the class in the main process.
        image_uri : str
            Docker URI for the container image (e.g., 'docker://docker.io/jakublala/boileroom-chai:0.3.0').
        config : dict | None
            Optional configuration mapping for the model.
        device : str | None
            Optional device identifier (e.g., 'cuda:0' or 'cpu').
        cache_dir : Path | str | None
            Optional cache directory for .sif files. If None, uses ~/.cache/boileroom.
        python_version : str
            Version of the interpreter in the image that runs the service (``/usr/local/bin/python<version>``).
        startup_timeout : float | None
            Seconds :meth:`startup` waits for the server to answer /health. None reads
            ``BOILEROOM_APPTAINER_STARTUP_TIMEOUT``, falling back to :data:`DEFAULT_STARTUP_TIMEOUT` (1800 s).

        Raises
        ------
        ValueError
            If apptainer is not available in PATH, or the startup timeout is not a positive number.
        """
        super().__init__()
        self._core_class_path = core_class_path
        self._config = dict(config) if config is not None else {}
        self._device = device or "cuda:0"
        self._image_uri = image_uri
        self._python_version = python_version
        self._startup_timeout = _resolve_startup_timeout(startup_timeout)

        # Check if apptainer is available
        if not _is_tool_available("apptainer"):
            raise ValueError(
                "To use the ApptainerBackend, you need to install Apptainer.\n"
                "See https://apptainer.org/docs/user/main/quick_start.html for installation instructions."
            )

        # Set up cache directory
        if cache_dir is None:
            model_dir = os.environ.get("MODEL_DIR")
            cache_dir = Path(model_dir).expanduser().resolve() if model_dir else ensure_cache_dir()
        else:
            cache_dir = Path(cache_dir)
        self._cache_dir = cache_dir
        self._sif_path = _get_cached_sif_path(image_uri, cache_dir)

        self._port: int | None = None
        self._base_url: str | None = None
        self._process: subprocess.Popen[str] | None = None
        self._client: httpx.Client | None = None
        self._log_file_path: Path | None = None
        self._transport_secret = secrets.token_hex(32)

    def startup(self) -> None:
        """Start the Apptainer container server and wait for it to be ready.

        This method:
        1. Pulls the Docker image if not cached locally
        2. Finds an available port
        3. Launches the server subprocess inside the container
        4. Waits for health check to confirm server is ready
        """
        if self._process is not None:
            logger.debug("ApptainerBackend.startup() called but process already exists, skipping startup")
            return

        # Set up log file path: MODEL_DIR/logs/apptainer_YYYY-MM-DD_HH-MM-SS.log
        model_dir = os.environ.get("MODEL_DIR")
        model_dir_path = Path(model_dir).expanduser().resolve() if model_dir else self._cache_dir

        log_dir = model_dir_path / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_filename = f"apptainer_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
        self._log_file_path = log_dir / log_filename

        logger.info(f"Starting ApptainerBackend with image_uri={self._image_uri}, device={self._device}")

        if not _is_image_cached(self._sif_path):
            _pull_pinned_image(self._image_uri, self._sif_path, log_file=self._log_file_path)
        else:
            host_arch = _get_host_architecture()
            cached_arch = _get_image_architecture(self._sif_path)
            if cached_arch and not _check_architecture_compatibility(host_arch, cached_arch):
                raise RuntimeError(
                    f"Architecture mismatch: host is {host_arch}, cached image is {cached_arch}. "
                    f"Remove {self._sif_path} and pull a compatible version."
                )

        # Find available port
        self._port = _find_available_port()
        self._base_url = f"http://127.0.0.1:{self._port}"
        logger.info(f"Selected port {self._port} for server, base_url={self._base_url}")

        # Get project root (parent of parent of parent of this file: boileroom/backend/apptainer.py)
        project_root = Path(__file__).parent.parent.parent
        server_path = Path(__file__).parent / "server.py"

        # Determine path in container (should match host path for bind mount)
        container_boileroom = str(project_root)
        container_server_path = str(server_path)

        # Build apptainer exec command.
        # --cleanenv: container starts with only the env we explicitly pass via --env.
        #   Without this, host PATH (with .venv/bin first under `uv run`), VIRTUAL_ENV,
        #   PYTHONHOME, and LD_LIBRARY_PATH leak into the container, allowing the host
        #   libpython to be loaded alongside the container's libpython and triggering
        #   `Fatal Python error: _PyImport_Init: global import state already initialized`.
        # --home /tmp: set the container's HOME to /tmp (writable, auto-mounted) and
        #   suppress the auto-mount of the host $HOME. `HOME` cannot be set via --env
        #   (apptainer ignores APPTAINERENV_HOME and warns); --home is the supported way.
        #   xet writes its logs/chunks under $HOME/.cache/huggingface, so HOME must
        #   resolve to a writable path inside the container.
        cmd = ["apptainer", "exec", "--cleanenv", "--home", "/tmp"]

        # Enable NVIDIA GPU support if device is CUDA
        device_number = _extract_device_number(self._device)
        if self._device.startswith("cuda"):
            cmd.append("--nv")

        # Bind mount only the boileroom package directory, read-only.
        # Binding the full project root would expose the host .venv/ inside the
        # container; the resulting host-built C extensions on the import path can
        # load a second libpython and crash interpreter startup.
        host_pkg_dir = project_root / "boileroom"
        cmd.extend(["-B", f"{host_pkg_dir}:{host_pkg_dir}:ro"])

        # Bind mount MODEL_DIR if present
        if model_dir:
            # Use the already-resolved model_dir_path from above
            try:
                model_dir_path.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                logger.warning(f"Failed to create MODEL_DIR {model_dir_path} on host: {e}")
                raise RuntimeError(
                    f"Cannot create MODEL_DIR {model_dir_path} on host. "
                    f"Ensure parent directories exist and you have write permissions."
                ) from e

            if not model_dir_path.exists():
                raise RuntimeError(f"MODEL_DIR {model_dir_path} does not exist after creation attempt")

            if not model_dir_path.is_dir():
                raise RuntimeError(f"MODEL_DIR {model_dir_path} exists but is not a directory")

            # Mount host MODEL_DIR to fixed container path for simplicity
            container_model_dir = "/.model_cache"
            cmd.extend(["-B", f"{model_dir_path}:{container_model_dir}"])
            logger.info(f"Bind mounting MODEL_DIR: {model_dir_path} -> {container_model_dir}")

        # Set environment variables. With --cleanenv, these are the *only* vars
        # the container sees, so we must set everything the server needs (PATH,
        # HOME, locale) ourselves rather than relying on inheritance.
        env_vars = {
            "MODEL_CLASS": self._core_class_path,
            "MODEL_CONFIG": json.dumps(self._config),
            "DEVICE": self._device,
            TRANSPORT_HMAC_KEY_ENV: self._transport_secret,
            "PYTHONPATH": container_boileroom,
            "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            # Override temp directory variables to use container's /tmp
            # This prevents issues when host TMPDIR points to a path that doesn't exist in container
            "TMPDIR": "/tmp",
            "TMP": "/tmp",
            "TEMP": "/tmp",
            # Set C compiler for Triton (needed for runtime CUDA kernel compilation)
            "CC": "gcc",
            "CXX": "g++",
            # The image this server runs, reported as ``image_ref`` in each prediction's provenance: by digest
            # (``<name>:<tag>@sha256:<hex>``) when the cached .sif's digest is recorded, so a moved tag still shows.
            IMAGE_REF_ENV: _cached_image_reference(self._image_uri, self._sif_path),
        }

        # Build LD_LIBRARY_PATH to include Python wheel CUDA libraries and driver paths.
        env_vars["LD_LIBRARY_PATH"] = _build_ld_library_path(self._python_version)

        if device_number is not None:
            env_vars["CUDA_VISIBLE_DEVICES"] = device_number

        # Set MODEL_DIR to container path if host MODEL_DIR is present
        if model_dir:
            # Use container path where MODEL_DIR is mounted
            container_model_dir_env = "/.model_cache"
            env_vars["MODEL_DIR"] = container_model_dir_env
            # Automatically derive CHAI_DOWNLOADS_DIR from MODEL_DIR/chai
            # Only set if not already explicitly set (allows override if needed)
            if "CHAI_DOWNLOADS_DIR" not in os.environ:
                # Directly construct container path since MODEL_DIR will be /.model_cache in container
                env_vars["CHAI_DOWNLOADS_DIR"] = f"{container_model_dir_env}/chai"

        # Add environment variables to command
        for key, value in env_vars.items():
            cmd.extend(["--env", f"{key}={value}"])

        # Add image and command. Use the absolute path to the container's
        # python so PATH-based resolution can never reach a host binary that
        # leaked through the bind mounts.
        cmd.append(str(self._sif_path))
        cmd.extend(
            [
                f"/usr/local/bin/python{self._python_version}",
                container_server_path,
                "--host",
                "0.0.0.0",
                "--port",
                str(self._port),
            ]
        )

        logger.debug("Launching Apptainer server with command: %s", " ".join(cmd))
        logger.info("Starting server in container...")

        # Launch subprocess with stdout/stderr redirected to log file
        with open(self._log_file_path, "a") as log_file:
            self._process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,  # Combine stderr into stdout
                text=True,
                cwd=str(project_root),
            )

        logger.info(f"Waiting for health check at {self._base_url} (port {self._port})")
        try:
            self._wait_for_health_check(timeout=self._startup_timeout)
        except BaseException:
            # Never leave a half-started server behind: it would hold the GPU and the port.
            self._stop_process()
            raise

        logger.info(f"Creating HTTP client for {self._base_url}")
        # Use 30 minute timeout to handle large responses (e.g., PAE matrices for long sequences)
        # This matches Modal backend timeout of 20 minutes with some buffer for serialization/transmission
        # Set explicit timeouts: connect=10s, read=1800s (30min), write=60s, pool=10s
        # The read timeout is the critical one for large response bodies
        timeout_config = httpx.Timeout(connect=10.0, read=1800.0, write=60.0, pool=10.0)
        self._client = httpx.Client(base_url=self._base_url, timeout=timeout_config)
        logger.info("Apptainer backend startup complete")

    def shutdown(self) -> None:
        """Shutdown the Apptainer container server gracefully.

        Sends SIGTERM to the subprocess and waits for graceful termination.
        If the process doesn't terminate within 10 seconds, sends SIGKILL.
        """
        if self._client is not None:
            self._client.close()
            self._client = None

        self._stop_process()
        self._base_url = None
        self._port = None

    def _stop_process(self) -> None:
        """Terminate the server process, kill it if it ignores SIGTERM for 10 s, and reap it."""
        if self._process is None:
            return
        try:
            if self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait()
        finally:
            self._process = None

    def get_model(self) -> Any:
        """Get the HTTP proxy object for making requests to the container server.

        Returns
        -------
        Any
            HTTP client proxy object with methods like embed() that serialize inputs,
            POST to /embed, and deserialize outputs.

        Raises
        ------
        RuntimeError
            If the backend has not been started.
        """
        if self._client is None:
            raise RuntimeError("Apptainer backend is not initialized. Call startup() before use.")
        return _ApptainerModelProxy(
            self._client,
            transport_secret=self._transport_secret,
            log_file_path=self._log_file_path,
        )

    def _wait_for_health_check(self, timeout: float = DEFAULT_STARTUP_TIMEOUT, poll_interval: float = 1.0) -> None:
        """Wait for the server to become ready by polling the /health endpoint.

        Parameters
        ----------
        timeout : float
            Maximum time to wait in seconds; :meth:`startup` passes the backend's startup timeout.
        poll_interval : float
            Time between health check attempts in seconds.

        Raises
        ------
        OptimizationUnavailableError
            If the server exits with the refusal code (:data:`boileroom.optimization.REFUSAL_PROCESS_EXIT_CODE`):
            the core refused the requested optimization mode on this GPU or in this image.
        RuntimeError
            If the server process exits otherwise, or does not become ready within the timeout. The errors quote the
            tail of the server log; :meth:`startup` stops the process before re-raising.
        """
        if self._base_url is None:
            raise RuntimeError("Base URL not set")

        start_time = time.time()
        health_check_count = 0
        while time.time() - start_time < timeout:
            elapsed = time.time() - start_time
            health_check_count += 1

            try:
                if (
                    health_check_count == 1 or elapsed % 5 < poll_interval
                ):  # Log first check and roughly every 5 seconds
                    logger.debug(f"Attempting health check to {self._base_url}/health (elapsed: {elapsed:.1f}s)")
                response = httpx.get(f"{self._base_url}/health", timeout=5.0)
                logger.debug(f"Health check response: status={response.status_code}, url={response.url}")
                if response.status_code == 200:
                    logger.info(
                        f"Health check passed at {self._base_url}, server is ready (check #{health_check_count}, elapsed: {elapsed:.1f}s)"
                    )
                    return
                else:
                    logger.warning(f"Health check returned non-200 status: {response.status_code}")
            except httpx.TimeoutException:
                # Only log failures at debug level to avoid spam
                if elapsed % 10 < poll_interval:  # Log roughly every 10 seconds
                    logger.debug(f"Health check timeout, base_url={self._base_url}, elapsed={elapsed:.1f}s")
            except httpx.RequestError as e:
                # Only log failures at debug level to avoid spam
                if elapsed % 10 < poll_interval:  # Log roughly every 10 seconds
                    logger.debug(f"Health check request error: {e}, base_url={self._base_url}, elapsed={elapsed:.1f}s")
            except Exception as e:
                # Catch any other unexpected exceptions
                logger.warning(
                    f"Unexpected error during health check: {e}, base_url={self._base_url}, elapsed={elapsed:.1f}s",
                    exc_info=True,
                )

            # Check if process has died
            returncode = self._process.poll() if self._process is not None else None
            if returncode is not None:
                log_tail = (
                    f"Last {_LOG_TAIL_LINES} lines of log ({self._log_file_path}):\n"
                    f"{_read_log_tail(self._log_file_path)}"
                )
                if returncode == REFUSAL_PROCESS_EXIT_CODE:
                    raise OptimizationUnavailableError(
                        f"The model server refused to load the model (exit code {returncode}): the requested "
                        f"optimization mode is not available with this GPU or image. {log_tail}"
                    )
                raise RuntimeError(f"Server process died (exit code {returncode}). {log_tail}")

            time.sleep(poll_interval)

        raise RuntimeError(
            f"Server did not become ready within {timeout:g} seconds (set {STARTUP_TIMEOUT_ENV} to wait longer). "
            f"Last {_LOG_TAIL_LINES} lines of log ({self._log_file_path}):\n{_read_log_tail(self._log_file_path)}"
        )


class _ApptainerModelProxy:
    """HTTP proxy for making requests to the Apptainer container server."""

    def __init__(self, client: httpx.Client, transport_secret: str, log_file_path: Path | None = None) -> None:
        """Initialize the proxy with an HTTP client.

        Parameters
        ----------
        client : httpx.Client
            HTTP client configured with the server's base URL.
        transport_secret : str
            Shared HMAC secret used to verify server responses.
        log_file_path : Path | None
            Optional path to the server log file for error reporting.
        """
        self._client = client
        self._transport_secret = transport_secret
        self._log_file_path = log_file_path

    def embed(self, sequences: str | list[str], options: dict | None = None) -> Any:
        """Embed sequences by making a POST request to /embed.

        Parameters
        ----------
        sequences : str | list[str]
            Single sequence or list of sequences. Multimer sequences with ':'
            separator are passed through as-is.
        options : dict | None
            Optional per-call configuration options.

        Returns
        -------
        Any
            Deserialized embedding output with numpy arrays reconstructed.

        Raises
        ------
        OptimizationUnavailableError
            If the core refused the call's optimization mode.
        RuntimeError
            If the server failed the call otherwise (HTTP 500).
        """
        return self._post("/embed", {"sequences": sequences, "options": options})

    def inverse_fold(self, sequence: str, backbone_coordinates: Any, positions: list[int]) -> Any:
        """Predict masked-residue amino-acid logits by making a POST request to /inverse_fold.

        Parameters
        ----------
        sequence : str
            Amino-acid sequence; chains separated by ``:``.
        backbone_coordinates : array-like
            ``(n_residues, 3, 3)`` N, CA, C coordinates. NaN entries (missing atoms) are sent as JSON ``null``.
        positions : list[int]
            Residue indices to mask.

        Returns
        -------
        Any
            Deserialized inverse-folding output with numpy arrays reconstructed.

        Raises
        ------
        RuntimeError
            If the server failed the call (HTTP 500).
        """
        coordinates = np.asarray(backbone_coordinates, dtype=np.float64)
        json_coordinates = coordinates.astype(object)
        json_coordinates[np.isnan(coordinates)] = None  # JSON has no NaN
        payload = {
            "sequence": sequence,
            "backbone_coordinates": json_coordinates.tolist(),
            "positions": [int(position) for position in positions],
        }
        return self._post("/inverse_fold", payload)

    def fold(self, sequences: str | list[str], options: dict | None = None) -> Any:
        """Fold sequences by making a POST request to /fold.

        Parameters
        ----------
        sequences : str | list[str]
            Single sequence or list of sequences. Multimer sequences with ':'
            separator are passed through as-is.
        options : dict | None
            Optional per-call configuration options.

        Returns
        -------
        Any
            Deserialized folding output with numpy arrays reconstructed.

        Raises
        ------
        OptimizationUnavailableError
            If the core refused the call's optimization mode.
        RuntimeError
            If the server failed the call otherwise (HTTP 500).
        """
        return self._post("/fold", {"sequences": sequences, "options": options})

    def _post(self, path: str, payload: dict[str, Any]) -> Any:
        """POST ``payload`` to ``path`` and return the verified, deserialized output.

        A 500 response names the exception type the core raised (``error_type``, see ``server.py``); a refusal is
        raised again as :class:`OptimizationUnavailableError`, anything else as ``RuntimeError``.
        """
        response = self._client.post(path, json=payload)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            if response.status_code != 500:
                raise
            try:
                body = response.json()
            except ValueError:
                body = None
            body = body if isinstance(body, dict) else {}
            detail = body.get("detail") or response.text
            log_file_msg = f"\n\nServer log file: {self._log_file_path}" if self._log_file_path is not None else ""
            if body.get("error_type") == OptimizationUnavailableError.__name__:
                raise OptimizationUnavailableError(f"{detail}{log_file_msg}") from e
            # Raise RuntimeError with log file path, chaining from the original HTTPStatusError
            raise RuntimeError(
                f"Internal server error (500) occurred: {detail}{log_file_msg}\nHTTP request failed: {e}"
            ) from e
        return _deserialize_output(response.json(), self._transport_secret)


def _deserialize_output(data: dict[str, Any], transport_secret: str) -> Any:
    """Deserialize and verify a signed JSON response payload.

    Parameters
    ----------
    data : dict[str, Any]
        JSON response containing a signed payload.
    transport_secret : str
        Shared HMAC secret negotiated with the server process.

    Returns
    -------
    Any
        Deserialized output object (e.g., ESM2Output).
    """
    return deserialize_transport_payload(data, transport_secret)
