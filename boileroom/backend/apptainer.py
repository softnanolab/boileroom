"""Apptainer backend for running models in containers via HTTP microservice."""

import json
import logging
import os
import platform
import shutil
import socket
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import httpx

from .base import Backend
from ..utils import ensure_cache_dir

logger = logging.getLogger(__name__)

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


def _extract_device_number(device: str) -> Optional[str]:
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


def _get_cached_sif_path(image_uri: str, cache_dir: Path) -> Path:
    """Get the cache path for a .sif file from an image URI.

    Parameters
    ----------
    image_uri : str
        Docker URI (e.g., 'docker://docker.io/jakublala/boileroom-chai:latest').
    cache_dir : Path
        Base cache directory.

    Returns
    -------
    Path
        Path to cached .sif file.
    """
    # Extract image name from URI
    # docker://docker.io/jakublala/boileroom-chai:latest -> boileroom-chai_latest.sif
    parsed = urlparse(image_uri.replace("docker://", "https://"))
    image_name = parsed.path.lstrip("/").replace("/", "-").replace(":", "_")
    if not image_name.endswith(".sif"):
        image_name = f"{image_name}.sif"

    return cache_dir / "images" / image_name


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
                data.get("data", {}).get("attributes", {}).get("arch")
                or data.get("arch")
                or data.get("architecture")
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


def _build_ld_library_path() -> str:
    """Build LD_LIBRARY_PATH for CUDA libraries in conda environments.

    Includes paths for:
    - cuequivariance_ops (libcue_ops.so)
    - NVIDIA conda packages (libcublas.so.12)
    - PyTorch CUDA libraries (libnvrtc.so.12)
    - System CUDA toolkit paths
    - NVIDIA driver libraries (added by --nv flag)

    Returns
    -------
    str
        Colon-separated LD_LIBRARY_PATH string.
    """
    conda_base = "/opt/conda/lib"
    python_version = "3.12"
    site_packages = f"{conda_base}/python{python_version}/site-packages"

    # Conda package-specific library paths (highest priority)
    conda_lib_paths = [
        f"{site_packages}/cuequivariance_ops/lib",  # libcue_ops.so
        f"{site_packages}/nvidia/cublas/lib",  # libcublas.so.12
        f"{site_packages}/torch/lib",  # libnvrtc.so.12 and other PyTorch CUDA libs
        f"{site_packages}/nvidia",  # Other NVIDIA conda packages
        conda_base,  # Base conda lib directory
    ]

    # System CUDA toolkit paths (fallback if not in conda)
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

    # Combine all paths in priority order
    ld_path_parts = conda_lib_paths + cuda_toolkit_paths + nvidia_driver_paths

    # Append host LD_LIBRARY_PATH if present (lowest priority)
    if "LD_LIBRARY_PATH" in os.environ:
        host_paths = [p for p in os.environ["LD_LIBRARY_PATH"].split(":") if p and p not in ld_path_parts]
        ld_path_parts.extend(host_paths)

    return ":".join(ld_path_parts)


def _pull_image(image_uri: str, sif_path: Path, log_file: Path | None = None) -> None:
    """Pull Docker image and convert to .sif format.

    Parameters
    ----------
    image_uri : str
        Docker URI to pull (e.g., 'docker://docker.io/jakublala/boileroom-chai:latest').
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
    ) -> None:
        """Initialize the ApptainerBackend with a Core class path and Docker image.

        Parameters
        ----------
        core_class_path : str
            Full module path to the Core class (e.g., 'boileroom.models.esm.esm2.ESM2Core').
            This is passed as a string to avoid importing the class in the main process.
        image_uri : str
            Docker URI for the container image (e.g., 'docker://docker.io/jakublala/boileroom-chai:latest').
        config : dict | None
            Optional configuration mapping for the model.
        device : str | None
            Optional device identifier (e.g., 'cuda:0' or 'cpu').
        cache_dir : Path | str | None
            Optional cache directory for .sif files. If None, uses ~/.cache/boileroom.

        Raises
        ------
        ValueError
            If apptainer is not available in PATH.
        """
        super().__init__()
        self._core_class_path = core_class_path
        self._config = dict(config) if config is not None else {}
        self._device = device or "cuda:0"
        self._image_uri = image_uri

        # Check if apptainer is available
        if not _is_tool_available("apptainer"):
            raise ValueError(
                "To use the ApptainerBackend, you need to install Apptainer.\n"
                "See https://apptainer.org/docs/user/main/quick_start.html for installation instructions."
            )

        # Set up cache directory
        if cache_dir is None:
            model_dir = os.environ.get("MODEL_DIR")
            if model_dir:
                cache_dir = Path(model_dir).expanduser().resolve()
            else:
                cache_dir = ensure_cache_dir()
        else:
            cache_dir = Path(cache_dir)
        self._cache_dir = cache_dir
        self._sif_path = _get_cached_sif_path(image_uri, cache_dir)

        self._port: int | None = None
        self._base_url: str | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._client: httpx.Client | None = None
        self._log_file_path: Path | None = None

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
        if model_dir:
            model_dir_path = Path(model_dir).expanduser().resolve()
        else:
            model_dir_path = self._cache_dir
        
        log_dir = model_dir_path / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_filename = f"apptainer_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
        self._log_file_path = log_dir / log_filename

        logger.info(f"Starting ApptainerBackend with image_uri={self._image_uri}, device={self._device}")

        if not _is_image_cached(self._sif_path):
            _pull_image(self._image_uri, self._sif_path, log_file=self._log_file_path)
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

        # Build apptainer exec command
        cmd = ["apptainer", "exec"]

        # Enable NVIDIA GPU support if device is CUDA
        device_number = _extract_device_number(self._device)
        if device_number is not None:
            cmd.append("--nv")

        # Bind mount boileroom source code (read-only)
        cmd.extend(["-B", f"{container_boileroom}:{container_boileroom}:ro"])

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

        # Set environment variables
        env_vars = {
            "MODEL_CLASS": self._core_class_path,
            "MODEL_CONFIG": json.dumps(self._config),
            "DEVICE": self._device,
            "PYTHONPATH": container_boileroom,
            # Override temp directory variables to use container's /tmp
            # This prevents issues when host TMPDIR points to a path that doesn't exist in container
            # CRITICAL: Must override TMPDIR before micromamba runs, otherwise it fails
            "TMPDIR": "/tmp",
            "TMP": "/tmp",
            "TEMP": "/tmp",
            # Set C compiler for Triton (needed for runtime CUDA kernel compilation)
            "CC": "gcc",
            "CXX": "g++",
        }

        # Build LD_LIBRARY_PATH to include all necessary CUDA library paths
        # This ensures conda-installed CUDA libraries (libcue_ops.so, libcublas.so.12, libnvrtc.so.12)
        # and system CUDA toolkit libraries are accessible
        env_vars["LD_LIBRARY_PATH"] = _build_ld_library_path()

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

        # Add image and command
        cmd.append(str(self._sif_path))
        cmd.extend(
            [
                "micromamba",
                "run",
                "-n",
                "base",
                "python",
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
        self._wait_for_health_check()
        
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

        if self._process is not None:
            try:
                self._process.terminate()
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
            finally:
                self._process = None

        self._base_url = None
        self._port = None

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
            raise RuntimeError("Apptainer backend is not initialized. Call start() before use.")
        return _ApptainerModelProxy(self._client, log_file_path=self._log_file_path)

    def _wait_for_health_check(self, timeout: float = 300.0, poll_interval: float = 1.0) -> None:
        """Wait for the server to become ready by polling the /health endpoint.

        Parameters
        ----------
        timeout : float
            Maximum time to wait in seconds (default 300s = 5 minutes for model downloads).
        poll_interval : float
            Time between health check attempts in seconds.

        Raises
        ------
        RuntimeError
            If the server doesn't become ready within the timeout period.
        """
        if self._base_url is None:
            raise RuntimeError("Base URL not set")

        start_time = time.time()
        health_check_count = 0
        while time.time() - start_time < timeout:
            elapsed = time.time() - start_time
            health_check_count += 1

            try:
                if health_check_count == 1 or elapsed % 5 < poll_interval:  # Log first check and roughly every 5 seconds
                    logger.debug(f"Attempting health check to {self._base_url}/health (elapsed: {elapsed:.1f}s)")
                response = httpx.get(f"{self._base_url}/health", timeout=5.0)
                logger.debug(f"Health check response: status={response.status_code}, url={response.url}")
                if response.status_code == 200:
                    logger.info(f"Health check passed at {self._base_url}, server is ready (check #{health_check_count}, elapsed: {elapsed:.1f}s)")
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
                logger.warning(f"Unexpected error during health check: {e}, base_url={self._base_url}, elapsed={elapsed:.1f}s", exc_info=True)

            # Check if process has died
            if self._process is not None and self._process.poll() is not None:
                # Read last 50 lines from log file for error context
                error_context = ""
                if self._log_file_path is not None and self._log_file_path.exists():
                    try:
                        with open(self._log_file_path) as f:
                            lines = f.readlines()
                            error_context = "".join(lines[-50:])
                    except Exception as e:
                        logger.debug(f"Failed to read log file for error context: {e}")
                        error_context = f"(Could not read log file: {e})"
                
                error_msg = f"Server process died. Last 50 lines of log:\n{error_context}"
                raise RuntimeError(error_msg)

            time.sleep(poll_interval)

        raise RuntimeError(f"Server did not become ready within {timeout} seconds")


class _ApptainerModelProxy:
    """HTTP proxy for making requests to the Apptainer container server."""

    def __init__(self, client: httpx.Client, log_file_path: Path | None = None) -> None:
        """Initialize the proxy with an HTTP client.

        Parameters
        ----------
        client : httpx.Client
            HTTP client configured with the server's base URL.
        log_file_path : Path | None
            Optional path to the server log file for error reporting.
        """
        self._client = client
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
        """
        payload = {"sequences": sequences, "options": options}
        response = self._client.post("/embed", json=payload)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            if response.status_code == 500:
                log_file_msg = ""
                if self._log_file_path is not None:
                    log_file_msg = f"\n\nServer log file: {self._log_file_path}"
                # Raise RuntimeError with log file path, chaining from the original HTTPStatusError
                raise RuntimeError(
                    f"Internal server error (500) occurred.{log_file_msg}\n"
                    f"HTTP request failed: {e}"
                ) from e
            raise
        return _deserialize_output(response.json())

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
        """
        payload = {"sequences": sequences, "options": options}
        response = self._client.post("/fold", json=payload)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            if response.status_code == 500:
                log_file_msg = ""
                if self._log_file_path is not None:
                    log_file_msg = f"\n\nServer log file: {self._log_file_path}"
                # Raise RuntimeError with log file path, chaining from the original HTTPStatusError
                raise RuntimeError(
                    f"Internal server error (500) occurred.{log_file_msg}\n"
                    f"HTTP request failed: {e}"
                ) from e
            raise
        return _deserialize_output(response.json())


def _deserialize_output(data: dict[str, Any]) -> Any:
    """Deserialize pickled output object from base64-encoded JSON response.

    Parameters
    ----------
    data : dict[str, Any]
        JSON response containing base64-encoded pickled data.

    Returns
    -------
    Any
        Deserialized output object (e.g., ESM2Output).
    """
    import base64
    import pickle

    if "pickled" not in data:
        raise ValueError("Response does not contain pickled data")

    base64_encoded = data["pickled"]
    pickled_data = base64.b64decode(base64_encoded.encode("utf-8"))
    return pickle.loads(pickled_data)
