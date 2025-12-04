"""Apptainer backend for running models in containers via HTTP microservice."""

import json
import logging
import os
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
        If image pull fails.
    """
    logger.info(f"Pulling image: {image_uri}")

    # Ensure cache directory exists
    sif_path.parent.mkdir(parents=True, exist_ok=True)

    # Build apptainer pull command
    cmd = ["apptainer", "pull", "--force", str(sif_path), image_uri]

    logger.debug(f"Running: {' '.join(cmd)}")
    
    if log_file is not None:
        with open(log_file, "a") as log_handle:
            result = subprocess.run(
                cmd,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
    else:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )

    if result.returncode != 0:
        error_msg = f"Failed to pull Apptainer image '{image_uri}':\n"
        error_msg += f"Command: {' '.join(cmd)}\n"
        error_msg += f"Return code: {result.returncode}\n"
        if log_file is None:
            if result.stdout:
                error_msg += f"stdout:\n{result.stdout}\n"
            if result.stderr:
                error_msg += f"stderr:\n{result.stderr}\n"
        else:
            error_msg += f"See log file: {log_file}\n"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from subprocess.CalledProcessError(result.returncode, cmd)

    if not _is_image_cached(sif_path):
        raise RuntimeError(f"Image pull completed but .sif file not found at {sif_path}")

    logger.info(f"Successfully pulled and cached image: {sif_path}")


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
        logger.info(f"Log file: {self._log_file_path}")

        # Pull image if not cached
        if not _is_image_cached(self._sif_path):
            logger.info("Image not cached, pulling...")
            _pull_image(self._image_uri, self._sif_path, log_file=self._log_file_path)

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
            model_dir_path = Path(model_dir).resolve()
            # Ensure directory exists
            model_dir_path.mkdir(parents=True, exist_ok=True)
            # Mount to same path in container
            container_model_dir = str(model_dir_path)
            cmd.extend(["-B", f"{container_model_dir}:{container_model_dir}"])

        # Bind mount CHAI_DOWNLOADS_DIR if present
        chai_dir = os.environ.get("CHAI_DOWNLOADS_DIR")
        if chai_dir:
            chai_dir_path = Path(chai_dir).resolve()
            chai_dir_path.mkdir(parents=True, exist_ok=True)
            container_chai_dir = str(chai_dir_path)
            cmd.extend(["-B", f"{container_chai_dir}:{container_chai_dir}"])

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

        # TODO: BOLTZ NEEDS CUDA 12 !!$@($_(!@#$!(@$@!)))
        
        # Build LD_LIBRARY_PATH to include conda libraries
        # libcue_ops.so is in /opt/conda/lib/python3.12/site-packages/cuequivariance_ops/lib/
        # libcublas.so.12 is in /opt/conda/lib/python3.12/site-packages/nvidia/cublas/lib/
        # Both need to be in LD_LIBRARY_PATH
        # Note: --nv will add /usr/local/nvidia/lib and /usr/local/nvidia/lib64 automatically
        # Also include the base conda lib directory as a fallback
        conda_base_lib_path = "/opt/conda/lib"
        conda_cue_ops_lib_path = "/opt/conda/lib/python3.12/site-packages/cuequivariance_ops/lib"
        conda_cuda_lib_path = "/opt/conda/lib/python3.12/site-packages/nvidia/cublas/lib"
        # Start with conda library paths (highest priority), then NVIDIA paths that --nv sets
        ld_path_parts = [
            conda_cue_ops_lib_path,  # Where libcue_ops.so is located
            conda_cuda_lib_path,  # Where libcublas.so.12 is located
            conda_base_lib_path,  # Base conda lib directory as fallback
            "/usr/local/nvidia/lib64",
            "/usr/local/nvidia/lib",
            "/.singularity.d/libs",
        ]
        # If host has LD_LIBRARY_PATH, append any additional paths (but conda path takes priority)
        if "LD_LIBRARY_PATH" in os.environ:
            host_paths = [p for p in os.environ["LD_LIBRARY_PATH"].split(":") if p]
            for path in host_paths:
                if path not in ld_path_parts:
                    ld_path_parts.append(path)
        env_vars["LD_LIBRARY_PATH"] = ":".join(ld_path_parts)

        if device_number is not None:
            env_vars["CUDA_VISIBLE_DEVICES"] = device_number

        # Pass through MODEL_DIR and CHAI_DOWNLOADS_DIR if present
        # TODO: this should be unified, and simplified
        # TODO: all env variables should be passed through to the container, in a programmatic way
        for key in ["MODEL_DIR", "CHAI_DOWNLOADS_DIR"]:
            if key in os.environ:
                env_vars[key] = os.environ[key]

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
        self._client = httpx.Client(base_url=self._base_url, timeout=300.0)
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
        return _ApptainerModelProxy(self._client)

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

    def __init__(self, client: httpx.Client) -> None:
        """Initialize the proxy with an HTTP client.

        Parameters
        ----------
        client : httpx.Client
            HTTP client configured with the server's base URL.
        """
        self._client = client

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
        response.raise_for_status()
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
        response.raise_for_status()
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
