"""Conda backend for running models in separate conda environments via HTTP microservice."""

import json
import logging
import os
import shutil
import socket
import subprocess
import time
import yaml  # type: ignore[import-not-found]
from pathlib import Path
from typing import Any, Optional

import httpx  # type: ignore[import-not-found]

from .base import Backend
from .progress import ProgressTracker
from ..utils import get_model_cache_dir

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


def _get_environment_name(environment_yml_path: Path, runner_command: str = "conda") -> str:
    """Extract environment name from environment.yml file or path.

    If the yml file specifies a name, uses that. Otherwise uses the pattern
    "boileroom-{directory_name}" as fallback.

    Parameters
    ----------
    environment_yml_path : Path
        Path to environment.yml file.
    runner_command : str
        Conda-compatible command to use for checking existing environments.

    Returns
    -------
    str
        Environment name from the 'name' field in environment.yml if present,
        otherwise "boileroom-{directory_name}".
    """
    directory_name = environment_yml_path.parent.name
    fallback_name = f"boileroom-{directory_name}"
    yml_name: str | None = None

    # Try to read name from environment.yml file
    try:
        with open(environment_yml_path, "r") as f:
            env_spec = yaml.safe_load(f)
            if env_spec and "name" in env_spec:
                yml_name = env_spec["name"]
    except Exception:
        # If we can't read the file, fall back to boileroom-{directory_name}
        pass

    # If no name in yml, use fallback pattern
    if yml_name is None:
        return fallback_name

    # Use yml name
    return yml_name


def _is_tool_available(tool_name: str) -> bool:
    """Check if a conda/mamba tool is available in PATH.

    Parameters
    ----------
    tool_name : str
        Name of the tool to check (e.g., 'conda', 'mamba', 'micromamba').

    Returns
    -------
    bool
        True if the tool is available and executable, False otherwise.
    """
    return shutil.which(tool_name) is not None


def _detect_available_tool() -> Optional[str]:
    """Detect which conda-compatible tool is available, preferring fastest options.

    Checks for tools in order of preference: micromamba > mamba > conda.
    This prioritizes speed and efficiency.

    Returns
    -------
    Optional[str]
        Name of the first available tool found, or None if none are available.
    """
    # Priority order: fastest to slowest
    tools = ["micromamba", "mamba", "conda"]
    for tool in tools:
        if _is_tool_available(tool):
            return tool
    return None


def _verify_conda_environment(runner_command: str, env_name: str, environment_yml_path: Path) -> tuple[bool, bool]:
    """Verify if a conda environment exists and has all required packages.

    Parameters
    ----------
    runner_command : str
        Conda-compatible command to use (e.g., 'conda', 'mamba', 'micromamba').
    env_name : str
        Name of the conda environment to check.
    environment_yml_path : Path
        Path to the environment.yml file containing required packages.

    Returns
    -------
    tuple[bool, bool]
        Tuple of (exists, is_valid) where:
        - exists: True if the environment exists, False otherwise.
        - is_valid: True if the environment exists and has all required packages, False otherwise.
        If exists is False, is_valid will also be False.
    """
    logger.debug(f"Verifying conda environment '{env_name}' using '{runner_command}'")

    # Check if environment exists
    result = subprocess.run(
        [runner_command, "env", "list"],
        capture_output=True,
        text=True,
        check=False,
    )

    logger.debug(f"Environment list command return code: {result.returncode}")
    logger.debug(f"Environment list stdout: {result.stdout}")
    if result.stderr:
        logger.debug(f"Environment list stderr: {result.stderr}")

    # Parse environment list output more robustly
    # Output format from conda env list:
    #   "env_name                    /path/to/envs/env_name"
    #   "env_name *                  /path/to/envs/env_name"  (if active)
    #   "base                        /path/to/base"
    environment_exists = False
    for line in result.stdout.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # Split by whitespace - first part is name, second part (if present) is path
        parts = line.split()
        if not parts:
            continue

        # Check first part (environment name)
        first_part = parts[0].strip()
        # Remove asterisk if present (indicates active environment)
        if first_part.endswith("*"):
            first_part = first_part[:-1].strip()

        if first_part == env_name:
            environment_exists = True
            logger.debug(f"Found environment '{env_name}' in list: {line}")
            break

        # Also check if path contains the environment name (for robustness)
        if len(parts) > 1:
            path_part = parts[1].strip()
            if "/" in path_part:
                # Extract name from path (last component)
                name_from_path = path_part.split("/")[-1]
                if name_from_path == env_name:
                    environment_exists = True
                    logger.debug(f"Found environment '{env_name}' in path: {path_part}")
                    break

    if not environment_exists:
        logger.info(f"Environment '{env_name}' does not exist")
        return (False, False)

    logger.info(f"Environment '{env_name}' exists")

    # If yaml is not available, we can't verify packages, so assume invalid
    if yaml is None:
        logger.warning("yaml module not available, cannot verify packages in environment")
        return (True, False)

    # Parse environment.yml to get required packages
    logger.debug(f"Parsing environment.yml from: {environment_yml_path}")
    try:
        with open(environment_yml_path, "r") as f:
            env_spec = yaml.safe_load(f)
        logger.debug(f"Parsed environment spec: {env_spec}")
    except Exception as e:
        logger.warning(f"Failed to parse environment.yml: {e}")
        # If we can't parse the file, assume invalid
        return (True, False)

    # Extract required packages from dependencies
    required_packages: set[str] = set()
    dependencies = env_spec.get("dependencies", [])
    logger.debug(f"Extracting required packages from {len(dependencies)} dependencies")
    for dep in dependencies:
        if isinstance(dep, str):
            # Conda package: extract package name (before version specifiers)
            # Handle formats like "package=1.0", "package>=1.0", "package<2.0", etc.
            package_name = dep.split()[0]
            # Remove version specifiers (check longer operators first)
            for operator in [">=", "<=", "!=", "==", ">", "<", "="]:
                if operator in package_name:
                    package_name = package_name.split(operator)[0]
                    break
            required_packages.add(package_name.lower())
            logger.debug(f"Added conda package: {package_name.lower()}")
        elif isinstance(dep, dict) and "pip" in dep:
            # Pip packages
            pip_packages = dep["pip"]
            logger.debug(f"Found {len(pip_packages)} pip packages")
            for pip_dep in pip_packages:
                if isinstance(pip_dep, str):
                    # Extract package name (before version specifiers)
                    # Handle formats like "package==1.0", "package>=1.0", "package~=1.0", etc.
                    package_name = pip_dep.split()[0]
                    # Remove version specifiers (check longer operators first)
                    for operator in [">=", "<=", "!=", "==", "~=", ">", "<", "="]:
                        if operator in package_name:
                            package_name = package_name.split(operator)[0]
                            break
                    required_packages.add(package_name.lower())
                    logger.debug(f"Added pip package: {package_name.lower()}")

    logger.debug(f"Total required packages: {len(required_packages)}")
    if not required_packages:
        # No packages specified, consider it valid
        logger.info("No packages specified in environment.yml, considering environment valid")
        return (True, True)

    # Get installed packages from the environment
    logger.debug(f"Listing installed packages in environment '{env_name}'")
    list_result = subprocess.run(
        [runner_command, "list", "-n", env_name],
        capture_output=True,
        text=True,
        check=False,
    )

    if list_result.returncode != 0:
        logger.warning(f"Failed to list packages in environment '{env_name}': {list_result.stderr}")
        # Can't list packages, assume invalid
        return (True, False)

    # Parse installed packages (skip header lines)
    installed_packages: set[str] = set()
    lines = list_result.stdout.split("\n")
    logger.debug(f"Parsing {len(lines)} lines from package list")
    for line in lines:
        # Skip empty lines and headers
        if not line.strip() or line.startswith("#"):
            continue
        # Format: package_name version build channel
        parts = line.split()
        if parts:
            installed_packages.add(parts[0].lower())

    logger.debug(f"Found {len(installed_packages)} installed packages")

    # Check if all required packages are installed
    missing_packages = required_packages - installed_packages
    is_valid = len(missing_packages) == 0

    if missing_packages:
        logger.info(f"Environment '{env_name}' is missing {len(missing_packages)} packages: {sorted(missing_packages)}")
    else:
        logger.info(f"Environment '{env_name}' has all required packages")

    return (True, is_valid)


class CondaBackend(Backend):
    """Backend that runs models in separate conda environments via HTTP microservice.

    This backend ensures complete dependency independence between Boiler Room
    and model-specific environments by accepting Core class paths as strings
    instead of importing them directly.
    """

    def __init__(
        self,
        core_class_path: str,
        config: dict | None = None,
        device: str | None = None,
        environment_yml_path: Path | str | None = None,
        runner_command: str | None = "auto",
    ) -> None:
        """Initialize the CondaBackend with a Core class path and configuration.

        Parameters
        ----------
        core_class_path : str
            Full module path to the Core class (e.g., 'boileroom.models.esm.esm2.ESM2Core').
            This is passed as a string to avoid importing the class in the main process.
        config : dict | None
            Optional configuration mapping for the model.
        device : str | None
            Optional device identifier (e.g., 'cuda:0' or 'cpu').
        environment_yml_path : Path | str | None
            Path to the conda environment.yml file. If None, environment management is skipped.
        runner_command : str | None
            Command to use for running conda environments. Options:
            - "auto" (default): Auto-detect available tool (micromamba > mamba > conda)
            - "conda": Use conda explicitly
            - "mamba": Use mamba explicitly
            - "micromamba": Use micromamba explicitly
            If None or "auto" and no tool is found, raises ValueError with installation instructions.

        Raises
        ------
        ValueError
            If runner_command is "auto" or None and no compatible tool is found, or if
            an explicit runner_command is specified but not available.
        """
        super().__init__()
        self._core_class_path = core_class_path
        self._config = dict(config) if config is not None else {}
        self._device = device or "cuda:0"
        self._environment_yml_path = Path(environment_yml_path) if environment_yml_path else None

        # Handle runner command selection
        if runner_command is None or runner_command == "auto":
            detected_tool = _detect_available_tool()
            if detected_tool is None:
                raise ValueError(
                    "To use the CondaBackend, you need to install one of: conda, mamba, or micromamba.\n"
                    "We recommend micromamba for lean and fast performance.\n"
                    "See https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html for installation instructions."
                )
            self._runner_command = detected_tool
            # Suggest faster alternatives if available but not selected
            if detected_tool == "conda":
                if _is_tool_available("micromamba"):
                    import logging

                    logger = logging.getLogger(__name__)
                    logger.info(
                        "Using conda backend. For faster performance, consider using backend='micromamba' "
                        "(micromamba is available on this system)."
                    )
                elif _is_tool_available("mamba"):
                    import logging

                    logger = logging.getLogger(__name__)
                    logger.info(
                        "Using conda backend. For faster performance, consider using backend='mamba' "
                        "(mamba is available on this system)."
                    )
        else:
            if not _is_tool_available(runner_command):
                raise ValueError(
                    f"Specified runner command '{runner_command}' is not available in PATH.\n"
                    f"Please install it or use 'auto' to auto-detect an available tool."
                )
            self._runner_command = runner_command

        self._env_name: str | None = None
        self._port: int | None = None
        self._base_url: str | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._client: httpx.Client | None = None

    def startup(self) -> None:
        """Start the conda environment server and wait for it to be ready.

        This method:
        1. Checks if conda environment exists, creates it if missing
        2. Finds an available port
        3. Launches the server subprocess with appropriate environment variables
        4. Waits for health check to confirm server is ready
        """
        if self._process is not None:
            return

        # Determine environment name
        if self._environment_yml_path is None:
            raise ValueError("environment_yml_path must be provided")
        self._env_name = _get_environment_name(self._environment_yml_path, self._runner_command)

        # Check/create conda environment
        self._ensure_conda_environment()

        # Find available port
        self._port = _find_available_port()
        self._base_url = f"http://127.0.0.1:{self._port}"

        # Build command - run server file directly to avoid importing boileroom.__init__.py
        # which would trigger imports of all models (including modal dependencies)
        project_root = Path(__file__).parent.parent.parent
        server_path = Path(__file__).parent / "server.py"
        cmd = [
            self._runner_command,
            "run",
            "-n",
            self._env_name,
            "python",
            str(server_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(self._port),
        ]

        # Build environment variables
        env = os.environ.copy()
        env["MODEL_CLASS"] = self._core_class_path
        env["MODEL_CONFIG"] = json.dumps(self._config)
        env["DEVICE"] = self._device

        device_number = _extract_device_number(self._device)
        if device_number is not None:
            env["CUDA_VISIBLE_DEVICES"] = device_number

        # Pass through MODEL_DIR if present
        if "MODEL_DIR" in os.environ:
            env["MODEL_DIR"] = os.environ["MODEL_DIR"]
            # Automatically derive CHAI_DOWNLOADS_DIR from MODEL_DIR/chai
            # Only set if not already explicitly set (allows override if needed)
            if "CHAI_DOWNLOADS_DIR" not in os.environ:
                chai_cache_dir = get_model_cache_dir("chai")
                env["CHAI_DOWNLOADS_DIR"] = str(chai_cache_dir)

        # Set PYTHONPATH to project root
        env["PYTHONPATH"] = str(project_root)

        # Launch subprocess
        self._process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(project_root),
        )

        # Wait for health check
        self._wait_for_health_check()

        # Create HTTP client
        # Use 30 minute timeout to handle large responses (e.g., PAE matrices for long sequences)
        # This matches Modal backend timeout of 20 minutes with some buffer for serialization/transmission
        # Set explicit timeouts: connect=10s, read=1800s (30min), write=60s, pool=10s
        # The read timeout is the critical one for large response bodies
        timeout_config = httpx.Timeout(connect=10.0, read=1800.0, write=60.0, pool=10.0)
        self._client = httpx.Client(base_url=self._base_url, timeout=timeout_config)

    def shutdown(self) -> None:
        """Shutdown the conda environment server gracefully.

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
        """Get the HTTP proxy object for making requests to the conda server.

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
            raise RuntimeError("Conda backend is not initialized. Call start() before use.")
        return _CondaModelProxy(self._client)

    def _ensure_conda_environment(self) -> None:
        """Ensure the conda environment exists and has all required packages.

        Checks if the environment exists and is valid using `_verify_conda_environment`.
        If the environment doesn't exist, creates it using `conda env create`.
        If the environment exists but is missing packages, updates it using `conda env update`.
        """
        if self._environment_yml_path is None or self._env_name is None:
            logger.debug("Skipping environment check: environment_yml_path or env_name is None")
            return

        runner_env = self._build_runner_env()

<<<<<<< HEAD
        # Verify environment status
        exists, is_valid = _verify_conda_environment(self._runner_command, self._env_name, self._environment_yml_path)
=======
        with ProgressTracker(logger_name="boileroom.backend.conda") as tracker:
            tracker.record_stage(f"Ensuring conda environment '{self._env_name}' exists and is valid")
>>>>>>> origin/feat/conda-progress-debug

            logger.info(f"Ensuring conda environment '{self._env_name}' exists and is valid")
            logger.debug(f"Environment YAML path: {self._environment_yml_path}")
            logger.debug(f"Runner command: {self._runner_command}")

            # Verify environment status
            exists, is_valid = _verify_conda_environment(
                self._runner_command, self._env_name, self._environment_yml_path
            )

<<<<<<< HEAD
        if exists and not is_valid:
            # Environment exists but is missing packages, update it
            logger.info(f"Updating environment '{self._env_name}' with missing packages")
            logger.debug(
                f"Running: {self._runner_command} env update -f {self._environment_yml_path} -n {self._env_name}"
            )
            try:
                result = subprocess.run(
                    [
                        self._runner_command,
                        "env",
                        "update",
                        "-f",
                        str(self._environment_yml_path),
                        "-n",
                        self._env_name,
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                logger.debug(f"Update command stdout: {result.stdout}")
                if result.stderr:
                    logger.debug(f"Update command stderr: {result.stderr}")
                logger.info(f"Successfully updated environment '{self._env_name}'")
            except subprocess.CalledProcessError as e:
                error_msg = f"Failed to update conda environment '{self._env_name}':\n"
                error_msg += f"Command: {' '.join(e.cmd)}\n"
                error_msg += f"Return code: {e.returncode}\n"
                if e.stdout:
                    error_msg += f"stdout:\n{e.stdout}\n"
                if e.stderr:
                    error_msg += f"stderr:\n{e.stderr}\n"
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e
        else:
            # Environment doesn't exist, create it
            logger.info(f"Creating environment '{self._env_name}'")
            logger.debug(
                f"Running: {self._runner_command} env create -f {self._environment_yml_path} -n {self._env_name}"
            )
            try:
                result = subprocess.run(
                    [
                        self._runner_command,
                        "env",
                        "create",
                        "-f",
                        str(self._environment_yml_path),
                        "-n",
                        self._env_name,
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                logger.debug(f"Create command stdout: {result.stdout}")
                if result.stderr:
                    logger.debug(f"Create command stderr: {result.stderr}")
                logger.info(f"Successfully created environment '{self._env_name}'")
            except subprocess.CalledProcessError as e:
                error_msg = f"Failed to create conda environment '{self._env_name}':\n"
                error_msg += f"Command: {' '.join(e.cmd)}\n"
                error_msg += f"Return code: {e.returncode}\n"
                if e.stdout:
                    error_msg += f"stdout:\n{e.stdout}\n"
                if e.stderr:
                    error_msg += f"stderr:\n{e.stderr}\n"
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e
=======
            logger.debug(f"Environment verification result: exists={exists}, is_valid={is_valid}")

            if exists and is_valid:
                # Environment exists and has all required packages
                logger.info(f"Environment '{self._env_name}' exists and is valid, no action needed")
                return

            if exists and not is_valid:
                # Environment exists but is missing packages, update it
                logger.info(f"Updating environment '{self._env_name}' with missing packages")
                logger.debug(f"Running: {self._runner_command} env update -f {self._environment_yml_path} -n {self._env_name}")
                try:
                    result = tracker.run_subprocess(
                        [
                            self._runner_command,
                            "env",
                            "update",
                            "-y",
                            "-f",
                            str(self._environment_yml_path),
                            "-n",
                            self._env_name,
                        ],
                        stage_label=f"Updating environment '{self._env_name}'",
                        subprocess_title="conda env update",
                        env=runner_env,
                    )
                    logger.debug(f"Update command stdout: {result.stdout}")
                    if result.stderr:
                        logger.debug(f"Update command stderr: {result.stderr}")
                    logger.info(f"Successfully updated environment '{self._env_name}'")
                except subprocess.CalledProcessError as e:
                    error_msg = f"Failed to update conda environment '{self._env_name}':\n"
                    error_msg += f"Command: {' '.join(e.cmd)}\n"
                    error_msg += f"Return code: {e.returncode}\n"
                    if e.stdout:
                        error_msg += f"stdout:\n{e.stdout}\n"
                    if e.stderr:
                        error_msg += f"stderr:\n{e.stderr}\n"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg) from e
            else:
                # Environment doesn't exist, create it
                logger.info(f"Creating environment '{self._env_name}'")
                logger.debug(f"Running: {self._runner_command} env create -f {self._environment_yml_path} -n {self._env_name}")
                try:
                    result = tracker.run_subprocess(
                        [
                            self._runner_command,
                            "env",
                            "create",
                            "-y",
                            "-f",
                            str(self._environment_yml_path),
                            "-n",
                            self._env_name,
                        ],
                        stage_label=f"Creating environment '{self._env_name}'",
                        subprocess_title="conda env create",
                        env=runner_env,
                    )
                    logger.debug(f"Create command stdout: {result.stdout}")
                    if result.stderr:
                        logger.debug(f"Create command stderr: {result.stderr}")
                    logger.info(f"Successfully created environment '{self._env_name}'")
                except subprocess.CalledProcessError as e:
                    error_msg = f"Failed to create conda environment '{self._env_name}':\n"
                    error_msg += f"Command: {' '.join(e.cmd)}\n"
                    error_msg += f"Return code: {e.returncode}\n"
                    if e.stdout:
                        error_msg += f"stdout:\n{e.stdout}\n"
                    if e.stderr:
                        error_msg += f"stderr:\n{e.stderr}\n"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg) from e

    def _build_runner_env(self) -> dict[str, str]:
        """Return a sanitized environment for micromamba commands.

        Removes PYTHONPATH to avoid conflicts where our source tree shadows stdlib
        modules (e.g., boileroom.models.esm.types vs python's built-in types).
        """
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        return env
>>>>>>> origin/feat/conda-progress-debug

    def _wait_for_health_check(self, timeout: float = 60.0, poll_interval: float = 1.0) -> None:
        """Wait for the server to become ready by polling the /health endpoint.

        Parameters
        ----------
        timeout : float
            Maximum time to wait in seconds.
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
        while time.time() - start_time < timeout:
            try:
                response = httpx.get(f"{self._base_url}/health", timeout=5.0)
                if response.status_code == 200:
                    return
            except (httpx.RequestError, httpx.TimeoutException):
                pass

            # Check if process has died
            if self._process is not None and self._process.poll() is not None:
                stdout, stderr = self._process.communicate()
                error_msg = f"Server process died. stdout: {stdout.decode()}, stderr: {stderr.decode()}"
                raise RuntimeError(error_msg)

            time.sleep(poll_interval)

        raise RuntimeError(f"Server did not become ready within {timeout} seconds")


class _CondaModelProxy:
    """HTTP proxy for making requests to the conda server."""

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
