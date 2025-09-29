"""Conda-based backend skeleton for running BoilerRoom models locally."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Type

from .base import Backend


logger = logging.getLogger(__name__)


@dataclass
class CondaEnvironmentSpec:
    """Configuration describing the target conda environment."""

    environment_name: str
    environment_file: Path | None = None
    environment_prefix_path: Path | None = None

    def __post_init__(self) -> None:
        if self.environment_file is not None:
            self.environment_file = self.environment_file.expanduser().resolve()
        if self.environment_prefix_path is not None:
            self.environment_prefix_path = self.environment_prefix_path.expanduser().resolve()


class CondaBackend(Backend):
    """Backend responsible for managing local execution via a conda environment."""

    def __init__(
        self,
        model_class: Type[Any],
        config: dict | None = None,
        device: str | None = None,
        conda_executable: str | Path | None = None,
        environment_spec: CondaEnvironmentSpec | None = None,
    ) -> None:
        super().__init__()
        self._model_class = model_class
        self._config = config or {}
        self._device = device
        self._conda_executable = self._resolve_conda_executable(conda_executable)
        self._environment_spec = environment_spec or CondaEnvironmentSpec(environment_name="boileroom-esmfold")
        self._environment_ready = False
        self.model: Any | None = None

    def startup(self) -> None:
        """Ensure the conda environment is ready and instantiate the model."""
        self._ensure_conda_environment()
        if self.model is None:
            self.model = self._instantiate_model()

    def shutdown(self) -> None:
        """Tear down the model instance and release resources."""
        if self.model is not None:
            self._teardown_model(self.model)
            self.model = None

    def _ensure_conda_environment(self) -> None:
        if self._environment_ready:
            return

        if self._conda_executable is None:
            logger.warning("Conda executable not located; skipping environment validation.")
            return

        if not self._environment_exists():
            logger.info("Conda environment '%s' missing; attempting to build.", self._environment_spec.environment_name)
            self._build_environment()

        if not self._environment_exists():
            raise RuntimeError(
                f"Conda environment '{self._environment_spec.environment_name}' could not be prepared."
            )

        self._environment_ready = True

    def _environment_exists(self) -> bool:
        """Check whether the configured conda environment already exists."""
        if self._environment_spec.environment_prefix_path is not None:
            return self._environment_spec.environment_prefix_path.exists()

        if self._conda_executable is None:
            return False

        command = [str(self._conda_executable), "env", "list", "--json"]
        result = self._run_conda_command(command)

        if result is None:
            return False

        try:
            parsed_output = json.loads(result)
        except json.JSONDecodeError as error:
            logger.warning("Failed to parse conda environment list: %s", error)
            return False

        environment_paths = parsed_output.get("envs", [])
        for environment_path in environment_paths:
            environment_candidate = Path(environment_path).resolve()
            if environment_candidate.name == self._environment_spec.environment_name:
                return True
        return False

    def _build_environment(self) -> None:
        """Create or update the conda environment using the configured specification."""
        if self._conda_executable is None:
            logger.error("Cannot build conda environment without locating the conda executable.")
            return

        if self._environment_spec.environment_file is not None:
            command = [
                str(self._conda_executable),
                "env",
                "update",
                "--name",
                self._environment_spec.environment_name,
                "--file",
                str(self._environment_spec.environment_file),
                "--prune",
            ]
        else:
            command = [
                str(self._conda_executable),
                "create",
                "--name",
                self._environment_spec.environment_name,
                "python=3.10",
            ]

        logger.info("Executing conda command: %s", " ".join(command))
        self._run_conda_command(command)

    def _instantiate_model(self) -> Any:
        """Instantiate the configured model class inside the prepared environment."""
        model_instance = self._model_class(self._config)

        if self._device is not None and hasattr(model_instance, "device"):
            setattr(model_instance, "device", self._device)

        initialize_method = getattr(model_instance, "_initialize", None)
        if callable(initialize_method):
            initialize_method()
        else:
            logger.debug("Model class %s does not expose an '_initialize' hook.", self._model_class.__name__)

        return model_instance

    def _teardown_model(self, model_instance: Any) -> None:
        """Perform best-effort cleanup on the model instance."""
        teardown_hooks = ["close", "shutdown", "teardown"]
        for hook_name in teardown_hooks:
            cleanup_hook = getattr(model_instance, hook_name, None)
            if callable(cleanup_hook):
                try:
                    cleanup_hook()
                except Exception as error:  # noqa: BLE001 - speculative cleanup
                    logger.debug("Error during model cleanup using hook '%s': %s", hook_name, error)
                break

    def _resolve_conda_executable(self, requested_executable: str | Path | None) -> Path | None:
        """Determine the absolute path to the appropriate conda executable."""
        candidate_paths: list[Path] = []

        if requested_executable is not None:
            candidate_paths.append(Path(requested_executable).expanduser().resolve())

        environment_override = os.environ.get("CONDA_EXE")
        if environment_override:
            candidate_paths.append(Path(environment_override).expanduser().resolve())

        discovered_executable = shutil.which("conda")
        if discovered_executable is not None:
            candidate_paths.append(Path(discovered_executable).resolve())

        for candidate_path in candidate_paths:
            if candidate_path.exists():
                return candidate_path

        logger.warning("Unable to resolve a conda executable from provided inputs.")
        return None

    def _run_conda_command(self, command: list[str]) -> str | None:
        """Execute a conda command, returning stdout when successful."""
        try:
            process = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as error:
            logger.error("Failed to execute command %s: %s", command, error)
            return None

        if process.returncode != 0:
            logger.warning("Conda command failed (%s): %s", process.returncode, process.stderr)
            return None

        return process.stdout

