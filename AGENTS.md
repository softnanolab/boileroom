# Repository Guidelines

## Project Structure & Modules
- `boileroom/models/<family>/` splits each model into `*Core`, `Modal*`, and user-facing classes (see `boltz/boltz2.py`, `chai/chai1.py`, `esm/esmfold.py`, `esm/esm2.py`).
- `boileroom/models/<family>/types.py` contains lightweight output type definitions (e.g., `Boltz2Output`, `Chai1Output`, `ESM2Output`, `ESMFoldOutput`) that only depend on `numpy`, `dataclasses`, and base classes. This enables dependency isolation—consumers can import output types without pulling in heavy ML dependencies (transformers, torch, etc.). Note, `biotite` is always across both sides (server, client).
- `boileroom/backend/` hosts backend adapters (`ModalBackend`, `LocalBackend`) plus the global Modal app; update these if you add new execution targets.
- `boileroom/images/` defines Modal images and shared volumes; duplicate per-model dependencies using the existing image factories.
- `boileroom/scripts/images/` contains Docker image build tooling (`build_model_images.py`, `build_model_images.sh`). New models should integrate with these scripts via their `Dockerfile` and optional `config.yaml`.
- `tests/` mirrors runtime features (`boltz/`, `chai/`, `esm/`) with shared fixtures in `conftest.py`; keep docstrings and assertions aligned with real world behaviour.
- `docs/` captures design notes (`architecture.md`, `backend_support_matrix.md`, `local_debugging.md`); update alongside feature work.

## Build, Test, and Dev Commands
- `uv python install 3.12 && uv sync` installs the pinned toolchain and dependencies.
- `uv run pytest` runs the full test suite; add `-n auto` when you need xdist parallelism.
- `uv run python script.py` exercises the orchestration demo; switch backends via `--backend modal|local`.
- `uv run pre-commit run --all-files` applies linting, formatting, and mypy checks before pushing.

## Coding Style & Naming
- Follow Python 3.12 conventions with 4-space indentation, descriptive `snake_case` for functions, and `PascalCase` for model classes.
- Type hints are required; CI enforces `mypy` (config in `.pre-commit-config.yaml`). Prefer builtin generics (list[int], dict[str, float]) and X | Y for unions (Python 3.9+ / 3.10+).
- `ruff` manages linting and formatting with a 120-character line budget; prefer auto-fixes via pre-commit.
- Place Modal configuration constants in `boileroom/constants.py`; avoid duplicating literal IDs across modules.
- Use f-strings and not any other way of doing strings, e.g., f"Today is {date}".
- Use Numpy Style docstring convention. Do not use Google ones or others.

## Testing Expectations
- Prefer pytest-style `test_*` functions and fixtures; see `tests/test_utils.py` for style cues.
- New features need positive and failure-path coverage in the matching backend folder and should prove Modal/local parity when relevant.
- Run `uv run pytest` (or targeted cases like `uv run pytest tests/test_basic.py::test_esmfold_batch -v -s`) before opening a PR.

## Backend Architecture & System Design
- Each model owns a Core class (`FoldingAlgorithm` or `EmbeddingAlgorithm`) that loads weights in `_load`, validates sequences, and exposes `fold` or `embed`. Keep Modal-side latency low by caching assets in `_initialize`.
- Modal wrappers (`ModalBoltz2`, `ModalChai1`, `ModalESMFold`, `ModalESM2`) live beside the core, use `@app.cls`, mount `model_weights`, and rehydrate the core inside `@modal.enter`. Keep method signatures thin—call the core directly and let Modal manage GPU scheduling.
- High-level classes (`Boltz2`, `Chai1`, `ESMFold`, `ESM2`) inherit `ModelWrapper`, choose `ModalBackend` by default, and fall back to `LocalBackend` for debugging. `LocalBackend` mirrors Modal calls but expects the environment to satisfy dependencies (`uv pip install .[boltz]` etc.).
- Backends share mechanisms through `boileroom/backend/base.py`; add new providers (e.g. Conda, Batch) by implementing `startup`, `shutdown`, and `get_model`.
- The goal of the package is to isolate the dependencies of individual models — hence, the boileroom dependencies should be independent of what is used within individual environemnts (conda, apptainer, Modal, etc.), and vice versa.
- Do not change crucial APIs, especially the high-level API that would be breaking the entire logic.

### Conda Backend Structure
- Each model that supports conda backend must have an `environment.yml` file in `boileroom/models/<family>/environment.yml`. The environment name follows the pattern `boileroom-<directory_name>` or uses the `name` field from environment.yml if present.
- Core classes are passed as string paths (e.g., `"boileroom.models.esm.core.ESM2Core"`) to `CondaBackend` to maintain dependency isolation between Boiler Room and model-specific environments.
- The conda backend uses HTTP microservice pattern: `boileroom/backend/server.py` runs in the conda environment and exposes `/health`, `/embed`, and `/fold` endpoints. The server dynamically loads the Core class specified via the `MODEL_CLASS` environment variable.
- Output types (e.g., `ESM2Output`, `Boltz2Output`) from `types.py` are serialized via pickle+base64 for JSON transport between the main process and the conda server.
- The `_CondaModelProxy` in `conda.py` provides the client-side interface that forwards method calls (`embed()` or `fold()`) to the HTTP server via POST requests.
- Models in the same family (e.g., ESM2 and ESMFold) can share the same `environment.yml` file.
- `CondaBackend` can auto-detect the runner (`micromamba` > `mamba` > `conda`) or use an explicit `runner_command` (`"conda"`, `"mamba"`, `"micromamba"`). It manages the environment by:
  - Deriving the environment name from `environment.yml` (`name:`) or falling back to `boileroom-<directory_name>`.
  - Verifying existence and required packages from `environment.yml` via `_verify_conda_environment`.
  - Creating or updating the environment with `env create` / `env update` when needed (including `pip` dependencies).
  - Launching `server.py` inside the environment via `runner_command run -n <env> python server.py`, passing `MODEL_CLASS`, `MODEL_CONFIG`, and `DEVICE` via environment variables.

### Apptainer Backend Structure
- Each model that supports apptainer backend uses pre-built Docker images from DockerHub (e.g., `docker://docker.io/jakublala/boileroom-<model>:latest`). Images are pulled and cached as `.sif` files in `~/.cache/boileroom/images/`.
- Core classes are passed as string paths (e.g., `"boileroom.models.esm.core.ESM2Core"`) to `ApptainerBackend` to maintain dependency isolation between Boiler Room and model-specific containers.
- The apptainer backend uses HTTP microservice pattern: `boileroom/backend/server.py` runs inside the Apptainer container and exposes `/health`, `/embed`, and `/fold` endpoints. The server dynamically loads the Core class specified via the `MODEL_CLASS` environment variable.
- Output types (e.g., `ESM2Output`, `Boltz2Output`) from `types.py` are serialized via pickle+base64 for JSON transport between the main process and the container server.
- The `_ApptainerModelProxy` in `apptainer.py` provides the client-side interface that forwards method calls (`embed()` or `fold()`) to the HTTP server via POST requests.
- Models in the same family (e.g., ESM2 and ESMFold) can share the same Docker image.
- `ApptainerBackend` expects a Docker image URI (e.g., `docker://docker.io/jakublala/boileroom-boltz:cuda12.6-dev`), pulls it as a `.sif` into a cache directory (default `~/.cache/boileroom/images` or `MODEL_DIR`), and starts `server.py` inside the container using `apptainer exec`. It:
  - Binds the repo source tree read-only and any model data directories (`MODEL_DIR`, `CHAI_DOWNLOADS_DIR`) into the container.
  - Sets `MODEL_CLASS`, `MODEL_CONFIG`, `DEVICE`, and CUDA-related env vars (`CUDA_VISIBLE_DEVICES`, `LD_LIBRARY_PATH`) before starting.
  - Uses `/health` polling to wait for readiness and exposes a thin HTTP client proxy for `embed()`/`fold()`.

## Adding a New Model

### File Structure Pattern
Each model family follows this structure for dependency isolation:
```
boileroom/models/<family>/
  ├── core.py          # Heavy dependencies (transformers, torch, biotite, etc.)
  ├── types.py         # Lightweight output types (numpy, dataclasses, base classes only)
  ├── <model>.py       # Wrapper (modal, backends) - imports from core and types
  └── ...
```

The `types.py` file contains output dataclasses (e.g., `MyFoldOutput`) that can be imported by consumers without pulling in heavy ML dependencies. This enables clean separation between Boiler Room's lightweight interface and model-specific runtime environments.

### Implementation Steps
- **Define output types in `types.py`**:
  ```python
  # boileroom/models/<family>/types.py
  from dataclasses import dataclass
  from typing import Optional, TYPE_CHECKING
  import numpy as np
  from ...base import StructurePrediction, PredictionMetadata

  if TYPE_CHECKING:
      from biotite.structure import AtomArray  # Use TYPE_CHECKING for heavy deps

  @dataclass
  class MyFoldOutput(StructurePrediction):
      metadata: PredictionMetadata
      atom_array: Optional[List["AtomArray"]] = None
      # ... other fields
  ```

- **Implement a core in `core.py`**:
  ```python
  # boileroom/models/<family>/core.py
  from .types import MyFoldOutput

  class MyFoldCore(FoldingAlgorithm):
      DEFAULT_CONFIG = {"device": "cuda:0"}
      def _load(self): ...
      def fold(self, sequences) -> MyFoldOutput:
          sequences = self._validate_sequences(sequences)
          ...
  ```

- **Provide a Modal wrapper in `<model>.py`**:
  ```python
  # boileroom/models/<family>/<model>.py
  from .types import MyFoldOutput
  from .core import MyFoldCore

  @app.cls(image=my_image, gpu="A10G", volumes={MODAL_MODEL_DIR: model_weights})
  class ModalMyFold:
      config: bytes = modal.parameter(default=b"{}")
      @modal.enter()
      def _initialize(self):
          self._core = MyFoldCore(json.loads(self.config))
          self._core._initialize()
      @modal.method()
      def fold(self, sequences) -> MyFoldOutput:
          return self._core.fold(sequences)
  ```

- **Expose the public API via `ModelWrapper`**:
  ```python
  # boileroom/models/<family>/<model>.py (continued)
  from .types import MyFoldOutput  # Re-export for convenience

  class MyFold(ModelWrapper):
      def __init__(self, backend="modal", device=None, config=None):
          if backend == "modal":
              self._backend = ModalBackend(ModalMyFold, config or {}, device=device)
          elif backend == "local":
              self._backend = LocalBackend(MyFoldCore, config or {}, device=device)
          elif backend == "conda":
              self._backend = CondaBackend(MyFoldCore, config or {}, device=device)
          else:
              raise ValueError("Backend not supported")
          self._backend.start()
  ```

- **Re-export output types** from wrapper files for convenience (consumers can import from either `types.py` or the wrapper module).

### Backend Support & Environments (using `boltz` as a concrete example)

- **Modal backend**
  - Implement a `Modal<FamilyModel>` class (e.g., `ModalBoltz2`) in `<model>.py` decorated with `@app.cls`.
  - Choose an image from `boileroom/images` via a small wrapper module (e.g., `boileroom/models/boltz/image.py` uses `base_image.pip_install("boltz==2.1.1")`).
  - Mount persistent volumes such as `MODAL_MODEL_DIR: model_weights` so that model weights, caches (e.g., MSA cache), and other artifacts survive across runs.
  - In `@modal.enter`, instantiate the core (`MyFoldCore`) with JSON-decoded config bytes, call `_initialize()`, and expose thin `@modal.method` wrappers that delegate directly to the core (`fold`, `embed`, etc.).

- **Local backend**
  - Use `LocalBackend(MyFoldCore, config, device=device)` in the high-level wrapper.
  - Expect the current Python environment to already have all heavy dependencies installed (matching what the core imports).

- **Conda backend**
  - Add `environment.yml` under the model directory. This file:
    - Pins Python and core dependencies (e.g., `python=3.12`, `numpy<2.0`, `torch`, `pytorch-lightning`, model library, `biotite`, `fastapi`, `uvicorn`, `pydantic` as for `boltz`).
    - Can optionally specify `name:`; if omitted, `CondaBackend` uses `boileroom-<directory_name>`.
  - In the high-level wrapper, construct a `CondaBackend` with:
    - `core_class_path="boileroom.models.<family>.core.MyFoldCore"` (string path, not class object).
    - `environment_yml_path=Path(__file__).parent / "environment.yml"`.
    - `runner_command` set to `"conda"`, `"mamba"`, `"micromamba"`, or left to `"auto"` for detection.
  - `CondaBackend` manages env creation/update and runs `boileroom/backend/server.py` with `MODEL_CLASS`, `MODEL_CONFIG`, and `DEVICE` in the environment.

- **Apptainer backend**
  - Provide a `Dockerfile` under `boileroom/models/<family>/Dockerfile`. It should:
    - Use a base image built from `boileroom/images/Dockerfile` (passed as `BASE_IMAGE` build-arg).
    - Install model-specific dependencies into the container (e.g., conda environment, CUDA support, model library).
  - Optionally add `config.yaml` under `boileroom/models/<family>/` with `supported_cuda: ["11.8", "12.6"]` to advertise supported CUDA versions to the build scripts.
  - Wire the high-level wrapper to `ApptainerBackend` with:
    - `core_class_path="boileroom.models.<family>.core.MyFoldCore"`.
    - `image_uri=f"docker://docker.io/jakublala/boileroom-<model>:{backend_tag}"`, where `backend_tag` encodes CUDA and version (e.g., `cuda12.6-dev` or `cuda12.6-latest`).
  - The backend:
    - Caches `.sif` images under `~/.cache/boileroom/images` or `MODEL_DIR`.
    - Binds the repo source tree and data directories (`MODEL_DIR`, `CHAI_DOWNLOADS_DIR`) into the container.
    - Starts `server.py` with `micromamba run -n base python server.py` and waits for `/health` before serving requests.

### Docker Images & Build Tooling

- **Model `Dockerfile` and `config.yaml`**
  - Each model that supports Apptainer should provide:
    - `boileroom/models/<family>/Dockerfile`: describes how to build `boileroom-<model>` images from the base image.
    - Optional `boileroom/models/<family>/config.yaml` with:
      ```yaml
      supported_cuda:
        - "11.8"
        - "12.6"
      ```
      This drives CUDA selection and skipping in the image build scripts.

- **Base and per-model images**
  - `boileroom/images/Dockerfile` defines a CUDA + micromamba base image (`boileroom-base:cuda<version>-<tag>`).
  - Model images are built on top of that base via `BASE_IMAGE` and `TORCH_WHEEL_INDEX` build args and tagged as:
    - `docker.io/jakublala/boileroom-<model>:cuda<version>-<tag>` (e.g., `boileroom-boltz:cuda12.6-dev`).

- **Build scripts**
  - `scripts/images/build_model_images.py`:
    - Discovers models (`boltz`, `chai`, `esm`) and their `Dockerfile`/`config.yaml`.
    - Builds base and per-model images for requested CUDA versions (`--cuda-version`, `--all-cuda`).
    - Supports parallel builds, optional `--push` to DockerHub, and tagging via `--tag`.
  - `scripts/images/build_model_images.sh`:
    - Shell wrapper with similar semantics, building:
      - `boileroom-base`
      - `boileroom-boltz`
      - `boileroom-chai1`
      - `boileroom-esm`
    - Controlled by `--platform`, `--no-cache`, `--tag`, `--cuda-version`, `--all-cuda`, and `--push`.

## Backend & Security Notes
- Modal is production default; ensure GPU type and timeouts are tuned per workload, and document image changes in `boileroom/images`.
- Local debugging often needs extras (`boltz`, `chai`) plus CUDA; capture new requirements in `docs/local_debugging.md`, and keep `docs/backend_support_matrix.md` current.
- Never commit Modal credentials or `.modal` state. Secrets belong in environment variables or Modal secrets, not source.
