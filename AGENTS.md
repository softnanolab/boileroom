# Repository Guidelines

## Project Structure & Modules
- `boileroom/models/<family>/` splits each model into `*Core`, `Modal*`, and user-facing classes (see `boltz/boltz2.py`, `chai/chai1.py`, `esm/esmfold.py`, `esm/esm2.py`).
- `boileroom/backend/` hosts backend adapters (`ModalBackend`, `LocalBackend`) plus the global Modal app; update these if you add new execution targets.
- `boileroom/images/` defines Modal images and shared volumes; duplicate per-model dependencies using the existing image factories.
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
- Output types (e.g., `ESM2Output`, `Boltz2Output`) are serialized via pickle+base64 for JSON transport between the main process and the conda server.
- The `_CondaModelProxy` in `conda.py` provides the client-side interface that forwards method calls (`embed()` or `fold()`) to the HTTP server via POST requests.
- Models in the same family (e.g., ESM2 and ESMFold) can share the same `environment.yml` file.

### Apptainer Backend Structure
- Each model that supports apptainer backend uses pre-built Docker images from DockerHub (e.g., `docker://docker.io/jakublala/boileroom-<model>:latest`). Images are pulled and cached as `.sif` files in `~/.cache/boileroom/images/`.
- Core classes are passed as string paths (e.g., `"boileroom.models.esm.core.ESM2Core"`) to `ApptainerBackend` to maintain dependency isolation between Boiler Room and model-specific containers.
- The apptainer backend uses HTTP microservice pattern: `boileroom/backend/server.py` runs inside the Apptainer container and exposes `/health`, `/embed`, and `/fold` endpoints. The server dynamically loads the Core class specified via the `MODEL_CLASS` environment variable.
- Output types (e.g., `ESM2Output`, `Boltz2Output`) are serialized via pickle+base64 for JSON transport between the main process and the container server.
- The `_ApptainerModelProxy` in `apptainer.py` provides the client-side interface that forwards method calls (`embed()` or `fold()`) to the HTTP server via POST requests.
- Models in the same family (e.g., ESM2 and ESMFold) can share the same Docker image.

## Adding a New Model
- Implement a core:
  ```python
  class MyFoldCore(FoldingAlgorithm):
      DEFAULT_CONFIG = {"device": "cuda:0"}
      def _load(self): ...
      def fold(self, sequences): sequences = self._validate_sequences(sequences); ...
  ```
- Provide a Modal wrapper next to the core:
  ```python
  @app.cls(image=my_image, gpu="A10G", volumes={MODAL_MODEL_DIR: model_weights})
  class ModalMyFold:
      config: bytes = modal.parameter(default=b"{}")
      @modal.enter()
      def _initialize(self): self._core = MyFoldCore(json.loads(self.config)); self._core._initialize()
      @modal.method()
      def fold(self, sequences): return self._core.fold(sequences)
  ```
- Expose the public API via `ModelWrapper`, wiring Modal and Local backends and adding `embed` when creating an embedding model:
  ```python
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

## Backend & Security Notes
- Modal is production default; ensure GPU type and timeouts are tuned per workload, and document image changes in `boileroom/images`.
- Local debugging often needs extras (`boltz`, `chai`) plus CUDA; capture new requirements in `docs/local_debugging.md`, and keep `docs/backend_support_matrix.md` current.
- Never commit Modal credentials or `.modal` state. Secrets belong in environment variables or Modal secrets, not source.
