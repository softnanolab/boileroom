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
- Type hints are required; CI enforces `mypy` (config in `.pre-commit-config.yaml`).
- `ruff` manages linting and formatting with a 120-character line budget; prefer auto-fixes via pre-commit.
- Place Modal configuration constants in `boileroom/constants.py`; avoid duplicating literal IDs across modules.
- Use f-strings and not any other way fo doing strings, e.g., f"Today is {date}".

## Testing Expectations
- Prefer pytest-style `test_*` functions and fixtures; see `tests/test_utils.py` for style cues.
- New features need positive and failure-path coverage in the matching backend folder and should prove Modal/local parity when relevant.
- Run `uv run pytest` (or targeted cases like `uv run pytest tests/test_basic.py::test_esmfold_batch -v -s`) before opening a PR.

## Backend Architecture & System Design
- Each model owns a Core class (`FoldingAlgorithm` or `EmbeddingAlgorithm`) that loads weights in `_load`, validates sequences, and exposes `fold` or `embed`. Keep Modal-side latency low by caching assets in `_initialize`.
- Modal wrappers (`ModalBoltz2`, `ModalChai1`, `ModalESMFold`, `ModalESM2`) live beside the core, use `@app.cls`, mount `model_weights`, and rehydrate the core inside `@modal.enter`. Keep method signatures thin—call the core directly and let Modal manage GPU scheduling.
- High-level classes (`Boltz2`, `Chai1`, `ESMFold`, `ESM2`) inherit `ModelWrapper`, choose `ModalBackend` by default, and fall back to `LocalBackend` for debugging. `LocalBackend` mirrors Modal calls but expects the environment to satisfy dependencies (`uv pip install .[boltz]` etc.).
- Backends share mechanisms through `boileroom/backend/base.py`; add new providers (e.g. Conda, Batch) by implementing `startup`, `shutdown`, and `get_model`.

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
          else:
              raise ValueError("Backend not supported")
          self._backend.start()
  ```

## Backend & Security Notes
- Modal is production default; ensure GPU type and timeouts are tuned per workload, and document image changes in `boileroom/images`.
- Local debugging often needs extras (`boltz`, `chai`) plus CUDA; capture new requirements in `docs/local_debugging.md`, and keep `docs/backend_support_matrix.md` current.
- Never commit Modal credentials or `.modal` state. Secrets belong in environment variables or Modal secrets, not source.
