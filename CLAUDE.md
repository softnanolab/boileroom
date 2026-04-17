# Repository Guidelines

## Structure
- `boileroom/models/<family>/` contains the public wrapper, Modal wrapper, core implementation, and lightweight output types for each model family.
- `boileroom/models/<family>/types.py` must stay lightweight. It may depend on `numpy`, dataclasses, and base protocols, but not heavy model libraries.
- `boileroom/backend/` contains shared backend machinery plus the Modal and Apptainer implementations.
- `boileroom/images/` contains shared runtime image metadata and Modal image helpers.
- `scripts/images/` contains Docker build, smoke, and promotion tooling.
- `docs/` contains user-facing and design documentation. Update it when behavior, release flow, or backend semantics change.

## Commands
- Use `uv run` for repo commands. Do not rely on system Python.
- Setup: `uv python install 3.12 && uv sync`
- Tests: `uv run pytest`
- Parallel tests: use `-n 4` if needed; do not use `-n auto`
- Lint/type checks: `uv run pre-commit run --all-files`

## Coding Rules
- Python 3.12, 4-space indentation, type hints required.
- Prefer builtin generics and `X | Y` unions.
- Use f-strings.
- Use NumPy-style docstrings.
- Keep the high-level public API stable unless a breaking change is intentional and documented.

## Testing
- Prefer pytest functions and fixtures.
- Add positive and failure-path coverage for behavior changes.
- Tests must not import heavy core modules at module scope. Use high-level wrappers or `pytest.importorskip()` inside the test.
- Contract tests should cover shared wrapper/image behavior without requiring model dependencies.

## Runtime Design
- Core classes own validation, loading, and `fold()` / `embed()` behavior.
- Modal wrappers should stay thin and delegate directly to the core.
- Apptainer runs the core in an HTTP microservice inside the container.
- Models in the same family may share a runtime image.
- Keep boiler room dependencies isolated from model-specific runtime dependencies.

## Image and Versioning Policy
- Dockerfiles are the canonical runtime definition for Docker, Modal, and Apptainer.
- Runtime image lookup defaults to the installed boileroom package version.
- `latest` is not published or used as a runtime default.
- Explicit image-tag overrides still work:
  - Modal via `BOILEROOM_MODAL_IMAGE_TAG`
  - Apptainer via `backend="apptainer:<tag>"`
- Canonical published tags are CUDA-qualified, for example `cuda12.6-0.3.0`.
- The default CUDA line `12.6` also gets an unqualified alias for the same version, for example `0.3.0`.
- Temporary validation tags such as `sha-<commit>` are allowed and should be deleted after use.

## Release Flow
- Merging to `main` publishes Docker Hub images for an automatically derived `0.3.x` version.
- The workflow first publishes a temporary `sha-<commit>` tag, verifies it, then promotes it to `0.3.x` tags.
- The `0.3.x` patch component is derived from commits after the configured CI baseline and grows with each main-branch commit.
- PyPI publication is separate and happens from the GitHub release workflow, which injects the `0.3.x` release tag into `pyproject.toml` before building.
- Changing `project.version` changes package release semantics, but Docker image tags are derived by CI.

## Apptainer Notes
- Apptainer images are pulled from Docker Hub and cached as `.sif` files.
- The backend binds the repo source tree read-only plus `MODEL_DIR`.
- Model-specific subdirectories under `MODEL_DIR` are derived automatically where needed.

## Branch and PR Conventions
- Branch names use `<type>/<short-description>` in kebab-case, e.g. `feat/profam`, `fix/ci-disk-space`, `chore/add-author`.
  - Allowed types: `feat`, `fix`, `chore`, `docs`, `refactor`, `ci`, `test`.
- Commit messages and PR titles follow [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/): `<type>: <lowercase description>`. No trailing period. Keep under 70 characters.
- Commits should be atomic: one meaningful change per commit.
- One feature per PR. Do not combine unrelated changes.
- Create a draft PR immediately when opening a new branch. Convert to ready when it is ready for review.
- Squash-and-merge into `main`. Delete the branch after merge.
- Never push directly to `main`. All changes go through a reviewed PR.
- All existing tests must pass before a PR is merged.

## Contributor Notes
- `AGENTS.md` is a symlink to this file.
- Keep this file concise. Put long-form implementation walkthroughs in `docs/` or Bagel docs instead of expanding always-loaded agent context.
