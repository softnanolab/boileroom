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
- Repo harness: `uv run python scripts/harness/check_repo.py`
- Tests: `uv run pytest`
- Parallel tests: use `-n 4` if needed; do not use `-n auto`
- Lint/type checks: `uv run --extra dev pre-commit run --all-files`

## Coding Rules
- Python 3.12, 4-space indentation, type hints required.
- Prefer builtin generics and `X | Y` unions.
- Use f-strings.
- Use NumPy-style docstrings.
- Keep the high-level public API stable unless a breaking change is intentional and documented.
- While the project is on a prerelease (no matching `vX.Y.Z` GitHub release for the pyproject version), back-compat shims are not required — make the breaking change cleanly and call it out in the PR.

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
- Canonical published tags are CUDA-qualified, for example `cuda12.6-0.3.0` or `cuda12.6-0.3.1-alpha.1`.
- The default CUDA line `12.6` also gets an unqualified alias for the same version, for example `0.3.0` or `0.3.1-alpha.1`.
- Temporary validation tags such as `sha-<commit>` are allowed and should be deleted after use.
- Renames of runtime-facing env vars (e.g. `BOILEROOM_*`) bake into Modal images at build time. After such a rename, Modal images may need to be rebuilt before consumers can rely on the new name; verify before assuming the override propagates.

## Release Flow
- Merging to `main` publishes Docker Hub images for an automatically derived alpha prerelease tag such as `0.3.1-alpha.1`.
- Promoted `main` and full-release workflow runs build and validate the final alpha or stable image tag directly.
- Validation-only manual workflow runs may still publish temporary `sha-<commit>` tags.
- Alpha numbers are counted from the latest reachable stable release tag; before the first stable release tag, they use the configured CI baseline.
- Publishing a full GitHub release from a `vX.Y.Z` tag publishes verified Docker images with the stable `X.Y.Z` tag.
- GitHub releases marked as pre-releases do not publish stable Docker or PyPI artifacts.
- PyPI publication is separate and happens from the GitHub release workflow, which injects the stable release tag into `pyproject.toml` before building.
- Changing `project.version` changes the stable target for subsequent alpha image tags and package release semantics.

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
- Before pushing or updating a PR, run `uv run --extra dev pre-commit run --all-files`.
- Squash-and-merge into `main`. Delete the branch after merge.
- Never push directly to `main`. All changes go through a reviewed PR.
- All existing tests must pass before a PR is merged.

## Contributor Notes
- `AGENTS.md` is a symlink to this file.
- Keep this file concise. Put long-form implementation walkthroughs in `docs/` or Bagel docs instead of expanding always-loaded agent context.
