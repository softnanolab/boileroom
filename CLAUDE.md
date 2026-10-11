# Repository Guidelines

## Start Here
- Read the current `boileroom/models/registry.py`, `harness/model_family_contract.yaml`, and relevant contracts before adapting an older branch or upstream example. Do not copy obsolete backend or image patterns.
- For new models, new task methods, or runtime changes, follow [the integration checklist](docs/agent_harness.md#model-integration-checklist). It covers generation as well as folding and embeddings; advertised capabilities must match the implementation.

## Structure
- `boileroom/models/<family>/` contains the public wrapper, Modal wrapper, core implementation, and lightweight output types for each model family.
- `boileroom/models/<family>/types.py` must stay lightweight. It may depend on `numpy`, dataclasses, and base protocols, but not heavy model libraries.
- `boileroom/backend/` contains shared backend machinery plus the Modal and Apptainer implementations.
- `boileroom/images/` contains shared runtime image metadata and Modal image helpers.
- `scripts/images/` contains Docker build, smoke, and promotion tooling.
- `docs/` contains user-facing and design documentation. Update it when behavior, release flow, or backend semantics change.

## Commands
- Use `uv run` for repo commands. Do not rely on system Python.
- Setup: `uv python install 3.12 && uv sync --extra dev`
- Repo harness: `uv run python scripts/harness/check_repo.py`
- Fast contracts: `uv run --extra dev pytest -q -m contract`
- Offline/unit suite: `uv run --extra dev pytest -n 4 -m "not integration"`
- Plain `uv run pytest` includes real-model integration tests and defaults to Modal. Run paid GPU tests/builds only within an approved scope and budget; kit tests additionally require `--run-kit` or `BOILEROOM_RUN_KIT=1`.
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
- Cover invalid inputs, unsupported capabilities, repeated calls, output filtering/serialization, and failure/cleanup paths. A mocked runner or import smoke is not evidence of working model inference or scientific parity.
- Report the exact commit, commands, image/reference, device, passes, skips, and unrun checks. See the checklist for image, reference-output, and memory validation.

## Runtime Design
- Core classes own validation, loading, and task behavior. Keep `DEFAULT_CONFIG`, `STATIC_CONFIG_KEYS`, capability flags, and the registry contract aligned; reject unknown keys and unsupported inputs instead of dropping them.
- Load models once per live runtime and reuse weights/runners across calls. Keep per-call options, output paths, metadata, and returned arrays independent; close workers and temporary resources on shutdown/failure.
- Modal classes use the shared guarded lifecycle in `boileroom/backend/modal_server.py`, a lazy `_build_core()`, and their own `config: bytes = modal.parameter(default=b"{}")`. Do not load models in an unguarded `@modal.enter()` hook.
- Apptainer runs the core in an HTTP microservice inside the container.
- Use `ModelWorker` when an upstream stack needs an isolated interpreter; preserve its hard deadlines, cleanup, and refusal propagation. Do not silently retry a standing optimization refusal or fall back to another mode/kernel.
- Keep output types dependency-light and materialize return values on CPU. Preserve sample order, confidence units, and chain/residue mappings; document genuinely unavailable fields rather than inventing scores.
- Record resolved runtime/weight/optimization provenance, not just requested settings. Follow `boileroom/optimization.py` for supported family modes: historical benchmarks do not authorize a withdrawn mode, and `exact` does not promise byte-identical stochastic outputs.
- Models in the same family may share a runtime image.
- Keep boiler room dependencies isolated from model-specific runtime dependencies.

## Image and Versioning Policy
- Dockerfiles are the canonical runtime definition for Docker, Modal, and Apptainer.
- Runtime image lookup defaults to the installed boileroom package version.
- `latest` is not published or used as a runtime default.
- Explicit image-tag overrides still work:
  - Pytest accepts `--image-tag <tag>` for both Modal and Apptainer test backends
  - Modal and Apptainer both honor `BOILEROOM_IMAGE_TAG` as the shared runtime image-tag override
  - Apptainer also accepts an inline tag via `backend="apptainer:<tag>"`, which wins over `BOILEROOM_IMAGE_TAG` and `--image-tag` in tests
- Canonical published tags are CUDA-qualified, for example `cuda12.6-0.3.0` or `cuda12.6-0.3.1-alpha.1`.
- The default CUDA line `12.6` also gets an unqualified alias for the same version, for example `0.3.0` or `0.3.1-alpha.1`.
- Temporary validation tags such as `sha-<commit>` are allowed and should be deleted after use.
- Separate kit images (`*-kit`, for supported non-vanilla modes) use their own Dockerfiles: Modal builds them from source by default; registry/Apptainer pulls use `KIT_IMAGE_DIGESTS` unless an explicit kit tag overrides them. `BOILEROOM_IMAGE_TAG` never applies; `--kit-image-tag` on Modal requires `BOILEROOM_KIT_IMAGE_SOURCE=registry`. OpenDDE carries its kit in its stock image. See [image policy](docs/docker_images.md) before changing pins or promotion.
- Pin upstream code/toolchains and identify the actual checkpoint revision or checksum. Check code and weight licenses separately before redistribution; do not commit downloaded weights, credentials, or runtime caches.
- Renames of runtime-facing env vars (e.g. `BOILEROOM_*`) bake into Modal images at build time. After such a rename, Modal images may need to be rebuilt before consumers can rely on the new name; verify before assuming the override propagates.

## Release Flow
- Merging to `main` publishes Docker Hub images for an automatically derived alpha prerelease tag such as `0.3.1-alpha.1`.
- Promoted `main` and full-release workflow runs build and validate a `sha-<commit>` candidate tag, then retag each verified image to the final alpha or stable tag; a full release also tags the pinned kit image digests with `X.Y.Z`.
- Validation-only manual workflow runs may still publish temporary `sha-<commit>` tags.
- Alpha numbers are counted from the latest reachable stable release tag; before the first stable release tag, they use the configured CI baseline.
- Publishing a full GitHub release from a `vX.Y.Z` tag publishes verified Docker images with the stable `X.Y.Z` tag.
- GitHub releases marked as pre-releases do not publish stable Docker or PyPI artifacts.
- PyPI publication is separate and happens from the GitHub release workflow, which injects the stable release tag into `pyproject.toml` before building.
- Changing `project.version` changes the stable target for subsequent alpha image tags and package release semantics.

## Apptainer Notes
- Apptainer images are pulled from Docker Hub and cached as `.sif` files.
- The backend binds the `boileroom/` package directory read-only plus `MODEL_DIR`, not the host virtualenv or full repo.
- Model-specific subdirectories under `MODEL_DIR` are derived automatically where needed.
- Preserve the loopback-only service default, HMAC verification before deserialization, and isolated interpreter/environment handling. Never weaken transport checks to make a smoke test pass.

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
- All existing tests must pass before a PR is merged. Revalidate the final head after integrating another branch; do not describe unrun GPU or backend checks as passed.

## Contributor Notes
- `AGENTS.md` is a symlink to this file.
- Keep this file concise. Put long-form implementation walkthroughs in `docs/` or Bagel docs instead of expanding always-loaded agent context.
