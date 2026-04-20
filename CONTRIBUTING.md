# Contributing to Boileroom

Full developer documentation lives at
**[bagel-docs](https://bagel.softnanolab.com/boileroom-api/development/contributing)**.
This guide covers the essentials to get started quickly.

## Quick setup

```bash
git clone <repo-url> && cd boileroom
uv python install 3.12 && uv python pin 3.12 && uv sync
```

## Running tests

```bash
uv run pytest
```

Tests use **Modal** by default. You need `MODAL_TOKEN_ID` and
`MODAL_TOKEN_SECRET` set in your environment.

## Linting

```bash
uv run pre-commit run --all-files
```

This runs **ruff** and **mypy**.

## Repo harness

Before opening a model-family, backend, or image-related PR, run:

```bash
uv run python scripts/harness/check_repo.py
```

The harness checks objective implementation contracts such as model-family files,
lightweight output types, registry/image links, public wrapper exports, and image
smoke target coverage. See [docs/agent_harness.md](docs/agent_harness.md).

## Pull request expectations

- All CI checks must pass.
- Add tests for new features.
- Update documentation when applicable.
- Version bump if required.

## Release and versioning

- Docker image version tags on `main` are prereleases for `project.version` in `pyproject.toml`, for example `0.3.1-alpha.1`.
- Runtime defaults also follow the installed boileroom package version unless you explicitly override the image tag.
- Merging to `main` triggers `.github/workflows/build-docker-images.yml`, which publishes Docker Hub tags for the current alpha prerelease.
- Stable Docker tags and PyPI publication are separate manual release steps and happen from GitHub release tags such as `v0.3.1`.
- In practice, Docker Hub can act as the earlier staging/public channel while PyPI remains the later package release step.
- After a stable release, bump `project.version` to the next intended stable version so subsequent `main` builds become the next alpha series.

## Further reading

- [Architecture overview](https://bagel.softnanolab.com/boileroom-api/development/architecture)
- [Adding a model](https://bagel.softnanolab.com/boileroom-api/development/adding-a-model)
- [Adding a backend](https://bagel.softnanolab.com/boileroom-api/development/adding-a-backend)
