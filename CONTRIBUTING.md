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

## Pull request expectations

- All CI checks must pass.
- Add tests for new features.
- Update documentation when applicable.
- Version bump if required.

## Release and versioning

- Docker image version tags follow `project.version` in `pyproject.toml`.
- Runtime defaults also follow the installed boileroom package version unless you explicitly override the image tag.
- Merging to `main` triggers `.github/workflows/build-docker-images.yml`, which publishes Docker Hub tags for the current package version.
- PyPI publication is a separate step and only happens when a GitHub release is published.
- In practice, Docker Hub can act as the earlier staging/public channel while PyPI remains the later package release step.
- If you change `project.version`, you are also changing the versioned Docker tags that will be published from `main`.

## Further reading

- [Architecture overview](https://bagel.softnanolab.com/boileroom-api/development/architecture)
- [Adding a model](https://bagel.softnanolab.com/boileroom-api/development/adding-a-model)
- [Adding a backend](https://bagel.softnanolab.com/boileroom-api/development/adding-a-backend)
