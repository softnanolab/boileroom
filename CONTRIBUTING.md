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
uv run pytest -n auto
```

Tests use **Modal** by default. You need `MODAL_TOKEN_ID` and
`MODAL_TOKEN_SECRET` set in your environment.

## Linting

```bash
pre-commit run --all-files
```

This runs **ruff** and **mypy**.

## Pull request expectations

- All CI checks must pass.
- Add tests for new features.
- Update documentation when applicable.
- Version bump if required.

## Further reading

- [Architecture overview](https://bagel.softnanolab.com/boileroom-api/development/architecture)
- [Adding a model](https://bagel.softnanolab.com/boileroom-api/development/adding-a-model)
- [Adding a backend](https://bagel.softnanolab.com/boileroom-api/development/adding-a-backend)
