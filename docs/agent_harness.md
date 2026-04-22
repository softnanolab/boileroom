# Boileroom agent harness

The Boileroom harness turns repo conventions into objective checks that an agent can run before opening a PR. It is intentionally narrow: Bagel docs remain the source for public narrative documentation, while this repo owns implementation contracts that must stay true for code changes.

## Local command

Run the harness before a model-family PR:

```bash
uv run python scripts/harness/check_repo.py
```

For machine-readable output:

```bash
uv run python scripts/harness/check_repo.py --json-output /tmp/boileroom-harness.json
```

## What it checks

- Every model family directory has the required runtime files declared in `harness/model_family_contract.yaml`.
- `types.py` files stay lightweight and do not import heavy model/runtime libraries at runtime.
- `ModelSpec` entries, `RuntimeImageSpec` entries, wrapper exports, and image smoke target enumeration agree.
- Apptainer core class paths resolve statically, without importing model core modules.

## Adding a model family

An agent adding a model family should update the family files, registry metadata, image metadata, public exports, contract tests, and user-facing docs. The harness catches missing structural links, but it does not replace model behavior tests or Docker/Modal/Apptainer smoke validation.

Use the normal verification path after the harness passes:

```bash
uv run pytest -q -m contract
uv run pytest -n 4 -m "not integration"
```

For Modal integration tests, prefer grouped parallel execution:

```bash
uv run pytest -v -n 4 --dist loadgroup -m integration
```

The integration test modules are marked with model-family `xdist_group`s, so `--dist loadgroup` keeps Boltz, Chai, ESM2, and ESMFold on separate workers and separate Modal apps. To run the same integration tests in series, use:

```bash
uv run pytest -v -m integration
```

Public-facing development pages live in the separate `bagel-docs` repo. If a code change affects public docs, update that repo in a separate docs PR.
