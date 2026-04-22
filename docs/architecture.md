# Architecture & Backend Alignment

The high-level API (`ESMFold`, `ESM2`) is a thin wrapper around two building blocks:

- **Core algorithms** (`ESMFoldCore`, `ESM2Core`) encapsulate all model-specific logic. They know nothing about Modal, which keeps the code portable.
- **Modal integration** registers each model entrypoint on its own `modal.App` (for example `boileroom-esmfold` and `boileroom-boltz2`) managed by `ModalAppManager`. `ModalBackend` acquires/releases the app attached to the selected Modal class and instantiates that class only once, so test processes can run model-specific apps without registering unrelated GPU functions.

When you construct `ESMFold(backend="modal", device="T4", config={...})`:

1. `ESMFold` creates a `ModalBackend(ModalESMFold, config, device)`. The optional `device` string maps directly to Modal GPU SKUs (`"T4"`, `"A100-40GB"`, etc.).
2. `ModalBackend.startup()` acquires a reference-counted token from the selected class app's `ModalAppManager`. If the Modal app isn’t running yet, it calls `app.run()` once and keeps the context alive until the last wrapper shuts down.
3. The backend instantiates `ModalESMFold` (with `.with_options(gpu=device)` when provided) and exposes its remote methods. Calls like `model.fold([...])` delegate to `self._backend.model.fold.remote(...)`.

Model-specific test shards can run in parallel without importing unrelated GPU functions into each Modal app. The integration suite marks each model with an xdist group and runs with `--dist loadgroup`, so each worker executes tests for only one Modal app. To run the same tests in series, omit xdist and run `uv run pytest -v -m integration`; the integration-test workflow also exposes a manual `series` execution mode. Parallelism still needs to match the number of remote classes under test, since Modal keeps a container alive per active class.
