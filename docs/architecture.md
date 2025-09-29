# Architecture & Backend Alignment

The high-level API (`ESMFold`, `ESM2`) is a thin wrapper around two building blocks:

- **Core algorithms** (`ESMFoldCore`, `ESM2Core`) encapsulate all model-specific logic. They know nothing about Modal, which keeps the code portable.
- **Modal integration** uses a single global `modal.App("boileroom")` managed by `ModalAppManager`. `ModalBackend` acquires/releases the shared app context and instantiates the corresponding Modal class (`ModalESMFold`, `ModalESM2`) only once, so multiple wrappers reuse the same remote container.

When you construct `ESMFold(backend="modal", device="T4", config={...})`:

1. `ESMFold` creates a `ModalBackend(ModalESMFold, config, device)`. The optional `device` string maps directly to Modal GPU SKUs (`"T4"`, `"A100-40GB"`, etc.).
2. `ModalBackend.startup()` acquires a reference-counted token from `ModalAppManager`. If the Modal app isn’t running yet, it calls `app.run()` once and keeps the context alive until the last wrapper shuts down.
3. The backend instantiates `ModalESMFold` (with `.with_options(gpu=device)` when provided) and exposes its remote methods. Calls like `model.fold([...])` delegate to `self._backend.model.fold.remote(...)`.

Because the same backend can serve multiple wrapper instances, avoid running many tests in parallel unless you provision enough GPUs—Modal keeps a container alive per remote class.
