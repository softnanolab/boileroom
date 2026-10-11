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

The family's decorated Modal class (often alongside its public wrapper, not necessarily in `modal.py`) subclasses `ModalFoldServer` or `ModalEmbedServer` from `boileroom/backend/modal_server.py`, declares `config: bytes = modal.parameter(default=b"{}")` on itself (Modal reads parameters only from the decorated class), and implements `_build_core()`, importing the core inside the method so the module stays importable without the model's libraries. Do not add your own `@modal.enter()` that loads the model: an exception there sends the container into a silent restart loop, which the shared enter hook avoids. Additional task methods must retain this guarded lifecycle and record GPU memory, as ESM3's `inverse_fold()` does.

Use the normal verification path after the harness passes:

```bash
uv run --extra dev pytest -q -m contract
uv run --extra dev pytest -n 4 -m "not integration"
uv run --extra dev pre-commit run --all-files
```

Real-model integration runs can incur GPU charges, download weights, and contact external MSA services. Select only the approved models, devices, tests, and budget. An unfiltered `pytest` run includes integration tests. Once that scope is approved, prefer grouped parallel execution for Modal:

```bash
uv run pytest -v -n 4 --dist loadgroup -m integration
```

The integration test modules are marked with model-family `xdist_group`s, so `--dist loadgroup` keeps Boltz, Chai, ESM2, ESMFold2, ESMFold, Protenix, OpenDDE, RF3, and AlphaFold2-Multimer on separate workers and separate Modal apps. To run the same integration tests in series, use:

```bash
uv run pytest -v -m integration
```

Public-facing development pages live in the separate `bagel-docs` repo. If a code change affects public docs, update that repo in a separate docs PR.

## Model integration checklist

Use this checklist for a new family, a new task such as sequence generation, or an older PR being brought onto current `main`. It describes the integration work and evidence expected; it does not claim every existing model supports every capability.

### 1. Connect the public API to the current runtime

- Read `boileroom/base.py`, `boileroom/models/registry.py`, `boileroom/images/metadata.py`, and the family contract before choosing an example. Check the relevant `tests/contracts/` cases and CI workflows rather than copying an old wrapper wholesale.
- Add the family files required by `harness/model_family_contract.yaml`, public exports, `ModelSpec`/`ModelContract`, runtime image metadata, and smoke targets together. Families sharing an image should reference its owner rather than duplicate its stack. Keep `types.py` within the harness's lightweight import rules; importing the public API must not require upstream model packages or weights.
- Declare real task/input/output semantics. Folding, embedding, and ESM3's additional `inverse_fold()` method illustrate distinct contracts. For a new primary task such as generation, update the shared protocol/registry, wrapper dispatch, both advertised backends, serialization, harness checks, and contract tests together. Do not disguise generated sequences as structures or embeddings just to satisfy existing checks.
- Keep Modal classes thin, with the guarded lifecycle described above. New task forwarding must retain cleanup and post-call memory recording. Verify Apptainer's matching request/response route and error handling; a Modal-only test does not establish Apptainer support.

### 2. Validate before expensive work

- List accepted keys in `DEFAULT_CONFIG`, classify initialization-only keys in `STATIC_CONFIG_KEYS`, and keep the registry copy equal. Use the shared option validation and add model-specific value/range checks. Test construction and per-call options, including typos and attempted static overrides.
- State whether a top-level list is a batch, whether `:` separates chains, and any limits on chains, length, samples, or model variants. Refuse unsupported inputs explicitly rather than truncating, concatenating, or silently dropping them. For generation, specify prompt/conditioning semantics and how length limits and special tokens map to returned sequences.
- `SUPPORTS_USER_MSA` and `SUPPORTS_USER_TEMPLATES` are promises to consume those inputs. Reuse `boileroom.inputs.parse_a3m` where appropriate, then enforce the family's query, alignment, and chain rules. Materialize caller-local files before crossing a backend/interpreter boundary. Test that supplied inputs actually reach the upstream runner and are not silently replaced by a search.
- Document external services that receive sequences and how to disable their use. Preserve provider-aware cache identity. Validate template compatibility rather than accepting templates that upstream will discard.
- Follow the current family mode policy in `boileroom/optimization.py`. Models without a kit support vanilla only; unsupported cards, missing kernels, inactive/partial kits, and unavailable modes must fail clearly. Preserve `OptimizationUnavailableError` through workers and HTTP. A standing refusal must not become a silent fallback or a restart loop.

### 3. Preserve output meaning and reproducibility

- Return dependency-light dataclasses with fresh `PredictionMetadata` and CPU materialized arrays/structures. The next call must not mutate a previously returned result. Keep minimal output fields when filtering and test default, selected, and wildcard `include_fields` through transport round trips.
- For structure outputs, normalize per-residue `plddt` to `[0, 1]` and scalar `ptm`/`iptm` to per-sample shape-`(1,)` arrays where available. Keep missing values missing. Ensure every structure, confidence array, seed, and sample rank uses the same ordering; distinguish token, residue, and atom axes.
- Check finite coordinates/confidences, PAE dimensions, and chain/residue correspondence, including repeated chains and multi-sample outputs. See `tests/contracts/test_confidence_metrics.py` and `tests/rf3/test_rf3_outputs.py`. For embeddings/logits or generated sequences, test residue/token alignment, masked-position order, amino-acid vocabulary order, and removal of padding/special tokens as applicable.
- Record the code/image, resolved stack, actual checkpoint identity, device, mode, and effective sampling settings needed to reproduce the run. Use shared provenance helpers and the existing `PredictionMetadata.runtime` key conventions in [optimization.md](optimization.md#what-a-prediction-records); worker facts must come from the worker, not the parent's interpreter.
- Pin upstream revisions and reproducible runtime dependencies in the canonical Dockerfile/requirements. Verify downloaded checkpoints when a trusted checksum is available. Check code and weights' licenses separately, especially before baking weights into published images.
- Checkpoints can contain executable pickle data. If upstream requires full-pickle/Lightning loading, establish trust in the official pinned artifact and verify its identity before loading. Do not silently retry a safe-load failure with `weights_only=False` or treat an arbitrary caller-selected checkpoint as trusted.
- Reference fixtures need a manifest with the upstream revision, checkpoint identity, inputs, seed/settings, device, and regeneration command. RF3, ESMFold2, and SAE fixtures under `tests/data/` are examples. Check upstream semantics as well as plausible shapes: SAE's reference tests guard normalization and layer selection. Use justified numeric/structural tolerances and seed-aware comparisons; stochastic samplers are not promised identical output solely by a shared seed or the name `exact`.

### 4. Keep repeated calls bounded and isolated

- Load weights/runners once per live runtime. Reuse `ModelWorker` for isolated upstream interpreters rather than shelling out to load a new model for every request. Preserve interpreter-compatible worker code and environment isolation; never import the parent's package into a worker that cannot run its Python version.
- Give each request fresh options, temporary outputs, and metadata. Test changing seeds, lengths, conditioning, and output selection across calls on the same instance. Close workers/resources on shutdown; retain hard timeout termination and the distinction between recoverable failures and standing refusals.
- Test model/cache lifecycle separately from inference correctness. Cached weights, graph-captured buffers, and geometry caches have different ownership: freeing a workspace still referenced by a CUDA graph can produce NaNs, while retaining every shape can exhaust memory. Recheck private upstream cache hooks whenever their pinned revision changes.
- Extend `tests/test_memory_stability.py` or add an appropriate task-specific case for new methods. Repeated shape-controlled calls after warm-up must show finite outputs and bounded allocated/reserved memory; worker allocator counters must be measured in the child. Compare device-wide memory as well, and distinguish bounded shape/cache warm-up from sustained growth. A single successful prediction is not a memory-stability test.
- Bound samples, runtime, GPU count, and concurrency for validation. Use at most `-n 4`, with `--dist loadgroup` and family groups for Modal integration. Run serially on a single local Apptainer GPU unless explicit device isolation prevents competing models. Do not launch broad sweeps or paid builds to fix an offline contract failure.

### 5. Verify in layers and report the gaps

1. Run the harness, offline contracts, full non-integration suite, and pre-commit commands above. Use mocks/fakes for validation, lifecycle, refusal, output parsing, and transport tests. Tests must collect without model-only libraries; `importorskip()` belongs inside tests that genuinely need them.
2. For image changes, use `scripts/images/` import and server-health checks against the actual target interpreter and built image. Preserve Apptainer's read-only package bind, clean environment, loopback service, and HMAC verification before unpickling. Never accept untrusted serialized output or print credentials to work around a failure.
3. When authorized, run scoped real-model tests on each backend/device/mode claimed by the PR. Separate image import/startup, weight loading, inference/output correctness, upstream-reference parity, and repeated-call memory results. Kit tests require an additional explicit `--run-kit` opt-in; on Modal, `--kit-image-tag` needs `BOILEROOM_KIT_IMAGE_SOURCE=registry`.
4. Identify the exact tested commit and image digest/reference, model/checkpoint, device, command, and result. Report skips, expected failures, unavailable credentials/hardware, and unrun checks separately from passes. Do not turn missing validation into a success claim or weaken a check to hide it.
5. Re-run affected checks after integration or conflict resolution and inspect CI for the final remote head. A docs-only change does not require paid inference, but guidance and examples must agree with current code. A draft PR or passing CI alone is not authorization to merge, publish a release, or start more compute.

See [architecture.md](architecture.md), [models.md](models.md), [docker_images.md](docker_images.md), and [optimization.md](optimization.md) for implementation detail. If a narrative example conflicts with executable contracts, verify the current implementation and update the affected documentation rather than restoring obsolete behavior.
