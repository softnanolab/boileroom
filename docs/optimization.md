# Kit optimization modes (`optimization`)

ESMFold2 (including ESMFold2-Fast) and Protenix accept a static init option
`optimization = "vanilla" | "exact" | "fast"` (default `"vanilla"`). `exact` and `fast` drive the
prebuilt kernels of [anthropics/uplifting-biomolecular-modeling](https://github.com/anthropics/uplifting-biomolecular-modeling)
(Apache-2.0, kit commit `f4f62fa`).

- `vanilla`: unchanged behavior.
- `exact`: byte-identical to the unoptimized path when the kit's `--det 1` recipe is applied.
- `fast`: numerically different, within seed-to-seed spread.

A mode is all of its levers on a GPU class. The mode is resolved against the visible GPU before any
weights load, and a GPU or stack the kit cannot serve fails by name with `OptimizationUnavailableError`.

| GPU | class | `exact` / `fast` |
| --- | --- | --- |
| A100 | sm80 | served (kit config `a100`) |
| H100 | sm90 | served (kit config `h100`) |
| L4, L40S | sm89 | refused by name (ESMFold2: kit levers need more shared memory or have no tile table; Protenix: no BLK2 launch cells) |

## Requirements

The kit ships its own pinned stack, so kit modes need a kit image, not the default boileroom image:

- ESMFold2: python 3.12, torch 2.13+cu130, esm 3.3.0 and the kit's transformers fork (`ESMFold2Model`),
  flash-attn / TransformerEngine built from source. Kit modes load the fork's model class and fold through
  `ESMFold2InputBuilder.fold()`, which the kit hooks; `vanilla` keeps loading esm's `EsmFold2Model`.
- Protenix: python 3.11, torch 2.13.0+cu130, cuequivariance 0.11.1, protenix 2.0.0, driver 580+.
  The kit is enabled inside the worker before any `protenix` import (the kit refuses late activation).
- ESM-C must be the 2026-06-03 snapshot (`45b0fa5d7fb0`); the 2026-09-14 re-upload renames every weight and
  loads as uninitialised memory (NaN structures).

`PredictionMetadata.optimization` records the requested and resolved mode, kit config and GPU.

## Measured (5 seeds x 3 sequences of 170-199 tokens, no MSA, bakeoff sampler settings)

Median seconds per fold after warm-up, and cost per fold at Modal list prices
(L4 $0.000222/s, L40S $0.000542/s, A100-80GB $0.000694/s, H100 $0.001097/s).

| Model | Eval GPU, vanilla | A100 `fast` | H100 `fast` |
| --- | --- | --- | --- |
| ESMFold2-Fast | L4: 2.45 s, $0.00054 | 0.26 s, $0.00018 | 0.13 s, $0.00014 |
| Protenix v2 | L40S: 12.1 s, $0.0066 | 3.4 s, $0.0024 | 1.3 s, $0.0015 |

The kit pays a one-time warm-up on the first fold of a process (Triton JIT, CUDA graphs: 20-40 s), so it
only lowers cost for containers that stay warm across many folds. Without a persistent
`MODEL_OPT_JIT_ROOT`, 15 folds average slightly worse than vanilla on the eval GPUs.
