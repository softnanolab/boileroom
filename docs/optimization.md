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
| H100, H200 | sm90 | served (kit config `h100`; Modal may schedule an H100 request onto an H200) |
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

## Measured (5 seeds x 3 complexes of 170-199 tokens, no MSA, bakeoff sampler settings)

Median warm wall seconds per fold through `ESMFold2Core` / `ProtenixCore` on Modal (wall includes pre- and
post-processing; the first fold of each process is excluded), and cost per fold at Modal list prices
(L4 $0.000222/s, L40S $0.000542/s, A100-80GB $0.000694/s, H100 $0.001097/s, H200 $0.001261/s).
The A100 runs landed on a mix of SXM4 and PCIe cards; the H100 rows are pinned real H100s (`H100!`).

| Model | Eval GPU, vanilla | A100 `fast` | H100 `fast` | H200 `fast` |
| --- | --- | --- | --- | --- |
| ESMFold2 (full) | L4: 4.40 s, $0.00098 | 0.47 s, $0.00032 | 0.26 s, $0.00029 | 0.23 s, $0.00028 |
| ESMFold2-Fast | L4: 2.51 s, $0.00056 | 0.33 s, $0.00023 | 0.19 s, $0.00021 | not run |
| Protenix v2 | L40S: 12.2 s, $0.0066 | 3.5 s, $0.0024 | 1.4 s, $0.0015 | not run |

`exact` (bit-identical with `--det 1`) lowers warm cost per fold by 1.8x for full ESMFold2 (H100: 0.48 s,
$0.00053), 1.3x for ESMFold2-Fast and 2.6-3.1x for Protenix, against the eval-GPU vanilla.

The kit pays a one-time warm-up on the first fold of a process (Triton JIT, CUDA graphs: 18-41 s for `fast`,
8-32 s for `exact`), so it only lowers cost for containers that stay warm. Against the eval GPU running
vanilla, `fast` on A100/H100 breaks even after about 28-41 folds per process for full ESMFold2, 44-55 for
ESMFold2-Fast and 2-7 for Protenix (no persistent `MODEL_OPT_JIT_ROOT`; a persistent cache was not measured).

![Average cost per fold vs folds per process](optimization-bench/cost_vs_folds.png)

Raw numbers: [`optimization-bench/summary.json`](optimization-bench/summary.json).
