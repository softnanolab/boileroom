# Kit optimization modes (`optimization`)

ESMFold2 (including ESMFold2-Fast), Protenix and OpenDDE accept a static init option
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
- OpenDDE: python 3.11, torch 2.7.1+cu126, cuequivariance 0.10.0, opendde 1.1.1, CUDA 12.6 `nvcc` and gcc at run time
  (fused LayerNorm and Triton JIT), libstdc++ from GCC 13 for `exact`, driver 560+. The Dockerfile installs the pinned kit
  commit; the kit is enabled in the worker before `runner` is imported.
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

Same GPU, vanilla vs kit (median warm wall time per fold and cost; speedup equals cost ratio because the GPU is the same):

| Model | GPU | vanilla | `exact` | `fast` |
| --- | --- | --- | --- | --- |
| ESMFold2 (full) | A100 | 2.04 s, $0.00141 | 0.79 s, $0.00055 (2.6x) | 0.47 s, $0.00032 (4.4x) |
| ESMFold2 (full) | H100 | 1.43 s, $0.00157 | 0.48 s, $0.00053 (3.0x) | 0.26 s, $0.00029 (5.5x) |
| ESMFold2-Fast | A100 | 1.51 s, $0.00105 | 0.62 s, $0.00043 (2.5x) | 0.33 s, $0.00023 (4.6x) |
| ESMFold2-Fast | H100 | 0.79 s, $0.00087 | 0.39 s, $0.00043 (2.0x) | 0.19 s, $0.00021 (4.2x) |
| Protenix v2 | A100 | 14.5 s, $0.0100 | 3.6 s, $0.0025 (4.0x) | 3.5 s, $0.0024 (4.1x) |
| Protenix v2 | H100 | 12.4 s, $0.0136 | 1.9 s, $0.0021 (6.4x) | 1.4 s, $0.0015 (9.1x) |
| OpenDDE | A100 | 16.1 s, $0.0111 | 12.7 s, $0.0088 (1.3x) | 9.3 s, $0.0064 (1.7x) |
| OpenDDE | H100 | 14.0 s, $0.0154 | 11.4 s, $0.0125 (1.2x) | 5.8 s, $0.0064 (2.4x) |

OpenDDE rows use the same 3 complexes but boileroom's default sampler settings (10 cycles, 200 steps, 5 samples per
seed, A100-80GB SXM4 / H100), so their absolute seconds are not comparable with the Protenix rows above, which used the
bakeoff's lighter settings. The first fold of a process takes 61-78 s in every mode (weight load, plus Triton JIT and CUDA
graph capture for the kit modes), and the kit modes add only 3-5 s to it, so they pay off from the second fold.
The fused LayerNorm CUDA extension is not built in the current image (the worker logs "Fast LayerNorm CUDA extension is
unavailable ... Ninja is required"), so all three modes use torch's `layer_norm`; installing ninja is the first thing to
try for a further speed-up and would change these numbers.

OpenDDE `exact` is not reproducible against vanilla at these settings: aligned RMSD of the same seed between vanilla and
`exact` has a median of 0.6-2.1 A and reaches 9-13 A on the flexible ubiquitin-barnase complex. This is not a kit
regression: two vanilla repeats of one seed already differ by up to 10 A and two `exact` repeats by up to 3.5 A,
because boileroom does not enable the runner's deterministic mode; the kit documents bit-identity only with
`--det 1`. Treat all modes as stochastic samplers until a `deterministic` option is added and re-measured.

Vanilla on A100/H100 costs more per fold than vanilla on the eval GPUs (L4 $0.00098 / $0.00056 for ESMFold2 / ESMFold2-Fast, L40S $0.0066 for Protenix), so the kit is what makes those GPUs worth using. A100 rows mix cards (ESMFold2-Fast and Protenix vanilla/exact on SXM4, `fast` on PCIe; full ESMFold2 vanilla/fast on PCIe, `exact` on SXM4), so A100 ratios carry a few percent of hardware noise.

`exact` (bit-identical with `--det 1`) lowers warm cost per fold by 1.8x for full ESMFold2 (H100: 0.48 s,
$0.00053), 1.3x for ESMFold2-Fast and 2.6-3.1x for Protenix, against the eval-GPU vanilla.

The kit pays a one-time warm-up on the first fold of a process (Triton JIT, CUDA graphs: 18-41 s for `fast`,
8-32 s for `exact`), so it only lowers cost for containers that stay warm. Against the eval GPU running
vanilla, `fast` on A100/H100 breaks even after about 28-41 folds per process for full ESMFold2, 44-55 for
ESMFold2-Fast and 2-7 for Protenix (no persistent `MODEL_OPT_JIT_ROOT`; a persistent cache was not measured).

![Average cost per fold vs folds per process](optimization-bench/cost_vs_folds.png)

Raw numbers: [`optimization-bench/summary.json`](optimization-bench/summary.json).
