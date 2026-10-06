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

The kit ships its own pinned stack, so `exact` and `fast` need a kit image, not the default boileroom image:

- ESMFold2: python 3.12, torch 2.13+cu130, esm 3.3.0 and the kit's transformers fork (`ESMFold2Model`),
  flash-attn / TransformerEngine built from source. Kit modes load the fork's model class and fold through
  `ESMFold2InputBuilder.fold()`, which the kit hooks; `vanilla` keeps loading esm's `EsmFold2Model`.
- Protenix: python 3.11, torch 2.13.0+cu130, cuequivariance 0.11.1, protenix 2.0.0, driver 580+.
  The kit is enabled inside the worker before any `protenix` import (the kit refuses late activation).
- OpenDDE: python 3.11, torch 2.7.1+cu126, cuequivariance 0.10.0, opendde 1.1.1, CUDA 12.6 `nvcc` and gcc at run time
  (Triton JIT), libstdc++ from GCC 13 for `exact`, driver 560+. Unlike the other two families it has no separate kit image: the
  stock `boileroom-opendde` image installs the pinned kit commit, and the kit is enabled in the worker before `runner` is imported.

For ESMFold2 and Protenix the repository carries the definition of that separate kit image, and `optimization="vanilla"` never
touches it:

| Family | Dockerfile | Image name | Base |
| --- | --- | --- | --- |
| ESMFold2 | [`boileroom/models/esmfold2/kit/Dockerfile`](../boileroom/models/esmfold2/kit/Dockerfile) (+ `kit_wheels.sh`, `kit_finish.sh`) | `boileroom-esmfold2-kit` | `python:3.12.10-slim-bookworm` |
| Protenix | [`boileroom/models/protenix/kit/Dockerfile`](../boileroom/models/protenix/kit/Dockerfile) | `boileroom-protenix-kit` | `nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04` |

Both Dockerfiles fetch the kit at the pinned commit `f4f62fa6592ae4938d49b1757bea0cfeff9f468e` and follow the kit's own
`<family>/environment/Dockerfile` at that commit: the same lock, versions, checksums and compile recipe. The changes are the
kit coming from git instead of a build context, the optional `_jitcache` `COPY` removed (a glob that Modal's Dockerfile parser
rejects), and boileroom's runtime dependencies on top (`fastapi`/`uvicorn` for the Apptainer service). These are the
definitions the benchmark images in [Measured](#measured-5-seeds-x-3-complexes-of-170-199-tokens-no-msa-bakeoff-sampler-settings)
were made from; the benchmark recipe itself lived outside the repository, so what is checked in is a reconstruction of it, not
the original files.

**Routing.** The wrapper picks the image from `optimization`: `vanilla` (the default) runs the stock `boileroom-esmfold2` /
`boileroom-protenix` image and Modal class as before, and `exact` / `fast` run `ModalESMFold2Kit` / `ModalProtenixKit` (Modal)
or the kit image (Apptainer). The kit classes live on their own Modal apps (`boileroom-esmfold2-kit`,
`boileroom-protenix-kit`), so using vanilla never builds a kit image. A kit class asks for an A100 (ESMFold2 `A100-80GB`,
Protenix `A100-40GB`); pass `device="H100"` (or `H200`) to run on those.

**Where the image comes from.** `BOILEROOM_KIT_IMAGE_SOURCE` selects it:

- `build` (default): Modal builds the image from the Dockerfile in your installed boileroom. Nothing is published, so the first
  kit-mode use pays the build and Modal caches it afterwards. Protenix compiles its fast-LayerNorm extension, which takes
  minutes. ESMFold2 compiles flash-attn, TransformerEngine and xformers for compute capability 8.0 and 9.0 on a 48-core,
  192 GiB build step with a 4 h limit (the benchmarked image compiled in 1893 s on 64 cores; budget 30-45 min on Modal, more
  on a smaller builder).
- `registry`: Modal (or Apptainer, as `docker://`) pulls `<repository>/boileroom-<family>-kit:<tag>` for an image you built and
  pushed. The repository comes from `BOILEROOM_DOCKER_REPOSITORY` and the tag from `BOILEROOM_IMAGE_TAG`, as for the stock
  images. This repository's CI does not build the kit images (the ESMFold2 compile takes hours on a GitHub-hosted
  runner); they are built locally and pushed by hand. Both are on `docker.io/jakublala` as `boileroom-esmfold2-kit` and
  `boileroom-protenix-kit` under the temporary tag `sha-dc652b0`, to be retagged to the release version after merge.

Apptainer only pulls from a registry, so a kit mode on the Apptainer backend needs a published image: set
`BOILEROOM_KIT_IMAGE_SOURCE=registry` or pass `backend="apptainer:<tag>"`. Without either, boileroom refuses the call up
front instead of failing on a pull that cannot succeed.

To build an image yourself on a many-core machine and use it from Modal or Apptainer:

```bash
docker build -f boileroom/models/protenix/kit/Dockerfile boileroom/models/protenix/kit -t <repository>/boileroom-protenix-kit:<tag>
docker build -f boileroom/models/esmfold2/kit/Dockerfile boileroom/models/esmfold2/kit -t <repository>/boileroom-esmfold2-kit:<tag>
# H100/H200 only, a smaller ESMFold2 compile: add --build-arg STACK=img_ef2_fa
docker push <repository>/boileroom-esmfold2-kit:<tag>   # and likewise for protenix

export BOILEROOM_KIT_IMAGE_SOURCE=registry BOILEROOM_DOCKER_REPOSITORY=<repository> BOILEROOM_IMAGE_TAG=<tag>
```

The Modal `build` path produces the same image through the same scripts: the Dockerfile is built with `WHEELS_FROM=defer` and
the compile and install (`kit_wheels.sh`, `kit_finish.sh`) run as a Modal build step on the larger builder. The image
identity is the Dockerfile plus those scripts at the pinned kit commit.

**ESMFold2 weights.** The kit loads its own pinned Hugging Face snapshots (`biohub/ESMFold2`, `ESMFold2-Fast`, `ESMC-6B`, about
27 GB), which differ from the single revision (`ESMFOLD2_HF_REVISION`) that `vanilla` loads. They are not in the image. On the
first kit-mode call the core downloads the snapshots for the requested variant into `$MODEL_DIR/esmfold2/kit-hf` (on Modal, the
`model_weights` volume) and reuses them afterwards; a failed download raises `OptimizationUnavailableError`. Protenix keeps
downloading its checkpoint into `PROTENIX_ROOT_DIR`, as in vanilla.

**What has been exercised.** The Dockerfiles, routing, builders, the Apptainer interpreter selection and the weight bootstrap
are covered by offline contract and unit tests (`tests/contracts/test_kit_images.py`, the kit tests in
`tests/esmfold2/test_esmfold2.py`). On Modal, through this repository's routing and images built from the Dockerfiles here
(2026-10-03; `PredictionMetadata.optimization` reported the requested and resolved mode, kit config and card each time):

| Model | Mode and card | Cold first fold | Warm fold |
| --- | --- | --- | --- |
| ESMFold2 | `fast` A100, `exact` A100, `fast` H100 | 98 s, 74 s, 166 s | 0.4 s, 0.5 s, 0.4 s (vanilla A100: 1.9 s) |
| Protenix | `fast` A100 and H100, `exact` A100 and H100, `fast` with a template (A100) | 65-208 s | `fast` 4.9 s A100 / 4.3 s H100, `exact` 3.1 s A100 / 2.3 s H100 (vanilla A100: 5.5 s) |
| OpenDDE | `exact` and `fast` A100 | 45 s, 62 s | 4.5 s, 4.3 s (vanilla: 7.0 s, 15 folds) |

The cold fold includes the model load (plus the one-off weight download the first time on a volume). ESMFold2 `fast` on A100
gave ptm 0.173 against vanilla's 0.176 for the same sequence. OpenDDE `exact` and `fast` differ from vanilla by up to several
angstroms of C-alpha RMSD at the same seed on two-chain complexes, but two same-seed runs of vanilla differ by 0.8-2.3 A, so
this is the sampler's run-to-run variation, not evidence the kit is wrong; `exact` is the more repeatable (0.5 A between runs).
Do not rely on bit-identical output to vanilla for OpenDDE.

Not exercised: the Apptainer kit path, `kit_msa` for ESMFold2, a Protenix or OpenDDE kit fold with a caller-supplied MSA, and
OpenDDE on H100. The benchmark numbers below were measured on images assembled from the original recipe. The Protenix
Dockerfile keeps `TORCH_CUDA_ARCH_LIST="9.0+PTX"` from that recipe, whose configured card was the H100; the A100 runs above
completed, but whether stock's fast-LayerNorm extension (built for compute capability 9.0) is used on an A100 or falls back was
not checked.

**ESMFold2 loads offline.** After the weight bootstrap the core sets `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE`, as the kit's
own launcher does. Online, `from_pretrained("biohub/ESMFold2")` would resolve upstream's newest commit, which this kit's
transformers fork cannot parse, and repoint the volume's `refs/main` at it. The kit's pinned `ccd.pkl` is read from the same
snapshot directory.

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
The OpenDDE image sets `LAYERNORM_TYPE=fast_layernorm` but does not install `ninja`, so upstream's fused LayerNorm CUDA
extension cannot be built (the worker logs "Fast LayerNorm CUDA extension is unavailable ... Ninja is required") and all
three modes use torch's `layer_norm`; installing ninja is the first thing to try for a further speed-up and would change
these numbers.

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
