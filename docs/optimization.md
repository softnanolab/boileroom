# Kit optimization modes (`optimization`)

ESMFold2 (including ESMFold2-Fast), Protenix and OpenDDE accept a static init option
`optimization = "vanilla" | "exact" | "fast"` (default `"vanilla"`). `exact` and `fast` drive the
prebuilt kernels of [anthropics/uplifting-biomolecular-modeling](https://github.com/anthropics/uplifting-biomolecular-modeling)
(Apache-2.0, kit commit `f4f62fa`).

- `vanilla`: the stock model on the stock image, unchanged.
- `exact`: the kit's kernels that reproduce the kit's own unoptimized route bit for bit, but only under the kit's
  deterministic `--det 1` recipe, which boileroom does not apply. Against boileroom's `vanilla` it is a different stack
  (ESMFold2: esm 3.3.0, the kit's transformers fork and the kit's pinned weight snapshots instead of esm 3.4.1 and
  boileroom's revision; Protenix: fused LayerNorm on torch 2.13/CUDA 13.0 instead of openfold LayerNorm on CUDA 12.6;
  OpenDDE: fused LayerNorm instead of torch's), so expect differences the size of seed-to-seed variation, not identical
  output.
- `fast`: numerically different kernels. Their effect on PAE or ipSAE has not been measured in this repository.

A mode is all of its levers on a GPU class. The mode is resolved against the visible GPU before any
weights load, and a GPU or stack the kit cannot serve fails by name with `OptimizationUnavailableError`.
A kit that refuses later, with its own exit code 3 (`SystemExit(3)` or `os._exit(3)`) or OpenDDE's kernel-census
`KernelsRefused` (`SystemExit(5)`), fails the same way; any other exit is a `RuntimeError` naming the code. On Modal, a load or config failure is raised from the first call rather than
restarting the container.

**Refusal semantics.** A refusal never falls back to slower kernels; a run either uses the mode it asked for or raises.

- A refusal stands for the life of that runtime: the worker is closed, nothing is retried, and every later call raises
  the same `OptimizationUnavailableError`. A `ValueError` from the config is permanent in the same way. Any other load
  failure (for example a failed weight download) is a `RuntimeError` that the next call retries. The one per-call
  refusal is an ESMFold2 kit fold whose lever set settled partial (see `metadata.runtime` below).
- Apptainer: the server loads the model before it accepts requests. A refusal exits with code 3 (`REFUSAL_PROCESS_EXIT_CODE`) and
  the client raises `OptimizationUnavailableError`; any other startup failure exits with 1 and raises `RuntimeError`.
- `boileroom.optimization.is_refusal(error)` tells a refusal from a failure: an exception class named
  `OptimizationUnavailableError`, or a kit's refusal `SystemExit` (code 3 or 5, `KIT_REFUSAL_EXIT_CODES`).

| GPU | class | `exact` / `fast` |
| --- | --- | --- |
| A100 | sm80 | served (kit config `a100`) |
| H100, H200 | sm90 | served (kit config `h100`; Modal may schedule an H100 request onto an H200) |
| L4, L40S | sm89 | refused by name (ESMFold2: kit levers need more shared memory or have no tile table; Protenix: no BLK2 launch cells) |
| other cards (A30, A10G, H800, GH200, MIG slices, ...) | any | refused by name: the kit was validated only on A100, H100 and H200 |

Cards are matched by whole name tokens (`NVIDIA H200` is served, `NVIDIA GH200` is not). Without torch the card is read
with `nvidia-smi`, mapping `CUDA_VISIBLE_DEVICES` (indices or `GPU-` UUIDs) the way CUDA does. On a node that mixes
card models an integer device (a `cuda:N` index or a `CUDA_VISIBLE_DEVICES` index) is refused, because CUDA numbers the
cards fastest first; set `CUDA_DEVICE_ORDER=PCI_BUS_ID` or select the GPU by UUID.

## Requirements

The kit ships its own pinned stack, so `exact` and `fast` need a kit image, not the default boileroom image:

- ESMFold2: python 3.12, torch 2.13+cu130, esm 3.3.0 and the kit's transformers fork (`ESMFold2Model`),
  flash-attn / TransformerEngine / xformers built from source. Kit modes load the fork's model class and fold through
  `ESMFold2InputBuilder.fold()`, which the kit hooks; `vanilla` keeps loading esm's `EsmFold2Model`.
- Protenix: python 3.11, torch 2.13.0+cu130, cuequivariance 0.11.1, protenix 2.0.0, driver 580+.
  The kit is enabled inside the worker before any `protenix` import (the kit refuses late activation).
- OpenDDE: python 3.11, torch 2.7.1+cu126, cuequivariance 0.10.0, opendde 1.1.1, CUDA 12.6 `nvcc` and gcc at run time
  (Triton JIT), libstdc++ from GCC 13 for `exact`, driver 560+. Unlike the other two families it has no separate kit image: the
  stock `boileroom-opendde` image installs the pinned kit commit, and the kit is enabled in the worker before `runner` is imported.

### LayerNorm per mode

Protenix and OpenDDE read `LAYERNORM_TYPE` when they import. The core's worker sets it from the mode, over any value an
image `ENV` or the caller's shell carries, so the mode alone decides it. The values are the cores' constants
`PROTENIX_LAYERNORM` (`boileroom/models/protenix/core.py`) and `OPENDDE_LAYERNORM` (`boileroom/models/opendde/core.py`):

| Family | `vanilla` | `exact` | `fast` |
| --- | --- | --- | --- |
| Protenix | `openfold` | `fast_layernorm` | `fast_layernorm` |
| OpenDDE | `torch` | `fast_layernorm` | `fast_layernorm` |

`fast_layernorm` is upstream's fused LayerNorm CUDA extension, which the kits' LayerNorm lever needs. The Protenix kit image
ships it compiled; OpenDDE builds it at first use with the venv's `ninja` and the CUDA 12.6 compiler. Upstream silently
serves torch LayerNorm when that extension does not load; boileroom refuses the kit mode instead, so a kit run whose
LayerNorm fell back raises rather than returning slower or different numbers. The worker also refuses, in every mode,
a `LAYERNORM_TYPE` other than the table's value (a worker started outside the core). The resolved value is recorded in
`metadata.runtime` (`worker.layernorm_type`).

### JIT compile caches

OpenDDE persists its compile caches next to its weights: the worker gets `MODEL_OPT_JIT_ROOT=$MODEL_DIR/opendde/jit`
(a value already set is kept), and the runtime keys the Triton and torch-extension caches under it by stack, as the
kit's `configs/<gpu>.env` does: `<root>/<stack key>/triton` and `<root>/<stack key>/torch_ext`, with the key from the
kit's `opt_core.jit_cache` (torch, CUDA and compute capability; the launcher's key on a healthy GPU box, and a
directory of that process alone when a part cannot be read). A100 and H100 workers sharing one volume, or a volume
kept across a torch or CUDA bump, therefore never load each other's builds. The key is recorded as
`worker.jit.stack_key` (`absent` without the kit core in the image, which leaves torch's own defaults). The Protenix and
ESMFold2 kit images keep the kit's ephemeral `MODEL_OPT_JIT_ROOT=/tmp/jit`: the Protenix fused LayerNorm is prebuilt
in the image, and a persistent cache was not validated for either, so their kit modes recompile their Triton kernels
on each cold start.

### Kit images

For ESMFold2 and Protenix the repository carries the definition of a separate kit image, and `optimization="vanilla"` never
touches it:

| Family | Dockerfile | Image name | Base |
| --- | --- | --- | --- |
| ESMFold2 | [`boileroom/models/esmfold2/kit/Dockerfile`](../boileroom/models/esmfold2/kit/Dockerfile) (+ `kit_wheels.sh`, `kit_sass.sh`, `kit_finish.sh`, `kit_smoke.py`) | `boileroom-esmfold2-kit` | `python:3.12.10-slim-bookworm` |
| Protenix | [`boileroom/models/protenix/kit/Dockerfile`](../boileroom/models/protenix/kit/Dockerfile) | `boileroom-protenix-kit` | `nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04` |

Both Dockerfiles fetch the kit at the pinned commit `f4f62fa6592ae4938d49b1757bea0cfeff9f468e` and follow the kit's own
`<family>/environment/Dockerfile` at that commit: the same lock, versions, checksums and compile recipe. The changes are the
kit coming from git instead of a build context, the optional `_jitcache` `COPY` removed (a glob that Modal's Dockerfile parser
rejects), the ESMFold2 compile toolchain pinned (nvcc 13.0.88, ninja 1.13.0, the versions the kit's `PINS.json` records),
boileroom's runtime dependencies on top (`fastapi`/`uvicorn` for the Apptainer service), and a CPU-only smoke check at the
end of each build (ESMFold2: `kit_sass.sh` checks with `cuobjdump` that each compiled wheel carries machine code for every
compute capability of `STACK`, then `kit_smoke.py` imports the compiled kernels; Protenix: the kernel packages, the fused
LayerNorm and the kit import, plus an `ldd` check of cuEquivariance's `libcue_ops.so`). The ESMFold2 image also sets
`ESMFOLD2_OPT_REQUIRE_FAST_ENV=1`, so the kit itself refuses a fast path it cannot load. Both record the kit commit
(`BOILEROOM_KIT_COMMIT`) for the runtime's provenance.

**Routing.** The wrapper picks the image from `optimization`: `vanilla` (the default) runs the stock `boileroom-esmfold2` /
`boileroom-protenix` image and Modal class as before, and `exact` / `fast` run `ModalESMFold2Kit` / `ModalProtenixKit` (Modal)
or the kit image (Apptainer). The kit classes live on their own Modal apps (`boileroom-esmfold2-kit`,
`boileroom-protenix-kit`), so using vanilla never builds a kit image. A kit class asks for an A100 (ESMFold2 `A100-80GB`,
Protenix `A100-40GB`); pass `device="H100"` (or `H200`) to run on those.

**Where the image comes from.** The published kit images are pinned by content digest in `KIT_IMAGE_DIGESTS`
(`boileroom/images/metadata.py`), so a re-pushed tag cannot change what a kit mode runs:

| Image | Pinned digest (kit `f4f62fa`) |
| --- | --- |
| `docker.io/jakublala/boileroom-esmfold2-kit` | `sha256:b6111c000142945e83ae2ceaef90d12857a315ed037286d6e3a1ad23d2518e3b` |
| `docker.io/jakublala/boileroom-protenix-kit` | `sha256:779a2040a2ffa3c4c05c801b44a776896073a0633d72a9a578beece694638436` |

These two images were built before the in-image `ESMFOLD2_OPT_REQUIRE_FAST_ENV` and the build-time smoke checks were added
to the Dockerfiles. boileroom's own guards still refuse a run whose fast path is missing; a re-push of the images must
update the digests and repeat the GPU validation of both kit modes.

- Modal, `BOILEROOM_KIT_IMAGE_SOURCE=build` (default): Modal builds the image from the Dockerfile in your installed
  boileroom. The first kit-mode use pays the build and Modal caches it afterwards. Protenix compiles its fused LayerNorm
  extension, which takes minutes. ESMFold2 compiles flash-attn, TransformerEngine and xformers for compute capability 8.0
  and 9.0 on a 48-core, 192 GiB build step with a 4 h limit (1893 s on 64 cores when last measured; budget 30-45 min on
  Modal, more on a smaller builder). The Dockerfile is built with `WHEELS_FROM=defer`, then `kit_wheels.sh` runs as that
  compile step and `kit_finish.sh` (the install and the smoke check) as a second build step on default resources, so a
  Modal build and a local `docker build` end in the same image. A failed finish step is retried without recompiling,
  but on Modal any change to the Dockerfile or its build context (`kit_finish.sh` and `kit_smoke.py` included)
  recompiles the extensions.
- Modal, `BOILEROOM_KIT_IMAGE_SOURCE=registry`: Modal pulls the published image by its pinned digest.
- Apptainer always pulls from the registry, by the pinned digest unless a tag is named.
- A tag replaces the digest: `backend="apptainer:<tag>"` (Apptainer, wins), then `BOILEROOM_KIT_IMAGE_TAG`, then pytest's
  `--kit-image-tag`. `BOILEROOM_IMAGE_TAG` never applies to kit images: their tags are outside the stock images'
  CUDA-qualified release scheme. On Modal a kit tag only takes effect with `BOILEROOM_KIT_IMAGE_SOURCE=registry`
  (`--kit-image-tag` with the `build` source is a usage error). `BOILEROOM_DOCKER_REPOSITORY` sets the repository as for
  the stock images.

This repository's CI does not build the kit images (the ESMFold2 compile takes hours on a GitHub-hosted runner); they are
built by hand. A full GitHub release tags each pinned digest with the stable `X.Y.Z` version (the workflow runs
`scripts/images/promote_image_tags.py --kit-images-only`; a manual promotion tags them too unless `--skip-kit-images`).
A kit tag that already names another digest is refused unless `--force-kit-tags` is passed. The Docker Hub cleanup keeps
every tag that points at any digest ever pinned (`KIT_IMAGE_DIGEST_HISTORY`), so older releases keep pulling theirs.

To build an image yourself on a many-core machine and use it from Modal or Apptainer:

```bash
docker build -f boileroom/models/protenix/kit/Dockerfile boileroom/models/protenix/kit -t <repository>/boileroom-protenix-kit:<tag>
docker build -f boileroom/models/esmfold2/kit/Dockerfile boileroom/models/esmfold2/kit -t <repository>/boileroom-esmfold2-kit:<tag>
docker push <repository>/boileroom-esmfold2-kit:<tag>   # and likewise for protenix

export BOILEROOM_KIT_IMAGE_SOURCE=registry BOILEROOM_DOCKER_REPOSITORY=<repository> BOILEROOM_KIT_IMAGE_TAG=<tag>
```

`--build-arg STACK=img_ef2_fa` compiles ESMFold2 for compute capability 9.0 only (H100/H200). That image cannot run on an
A100, including the A100 GPU class of `ModalESMFold2Kit`, so keep the default `STACK=img_esmfold2_a100` for any image
that may be scheduled there. The image records its stack in `BOILEROOM_KIT_STACK`, and on an A100 the runtime refuses an
`img_ef2_fa` image with `OptimizationUnavailableError` before fetching any weights.

**Build identity.** A Modal-built kit image reports `image_ref` as `build:<dockerfile path>@<hash>`. The 12-hex hash
covers the Dockerfile, every file of its build context, the build arguments, and for ESMFold2 the compile step's
environment, memory request and source, so it changes whenever any input of the build in this repository does.

**ESMFold2 weights.** The kit loads its own pinned Hugging Face snapshots (`biohub/ESMFold2`, `ESMFold2-Fast`, `ESMC-6B`, about
27 GB), which differ from the single revision (`ESMFOLD2_HF_REVISION`) that `vanilla` loads. They are not in the image. On the
first kit-mode call the core downloads the snapshots for the requested variant into `$MODEL_DIR/esmfold2/kit-hf` (on Modal, the
`model_weights` volume) and reuses them afterwards.

- Files already present are not fetched again, but their `refs/main` is re-pointed at the pin and their digests are
  checked. A file off its pin raises `OptimizationUnavailableError`; remove it so the next load fetches it again.
- A failed transfer raises `RuntimeError` and the next load tries again. Files absent or off their pin after the fetch
  raise `OptimizationUnavailableError`.
- The fetch lifts the Hugging Face offline switches for its duration. Afterwards the core sets `HF_HUB_OFFLINE` /
  `TRANSFORMERS_OFFLINE` (in the environment and in the already-imported modules), as the kit's own launcher does.
  Online, `from_pretrained("biohub/ESMFold2")` would resolve upstream's newest commit, which this kit's transformers fork
  cannot parse, and repoint the volume's `refs/main` at it. The kit's pinned `ccd.pkl` is read from the same snapshot
  directory.
- The kit memoizes each file's digest (keyed by path, size, mtime and inode) in `$MODEL_DIR/esmfold2/kit-digests`, so a
  cold start checks the weights without re-hashing about 27 GB. An explicit `ESMFOLD2_OPT_WEIGHTS_MEMO_DIR` wins. A memo
  miss only re-hashes the file.

Protenix keeps downloading its checkpoint into `PROTENIX_ROOT_DIR`, as in vanilla.

**Protenix's fused LayerNorm on A100.** The Protenix kit image keeps the kit's `TORCH_CUDA_ARCH_LIST="9.0+PTX"`, but the
one extension compiled in it ignores that value: Protenix's `torch_ext_compile.py` passes explicit `-gencode` flags for
every architecture among sm_70/80/86/89/90/100 that the image's nvcc lists (CUDA 13.0: sm_80 to sm_100), so the
extension carries native code for both A100 and H100. The image build checks that with `cuobjdump --list-elf` and fails
when the extension lacks sm_80 or sm_90 code. A kit run whose fused LayerNorm does not load is refused (see
[LayerNorm per mode](#layernorm-per-mode)).

**What has been exercised.** The Dockerfiles, routing, builders, the Apptainer interpreter selection, the weight bootstrap
and the refusal paths are covered by offline contract and unit tests (`tests/contracts/test_kit_images.py`,
`tests/contracts/test_optimization.py`, the kit tests in `tests/esmfold2/test_esmfold2.py`). On Modal, single smoke folds
went through this repository's routing on images built from these Dockerfiles before this round of fixes (2026-10-03):
ESMFold2 `exact` and `fast` on A100 and `fast` on H100; Protenix `exact` and `fast` on A100 and H100, and `fast` with a
template on A100; OpenDDE `exact` and `fast` on A100. `PredictionMetadata.optimization` reported the mode, kit config and
card each time. Their timings were single runs and are not benchmark data.

Not exercised: the Apptainer kit path, `kit_msa` for ESMFold2, a Protenix or OpenDDE kit fold with a caller-supplied MSA,
OpenDDE kit modes on H100 with the current OpenDDE image, and any kit mode with the code of this round of fixes on a GPU.

### Memory across calls

The kit's native TriMul adapter keeps the packed weights of the last two input lengths per device and evicts older
ones, but the finalizer it registers on each model weight still holds every evicted cache until the model is freed. A
runtime that folds many different lengths then grows by about 0.3 GB per new length until it runs out of memory. This
was measured with ESMFold2 `fast` on A100-80GB at kit commit `f4f62fa`: allocated memory rose from 15.4 to 19.3 GB
over 16 folds cycling four lengths, while `exact` and `vanilla` stayed flat. `fast` uses the native adapter on A100 at
every length and on H100 only at the lengths it serves natively; OpenDDE `fast` loads the same adapter and was not
measured. boileroom frees the evicted caches after every fold (`boileroom.models._worker.release_evicted_kit_caches`,
called by the ESMFold2 core and by every worker child), which kept the same 16 folds flat. The release reads the
adapter's private state, so re-check it when the kit commit changes; the upstream fix is for the kit to hold the cache
weakly or to detach its finalizers on eviction. `gpu.mem.used_mib` in each output shows whether a runtime's memory
grows.

## What a prediction records

`PredictionMetadata.optimization` records `mode`, `kit_config` (`a100`, `h100` or `null` for vanilla), `gpu_name` and
`capability` (e.g. `sm80`; vanilla records the card too when it runs on one). A mode that cannot be served raises
instead of falling back, so `mode` is both what was asked for and what ran. Models without a kit (everything but
ESMFold2, Protenix and OpenDDE) accept only `optimization="vanilla"` and refuse `exact` / `fast` up front.

`PredictionMetadata.runtime` is a flat `dict[str, str]` describing the code, image, stack and GPU that produced the
prediction (`boileroom.provenance.runtime_provenance`). Every model that fills it starts with:

| Key | Value |
| --- | --- |
| `boileroom`, `python` | versions of the process that ran the core |
| `image_ref` | the image's own reference, from `BOILEROOM_IMAGE_REF`: a registry reference (Apptainer adds the digest the `.sif` was pulled at), `build:<dockerfile>@<hash>` for a Modal-built kit image, or `unknown` |
| `torch`, `cuda` | the loaded torch and its CUDA version (`not-loaded` if torch is not imported, `cuda` is `none` for a CPU build) |
| `gpu`, `gpu_capability` | the card and its class (`none` without a GPU) |
| one key per recorded package | its installed version, or `absent` |

In `exact` and `fast` every kit family (ESMFold2, Protenix, OpenDDE) also records the same kit entries, under the same
names and in the same format (`boileroom.provenance.kit_provenance`); `vanilla` records none of them:

| Key | Value |
| --- | --- |
| `kit.commit` | the kit's source commit (the kit images and the OpenDDE image set `BOILEROOM_KIT_COMMIT`; no `.git` is read), or `unknown` |
| `kit.levers_applied` | the levers that served, comma-joined (`none` when empty) |
| `kit.levers_fallback` | the levers that fell back to the stock path, comma-joined (`none` when empty) |
| `kit.partial` | `true` or `false`: whether the kit reported a partial lever set (refused, so `false` on any output) |

The lever entries are the kit's report as settled after the fold or prediction, not the load-time snapshot.

Then each family adds its own entries:

- ESMFold2 `vanilla`: `weights` (`<model_name>@<revision>`) and esm's import-time kernel switches as `esm.<FLAG>`
  (`esm.FLASH_ATTN_AVAILABLE`, `esm.CUE_AVAILABLE`, `esm.TRITON_KERNELS_AVAILABLE`, `esm.TE_AVAILABLE` and the ESM-C
  kernel flags), each `True`, `False`, `not-loaded` or `absent`. They are recorded, not enforced: vanilla runs whatever
  the image provides.
- ESMFold2 kit modes: `kit.stack`, `kit.package`, `kit.variant`, `kit.weights` (each repo at its snapshot commit),
  `kit.require_fast_env`, `kit.weights_memo`, `kit.metadata_words`, `kit.attn` (the kit's whole word line) and one
  entry per kernel word, `kit.attn.atom_attn`, `kit.attn.esmc_mlp`, `kit.attn.esmc_attn`, `kit.attn.esmc_rope`
  (`unknown` when the kit did not report it), and `kit.levers_gated`, `kit.gated`, `kit.guards` (each data-dependent
  guard's verdict as `<lever>=<kind>: <note>`), `kit.scope` (`none` when empty). The tally is settled after every fold,
  as the kit's CLI does, and a partial lever set is refused. The kernel words are also read again on the model after
  every fold, because the kit's `status()` repeats the words it read at the load: a path switched off since then is
  refused too, so the `kit.attn.*` entries hold for every fold they are reported on. That refusal stands for the
  runtime, and later folds raise it before reaching the GPU. A partial lever set depends on the call (the kit scopes
  levers by sample count), so it refuses that fold only. One exception: when no Transition
  call was served by the t16 kernel but every call reached it and stepped aside by name to the previous statements
  (same values, slower), `t16` is recorded in `kit.levers_gated` instead of being refused; with no call at all it is
  still refused.
- Protenix and OpenDDE: `worker.<key>` for what the worker reported when it loaded (its interpreter, `layernorm_type`,
  requested and resolved triangle kernels, and in kit modes the kit's full activation report as `worker.kit.<field>`)
  and `predict.<key>` for what this request reported (resolved kernels, templates, the settled kit report as
  `predict.kit.<field>`, and for OpenDDE kit modes `kit.lnstream` and `kit.kernels`). Inside these prefixed entries a
  missing fact uses the same words as above (`not-loaded`, `absent`, `none`, `unknown`; an empty list is `none`).
  OpenDDE also records `worker.kernel.cc7_fallback` (`true` when `vanilla` ran upstream's compute-capability-7.x
  fallback) and `worker.jit.stack_key`.

Every output a backend returns, for every model, also records the GPU memory in use right after the call
(`boileroom.provenance.record_gpu_memory`; the Modal and Apptainer servers add it, a core called directly does not):

| Key | Value |
| --- | --- |
| `gpu.mem.used_mib`, `gpu.mem.total_mib` | the device's used and total memory in MiB, every process on it counted (a model's worker child too) |
| `gpu.mem.allocated_mib`, `gpu.mem.reserved_mib` | the serving process's live tensors and all the memory its torch caching allocator holds, in MiB; only when that process runs CUDA itself, so not for AlphaFold2-Multimer, Protenix and OpenDDE, which predict in a worker child |

The entries are missing when no GPU memory can be read (a CPU run, no `nvidia-smi`). Within one runtime, calls of a
repeating input size should leave `gpu.mem.used_mib` flat after the first few, since torch reuses the memory it
reserved; growth from call to call is a leak (see [Memory across calls](#memory-across-calls)).

**Spotting a degraded run.** A kit mode never degrades silently: a missing kernel, a partial lever set or a fallen-back
LayerNorm raises `OptimizationUnavailableError`. `vanilla` is not guarded, because it runs whatever the image provides,
so compare `runtime` between runs before comparing their scores. In particular an ESMFold2 `vanilla` run with
`esm.FLASH_ATTN_AVAILABLE` other than `True`, or an `image_ref` you do not expect, is not comparable with a run that had
flash-attn. An ESMFold2 image without flash-attn once dropped the model's sliding-window attention and lowered the
p53/MDM2 ipSAE from about 0.39 to 0.29.

## Measured

Provenance: `summary.json` records only these cells, measured on Modal through `ESMFold2Core`, `ProtenixCore` and
`OpenDDECore` with 3 complexes x 5 seeds and no MSA. ESMFold2 and Protenix used the bakeoff's sampler settings; OpenDDE
used boileroom's defaults (10 cycles, 200 steps, 5 samples) on A100-80GB SXM4 and H100, so its seconds are not comparable
with the Protenix rows. The OpenDDE cells ran on the image `cuda12.6-sha-bdc9a27`, whose OpenDDE Dockerfile differs from
the current one; the image of the other cells is not recorded. All cells were measured with kit commit `f4f62fa` and
predate this round of fixes, and the code that ran them did not refuse a partial lever set. Which levers and which
LayerNorm served the OpenDDE kit rows was not recorded, so treat their lever set as unverified. Prices are Modal list
prices per second of the card.

Warm is the median wall time of a fold after the first in a process (it includes pre- and post-processing); first is
the first fold of a process (model load, plus Triton JIT and CUDA graph capture in the kit modes); cost is warm times
price.

| Model | GPU | Mode | Warm (s) | First (s) | Cost per warm fold (USD) |
| --- | --- | --- | --- | --- | --- |
| ESMFold2 | L4 | vanilla | 4.40 | 7.3 | 0.00098 |
| ESMFold2 | A100 | exact | 0.79 | 12.6 | 0.00055 |
| ESMFold2 | A100 | fast | 0.47 | 27.2 | 0.00032 |
| ESMFold2 | H100 | exact | 0.48 | 9.0 | 0.00053 |
| ESMFold2 | H100 | fast | 0.26 | 26.3 | 0.00029 |
| ESMFold2 | H200 | exact | 0.47 | 9.9 | 0.00059 |
| ESMFold2 | H200 | fast | 0.23 | 21.8 | 0.00028 |
| ESMFold2-Fast | L4 | vanilla | 2.51 | 4.6 | 0.00056 |
| ESMFold2-Fast | A100 | exact | 0.62 | 9.3 | 0.00043 |
| ESMFold2-Fast | A100 | fast | 0.33 | 21.6 | 0.00023 |
| ESMFold2-Fast | H100 | exact | 0.39 | 8.0 | 0.00043 |
| ESMFold2-Fast | H100 | fast | 0.19 | 18.0 | 0.00021 |
| Protenix v2 | L40S | vanilla | 12.19 | 34.5 | 0.0066 |
| Protenix v2 | A100 | exact | 3.61 | 32.2 | 0.0025 |
| Protenix v2 | A100 | fast | 3.48 | 27.8 | 0.0024 |
| Protenix v2 | H100 | exact | 1.92 | 22.6 | 0.0021 |
| Protenix v2 | H100 | fast | 1.36 | 41.1 | 0.0015 |
| OpenDDE | A100 | vanilla | 16.07 | 71.9 | 0.011 |
| OpenDDE | A100 | exact | 12.71 | 77.5 | 0.0088 |
| OpenDDE | A100 | fast | 9.28 | 78.2 | 0.0064 |
| OpenDDE | H100 | vanilla | 14.05 | 61.0 | 0.015 |
| OpenDDE | H100 | exact | 11.43 | 64.7 | 0.013 |
| OpenDDE | H100 | fast | 5.81 | 65.6 | 0.0064 |

What the table supports:

- ESMFold2, ESMFold2-Fast and Protenix were measured against the eval GPU running `vanilla` (L4, L4, L40S); there is no
  same-GPU `vanilla` cell for them. Against those, warm cost per fold drops 1.8x (A100 `exact`), 1.9x (H100 `exact`),
  3.0x (A100 `fast`) and 3.4x (H100 `fast`) for ESMFold2; 1.3x (`exact`) and 2.5-2.7x (`fast`) for ESMFold2-Fast; and
  2.6x (A100 `exact`), 3.1x (H100 `exact`), 2.7x (A100 `fast`) and 4.4x (H100 `fast`) for Protenix.
- OpenDDE is the only same-GPU comparison: warm speedup over `vanilla` is 1.26x (`exact`) and 1.73x (`fast`) on A100, and
  1.23x and 2.42x on H100. The kit modes add 3.7-6.3 s to the first fold.
- The first fold is paid once per process, so the kit lowers cost only for containers that stay warm. Including the
  first fold on both sides, `fast` on A100/H100 breaks even with the eval GPU after about 28-41 folds per process for
  ESMFold2, 43-55 for ESMFold2-Fast and 2-7 for Protenix (no persistent `MODEL_OPT_JIT_ROOT`; a persistent cache was not
  measured).
- Same-seed structures from different modes, and from repeated runs of one mode, are not identical, because boileroom
  does not enable the runner's deterministic mode. Treat every mode as a stochastic sampler.

![Average cost per fold vs folds per process](optimization-bench/cost_vs_folds.png)

The figure plots the ESMFold2, ESMFold2-Fast and Protenix cells only (eval-GPU `vanilla` against kit `exact` and
`fast`); it predates the OpenDDE cells.

Raw numbers: [`optimization-bench/summary.json`](optimization-bench/summary.json).
