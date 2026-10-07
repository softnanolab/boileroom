# ESMFold2 optimization-kit runtime image: the stack behind `optimization="exact"` and `"fast"`.
#
# The default `boileroom-esmfold2` image (../Dockerfile) cannot run these modes: it carries torch 2.11 and esm
# 3.4.1.post1, the kit needs the stack it was tested on (torch 2.13.0+cu130, esm 3.3.0 and the transformers 4.57.6 fork
# at their pinned commits, and three CUDA extensions compiled from source). `optimization="vanilla"` keeps using the
# stock image; this one is opt-in and only used when a kit mode is requested.
#
# Source: anthropics/uplifting-biomolecular-modeling (Apache-2.0), fetched at KIT_COMMIT. The pinned stack, the install
# recipe and the compile recipe (kit_wheels.sh) follow the kit's own `esmfold2/environment/Dockerfile` at that commit,
# with these changes: the kit comes from git instead of the build context, the optional `_jitcache` COPY (a glob that
# Modal's Dockerfile parser rejects) is gone, the compile toolchain (nvcc 13.0.88, ninja 1.13.0) is pinned to the
# versions the kit's PINS.json records, the compiled wheels must carry machine code for every compute capability of the
# stack (kit_sass.sh), boileroom's runtime dependencies are added (kit_finish.sh), and the build ends with a CPU-only
# smoke check of the compiled kernels (kit_smoke.py).
#
# Build, from the repository root (about half an hour on a 48-64 core builder; the compile is the cost; flash-attn's
# nvcc jobs are capped at one per 9 GB of available memory and at most 14, so memory sets its pace):
#   docker build -f boileroom/models/esmfold2/kit/Dockerfile boileroom/models/esmfold2/kit -t boileroom-esmfold2-kit:local
# Compute capability 9.0 only (H100/H200, a smaller compile): add --build-arg STACK=img_ef2_fa. That image cannot run on
# an A100 (8.0) -- including the A100 GPU class of the Modal wrapper -- so keep the default STACK for any image that may
# be scheduled there; the ESMFold2 core refuses it on an A100 (BOILEROOM_KIT_STACK) before fetching any weight.
# On Modal the wrapper builds this file itself, compiling the extensions on a 48-core step (WHEELS_FROM=defer):
# see docs/optimization.md.
#
# Host: an NVIDIA driver that runs CUDA 13.0 (580 or newer). No CUDA toolkit stays in the image: every CUDA library
# comes from the nvidia-* wheels of the kit's lock. No weights are inside: the first kit run downloads the kit's pinned
# Hugging Face snapshots (~27 GB) into $MODEL_DIR/esmfold2/kit-hf.
FROM python:3.12.10-slim-bookworm

ARG KIT_REPOSITORY=https://github.com/anthropics/uplifting-biomolecular-modeling.git
# Keep in sync with KIT_COMMIT in boileroom/images/metadata.py (tests/contracts/test_kit_images.py checks it).
ARG KIT_COMMIT=f4f62fa6592ae4938d49b1757bea0cfeff9f468e
# img_esmfold2_a100: compute capability 8.0 and 9.0 (A100, H100, H200). img_ef2_fa: 9.0 only (H100, H200).
ARG STACK=img_esmfold2_a100
# build: compile the CUDA extensions in this image build. defer: leave that step and the install to the caller.
ARG WHEELS_FROM=build
ARG BUILD_JOBS=0

# The environment of the tested stack that does not depend on the machine, plus the model directory. HF_HOME is not set
# here: the ESMFold2 core points it at $MODEL_DIR/esmfold2/kit-hf (also under Apptainer, where MODEL_DIR is rebound).
# MODEL_OPT_JIT_ROOT is the kit's compile cache (Triton and torch extension builds on first use).
ENV PYTHONHASHSEED=0 \
    CFLAGS=-g0 \
    NVTE_FRAMEWORK=pytorch \
    XFORMERS_IGNORE_FLASH_VERSION_CHECK=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    MODEL_DIR=/mnt/models \
    MODEL_OPT_JIT_ROOT=/tmp/jit

# The toolchain of the tested image (gcc/g++ 12.2: Triton compiles its launchers at run time) and git.
RUN apt-get update && apt-get install -y gcc gfortran build-essential git && rm -rf /var/lib/apt/lists/* \
 && echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# The kit at the pinned commit: its shared core and the esmfold2 tree, under /kit as its own image has them.
RUN git init -q /tmp/kit-src \
 && git -C /tmp/kit-src fetch -q --depth 1 "$KIT_REPOSITORY" "$KIT_COMMIT" \
 && git -C /tmp/kit-src checkout -q --detach FETCH_HEAD \
 && test "$(git -C /tmp/kit-src rev-parse HEAD)" = "$KIT_COMMIT" \
 && mkdir -p /kit/common \
 && cp -r /tmp/kit-src/common/opt_core /kit/common/opt_core \
 && cp -r /tmp/kit-src/esmfold2 /kit/esmfold2 \
 && rm -rf /tmp/kit-src

# The pinned stack: the lock's own pip first, then every distribution at the lock's version (--no-deps: the lock is
# complete), less the three CUDA extensions that no index carries; kit_wheels.sh builds those.
RUN grep -E '^pip==' /kit/esmfold2/environment/requirements.lock | xargs python -m pip install --no-cache-dir --no-deps \
 && grep -v -E '^(#|(flash_attn|transformer_engine|xformers)==)' /kit/esmfold2/environment/requirements.lock > /tmp/stack.txt \
 && python -m pip install --no-cache-dir --no-deps -r /tmp/stack.txt \
 && rm -f /tmp/stack.txt

WORKDIR /kit/esmfold2

# The compile and the install are separate layers, so that in a `docker build` a change to kit_finish.sh or kit_smoke.py
# does not recompile the extensions. Modal (WHEELS_FROM=defer) builds this whole file as one cached step and runs the two
# scripts as two build steps on top of it (boileroom/images/modal.py): a failed install step is retried on its own, but
# any change to this file or to a file it copies, kit_finish.sh and kit_smoke.py included, recompiles the extensions.
COPY kit_wheels.sh kit_sass.sh /opt/boileroom-kit/
RUN case "$WHEELS_FROM" in \
      build) sh /opt/boileroom-kit/kit_wheels.sh ;; \
      defer) echo "WHEELS_FROM=defer: kit_wheels.sh is run by the caller" ;; \
      *) echo "WHEELS_FROM must be build or defer, got $WHEELS_FROM" >&2; exit 2 ;; \
    esac

COPY kit_finish.sh kit_smoke.py /opt/boileroom-kit/
RUN case "$WHEELS_FROM" in \
      build) sh /opt/boileroom-kit/kit_finish.sh ;; \
      defer) echo "WHEELS_FROM=defer: kit_finish.sh (and its kit_smoke.py check) is run by the caller" ;; \
      *) echo "WHEELS_FROM must be build or defer, got $WHEELS_FROM" >&2; exit 2 ;; \
    esac

# Fail loud at run time: with this set the kit refuses (exit 3, which boileroom raises as OptimizationUnavailableError)
# instead of falling back to its slow attention and MLP paths. The ESMFold2 core sets it as well, so that a kit image
# built without this line cannot fall back either. The stack and commit are recorded in each output's metadata.runtime.
ENV ESMFOLD2_OPT_REQUIRE_FAST_ENV=1 \
    BOILEROOM_KIT_STACK=$STACK \
    BOILEROOM_KIT_COMMIT=$KIT_COMMIT
LABEL org.opencontainers.image.revision=$KIT_COMMIT \
      io.boileroom.kit.stack=$STACK
