# ESMFold2 optimization-kit runtime image: the stack behind `optimization="exact"` and `"fast"`.
#
# The default `boileroom-esmfold2` image (../Dockerfile) cannot run these modes: it carries torch 2.11 and esm
# 3.4.1.post1, the kit needs the stack it was tested on (torch 2.13.0+cu130, esm 3.3.0 and the transformers 4.57.6 fork
# at their pinned commits, and three CUDA extensions compiled from source). `optimization="vanilla"` keeps using the
# stock image; this one is opt-in and only used when a kit mode is requested.
#
# Source: anthropics/uplifting-biomolecular-modeling (Apache-2.0), fetched at KIT_COMMIT. The pinned stack, the install
# recipe and the compile recipe (kit_wheels.sh) are the kit's own `esmfold2/environment/Dockerfile` at that commit, with
# three changes: the kit comes from git instead of the build context, the optional `_jitcache` COPY (a glob that Modal's
# Dockerfile parser rejects) is gone, and boileroom's runtime dependencies are added (kit_finish.sh).
#
# Build, from the repository root (about half an hour on a 48-64 core builder; the compile is the cost):
#   docker build -f boileroom/models/esmfold2/kit/Dockerfile boileroom/models/esmfold2/kit -t boileroom-esmfold2-kit:local
# H100/H200 only (compute capability 9.0, a smaller compile): add --build-arg STACK=img_ef2_fa
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

COPY kit_wheels.sh kit_finish.sh /opt/boileroom-kit/
WORKDIR /kit/esmfold2

RUN case "$WHEELS_FROM" in \
      build) sh /opt/boileroom-kit/kit_wheels.sh && sh /opt/boileroom-kit/kit_finish.sh ;; \
      defer) echo "WHEELS_FROM=defer: kit_wheels.sh and kit_finish.sh are run by the caller" ;; \
      *) echo "WHEELS_FROM must be build or defer, got $WHEELS_FROM" >&2; exit 2 ;; \
    esac
