# Protenix optimization-kit runtime image: the stack behind `optimization="exact"` and `"fast"`.
#
# The default `boileroom-protenix` image (../Dockerfile) cannot run these modes: it is the stock stack on CUDA 12.6, the
# kit needs the stack it was tested on (Python 3.11.5, torch 2.13.0+cu130, cuEquivariance 0.11.1, a CUDA 13.0 toolkit with
# nvcc for the compiles at build and first use). `optimization="vanilla"` keeps using the stock image; this one is opt-in
# and only used when a kit mode is requested.
#
# Source: anthropics/uplifting-biomolecular-modeling (Apache-2.0), fetched at KIT_COMMIT. Apart from the two changes below
# this is the kit's own `protenix_v2/environment/Dockerfile` at that commit: the kit comes from git instead of the build
# context, and the optional `_jitcache` COPY (a glob that Modal's Dockerfile parser rejects) is gone. Added on top:
# boileroom's runtime dependencies and the model directory.
#
# Build, from the repository root (compiles stock Protenix's fast-LayerNorm extension; minutes on a many-core builder):
#   docker build -f boileroom/models/protenix/kit/Dockerfile boileroom/models/protenix/kit -t boileroom-protenix-kit:local
# On Modal the wrapper builds this file itself: see docs/optimization.md.
#
# Python here is 3.11 at /usr/local/bin/python, not boileroom's usual 3.12. Host: an NVIDIA driver that runs CUDA 13.0
# (580 or newer). No weights are inside: Protenix downloads its checkpoint and data caches into PROTENIX_ROOT_DIR, which
# boileroom points at the model directory.
FROM nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04

ARG KIT_REPOSITORY=https://github.com/anthropics/uplifting-biomolecular-modeling.git
# Keep in sync with KIT_COMMIT in boileroom/images/metadata.py (tests/contracts/test_kit_images.py checks it).
ARG KIT_COMMIT=f4f62fa6592ae4938d49b1757bea0cfeff9f468e

# The process environment of the tested stack: hash seed fixed, no debug info from the C compiler, nvcc found through
# CUDA_HOME by torch's extension builder (which compiles at run time for compute capability 9.0 plus PTX, up to 32 jobs),
# the loader path every tested run had, and stock's fast LayerNorm (which the kit's layernorm lever requires; the vanilla
# image uses openfold). MODEL_OPT_JIT_ROOT is the kit's compile cache.
ENV PYTHONHASHSEED=0 \
    CFLAGS=-g0 \
    CUDA_HOME=/usr/local/cuda \
    LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64 \
    TORCH_CUDA_ARCH_LIST="9.0+PTX" \
    MAX_JOBS=32 \
    LAYERNORM_TYPE=fast_layernorm \
    MODEL_DIR=/mnt/models \
    MODEL_OPT_JIT_ROOT=/tmp/jit

# The C/C++ toolchain (gcc 13.3, make, ninja: torch's and Triton's run-time builds), git, the download tool of this build,
# and kalign with its stdin-closing entry ahead of /usr/bin on PATH (template realignment waits on an open stdin otherwise).
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates wget git build-essential ninja-build kalign && rm -rf /var/lib/apt/lists/* \
 && printf '#!/bin/sh\nexec /usr/bin/kalign "$@" </dev/null\n' > /usr/local/bin/kalign && chmod 755 /usr/local/bin/kalign

# Python 3.11.5: the python-build-standalone release the stack was assembled with, unpacked into /usr/local with its
# links made plain files (no system Python is used).
RUN wget -q -O /tmp/python.tar.gz "https://github.com/indygreg/python-build-standalone/releases/download/20230826/cpython-3.11.5+20230826-x86_64-unknown-linux-gnu-install_only.tar.gz" \
 && echo "fbed6f7694b2faae5d7c401a856219c945397f772eea5ca50c6eb825cbc9d1e1  /tmp/python.tar.gz" | sha256sum -c - \
 && tar -xzf /tmp/python.tar.gz -C /tmp && cp -rL /tmp/python/. /usr/local/ && rm -rf /tmp/python /tmp/python.tar.gz \
 && ln -sf /usr/local/bin/python3 /usr/local/bin/python && python -V

# The kit at the pinned commit: its shared core and the protenix_v2 tree, under /kit as its own image has them.
RUN git init -q /tmp/kit-src \
 && git -C /tmp/kit-src fetch -q --depth 1 "$KIT_REPOSITORY" "$KIT_COMMIT" \
 && git -C /tmp/kit-src checkout -q --detach FETCH_HEAD \
 && test "$(git -C /tmp/kit-src rev-parse HEAD)" = "$KIT_COMMIT" \
 && mkdir -p /kit/common \
 && cp -r /tmp/kit-src/common/opt_core /kit/common/opt_core \
 && cp -r /tmp/kit-src/protenix_v2 /kit/protenix_v2 \
 && rm -rf /tmp/kit-src

# The pinned stack: the lock's own pip first, then every line of the lock with --no-deps (the lock is complete), then
# stock Protenix from the bundled wheel (--no-deps: its declared torch / triton / cuequivariance pins do not hold on this
# stack, so `pip check` reports those lines and nothing else), then stock's fast-LayerNorm CUDA extension compiled the
# way stock compiles it at first import, so the image carries it built. BUILD_JOBS = parallel compile jobs; 0 = every core.
ARG BUILD_JOBS=0
RUN J=${BUILD_JOBS:-0}; [ "$J" -gt 0 ] || J=$(nproc --all); export MAX_JOBS=$J CMAKE_BUILD_PARALLEL_LEVEL=$J MAKEFLAGS=-j$J NVCC_THREADS=4 \
 && echo "build jobs: $J (cores: $(nproc --all))" \
 && grep -E '^pip==' /kit/protenix_v2/environment/requirements.lock | xargs python -m pip install --no-cache-dir \
 && grep -v -E '^(#|protenix==)' /kit/protenix_v2/environment/requirements.lock > /tmp/stack.txt \
 && python -m pip install --no-cache-dir --no-deps -r /tmp/stack.txt \
 && python -m pip install --no-cache-dir --no-deps /kit/protenix_v2/stock/protenix-2.0.0-py3-none-any.whl \
 && rm -f /tmp/stack.txt \
 && python -c "import protenix.model.layer_norm"

# The kit: the shared core, then protenix_opt, both editable, and the install checked against the pins. fastapi and
# uvicorn are what boileroom's Apptainer service imports on top of the kit's lock (numpy, biotite and pydantic are in it);
# the pin check runs again afterwards so that nothing the kit pinned moved.
WORKDIR /kit/protenix_v2
RUN bash run.sh install \
 && python -m pip install --no-cache-dir fastapi==0.136.3 uvicorn==0.48.0 \
 && python -I stock/check_pins.py \
 && mkdir -p /opt/jit_cache && chmod -R a+rX,u+w /opt/jit_cache
