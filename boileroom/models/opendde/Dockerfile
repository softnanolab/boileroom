# syntax=docker/dockerfile:1.7
ARG TARGETPLATFORM
ARG TARGETARCH
ARG TARGETOS
ARG BUILDPLATFORM
ARG BASE_IMAGE=docker.io/jakublala/boileroom-base:local

# libstdc++ from GCC 13+: the kit's bit-exact ``exact`` kernels are linked against GLIBCXX_3.4.32. Take it from
# conda-forge, which builds against an old glibc: Ubuntu 24.04's copy needs glibc 2.38 and cannot load on the
# Debian bookworm (glibc 2.36) base. micromamba and both packages are pinned to exact builds and checked by sha256.
FROM debian:bookworm-slim AS libstdcxx
ARG MICROMAMBA_URL=https://conda.anaconda.org/conda-forge/linux-64/micromamba-2.9.0-0.tar.bz2
ARG MICROMAMBA_SHA256=8761c382127e6363bd9e0a2451aa3ef90d071a79133f736e2f759a3bf13040dd
RUN apt-get update --quiet \
    && apt-get install --no-install-recommends --yes --quiet curl bzip2 ca-certificates \
    && curl -fsSL -o /tmp/micromamba.tar.bz2 "${MICROMAMBA_URL}" \
    && echo "${MICROMAMBA_SHA256}  /tmp/micromamba.tar.bz2" | sha256sum -c - \
    && tar -xjf /tmp/micromamba.tar.bz2 -C /usr/local bin/micromamba \
    && MAMBA_ROOT_PREFIX=/tmp/mamba micromamba create --yes --quiet --prefix /tmp/cf --channel conda-forge \
        --override-channels "libstdcxx=13.4.0=h934c35e_20" "libgcc=13.4.0=he0feb66_20" \
    && cd "$(dirname "$(find /tmp/mamba/pkgs -name libstdcxx-13.4.0-h934c35e_20.conda)")" \
    && printf '%s  %s\n' b16c1504aeeaad7a583c010e300c061faa07cce65f5137583c81e4e611e60e24 libstdcxx-13.4.0-h934c35e_20.conda \
        3aea71288652e300feae0b95131468079a6c59a417c2a947e1526e6a265e2414 libgcc-13.4.0-he0feb66_20.conda | sha256sum -c - \
    && mkdir -p /out \
    && cp -aL /tmp/cf/lib/libstdc++.so.6 /tmp/cf/lib/libgcc_s.so.1 /out/

FROM ${BASE_IMAGE}

COPY requirements.txt /tmp/requirements.txt

ARG UV_CACHE_ID=boileroom-uv

# boileroom runtime postprocessing dependencies live in the Python 3.12 system interpreter.
RUN --mount=type=cache,id=${UV_CACHE_ID},target=/root/.cache/uv \
    uv pip install --system -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# OpenDDE 1.1.1 and its Anthropic kit (uplifting-biomolecular-modeling) pin Python 3.11, torch 2.7.1+cu126,
# triton 3.3.1 and cuequivariance 0.10.0, which cannot share the boileroom interpreter. They live in an
# isolated interpreter under /opt/opendde; the core runs a persistent worker with it and never imports OpenDDE
# into Python 3.12. The worker needs the CUDA 12.6 compiler (upstream's fused LayerNorm extension and Triton
# JIT), gcc (build-essential in the base), hmmer and kalign (upstream's search and template binaries).
# This mirrors the kit's own opendde/environment/Dockerfile at KIT_COMMIT (Ubuntu 24.04, CUDA 12.6.3 devel) on
# boileroom's Debian bookworm base, with what the kit pins pinned the same: kalign 3.3.5, the CUDA 12.6 compiler,
# Python 3.11.5 (python-build-standalone, by sha256), the lock, the stock wheel, `run.sh install` and its pin
# check, and the kit's process environment. Not the same as the kit: gcc 12.2 / glibc 2.36 (Triton's launcher
# builds; the kit has gcc 13.2), hmmer 3.3.2 (the kit has Ubuntu 24.04's 3.4) and GCC 13's libstdc++ from
# conda-forge instead of the system one.
ARG CUDA_KEYRING=https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb
ARG CUDA_KEYRING_SHA256=e7f219eab6fe4819cdb5c15b98233dc3420302d9c00883219cd3d896857cf48d
RUN apt-get update --quiet \
    && apt-get install --no-install-recommends --yes --quiet hmmer kalign=1:3.3.5-1 git curl libglib2.0-0 libgl1 libxrender1 \
    && curl -fsSL -o /tmp/cuda-keyring.deb "${CUDA_KEYRING}" \
    && echo "${CUDA_KEYRING_SHA256}  /tmp/cuda-keyring.deb" | sha256sum -c - \
    && dpkg -i /tmp/cuda-keyring.deb && rm /tmp/cuda-keyring.deb \
    && apt-get update --quiet \
    && apt-get install --no-install-recommends --yes --quiet cuda-nvcc-12-6=12.6.85-1 cuda-cudart-dev-12-6=12.6.77-1 \
    && rm -rf /var/lib/apt/lists/*

ARG KIT_REPOSITORY=https://github.com/anthropics/uplifting-biomolecular-modeling.git
# Keep in sync with KIT_COMMIT in boileroom/images/metadata.py (tests/contracts/test_kit_images.py checks it).
ARG KIT_COMMIT=f4f62fa6592ae4938d49b1757bea0cfeff9f468e
ARG OPENDDE_HOME=/opt/opendde
ARG PYTHON_URL=https://github.com/indygreg/python-build-standalone/releases/download/20230826/cpython-3.11.5+20230826-x86_64-unknown-linux-gnu-install_only.tar.gz
ARG PYTHON_SHA256=fbed6f7694b2faae5d7c401a856219c945397f772eea5ca50c6eb825cbc9d1e1
ENV OPENDDE_HOME=${OPENDDE_HOME}
# The kit at the pinned commit (one shallow fetch, verified), Python 3.11.5 as the kit's image has it (links
# written out as files) at OPENDDE_HOME, the lock with --no-deps (it is complete), the stock wheel, then the
# kit's own install step: `run.sh install` installs opt_core and opendde_opt editable into the python on PATH
# and refuses (exit 3) unless stock/check_pins.py finds the pinned wheel's bytes installed.
RUN --mount=type=cache,id=${UV_CACHE_ID}-pip,target=/root/.cache/pip \
    git init --quiet /kit \
    && git -C /kit fetch --quiet --depth 1 "${KIT_REPOSITORY}" "${KIT_COMMIT}" \
    && git -C /kit checkout --quiet --detach FETCH_HEAD \
    && test "$(git -C /kit rev-parse HEAD)" = "${KIT_COMMIT}" \
    && curl -fsSL -o /tmp/python.tar.gz "${PYTHON_URL}" \
    && echo "${PYTHON_SHA256}  /tmp/python.tar.gz" | sha256sum -c - \
    && tar -xzf /tmp/python.tar.gz -C /tmp && mkdir -p "${OPENDDE_HOME}" \
    && cp -aL /tmp/python/. "${OPENDDE_HOME}/" && rm -rf /tmp/python /tmp/python.tar.gz \
    && ln -sf python3 "${OPENDDE_HOME}/bin/python" \
    && "${OPENDDE_HOME}/bin/python" -c "import sys; assert sys.version_info[:3] == (3, 11, 5), sys.version" \
    && grep -v -E '^(#|opendde==)' /kit/opendde/environment/requirements.lock > /tmp/stack.txt \
    && "${OPENDDE_HOME}/bin/python" -m pip install --no-deps -r /tmp/stack.txt \
    && "${OPENDDE_HOME}/bin/python" -m pip install --no-deps /kit/opendde/stock/opendde-1.1.1-py3-none-any.whl \
    && rm /tmp/stack.txt \
    && cd /kit/opendde && PATH="${OPENDDE_HOME}/bin:${PATH}" bash run.sh install

# Not on the image-wide LD_LIBRARY_PATH: that would load it into the system Python 3.12 too. The core adds
# ``${OPENDDE_HOME}/lib`` to the worker environment only.
COPY --from=libstdcxx /out/ /opt/opendde/lib/

# Kernel smoke, CPU-safe (a builder has no GPU and no libcuda.so.1): the kernel packages, the kit with its LayerNorm
# levers and the runner import, GCC 13's libstdc++ carries GLIBCXX_3.4.32, and cuEquivariance's kernel library
# libcue_ops.so is present with every dependency but the driver's libcuda.so.1 resolved (against the torch and
# nvidia-* wheel libraries torch loads at run time). cuequivariance_ops_torch itself is not imported: it loads
# libcue_ops.so, which needs libcuda.so.1.
RUN export LD_LIBRARY_PATH="${OPENDDE_HOME}/lib" \
    && "${OPENDDE_HOME}/bin/python" -I -c "import torch, cuequivariance_torch, opendde_opt, opendde_opt.lnstream, opendde_opt.lncensus, runner.batch_inference" \
    && grep -q GLIBCXX_3.4.32 "${OPENDDE_HOME}/lib/libstdc++.so.6" \
    && "${OPENDDE_HOME}/bin/python" -I -c "import importlib.util as u, os, pathlib as p, subprocess as s, sys; \
loc = lambda n: list(u.find_spec(n).submodule_search_locations) if u.find_spec(n) else []; \
libs = sorted(f for d in loc('cuequivariance_ops') for f in p.Path(d).rglob('libcue_ops.so')); \
dirs = [str(d) for n, g in (('torch', 'lib'), ('nvidia', '*/lib')) for r in loc(n) for d in sorted(p.Path(r).glob(g))]; \
run = lambda f: s.run(['ldd', str(f)], capture_output=True, text=True, env=dict(os.environ, LD_LIBRARY_PATH=':'.join([str(f.parent), *dirs, os.environ.get('LD_LIBRARY_PATH', '')]))); \
bad = {str(f): sorted({l.split()[0] for l in r.stdout.splitlines() if 'not found' in l} - {'libcuda.so.1'}) or r.returncode for f in libs for r in [run(f)]}; \
print('libcue_ops.so:', bad or 'missing'); \
sys.exit(0 if libs and not any(bad.values()) else 1)" \
    && python -c "import numpy"

# Neither the kit's process environment (hash seed, unbuffered output, no bytecode, CFLAGS=-g0) nor LAYERNORM_TYPE
# is set image-wide: the core's worker environment sets them (OPENDDE_PROCESS_ENV, and OPENDDE_LAYERNORM per mode:
# torch for vanilla, upstream's fused LayerNorm for the kit modes), so boileroom's Python 3.12 keeps its defaults.
# BOILEROOM_KIT_COMMIT names the kit for the runtime's provenance (the runtime reads no .git).
ENV CUDA_HOME=/usr/local/cuda-12.6 \
    PATH=/usr/local/cuda-12.6/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/nvidia/lib \
    MODEL_OPT=/kit/opendde \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    DISABLE_PANDERA_IMPORT_WARNING=True \
    BOILEROOM_KIT_COMMIT=${KIT_COMMIT}
