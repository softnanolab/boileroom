# syntax=docker/dockerfile:1.7
ARG TARGETPLATFORM
ARG TARGETARCH
ARG TARGETOS
ARG BUILDPLATFORM

# libstdc++ from GCC 13: the kit's bit-exact ``exact`` kernels are linked against GLIBCXX_3.4.32.
FROM ubuntu:24.04 AS libstdcxx
RUN mkdir -p /out && cp -aL /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /out/

ARG BASE_IMAGE=docker.io/jakublala/boileroom-base:local
FROM ${BASE_IMAGE}

COPY requirements.txt /tmp/requirements.txt

ARG UV_CACHE_ID=boileroom-uv

# boileroom runtime postprocessing dependencies live in the Python 3.12 system interpreter.
RUN --mount=type=cache,id=${UV_CACHE_ID},target=/root/.cache/uv \
    uv pip install --system -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# OpenDDE 1.1.1 and its Anthropic kit (uplifting-biomolecular-modeling) pin Python 3.11, torch 2.7.1+cu126,
# triton 3.3.1 and cuequivariance 0.10.0, which cannot share the boileroom interpreter. They live in an
# isolated virtualenv; the core runs a persistent worker with that interpreter and never imports OpenDDE
# into Python 3.12. Runtime needs the CUDA 12.6 compiler (upstream's fused LayerNorm extension and Triton
# JIT), gcc (build-essential in the base), hmmer and kalign (upstream's search and template binaries).
ARG CUDA_KEYRING=https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb
RUN apt-get update --quiet \
    && apt-get install --no-install-recommends --yes --quiet hmmer kalign git curl libglib2.0-0 libgl1 libxrender1 \
    && curl -fsSL -o /tmp/cuda-keyring.deb "${CUDA_KEYRING}" \
    && dpkg -i /tmp/cuda-keyring.deb && rm /tmp/cuda-keyring.deb \
    && apt-get update --quiet \
    && apt-get install --no-install-recommends --yes --quiet cuda-nvcc-12-6 cuda-cudart-dev-12-6 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=libstdcxx /out/libstdc++.so.6 /opt/opendde/lib/libstdc++.so.6

ARG KIT_REF=f4f62fa
ARG OPENDDE_HOME=/opt/opendde
ENV OPENDDE_HOME=${OPENDDE_HOME}
RUN --mount=type=cache,id=${UV_CACHE_ID}-pip,target=/root/.cache/pip \
    git clone --quiet https://github.com/anthropics/uplifting-biomolecular-modeling /kit \
    && git -C /kit checkout --quiet ${KIT_REF} \
    && UV_SYSTEM_PYTHON=0 uv venv --python 3.11 --seed ${OPENDDE_HOME} \
    && grep -v -E '^(#|opendde==)' /kit/opendde/environment/requirements.lock > /tmp/stack.txt \
    && "${OPENDDE_HOME}/bin/pip" install --no-deps -r /tmp/stack.txt \
    && "${OPENDDE_HOME}/bin/pip" install --no-deps /kit/opendde/stock/opendde-1.1.1-py3-none-any.whl \
    && "${OPENDDE_HOME}/bin/pip" install --no-deps -e /kit/common/opt_core -e /kit/opendde/opt \
    && rm /tmp/stack.txt

# Fail the build early if the stack, the kit or the runner cannot be imported.
RUN "${OPENDDE_HOME}/bin/python" -c "import opendde_opt, runner.batch_inference"

ENV CUDA_HOME=/usr/local/cuda-12.6 \
    PATH=/usr/local/cuda-12.6/bin:${PATH} \
    LD_LIBRARY_PATH=/opt/opendde/lib:/usr/local/nvidia/lib64:/usr/local/nvidia/lib \
    MODEL_OPT=/kit/opendde \
    LAYERNORM_TYPE=fast_layernorm \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    DISABLE_PANDERA_IMPORT_WARNING=True
