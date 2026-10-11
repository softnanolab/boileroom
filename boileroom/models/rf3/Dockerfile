# syntax=docker/dockerfile:1.7
ARG TARGETPLATFORM
ARG TARGETARCH
ARG TARGETOS
ARG BUILDPLATFORM
ARG BASE_IMAGE=docker.io/jakublala/boileroom-base:local

FROM ${BASE_IMAGE}

COPY requirements.txt /tmp/requirements.txt

ARG UV_CACHE_ID=boileroom-uv
ARG TORCH_WHEEL_INDEX=https://download.pytorch.org/whl/cu126

# boileroom runtime postprocessing dependencies live in the Python 3.12 system interpreter.
RUN --mount=type=cache,id=${UV_CACHE_ID},target=/root/.cache/uv \
    uv pip install --system -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# RoseTTAFold 3 (rc-foundry, RosettaCommons/foundry, BSD-3-Clause) is validated upstream against torch 2.7.1 and
# rdkit 2025.3.6 (its ci-constraints.txt). atomworks is held at 2.2.1, the release the kit lock tests: its unpinned
# resolution picks atomworks 3, which removed what rf3 imports. cuEquivariance 0.10.0 is the newest release whose torch
# ops do not require torch >= 2.11. The stack lives in an isolated virtualenv; the core runs a persistent worker with its
# interpreter and never imports RF3 into the boileroom interpreter.
# git: foundry is installed from its repository at the pinned commit.
ARG RF3_COMMIT=4010e3e2e7350edada3e25a45c908c6bf407df4d
ARG RF3_HOME=/opt/rf3
ENV RF3_HOME=${RF3_HOME}
RUN apt-get update --quiet \
    && apt-get install --no-install-recommends --yes --quiet git \
    && rm -rf /var/lib/apt/lists/*
RUN --mount=type=cache,id=${UV_CACHE_ID},target=/root/.cache/uv \
    printf 'torch==2.7.1\nrdkit==2025.3.6\natomworks==2.2.1\nbiotite==1.4.0\n' > /tmp/constraints.txt \
    && UV_SYSTEM_PYTHON=0 uv venv --python 3.12 --seed ${RF3_HOME} \
    && uv pip install --python ${RF3_HOME}/bin/python \
        --extra-index-url ${TORCH_WHEEL_INDEX} --index-strategy unsafe-best-match \
        -c /tmp/constraints.txt \
        "rc-foundry[rf3] @ git+https://github.com/RosettaCommons/foundry.git@${RF3_COMMIT}" \
        cuequivariance==0.10.0 cuequivariance-torch==0.10.0 \
        cuequivariance-ops-cu12==0.10.0 cuequivariance-ops-torch-cu12==0.10.0 \
    && rm /tmp/constraints.txt

# Fail the build early if the stack or the inference engine cannot be imported.
RUN "${RF3_HOME}/bin/python" -c "import torch, rf3.inference_engines.rf3" \
    && python -c "import numpy"
