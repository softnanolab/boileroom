# syntax=docker/dockerfile:1.7
ARG TARGETPLATFORM
ARG TARGETARCH
ARG TARGETOS
ARG BUILDPLATFORM

ARG BASE_IMAGE=docker.io/jakublala/boileroom-base:local
FROM ${BASE_IMAGE}

COPY requirements.txt /tmp/requirements.txt

ARG ALPHAFOLD_REF=c77e5d2a8961d1a353632c462914ff0a32a950f6
ARG TORCH_WHEEL_INDEX=https://download.pytorch.org/whl/cu126
ARG UV_CACHE_ID=boileroom-uv
ENV PIP_EXTRA_INDEX_URL=${TORCH_WHEEL_INDEX}
ENV UV_INDEX=${TORCH_WHEEL_INDEX}
ENV UV_INDEX_STRATEGY=unsafe-best-match

RUN apt-get update --quiet \
    && apt-get install --no-install-recommends --yes --quiet \
        build-essential \
        cmake \
        git \
        hmmer \
        hhsuite \
        kalign \
        wget \
    && rm -rf /var/lib/apt/lists/*

RUN git init /app/alphafold \
    && git -C /app/alphafold remote add origin https://github.com/google-deepmind/alphafold.git \
    && git -C /app/alphafold fetch --depth 1 origin ${ALPHAFOLD_REF} \
    && git -C /app/alphafold checkout --detach FETCH_HEAD \
    && wget -q -P /app/alphafold/alphafold/common/ \
        https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt

RUN --mount=type=cache,id=${UV_CACHE_ID},target=/root/.cache/uv \
    uv pip install --system -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

RUN echo '#!/usr/bin/env bash\nset -euo pipefail\npython /app/alphafold/run_alphafold.py "$@"' > /app/run_alphafold.sh \
    && chmod +x /app/run_alphafold.sh

ENV PYTHONPATH=/app/alphafold
