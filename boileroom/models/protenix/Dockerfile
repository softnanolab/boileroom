# syntax=docker/dockerfile:1.7
ARG TARGETPLATFORM
ARG TARGETARCH
ARG TARGETOS
ARG BUILDPLATFORM

ARG BASE_IMAGE=docker.io/jakublala/boileroom-base:local
FROM ${BASE_IMAGE}

COPY requirements.txt /tmp/requirements.txt

ARG TORCH_WHEEL_INDEX=https://download.pytorch.org/whl/cu126
ARG UV_CACHE_ID=boileroom-uv
ENV PIP_EXTRA_INDEX_URL=${TORCH_WHEEL_INDEX}
ENV UV_INDEX=${TORCH_WHEEL_INDEX}
ENV UV_INDEX_STRATEGY=unsafe-best-match

RUN apt-get update --quiet \
    && apt-get install --no-install-recommends --yes --quiet hmmer kalign git \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,id=${UV_CACHE_ID},target=/root/.cache/uv \
    uv pip install --system -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

ENV HF_HUB_ENABLE_HF_TRANSFER=1 \
    DISABLE_PANDERA_IMPORT_WARNING=True
