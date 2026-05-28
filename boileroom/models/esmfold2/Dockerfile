# syntax=docker/dockerfile:1.7
ARG TARGETPLATFORM
ARG TARGETARCH
ARG TARGETOS
ARG BUILDPLATFORM

ARG BASE_IMAGE=docker.io/jakublala/boileroom-base:local
FROM ${BASE_IMAGE}

COPY requirements.txt /tmp/requirements.txt

ARG TORCH_WHEEL_INDEX=https://download.pytorch.org/whl/cu118
ARG UV_CACHE_ID=boileroom-uv
ENV PIP_EXTRA_INDEX_URL=${TORCH_WHEEL_INDEX}
ENV UV_INDEX=${TORCH_WHEEL_INDEX}
ENV UV_INDEX_STRATEGY=unsafe-best-match
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN --mount=type=cache,id=${UV_CACHE_ID},target=/root/.cache/uv \
    uv pip install --system -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

RUN groupadd --gid 10001 boileroom \
    && useradd --uid 10001 --gid boileroom --create-home --home-dir /home/boileroom --shell /usr/sbin/nologin boileroom \
    && mkdir -p /mnt/models \
    && chown -R boileroom:boileroom /home/boileroom /mnt/models

WORKDIR /home/boileroom
USER boileroom
