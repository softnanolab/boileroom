# BuildKit platform arguments
ARG TARGETPLATFORM
ARG TARGETARCH
ARG TARGETOS
ARG BUILDPLATFORM

ARG PYTHON_BASE=python:3.12-slim-bookworm
FROM ${PYTHON_BASE}

ENV PIP_NO_CACHE_DIR=1
ENV UV_SYSTEM_PYTHON=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends eatmydata \
    && eatmydata apt-get install -y --no-install-recommends \
      git wget ca-certificates tini build-essential \
    && apt-get purge -y eatmydata \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip uv \
    && rm -rf /root/.cache/pip

ENV MODEL_DIR=/mnt/models
ENV TINI_SUBREAPER=1

ENTRYPOINT ["/usr/bin/tini", "--"]
