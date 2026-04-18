# BuildKit platform arguments
ARG TARGETPLATFORM
ARG TARGETARCH
ARG TARGETOS
ARG BUILDPLATFORM

ARG MICROMAMBA_BASE=mambaorg/micromamba:2.5-cuda12.6.3-ubuntu24.04
FROM ${MICROMAMBA_BASE}

USER ${MAMBA_USER}

ARG MAMBA_DOCKERFILE_ACTIVATE=1
ENV PIP_NO_CACHE_DIR=1

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends eatmydata \
    && eatmydata apt-get install -y --no-install-recommends \
      git wget ca-certificates tini build-essential \
    && apt-get purge -y eatmydata \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*
USER ${MAMBA_USER}

RUN micromamba install -y -c conda-forge python=3.12 pip uv \
    && micromamba clean --all --yes \
    && rm -rf "${HOME}/.cache/pip"

ENV MODEL_DIR=/mnt/models
ENV TINI_SUBREAPER=1
ENV PATH=/opt/conda/bin:${PATH}

ENTRYPOINT ["/usr/bin/tini", "--"]
