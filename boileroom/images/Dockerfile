# BuildKit platform arguments
ARG TARGETPLATFORM
ARG TARGETARCH
ARG TARGETOS
ARG BUILDPLATFORM

ARG MICROMAMBA_BASE=mambaorg/micromamba:2.4-cuda12.6.3-ubuntu22.04
FROM ${MICROMAMBA_BASE}

USER root

USER ${MAMBA_USER}

ARG MAMBA_DOCKERFILE_ACTIVATE=1

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
      git wget ca-certificates tini \
      build-essential \
    && rm -rf /var/lib/apt/lists/*
USER ${MAMBA_USER}

RUN micromamba install -y -c conda-forge python=3.12 pip \
    && python -m pip install --upgrade pip

ENV MODEL_DIR=/mnt/models
ENV PATH=/opt/conda/bin:${PATH}

ENTRYPOINT ["/usr/bin/tini", "--"]
