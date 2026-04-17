# BuildKit platform arguments
ARG TARGETPLATFORM
ARG TARGETARCH
ARG TARGETOS
ARG BUILDPLATFORM

ARG MICROMAMBA_BASE=mambaorg/micromamba:2.4-cuda12.6.3-ubuntu22.04
FROM ${MICROMAMBA_BASE}

USER ${MAMBA_USER}

ARG MAMBA_DOCKERFILE_ACTIVATE=1
ENV PIP_NO_CACHE_DIR=1

USER root
RUN cp /etc/apt/sources.list /tmp/sources.list \
    && sed -i \
      -e "s|http://archive.ubuntu.com/ubuntu|http://mirrors.ocf.berkeley.edu/ubuntu|g" \
      -e "s|http://security.ubuntu.com/ubuntu|http://mirrors.ocf.berkeley.edu/ubuntu|g" \
      /etc/apt/sources.list \
    && apt-get update \
      -o Acquire::Languages=none \
      -o Acquire::PDiffs=false \
      -o Acquire::QueueMode=access \
      -o Acquire::http::Pipeline-Depth=5 \
    && apt-get install -y --no-install-recommends eatmydata \
    && eatmydata apt-get install -y --no-install-recommends \
      git wget ca-certificates tini build-essential \
    && apt-get purge -y eatmydata \
    && apt-get autoremove -y \
    && mv /tmp/sources.list /etc/apt/sources.list \
    && rm -rf /var/lib/apt/lists/*
USER ${MAMBA_USER}

RUN micromamba install -y -c conda-forge python=3.12 pip \
    && python -m pip install --upgrade pip \
    && micromamba clean --all --yes \
    && rm -rf "${HOME}/.cache/pip"

ENV MODEL_DIR=/mnt/models
ENV TINI_SUBREAPER=1
ENV PATH=/opt/conda/bin:${PATH}

ENTRYPOINT ["/usr/bin/tini", "--"]
