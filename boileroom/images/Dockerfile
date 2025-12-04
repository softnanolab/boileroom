ARG MICROMAMBA_BASE=mambaorg/micromamba:2.3.2-cuda11.8.0-ubuntu20.04
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

ENTRYPOINT ["/usr/bin/tini", "--"]
