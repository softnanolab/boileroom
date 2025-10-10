FROM mambaorg/micromamba:2.3.2-cuda11.8.0-ubuntu20.04

USER root

USER ${MAMBA_USER}

ARG MAMBA_DOCKERFILE_ACTIVATE=1

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
      git wget ca-certificates tini \
    && rm -rf /var/lib/apt/lists/*
USER ${MAMBA_USER}

RUN micromamba install -y -c conda-forge python=3.12 pip \
    && python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir "biotite>=1.0.1"

ENV MODEL_DIR=/mnt/models

# TODO: unclear whether this is the way to gos
USER root
RUN mkdir -p "/mnt/models" \
    && chown -R ${MAMBA_USER}:${MAMBA_USER} /mnt/models
USER ${MAMBA_USER}

ENTRYPOINT ["/usr/bin/tini", "--"]

