FROM mambaorg/micromamba:git-c7619c9-cuda11.8.0-ubuntu22.04

USER root

RUN apt-get update \
    && apt-get install --yes --no-install-recommends git wget \
    && rm -rf /var/lib/apt/lists/*

USER ${MAMBA_USER}

ARG MAMBA_DOCKERFILE_ACTIVATE=1

RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir "biotite>=1.0.1"

ENV MODEL_DIR=/mnt/models

ENTRYPOINT ["/usr/bin/tini", "--"]

