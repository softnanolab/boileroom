# BuildKit platform arguments
ARG TARGETPLATFORM
ARG TARGETARCH
ARG TARGETOS
ARG BUILDPLATFORM

ARG BASE_IMAGE=docker.io/jakublala/boileroom-base:latest
FROM ${BASE_IMAGE}

ARG ENV_FILE=environment.yml
COPY ${ENV_FILE} /tmp/environment.yml
USER root
RUN chown ${MAMBA_USER}:${MAMBA_USER} /tmp/environment.yml
USER ${MAMBA_USER}

ARG TORCH_WHEEL_INDEX=https://download.pytorch.org/whl/cu118
ENV PIP_EXTRA_INDEX_URL=${TORCH_WHEEL_INDEX}

RUN micromamba env update -n base -f /tmp/environment.yml \
    && micromamba clean --all --yes \
    && rm /tmp/environment.yml
