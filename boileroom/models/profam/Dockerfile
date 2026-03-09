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

# Clone profam, patch missing __init__.py files, install
RUN git clone --depth 1 https://github.com/alex-hh/profam.git /tmp/profam \
    && touch /tmp/profam/src/sequence/__init__.py \
             /tmp/profam/src/evaluators/__init__.py \
             /tmp/profam/src/pipelines/__init__.py \
    && micromamba run -n base pip install /tmp/profam \
    && rm -rf /tmp/profam

# Download gated ProFam-1 checkpoint from HuggingFace
ARG HF_TOKEN
USER root
RUN mkdir -p /models/profam && chown ${MAMBA_USER}:${MAMBA_USER} /models/profam
USER ${MAMBA_USER}
RUN HF_TOKEN=${HF_TOKEN} micromamba run -n base python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('judewells/ProFam-1', \
local_dir='/models/profam', \
local_dir_use_symlinks=False)"
