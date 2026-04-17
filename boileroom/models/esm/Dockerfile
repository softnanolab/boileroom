ARG TARGETPLATFORM
ARG TARGETARCH
ARG TARGETOS
ARG BUILDPLATFORM

ARG BASE_IMAGE=docker.io/jakublala/boileroom-base:local
FROM ${BASE_IMAGE}

ARG ENV_FILE=environment.yml
COPY ${ENV_FILE} /tmp/environment.yml
COPY requirements.txt /tmp/requirements.txt
USER root
RUN chown ${MAMBA_USER}:${MAMBA_USER} /tmp/environment.yml /tmp/requirements.txt
USER ${MAMBA_USER}

ARG TORCH_WHEEL_INDEX=https://download.pytorch.org/whl/cu118
ENV PIP_EXTRA_INDEX_URL=${TORCH_WHEEL_INDEX}
ENV UV_INDEX=${TORCH_WHEEL_INDEX}
ENV UV_INDEX_STRATEGY=unsafe-best-match
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN micromamba env update -n base -f /tmp/environment.yml \
    && uv pip install --python /opt/conda/bin/python -r /tmp/requirements.txt \
    && micromamba clean --all --yes \
    && rm -rf "${HOME}/.cache/pip" "${HOME}/.cache/uv" \
    && rm /tmp/environment.yml /tmp/requirements.txt
