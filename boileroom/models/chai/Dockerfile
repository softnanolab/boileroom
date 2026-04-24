# BuildKit platform arguments
ARG TARGETPLATFORM
ARG TARGETARCH
ARG TARGETOS
ARG BUILDPLATFORM

ARG BASE_IMAGE=docker.io/jakublala/boileroom-base:local
FROM ${BASE_IMAGE}

COPY requirements.txt /tmp/requirements.txt

ARG TORCH_WHEEL_INDEX=https://download.pytorch.org/whl/cu118
ENV PIP_EXTRA_INDEX_URL=${TORCH_WHEEL_INDEX}
ENV UV_INDEX=${TORCH_WHEEL_INDEX}
ENV UV_INDEX_STRATEGY=unsafe-best-match

RUN uv pip install --system -r /tmp/requirements.txt \
    && rm -rf /root/.cache/pip /root/.cache/uv \
    && rm /tmp/requirements.txt

ENV HF_HUB_ENABLE_HF_TRANSFER=1 \
    DISABLE_PANDERA_IMPORT_WARNING=True
