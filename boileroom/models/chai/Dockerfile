ARG BASE_IMAGE=ghcr.io/boileroom/base:latest
FROM ${BASE_IMAGE}

ARG CHAI_DOWNLOADS_DIR=/mnt/models/chai1

RUN python -m pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu118 \
    "torch==2.5.1+cu118" \
    hf_transfer==0.1.8 \
    chai_lab==0.6.1

ENV CHAI_DOWNLOADS_DIR=${CHAI_DOWNLOADS_DIR} \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    DISABLE_PANDERA_IMPORT_WARNING=True

RUN mkdir -p "${CHAI_DOWNLOADS_DIR}"

