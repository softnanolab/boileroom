# syntax=docker/dockerfile:1.7
ARG BASE_IMAGE=docker.io/jakublala/boileroom-base:local
FROM ${BASE_IMAGE}

COPY requirements.txt /tmp/requirements.txt
ARG TORCH_WHEEL_INDEX=https://download.pytorch.org/whl/cu126
ARG UV_CACHE_ID=boileroom-uv
ENV UV_INDEX=${TORCH_WHEEL_INDEX} \
    UV_INDEX_STRATEGY=unsafe-best-match
RUN --mount=type=cache,id=${UV_CACHE_ID},target=/root/.cache/uv \
    uv pip install --system -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# Runtime download into MODEL_DIR, SHA256 checked before loading. No weights or
# Hugging Face credentials are baked into the image.
ENV BOILEROOM_PROFAM_COMMIT=a19061606dc6469424035db3fdcaae6e63688017
RUN python -c "from profam import ProFam; from profam.api import GenerationResult; assert callable(ProFam.generate)"
