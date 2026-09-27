# syntax=docker/dockerfile:1.7
ARG TARGETPLATFORM
ARG TARGETARCH
ARG TARGETOS
ARG BUILDPLATFORM

ARG BASE_IMAGE=docker.io/jakublala/boileroom-base:local
FROM ${BASE_IMAGE}

COPY requirements.txt /tmp/requirements.txt

ARG UV_CACHE_ID=boileroom-uv

# ColabFold pulls AlphaFold parameters and MSAs at runtime (via the ColabFold
# MMseqs2 server), so the image only needs git (for the pip source install) and
# the Python runtime dependencies.
RUN apt-get update --quiet \
    && apt-get install --no-install-recommends --yes --quiet git \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,id=${UV_CACHE_ID},target=/root/.cache/uv \
    uv pip install --system -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# ColabFold caches AlphaFold weights under its --data directory (MODEL_DIR/alphafold).
ENV MPLBACKEND=Agg
