# syntax=docker/dockerfile:1.7
ARG TARGETPLATFORM
ARG TARGETARCH
ARG TARGETOS
ARG BUILDPLATFORM
ARG BASE_IMAGE=docker.io/jakublala/boileroom-base:local
FROM ${BASE_IMAGE}

COPY requirements.txt /tmp/requirements.txt
COPY patch_upstream.py /tmp/patch_upstream.py
ARG TORCH_WHEEL_INDEX=https://download.pytorch.org/whl/cu126
ARG UV_CACHE_ID=boileroom-uv
ENV UV_INDEX=${TORCH_WHEEL_INDEX}
ENV UV_INDEX_STRATEGY=unsafe-best-match
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ARG MINIFOLD_COMMIT=63db8b91f5bb64535f20810f014e69df66b0faad
ENV MINIFOLD_COMMIT=${MINIFOLD_COMMIT}

RUN --mount=type=cache,id=${UV_CACHE_ID},target=/root/.cache/uv \
    uv pip install --system -r /tmp/requirements.txt \
    && git clone https://github.com/jwohlwend/minifold.git /opt/minifold \
    && git -C /opt/minifold checkout ${MINIFOLD_COMMIT} \
    && test "$(git -C /opt/minifold rev-parse HEAD)" = "${MINIFOLD_COMMIT}" \
    && python /tmp/patch_upstream.py /opt/minifold \
    && uv pip install --system --no-deps /opt/minifold \
    && python -c "from minifold.model.model import MiniFoldModel; from minifold.data.of_data import of_inference; from importlib.resources import files; assert (files('minifold') / 'utils/stereo_chemical_props.txt').is_file()" \
    && rm /tmp/requirements.txt /tmp/patch_upstream.py
