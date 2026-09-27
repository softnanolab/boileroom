# syntax=docker/dockerfile:1.7
ARG TARGETPLATFORM
ARG TARGETARCH
ARG TARGETOS
ARG BUILDPLATFORM

ARG BASE_IMAGE=docker.io/jakublala/boileroom-base:local
FROM ${BASE_IMAGE}

COPY requirements.txt /tmp/requirements.txt

ARG UV_CACHE_ID=boileroom-uv

# ColabFold only supports Python < 3.12 and pins pandas < 2.0, which has no
# cp312 wheel; it therefore cannot be installed into the boileroom (Python 3.12)
# system interpreter. It is instead installed into an isolated Python 3.10
# environment and only the ``colabfold_batch`` CLI is exposed on PATH. The
# boileroom core shells out to that CLI and never imports ColabFold, so the two
# interpreters stay independent. MSAs and AlphaFold parameters are fetched at
# runtime (via the ColabFold MMseqs2 server and ``--data`` cache), so the image
# only needs ``git`` for the source install.
RUN apt-get update --quiet \
    && apt-get install --no-install-recommends --yes --quiet git \
    && rm -rf /var/lib/apt/lists/*

# boileroom runtime postprocessing dependencies live in the Python 3.12 system.
RUN --mount=type=cache,id=${UV_CACHE_ID},target=/root/.cache/uv \
    uv pip install --system -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# Isolated ColabFold environment. The install steps mirror the upstream
# localcolabfold recipe: install ColabFold (alphafold code without JAX) from a
# pinned commit, then force the CUDA 12 JAX build and TensorFlow with pip's
# sequential --upgrade, which deliberately overrides ColabFold's older JAX cap.
ARG COLABFOLD_REF=efbf31c37cedb38cd09c69c1b991910a9866480e
ENV COLABFOLD_HOME=/opt/colabfold
RUN --mount=type=cache,id=${UV_CACHE_ID}-pip,target=/root/.cache/pip \
    UV_SYSTEM_PYTHON=0 uv venv --python 3.10 --seed ${COLABFOLD_HOME} \
    && "${COLABFOLD_HOME}/bin/pip" install --no-warn-conflicts \
        "colabfold[alphafold-minus-jax] @ git+https://github.com/sokrypton/ColabFold@${COLABFOLD_REF}" \
    && "${COLABFOLD_HOME}/bin/pip" install "colabfold[alphafold]" \
    && "${COLABFOLD_HOME}/bin/pip" install --upgrade "jax[cuda12]==0.5.3" \
    && "${COLABFOLD_HOME}/bin/pip" install --upgrade tensorflow silence_tensorflow \
    && ln -sf "${COLABFOLD_HOME}/bin/colabfold_batch" /usr/local/bin/colabfold_batch

# Fail the build early if the ColabFold CLI cannot import its dependencies
# (e.g. a JAX/Haiku/AlphaFold version mismatch) rather than at inference time.
RUN colabfold_batch --help > /dev/null

ENV MPLBACKEND=Agg
