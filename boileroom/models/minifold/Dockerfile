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

# minifold's pyproject.toml only declares packages=["minifold"], missing subpackages.
# Clone, patch to use find_packages discovery, then install.
RUN micromamba run -n base bash -c "\
    git clone https://github.com/jwohlwend/minifold.git /tmp/minifold \
    && cd /tmp/minifold \
    && sed -i 's/packages = \[\"minifold\"\]/packages = {find = {}}/' pyproject.toml \
    && pip install --no-cache-dir . \
    && rm -rf /tmp/minifold"
