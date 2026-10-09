# RoseTTAFold 3 optimization-kit runtime image: the stack behind `optimization="exact"`.
#
# The default `boileroom-rf3` image (../Dockerfile) cannot run this mode: it is the stock stack on CUDA 12.6 with torch
# 2.7.1, the kit needs the stack it was tested on (Python 3.12.1, torch 2.13.0+cu130, triton 3.7.1, cuEquivariance 0.11.1
# on its cu13 ops, atomworks 2.2.1). `optimization="vanilla"` keeps using the stock image; this one is opt-in and only
# used when a kit mode is requested.
#
# Source: anthropics/uplifting-biomolecular-modeling (Apache-2.0), fetched at KIT_COMMIT. Apart from the changes below this
# is the kit's own `rosettafold3/environment/Dockerfile` at that commit: the kit comes from git instead of the build
# context, and the optional `_jitcache` COPY (a glob that Modal's Dockerfile parser rejects) is gone. Added on top:
# boileroom's runtime dependencies and the model directory.
#
# Build, from the repository root:
#   docker build -f boileroom/models/rf3/kit/Dockerfile boileroom/models/rf3/kit -t boileroom-rf3-kit:local
# On Modal the wrapper builds this file itself: see docs/optimization.md.
#
# Two interpreters, as the kit requires: /usr/local/bin/python is the pristine stock one (`optimization="vanilla"` semantics),
# /kit/rosettafold3/opt/venv/bin/python is the patched one the kit modes run in. Host: an NVIDIA driver that runs CUDA 13.0
# (580 or newer). No weights are inside: RF3 downloads its checkpoint (2.8 GiB) into RF3_ROOT_DIR, which boileroom points
# at the model directory.
FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04

ARG KIT_REPOSITORY=https://github.com/anthropics/uplifting-biomolecular-modeling.git
# Keep in sync with KIT_COMMIT in boileroom/images/metadata.py (tests/contracts/test_kit_images.py checks it).
ARG KIT_COMMIT=f4f62fa6592ae4938d49b1757bea0cfeff9f468e

# The C toolchain Triton needs at run time (gcc, libc headers), git for the stock pin and the kit, and the download tools of
# this build.
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates wget git build-essential && rm -rf /var/lib/apt/lists/*

# Python 3.12.1: the python-build-standalone release (x86_64_v3 build) this stack was assembled with, unpacked into
# /usr/local (no system Python is used); it carries pip 23.3.2.
RUN wget -q -O /tmp/python.tar.gz "https://github.com/indygreg/python-build-standalone/releases/download/20240107/cpython-3.12.1+20240107-x86_64_v3-unknown-linux-gnu-install_only.tar.gz" \
 && echo "a7bcc6c9f66dbd47ea99615f30f101c5d2dd0084ca333d2f7336e64050951338  /tmp/python.tar.gz" | sha256sum -c - \
 && tar -xzf /tmp/python.tar.gz -C /tmp && cp -a /tmp/python/. /usr/local/ && rm -rf /tmp/python /tmp/python.tar.gz \
 && ln -sf /usr/local/bin/python3 /usr/local/bin/python && python -V

# The kit at the pinned commit: its shared core and the rosettafold3 tree, under /kit as its own image has them.
RUN git init -q /tmp/kit-src \
 && git -C /tmp/kit-src fetch -q --depth 1 "$KIT_REPOSITORY" "$KIT_COMMIT" \
 && git -C /tmp/kit-src checkout -q --detach FETCH_HEAD \
 && test "$(git -C /tmp/kit-src rev-parse HEAD)" = "$KIT_COMMIT" \
 && mkdir -p /kit/common \
 && cp -r /tmp/kit-src/common/opt_core /kit/common/opt_core \
 && cp -r /tmp/kit-src/rosettafold3 /kit/rosettafold3 \
 && rm -rf /tmp/kit-src

# The pinned stack, every distribution at the lock's version (--no-deps: the lock is complete; stock rc-foundry is the lock's
# `git+https` line, cloned at the pinned commit and built with the hatchling release that built the tested wheel), then the
# cu13 cuDNN / NCCL wheels laid down once more: they share their nvidia/cudnn and nvidia/nccl directories with the cu12 wheels
# of the lock, and the tested tree holds the cu13 files there.
RUN echo "build cores: $(nproc --all)" && grep -v -E '^#' /kit/rosettafold3/environment/requirements.lock > /tmp/stack.txt \
 && printf 'hatchling==1.31.0\nhatch-vcs==0.4.0\n' > /tmp/build-pins.txt \
 && PIP_CONSTRAINT=/tmp/build-pins.txt python -m pip install --no-cache-dir --no-deps -r /tmp/stack.txt \
 && python -m pip install --no-cache-dir --no-deps --force-reinstall nvidia-cudnn-cu13==9.20.0.48 nvidia-nccl-cu13==2.29.7 \
 && rm -f /tmp/stack.txt /tmp/build-pins.txt

# The kit: the patched interpreter /kit/rosettafold3/opt/venv derived from the stock one, the shared core and rosettafold3_opt
# installed editable into it, the pin checked on both interpreters, then the add-on's five rf3 files applied into the venv's
# copy of the package: the one install step, `run.sh install --make-venv`. Only the venv's copy is written, so
# /usr/local/bin/python stays pristine. fastapi and uvicorn are what boileroom's Apptainer service imports on top of the
# kit's lock (numpy, biotite and pydantic are in it); they go to the pristine interpreter, where boileroom's core runs.
WORKDIR /kit/rosettafold3
RUN bash run.sh install --make-venv \
 && python -m pip install --no-cache-dir fastapi==0.136.3 uvicorn==0.48.0 \
 && mkdir -p /opt/jit_cache && chmod -R a+rX,u+w /opt/jit_cache

# The process environment of the tested stack: hash seed fixed, no debug info in the launchers Triton compiles, the kit
# directory the package locates its kits through, and the model directory. The pristine stock interpreter is this image's own
# python (configs/*.env fill it only when unset). The checkpoint is never baked in.
ENV PYTHONHASHSEED=0 \
    CFLAGS=-g0 \
    MODEL_DIR=/mnt/models \
    MODEL_OPT=/kit/rosettafold3 \
    ROSETTAFOLD3_OPT_STOCK_PYTHON=/usr/local/bin/python
