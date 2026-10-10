#!/bin/sh
# Compile flash-attn, TransformerEngine and xformers for the selected stack and install them (step of the ESMFold2 kit image).
#
# Adapted from the wheel step of esmfold2/environment/Dockerfile in anthropics/uplifting-biomolecular-modeling (Apache-2.0),
# at the commit pinned by KIT_COMMIT in the Dockerfile next to this file. The compile recipe, source versions and checksums are
# the kit's. Changed: the compile toolchain is pinned to what the kit's PINS.json records for its build (nvcc 13.0.88, with
# the crt, libnvvm and libnvptxcompiler packages cuda-compiler-13-0 pulls in at the same version, and ninja 1.13.0), where
# the kit's recipe installs whatever the CUDA repository and PyPI serve that day. The cuda-keyring is checksummed; the other
# CUDA 13.0 development packages (headers and stub libraries) stay at the repository's 13.0 line and the base image's tag is
# not pinned by digest: neither changes the generated device code the way the compiler does, kit_sass.sh refuses a wheel
# without machine code for every compute capability of the stack, and the smoke check of kit_finish.sh (kit_smoke.py)
# refuses a build whose kernels do not import. Dropped: the WHEELS_FROM=prebuilt route (it
# needs wheel files that are not in the public kit repository).
#
# Environment: STACK (img_ef2_fa: compute capability 9.0, H100/H200 | img_esmfold2_a100: 8.0 and 9.0, A100 as well),
# BUILD_JOBS (parallel jobs, 0 = every core; flash-attn's nvcc jobs are further capped at one per 9 GB of available memory,
# at most 14). The benchmarked image compiled in 1893 s on 64 cores; expect hours on a small builder. The CUDA 13.0 compiler
# it installs for the compile is removed again in the same step, so the finished image carries the wheels and their
# installation only. Run from /kit/esmfold2 as root, after the pinned stack of requirements.lock is installed.
set -eu
W=/kit/esmfold2/stock/wheels
mkdir -p "$W"
case "$STACK" in
  img_ef2_fa)        TORCH_CUDA_ARCH_LIST="9.0" ;;
  img_esmfold2_a100) TORCH_CUDA_ARCH_LIST="8.0;9.0" ;;
  *) echo "esmfold2: no such stack: STACK=$STACK (img_ef2_fa: compute capability 9.0, H100 / H200 | img_esmfold2_a100: 8.0 and 9.0, A100 80GB as well)" >&2; exit 2 ;;
esac; export TORCH_CUDA_ARCH_LIST; SM_ARCHS=$(printf %s "$TORCH_CUDA_ARCH_LIST" | tr -d .)
T0=$(date +%s); J=${BUILD_JOBS:-0}; [ "$J" -gt 0 ] || J=$(nproc --all)
MEM_GB=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo); JFA=$(( MEM_GB / 9 )); [ "$JFA" -ge 1 ] || JFA=1; [ "$JFA" -le "$J" ] || JFA=$J; [ "$JFA" -le 14 ] || JFA=14
echo "esmfold2 wheels (STACK=$STACK): building xformers, transformer_engine, flash_attn for compute capability $TORCH_CUDA_ARCH_LIST with $J jobs (nproc --all = $(nproc --all), nproc = $(nproc), ${MEM_GB} GB available; flash-attn capped at $JFA jobs)"
dpkg-query -W -f '${Package}\n' | sort > /tmp/dpkg.before
apt-get update; apt-get install -y --no-install-recommends --no-upgrade ca-certificates curl
curl -fsSL -o /tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb
echo "e7f219eab6fe4819cdb5c15b98233dc3420302d9c00883219cd3d896857cf48d  /tmp/cuda-keyring.deb" | sha256sum -c -
dpkg -i /tmp/cuda-keyring.deb; apt-get update
apt-get install -y --no-install-recommends cuda-nvcc-13-0=13.0.88-1 cuda-crt-13-0=13.0.88-1 libnvvm-13-0=13.0.88-1 libnvptxcompiler-13-0=13.0.88-1 cuda-compiler-13-0 cuda-cuobjdump-13-0 cuda-libraries-dev-13-0 cuda-cudart-dev-13-0 cuda-nvtx-13-0 cuda-profiler-api-13-0 cuda-nvml-dev-13-0
export CUDA_HOME=/usr/local/cuda-13.0; export PATH="$CUDA_HOME/bin:$PATH" MAX_JOBS=$J CMAKE_BUILD_PARALLEL_LEVEL=$J MAKEFLAGS=-j$J NVCC_THREADS=2
nvcc --version | tail -1; nvcc --version | grep -q 'V13.0.88' || { echo "esmfold2 wheels: nvcc is not 13.0.88, the compiler the kit was built with" >&2; exit 2; }
python -m pip install --no-cache-dir cmake==4.4.3 ninja==1.13.0 'pybind11[global]==3.1.0' nvidia-cudnn-frontend==1.28.0
mkdir -p /tmp/src
git clone --quiet --depth 1 --branch v0.0.35 --recurse-submodules --shallow-submodules https://github.com/facebookresearch/xformers.git /tmp/src/xformers
test "$(git -C /tmp/src/xformers rev-parse HEAD)" = 03b91d7d9ff295ae68a320e2e733dd6c2ef8f342
T1=$(date +%s); ( cd /tmp/src/xformers && BUILD_VERSION=0.0.35+03b91d7.d20260904 FORCE_CUDA=1 XFORMERS_BUILD_TYPE=Release python -m pip wheel --no-cache-dir --no-build-isolation --no-deps -w "$W" . ); echo "xformers built in $(( $(date +%s) - T1 ))s"
git clone --quiet --depth 1 --branch v2.15 --recurse-submodules --shallow-submodules https://github.com/NVIDIA/TransformerEngine.git /tmp/src/TransformerEngine
test "$(git -C /tmp/src/TransformerEngine rev-parse HEAD)" = 42b840051647eef89761a16dfdff87e82bb253ab; git -C /tmp/src/TransformerEngine config core.abbrev 7; test "$(git -C /tmp/src/TransformerEngine rev-parse --short HEAD)" = 42b8400
NCCL_HOME=$(python -c "import nvidia.nccl; print(list(nvidia.nccl.__path__)[0])"); CUDNN_HOME=$(python -c "import nvidia.cudnn; print(list(nvidia.cudnn.__path__)[0])"); test -f "$NCCL_HOME/include/nccl.h"; test -f "$CUDNN_HOME/include/cudnn.h"
T1=$(date +%s); ( cd /tmp/src/TransformerEngine && export CPATH="$NCCL_HOME/include:$CUDNN_HOME/include" LIBRARY_PATH="$NCCL_HOME/lib" && NVTE_FRAMEWORK=pytorch NVTE_CUDA_ARCHS="$SM_ARCHS" NVTE_WITH_NCCL_EP=0 NVTE_BUILD_MAX_JOBS=$J CUDNN_PATH=$CUDNN_HOME python -m pip wheel --no-cache-dir --no-build-isolation --no-deps -w "$W" . ); echo "transformer_engine built in $(( $(date +%s) - T1 ))s"
python -m pip download --no-cache-dir --no-binary :all: --no-deps --no-build-isolation -d /tmp/src flash-attn==2.8.3.post1
echo "55d5103ed846da8b56e0797acf4bde07dee4b1c7e8907fcfc6699c203030c348  /tmp/src/flash_attn-2.8.3.post1.tar.gz" | sha256sum -c -
T1=$(date +%s); ( cd /tmp/src && MAX_JOBS=$JFA FLASH_ATTN_CUDA_ARCHS="$SM_ARCHS" FLASH_ATTENTION_FORCE_BUILD=TRUE python -m pip wheel --no-cache-dir --no-build-isolation --no-deps -w "$W" /tmp/src/flash_attn-2.8.3.post1.tar.gz ); echo "flash_attn built in $(( $(date +%s) - T1 ))s"
# Before the toolkit is removed: every wheel must carry machine code for each compute capability of the stack.
sh "$(dirname "$0")/kit_sass.sh" "$W" "$SM_ARCHS"
python -m pip uninstall -y cmake ninja pybind11 pybind11-global nvidia-cudnn-frontend
dpkg-query -W -f '${Package}\n' | sort > /tmp/dpkg.after; comm -13 /tmp/dpkg.before /tmp/dpkg.after > /tmp/dpkg.added
echo "removing the $(wc -l < /tmp/dpkg.added) packages this step installed"; xargs -r apt-get purge -y -qq < /tmp/dpkg.added
rm -rf /var/lib/apt/lists/* /etc/apt/sources.list.d/cuda* /usr/local/cuda* /tmp/src /tmp/cuda-keyring.deb /tmp/dpkg.before /tmp/dpkg.after /tmp/dpkg.added /root/.cache
ls -l "$W"; echo "esmfold2 wheels (STACK=$STACK, compute capability $TORCH_CUDA_ARCH_LIST): built in $(( $(date +%s) - T0 ))s on $(nproc --all) cores"
python -m pip install --no-cache-dir --no-index --no-deps "$W"/*.whl; python -m pip check
