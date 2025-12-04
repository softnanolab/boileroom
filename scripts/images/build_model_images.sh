#!/usr/bin/env bash

set -euo pipefail

# Build base plus all per-model Docker images using Docker.
# Usage mirrors the previous helpers but builds everything in one pass.

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Build order:
  1) boileroom-base   (boileroom/boileroom/images/Dockerfile)
  2) boileroom-boltz  (boileroom/boileroom/models/boltz/Dockerfile)
  3) boileroom-chai1  (boileroom/boileroom/models/chai/Dockerfile)
  4) boileroom-esm    (boileroom/boileroom/models/esm/Dockerfile)

Options:
  --platform=<value>       Target platform (default: linux/amd64)
  --no-cache               Build without using cache
  --tag=<value>            Tag suffix for all images (default: dev)
  --cuda-version=<value>   CUDA version to build (11.8 or 12.6, can be repeated)
  --all-cuda               Build both CUDA versions (11.8 and 12.6)
  --push                   Push images to Docker Hub after building
  -h, --help               Show this help and exit
EOF
}

PLATFORM="linux/amd64"
NO_CACHE=false
REQUESTED_TAG="dev"
DO_PUSH=false
declare -a CUDA_VERSIONS=()

for arg in "$@"; do
  case "$arg" in
    --platform=*) PLATFORM="${arg#*=}" ;;
    --no-cache) NO_CACHE=true ;;
    --tag=*) REQUESTED_TAG="${arg#*=}" ;;
    --cuda-version=*) CUDA_VERSIONS+=("${arg#*=}") ;;
    --all-cuda) CUDA_VERSIONS=("11.8" "12.6") ;;
    --push) DO_PUSH=true ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown option: $arg" >&2
      usage
      exit 1
      ;;
  esac
done

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is required but was not found on PATH." >&2
  exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
BOILEROOM_DIR="${REPO_ROOT}/boileroom"
IMAGES_DIR="${BOILEROOM_DIR}/images"

declare -r DOCKER_REGISTRY="docker.io/jakublala"
declare -r BASE_DOCKERFILE="${IMAGES_DIR}/Dockerfile"

declare -A MODEL_DOCKERFILES=(
  ["boltz"]="${BOILEROOM_DIR}/models/boltz/Dockerfile"
  ["chai"]="${BOILEROOM_DIR}/models/chai/Dockerfile"
  ["esm"]="${BOILEROOM_DIR}/models/esm/Dockerfile"
)

declare -a MODEL_ORDER=("boltz" "chai" "esm")

# CUDA version mapping: version -> (micromamba_base, torch_wheel_index)
declare -A CUDA_MICROMAMBA_BASE=(
  ["11.8"]="mambaorg/micromamba:2.4-cuda11.8.0-ubuntu22.04"
  ["12.6"]="mambaorg/micromamba:2.4-cuda12.6.3-ubuntu22.04"
)

declare -A CUDA_TORCH_WHEEL_INDEX=(
  ["11.8"]="https://download.pytorch.org/whl/cu118"
  ["12.6"]="https://download.pytorch.org/whl/cu126"
)

# Validate CUDA versions
for cuda_version in "${CUDA_VERSIONS[@]}"; do
  if [[ ! -v CUDA_MICROMAMBA_BASE["${cuda_version}"] ]]; then
    echo "Error: Unsupported CUDA version: ${cuda_version}. Supported versions: 11.8, 12.6" >&2
    exit 1
  fi
done

# Build for each CUDA version
declare -a ALL_BUILT_IMAGES=()

for cuda_version in "${CUDA_VERSIONS[@]}"; do
  echo ""
  echo "=========================================="
  echo "Building for CUDA ${cuda_version}"
  echo "=========================================="
  
  CUDA_SUFFIX="cuda${cuda_version}"
  # Build tag: if REQUESTED_TAG is "latest", use just CUDA suffix, otherwise append tag
  if [ "${REQUESTED_TAG}" = "latest" ]; then
    BASE_IMAGE_TAG="${DOCKER_REGISTRY}/boileroom-base:${CUDA_SUFFIX}"
  else
    BASE_IMAGE_TAG="${DOCKER_REGISTRY}/boileroom-base:${CUDA_SUFFIX}-${REQUESTED_TAG}"
  fi
  MICROMAMBA_BASE="${CUDA_MICROMAMBA_BASE[${cuda_version}]}"
  TORCH_WHEEL_INDEX="${CUDA_TORCH_WHEEL_INDEX[${cuda_version}]}"
  
  # Build base image
  echo "Building base image: ${BASE_IMAGE_TAG}"
  echo "  Using micromamba base: ${MICROMAMBA_BASE}"
  BASE_BUILD_CMD=("docker" build --platform "${PLATFORM}" --build-arg "MICROMAMBA_BASE=${MICROMAMBA_BASE}" -t "${BASE_IMAGE_TAG}" -f "${BASE_DOCKERFILE}")
  if [ "${NO_CACHE}" = true ]; then
    BASE_BUILD_CMD+=(--no-cache)
  fi
  BASE_BUILD_CMD+=("${IMAGES_DIR}")
  "${BASE_BUILD_CMD[@]}"
  if [ "${DO_PUSH}" = true ]; then
    docker push "${BASE_IMAGE_TAG}"
  fi
  ALL_BUILT_IMAGES+=("${BASE_IMAGE_TAG}")
  
  # Build model images
  for model in "${MODEL_ORDER[@]}"; do
    if [ "${REQUESTED_TAG}" = "latest" ]; then
      MODEL_TAG="${DOCKER_REGISTRY}/boileroom-${model}:${CUDA_SUFFIX}"
    else
      MODEL_TAG="${DOCKER_REGISTRY}/boileroom-${model}:${CUDA_SUFFIX}-${REQUESTED_TAG}"
    fi
    if [ "${model}" = "chai" ]; then
      if [ "${REQUESTED_TAG}" = "latest" ]; then
        MODEL_TAG="${DOCKER_REGISTRY}/boileroom-chai1:${CUDA_SUFFIX}"
      else
        MODEL_TAG="${DOCKER_REGISTRY}/boileroom-chai1:${CUDA_SUFFIX}-${REQUESTED_TAG}"
      fi
    fi
    dockerfile="${MODEL_DOCKERFILES[${model}]}"
    context_dir="$(dirname "${dockerfile}")"
    
    echo "Building ${model} image: ${MODEL_TAG}"
    MODEL_BUILD_CMD=("docker" build --platform "${PLATFORM}" --build-arg "BASE_IMAGE=${BASE_IMAGE_TAG}" --build-arg "TORCH_WHEEL_INDEX=${TORCH_WHEEL_INDEX}" -t "${MODEL_TAG}" -f "${dockerfile}")
    if [ "${NO_CACHE}" = true ]; then
      MODEL_BUILD_CMD+=(--no-cache)
    fi
    MODEL_BUILD_CMD+=("${context_dir}")
    "${MODEL_BUILD_CMD[@]}"
    if [ "${DO_PUSH}" = true ]; then
      docker push "${MODEL_TAG}"
    fi
    ALL_BUILT_IMAGES+=("${MODEL_TAG}")
  done
done

echo ""
echo "=========================================="
echo "Build complete:"
echo "=========================================="
for image in "${ALL_BUILT_IMAGES[@]}"; do
  echo "  ${image}"
done

