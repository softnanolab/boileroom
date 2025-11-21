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
  --push                   Push images to Docker Hub after building
  -h, --help               Show this help and exit
EOF
}

PLATFORM="linux/amd64"
NO_CACHE=false
REQUESTED_TAG="dev"
DO_PUSH=false

for arg in "$@"; do
  case "$arg" in
    --platform=*) PLATFORM="${arg#*=}" ;;
    --no-cache) NO_CACHE=true ;;
    --tag=*) REQUESTED_TAG="${arg#*=}" ;;
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
declare -r BASE_IMAGE_TAG="${DOCKER_REGISTRY}/boileroom-base:${REQUESTED_TAG}"
declare -r BASE_DOCKERFILE="${IMAGES_DIR}/Dockerfile"

declare -A MODEL_TAGS=(
  ["boltz"]="${DOCKER_REGISTRY}/boileroom-boltz:${REQUESTED_TAG}"
  ["chai"]="${DOCKER_REGISTRY}/boileroom-chai1:${REQUESTED_TAG}"
  ["esm"]="${DOCKER_REGISTRY}/boileroom-esm:${REQUESTED_TAG}"
)

declare -A MODEL_DOCKERFILES=(
  ["boltz"]="${BOILEROOM_DIR}/models/boltz/Dockerfile"
  ["chai"]="${BOILEROOM_DIR}/models/chai/Dockerfile"
  ["esm"]="${BOILEROOM_DIR}/models/esm/Dockerfile"
)

declare -a MODEL_ORDER=("boltz" "chai" "esm")

echo "Building base image: ${BASE_IMAGE_TAG}"
BASE_BUILD_CMD=("docker" build --platform "${PLATFORM}" -t "${BASE_IMAGE_TAG}" -f "${BASE_DOCKERFILE}")
if [ "${NO_CACHE}" = true ]; then
  BASE_BUILD_CMD+=(--no-cache)
fi
BASE_BUILD_CMD+=("${IMAGES_DIR}")
"${BASE_BUILD_CMD[@]}"
if [ "${DO_PUSH}" = true ]; then
  docker push "${BASE_IMAGE_TAG}"
fi

BASE_ARG_IMAGE="${BASE_IMAGE_TAG}"

for model in "${MODEL_ORDER[@]}"; do
  tag="${MODEL_TAGS[${model}]}"
  dockerfile="${MODEL_DOCKERFILES[${model}]}"
  context_dir="$(dirname "${dockerfile}")"

  echo "Building ${model} image: ${tag}"
  MODEL_BUILD_CMD=("docker" build --platform "${PLATFORM}" --build-arg "BASE_IMAGE=${BASE_ARG_IMAGE}" -t "${tag}" -f "${dockerfile}")
  if [ "${NO_CACHE}" = true ]; then
    MODEL_BUILD_CMD+=(--no-cache)
  fi
  MODEL_BUILD_CMD+=("${context_dir}")
  "${MODEL_BUILD_CMD[@]}"
  if [ "${DO_PUSH}" = true ]; then
    docker push "${tag}"
  fi
done

echo ""
echo "Build complete:"
echo "  Base : ${BASE_IMAGE_TAG}"
for model in "${MODEL_ORDER[@]}"; do
  printf "  %s: %s\n" "${model^}" "${MODEL_TAGS[${model}]}"
done

