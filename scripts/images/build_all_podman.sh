#!/usr/bin/env bash

set -euo pipefail

# Build all Boileroom Podman images locally for development, in dependency order.
#
# Images built:
#   - docker.io/jakublala/boileroom-base:dev
#   - docker.io/jakublala/boileroom-chai1:dev
#
# Options:
#   --platform=linux/amd64     Target platform (default linux/amd64)
#   --no-cache                 Do not use build cache
#   -h|--help                  Show help

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Build order:
  1) boileroom-base (from boileroom/boileroom/images/Dockerfile)
  2) boileroom-chai1 (from boileroom/boileroom/models/chai/Dockerfile)

Options:
  --platform=<value>       Default: linux/amd64
  --no-cache               Build without using cache
  -h, --help               Show this help and exit
EOF
}

PLATFORM="linux/amd64"
NO_CACHE=false

for arg in "$@"; do
  case "$arg" in
    --platform=*) PLATFORM="${arg#*=}" ;;
    --no-cache) NO_CACHE=true ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; usage; exit 1 ;;
  esac
done

# Resolve absolute paths relative to this script's directory
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# Repository root is two levels above scripts/images
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
# Point explicitly to the boileroom package directory inside the repo
BOILEROOM_DIR="${REPO_ROOT}/boileroom"
IMAGES_DIR="${BOILEROOM_DIR}/images"
CHAI_DIR="${BOILEROOM_DIR}/models/chai"
DOCKER_REGISTRY="docker.io/jakublala"

BASE_IMAGE_TAG="${DOCKER_REGISTRY}/boileroom-base:dev"
CHAI1_IMAGE_TAG="${DOCKER_REGISTRY}/boileroom-chai1:dev"

echo "Building base image: ${BASE_IMAGE_TAG}"
BASE_BUILD_CMD=(podman build --platform "${PLATFORM}" -t "${BASE_IMAGE_TAG}" -f "${IMAGES_DIR}/Dockerfile")
if [ "${NO_CACHE}" = true ]; then
  BASE_BUILD_CMD+=(--no-cache)
fi
BASE_BUILD_CMD+=("${IMAGES_DIR}")
"${BASE_BUILD_CMD[@]}"

BASE_ARG_IMAGE="${BASE_IMAGE_TAG}"

echo "Building chai1 image: ${CHAI1_IMAGE_TAG}"
CHAI_BUILD_CMD=(podman build --platform "${PLATFORM}" --build-arg "BASE_IMAGE=${BASE_ARG_IMAGE}" -t "${CHAI1_IMAGE_TAG}" -f "${CHAI_DIR}/Dockerfile")
if [ "${NO_CACHE}" = true ]; then
  CHAI_BUILD_CMD+=(--no-cache)
fi
CHAI_BUILD_CMD+=("${CHAI_DIR}")
"${CHAI_BUILD_CMD[@]}"

echo ""
echo "Build complete:"
echo "  Base : ${BASE_IMAGE_TAG}"
echo "  Chai1: ${CHAI1_IMAGE_TAG}"




