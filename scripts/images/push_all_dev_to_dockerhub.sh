#!/usr/bin/env bash

set -euo pipefail

# Push all local :dev images to Docker Hub.
#
# Images pushed:
#   - docker.io/jakublala/boileroom-base:dev
#   - docker.io/jakublala/boileroom-chai1:dev
#
# Options:
#   --tool=docker|podman      Container tool to use (default docker)
# usage prints the script usage, available options, and prerequisites to stdout.

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --tool=docker|podman   Which CLI to use (default docker)
  -h, --help             Show this help and exit

Prerequisites:
  - Authenticate with Docker Hub first, e.g.: docker login
  - Ensure the :dev images exist locally (build scripts can create them).
EOF
}

TOOL="docker"

for arg in "$@"; do
  case "$arg" in
    --tool=*) TOOL="${arg#*=}" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; usage; exit 1 ;;
  esac
done

if ! command -v "$TOOL" >/dev/null 2>&1; then
  echo "Error: $TOOL is not installed or not in PATH" >&2
  exit 1
fi

IMAGES=(
  "docker.io/jakublala/boileroom-base:dev"
  "docker.io/jakublala/boileroom-chai1:dev"
)

for image in "${IMAGES[@]}"; do
  if ! "$TOOL" image inspect "$image" >/dev/null 2>&1; then
    echo "Skipping push; local image not found: $image" >&2
    continue
  fi
  echo "Pushing $image to Docker Hub using $TOOL..."
  "$TOOL" push "$image"
done

echo "Done."

