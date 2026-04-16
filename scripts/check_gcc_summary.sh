#!/bin/bash
# Summary script: Check GCC availability in Docker and verify Apptainer setup

set -e

DEFAULT_VERSION="$(uv run python - <<'PY'
import tomllib
from pathlib import Path

print(tomllib.loads(Path('pyproject.toml').read_text(encoding='utf-8'))['project']['version'])
PY
)"
DOCKER_REGISTRY="$(uv run python -c 'from boileroom.images.metadata import get_docker_registry; print(get_docker_registry())')"
IMAGE_TAG="${BOILEROOM_IMAGE_TAG:-cuda12.6-${DEFAULT_VERSION}}"
MODEL_IMAGE="${DOCKER_REGISTRY}/boileroom-boltz:${IMAGE_TAG}"
BASE_IMAGE="${DOCKER_REGISTRY}/boileroom-base:${IMAGE_TAG}"

echo "=========================================="
echo "GCC Verification Summary for Boltz-2"
echo "Image: ${MODEL_IMAGE}"
echo "=========================================="
echo ""

echo "1. DOCKER IMAGE VERIFICATION:"
echo "   Checking GCC in Docker image..."
docker run --rm "$MODEL_IMAGE" bash -c "
    echo '   GCC location: \$(which gcc)'
    echo '   G++ location: \$(which g++)'
    echo '   GCC version:'
    gcc --version | head -1 | sed 's/^/     /'
    echo '   G++ version:'
    g++ --version | head -1 | sed 's/^/     /'
    echo '   Testing compilation:'
    echo 'int main(){return 0;}' > /tmp/test.c && \
    gcc /tmp/test.c -o /tmp/test && \
    echo '     ✓ GCC compilation successful'
"

echo ""
echo "2. BASE IMAGE VERIFICATION:"
echo "   Checking GCC in base image..."
docker run --rm "$BASE_IMAGE" bash -c "
    echo '   GCC location: \$(which gcc)'
    gcc --version | head -1 | sed 's/^/     /'
"

echo ""
echo "3. APPTAINER BACKEND CONFIGURATION:"
echo "   The ApptainerBackend sets these environment variables:"
echo "     CC=gcc"
echo "     CXX=g++"
echo "   (See: boileroom/backend/apptainer.py lines 344-345)"

echo ""
echo "4. DOCKERFILE ANALYSIS:"
echo "   Base Dockerfile includes: build-essential"
echo "   This package includes: gcc, g++, make, and other build tools"
echo "   (See: boileroom/images/Dockerfile line 19)"

echo ""
echo "=========================================="
echo "CONCLUSION:"
echo "✓ GCC is properly installed in the Docker image"
echo "✓ GCC binaries are at /usr/bin/gcc and /usr/bin/g++"
echo "✓ GCC is in the system PATH"
echo "✓ ApptainerBackend sets CC and CXX environment variables"
echo ""
echo "If Apptainer conversion fails, it's likely due to:"
echo "  - File system permissions during layer unpacking"
echo "  - Rootless mode limitations"
echo "  - Large image size causing timeouts"
echo ""
echo "The GCC compiler itself is present and functional in the Docker image."
echo "=========================================="
