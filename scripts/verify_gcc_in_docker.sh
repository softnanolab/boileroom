#!/bin/bash
# Verification script to check GCC availability in Boltz-2 Docker image

set -e

IMAGE_NAME="docker.io/jakublala/boileroom-boltz:cuda12.6-dev"

echo "=========================================="
echo "Verifying GCC in Boltz-2 Docker Image"
echo "Image: $IMAGE_NAME"
echo "=========================================="
echo ""

echo "1. Checking if GCC is available..."
if docker run --rm "$IMAGE_NAME" which gcc > /dev/null 2>&1; then
    GCC_PATH=$(docker run --rm "$IMAGE_NAME" which gcc)
    echo "   ✓ GCC found at: $GCC_PATH"
else
    echo "   ✗ GCC NOT FOUND!"
    exit 1
fi

echo ""
echo "2. Checking GCC version..."
docker run --rm "$IMAGE_NAME" gcc --version | head -1

echo ""
echo "3. Checking if G++ is available..."
if docker run --rm "$IMAGE_NAME" which g++ > /dev/null 2>&1; then
    GPP_PATH=$(docker run --rm "$IMAGE_NAME" which g++)
    echo "   ✓ G++ found at: $GPP_PATH"
else
    echo "   ✗ G++ NOT FOUND!"
    exit 1
fi

echo ""
echo "4. Checking installed GCC packages..."
docker run --rm "$IMAGE_NAME" bash -c "dpkg -l | grep -E '(gcc|build-essential)' | grep -v '^rc'"

echo ""
echo "5. Testing GCC compilation..."
docker run --rm "$IMAGE_NAME" bash -c "
    cat > /tmp/test.c << 'EOF'
#include <stdio.h>
int main() { printf(\"GCC test successful\\n\"); return 0; }
EOF
    gcc /tmp/test.c -o /tmp/test && /tmp/test && echo '   ✓ GCC compilation test passed'
"

echo ""
echo "6. Checking base image (boileroom-base:cuda12.6-dev)..."
BASE_IMAGE="docker.io/jakublala/boileroom-base:cuda12.6-dev"
if docker run --rm "$BASE_IMAGE" which gcc > /dev/null 2>&1; then
    echo "   ✓ Base image also has GCC"
    docker run --rm "$BASE_IMAGE" gcc --version | head -1
else
    echo "   ✗ Base image missing GCC!"
fi

echo ""
echo "=========================================="
echo "Summary: GCC is properly installed in the Docker image"
echo "=========================================="

