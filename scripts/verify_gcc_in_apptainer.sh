#!/bin/bash
# Verification script to check GCC availability in Boltz-2 Apptainer container

set -e

IMAGE_URI="docker://docker.io/jakublala/boileroom-boltz:cuda12.6-dev"
CACHE_DIR="${HOME}/.cache/boileroom/images"
SIF_NAME="boltz_cuda12.6-dev.sif"
SIF_PATH="${CACHE_DIR}/${SIF_NAME}"

echo "=========================================="
echo "Verifying GCC in Boltz-2 Apptainer Container"
echo "Image URI: $IMAGE_URI"
echo "=========================================="
echo ""

# Create cache directory if it doesn't exist
mkdir -p "$CACHE_DIR"

# Pull the image if it doesn't exist
if [ ! -f "$SIF_PATH" ]; then
    echo "Pulling Apptainer image (this may take a while)..."
    apptainer pull --force "$SIF_PATH" "$IMAGE_URI"
    echo ""
fi

echo "1. Checking if GCC is available in Apptainer container..."
if apptainer exec "$SIF_PATH" which gcc > /dev/null 2>&1; then
    GCC_PATH=$(apptainer exec "$SIF_PATH" which gcc)
    echo "   ✓ GCC found at: $GCC_PATH"
else
    echo "   ✗ GCC NOT FOUND!"
    exit 1
fi

echo ""
echo "2. Checking GCC version..."
apptainer exec "$SIF_PATH" gcc --version | head -1

echo ""
echo "3. Checking if G++ is available..."
if apptainer exec "$SIF_PATH" which g++ > /dev/null 2>&1; then
    GPP_PATH=$(apptainer exec "$SIF_PATH" which g++)
    echo "   ✓ G++ found at: $GPP_PATH"
else
    echo "   ✗ G++ NOT FOUND!"
    exit 1
fi

echo ""
echo "4. Checking installed GCC packages..."
apptainer exec "$SIF_PATH" bash -c "dpkg -l | grep -E '(gcc|build-essential)' | grep -v '^rc'" || echo "   (Note: dpkg may not work in Apptainer, but GCC binaries are present)"

echo ""
echo "5. Testing GCC compilation in Apptainer..."
apptainer exec "$SIF_PATH" bash -c "
    cat > /tmp/test.c << 'EOF'
#include <stdio.h>
int main() { printf(\"GCC test successful in Apptainer\\n\"); return 0; }
EOF
    gcc /tmp/test.c -o /tmp/test && /tmp/test && echo '   ✓ GCC compilation test passed in Apptainer'
"

echo ""
echo "6. Checking environment variables (CC, CXX)..."
apptainer exec "$SIF_PATH" bash -c 'echo "   CC=${CC:-not set}, CXX=${CXX:-not set}"'

echo ""
echo "=========================================="
echo "Summary: GCC is properly available in the Apptainer container"
echo "=========================================="

