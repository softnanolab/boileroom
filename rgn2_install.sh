#!/usr/bin/env bash
# setup_rgn2.sh – install RGN2 environment, weights, and ModRefiner
# Prerequisites: mamba (or micromamba), git, git-lfs, wget, unzip

set -euo pipefail

# ─────────────── configuration ────────────────
WORKDIR="rgn2"
RGN2_REPO="https://github.com/aqlaboratory/rgn2.git"

RGN2_HF_WEIGHTS="https://huggingface.co/christinafl/rgn2"
MODREFINER_URL="https://zhanggroup.org/ModRefiner/ModRefiner-l.zip"
# ───────────────────────────────────────────────

for bin in git git-lfs wget unzip mamba; do
  command -v "$bin" >/dev/null || { echo "ERROR: $bin not found in PATH."; exit 1; }
done

echo "=== Cloning RGN2 repository =================================================="
rm -rf "$WORKDIR"
git clone "$RGN2_REPO" "$WORKDIR"

echo "=== Creating conda env: rgn2 (via mamba) ====================================="
mamba env remove -y -n rgn2 2>/dev/null || true
mamba env create -f "${WORKDIR}/environment.yml"

echo "=== Fetching RGN2 weights from HuggingFace (git-lfs) =========================="
RGN2_RES_DIR="${WORKDIR}/resources"
rm -rf "$RGN2_RES_DIR"
GIT_LFS_SKIP_SMUDGE=1 git clone "$RGN2_HF_WEIGHTS" "$RGN2_RES_DIR"
(
  cd "$RGN2_RES_DIR"
  git lfs pull
)
mv "${RGN2_RES_DIR}/rgn2_runs" "${WORKDIR}/runs"

echo "=== Downloading ModRefiner ===================================================="
REFINER_DIR="${WORKDIR}/ter2pdb"
mkdir -p "$REFINER_DIR"
wget -O "$REFINER_DIR/ModRefiner-l.zip" "$MODREFINER_URL"
unzip -o "$REFINER_DIR/ModRefiner-l.zip" -d "$REFINER_DIR"
rm "$REFINER_DIR/ModRefiner-l.zip"

echo "=========================================================================="
echo "✅  All done!"
echo "   • Activate RGN2:   conda activate rgn2"
echo "=========================================================================="
