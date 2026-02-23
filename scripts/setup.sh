#!/bin/bash
# scripts/setup.sh
# =================
# One-time setup for the PIR pipeline on Lightning.ai.
# Installs system deps, Python packages, and downloads model weights.
#
# Usage:
#   bash scripts/setup.sh

set -e

echo "============================================"
echo " PIR Pipeline — Environment Setup"
echo "============================================"

# ── System ────────────────────────────────────────────────
echo ""
echo "[1/4] System packages..."

# ── Python ────────────────────────────────────────────────
echo ""
echo "[2/4] Python packages..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# SAM2 from source (not on PyPI)
echo "  Installing SAM2..."
git clone https://github.com/facebookresearch/sam2.git && cd sam2

pip install -e .

# ── Model weights ─────────────────────────────────────────
echo ""
echo "[3/4] Downloading Cosmos-Reason2-8B (~16GB, cached after first run)..."
cd ..
export HF_TOKEN=$(grep HF_TOKEN .env | cut -d '=' -f2)
hf auth login --token $HF_TOKEN
hf download nvidia/Cosmos-Reason2-8B --cache-dir ~/.cache/huggingface/hub --exclude "*.pt" "original/*"

echo ""
echo "[4/4] Downloading SAM2 weights..."
hf download facebook/sam2-hiera-small && echo "  ✓ SAM2 cached"

echo ""
echo "============================================"
echo " Setup complete."
echo ""
echo " Next: start the Cosmos vLLM server"
echo "   bash scripts/start_server.sh"
echo "============================================"
