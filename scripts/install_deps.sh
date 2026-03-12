#!/usr/bin/env bash
# ==============================================================
# Install Apollo dependencies on an H100 server.
# Handles mamba-ssm's nvcc requirement.
#
# Usage:
#   bash scripts/install_deps.sh
# ==============================================================
set -euo pipefail

# ── Find nvcc ─────────────────────────────────────────────────
if ! command -v nvcc &>/dev/null; then
  echo "nvcc not in PATH — searching for it..."
  NVCC_PATH=$(find /usr/local/cuda* /usr/cuda* -name nvcc -type f 2>/dev/null | head -1 || true)
  if [[ -z "$NVCC_PATH" ]]; then
    echo "ERROR: nvcc not found. Install CUDA toolkit (devel image) or add it to PATH manually."
    echo "  e.g. export PATH=/usr/local/cuda/bin:\$PATH"
    exit 1
  fi
  CUDA_BIN="$(dirname "$NVCC_PATH")"
  export PATH="$CUDA_BIN:$PATH"
  export LD_LIBRARY_PATH="$(dirname "$CUDA_BIN")/lib64:${LD_LIBRARY_PATH:-}"
  echo "Found nvcc at $NVCC_PATH"
fi

echo "nvcc: $(nvcc --version | head -1)"
echo "torch: $(python -c 'import torch; print(torch.__version__)')"

# ── Install everything except mamba-ssm first ─────────────────
echo ""
echo "Installing base dependencies..."
uv pip install \
  pytorch-lightning \
  hydra-core \
  omegaconf \
  h5py \
  thop \
  wandb \
  soundfile \
  fast-bss-eval \
  torch-mir-eval \
  torchmetrics \
  torch-optimizer \
  huggingface-hub \
  torch-complex \
  pyyaml \
  torchaudio \
  librosa \
  torchcodec

# ── Install mamba-ssm (requires nvcc) ─────────────────────────
echo ""
echo "Installing mamba-ssm (builds from source, takes a few minutes)..."
uv pip install mamba-ssm

echo ""
echo "All dependencies installed successfully."
