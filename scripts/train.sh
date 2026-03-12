#!/usr/bin/env bash
# ==============================================================
# Apollo training script for H100 server
#
# GPU selection via GPUS env var:
#   GPUS=0        → single GPU 0 (default)
#   GPUS=1        → single GPU 1
#   GPUS=0,1      → both GPUs (DDP)
#
# Source env.sh first, or set vars inline:
#   source scripts/env.sh && bash scripts/train.sh
#   GPUS=0,1 DATASET_PATH=/data/musdb18-hq bash scripts/train.sh
# ==============================================================
set -euo pipefail

# ── Resolve repo root (script lives in scripts/) ─────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

# ── GPU selection ─────────────────────────────────────────────
GPUS="${GPUS:-0}"

case "$GPUS" in
  "0")
    export CUDA_VISIBLE_DEVICES=0
    DEVICES="0"
    ;;
  "1")
    export CUDA_VISIBLE_DEVICES=1
    DEVICES="0"   # remapped to 0 inside the process
    ;;
  "0,1"|"both")
    export CUDA_VISIBLE_DEVICES=0,1
    DEVICES="0,1"
    ;;
  *)
    echo "ERROR: GPUS must be 0, 1, or 0,1 (got '$GPUS')"
    exit 1
    ;;
esac

echo "GPUS=${GPUS}  →  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}  devices=${DEVICES}"

# ── Required args ─────────────────────────────────────────────
if [[ -z "${DATASET_PATH:-}" ]]; then
  echo "ERROR: DATASET_PATH is required"
  echo "  e.g. DATASET_PATH=/data/musdb18-hq bash scripts/train.sh"
  exit 1
fi

# ── Hyperparams ───────────────────────────────────────────────
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_EPOCHS="${MAX_EPOCHS:-100}"
NUM_SAMPLES="${NUM_SAMPLES:-8000}"
NUM_WORKERS="${NUM_WORKERS:-16}"
FEATURE_DIM="${FEATURE_DIM:-256}"
LAYERS="${LAYERS:-6}"
USE_MAMBA="${USE_MAMBA:-1}"
MAMBA_EXPAND="${MAMBA_EXPAND:-2}"
MAMBA_DSTATE="${MAMBA_DSTATE:-16}"
MAMBA_DCONV="${MAMBA_DCONV:-4}"
BANDWIDTH_D="${BANDWIDTH_D:-80}"
BANDWIDTH_N="${BANDWIDTH_N:-40}"
WIN_PARTS="${WIN_PARTS:-240}"
PRECISION="${PRECISION:-bf16-mixed}"
WANDB_OFFLINE="${WANDB_OFFLINE:-1}"

EXP_DIR="${EXP_DIR:-$REPO_DIR/Exps}"
EXP_NAME="${EXP_NAME:-Apollo}"

CONFIG_PATH="$EXP_DIR/${EXP_NAME}/config.yaml"
mkdir -p "$EXP_DIR/$EXP_NAME"

# ── Generate config ───────────────────────────────────────────
echo "Generating config → $CONFIG_PATH"
python scripts/gen_config.py \
  --out "$CONFIG_PATH" \
  --exp_dir "$EXP_DIR" \
  --exp_name "$EXP_NAME" \
  --dataset_path "$DATASET_PATH" \
  --batch_size "$BATCH_SIZE" \
  --max_epochs "$MAX_EPOCHS" \
  --num_samples "$NUM_SAMPLES" \
  --num_workers "$NUM_WORKERS" \
  --feature_dim "$FEATURE_DIM" \
  --layers "$LAYERS" \
  --mamba_expand "$MAMBA_EXPAND" \
  --mamba_dstate "$MAMBA_DSTATE" \
  --mamba_dconv "$MAMBA_DCONV" \
  --bandwidth_d "$BANDWIDTH_D" \
  --bandwidth_n "$BANDWIDTH_N" \
  --win_parts "$WIN_PARTS" \
  --devices "$DEVICES" \
  --precision "$PRECISION" \
  ${WANDB_OFFLINE:+--wandb_offline}

# ── Model env vars (read by apollo.py) ───────────────────────
export A_USE_MAMBA="$USE_MAMBA"
export A_FEATURE_DIM="$FEATURE_DIM"
export A_LAYERS="$LAYERS"
export A_BANDWIDTH_D="$BANDWIDTH_D"
export A_BANDWIDTH_N="$BANDWIDTH_N"
export A_WIN_PARTS="$WIN_PARTS"
export A_MAMBA_EXPAND="$MAMBA_EXPAND"
export A_MAMBA_DSTATE="$MAMBA_DSTATE"
export A_MAMBA_DCONV="$MAMBA_DCONV"
export WANDB_MODE="${WANDB_OFFLINE:+offline}"
export WANDB_MODE="${WANDB_MODE:-online}"

# ── Launch ────────────────────────────────────────────────────
echo "Starting training (devices=$DEVICES, batch=$BATCH_SIZE, epochs=$MAX_EPOCHS)"
python train.py --conf_dir="$CONFIG_PATH" "$@"
