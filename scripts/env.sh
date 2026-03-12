#!/usr/bin/env bash
# ==============================================================
# Edit these values, then source this file before training:
#   source scripts/env.sh && bash scripts/train.sh
# ==============================================================

# GPU: 0, 1, or 0,1
export GPUS=0,1

# Path to HDF5 dataset directory
export DATASET_PATH=/data/musdb18-hq

# Experiment output
export EXP_DIR=./Exps
export EXP_NAME=Apollo

# Training
export BATCH_SIZE=8
export MAX_EPOCHS=100
export NUM_SAMPLES=8000
export NUM_WORKERS=4

# Model
export FEATURE_DIM=256
export LAYERS=6
export USE_MAMBA=1
export MAMBA_EXPAND=2
export MAMBA_DSTATE=16
export MAMBA_DCONV=4
export BANDWIDTH_D=80
export BANDWIDTH_N=40
export WIN_PARTS=240

# Trainer
export PRECISION=16-mixed
export WANDB_OFFLINE=1   # set to "" to use online wandb

# Force torchaudio to use legacy sox/soundfile backend instead of torchcodec
# (torchcodec requires FFmpeg shared libs which may not be installed)
export TORCHAUDIO_USE_TORCHCODEC=0
