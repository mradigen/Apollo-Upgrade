#!/bin/sh

export WANDB_MODE=offline
export A_FEATURE_DIM=256
export A_LAYERS=6
export A_BANDWIDTH_D=80
export A_BANDWIDTH_N=40
export A_WIN_PARTS=240
export A_USE_MAMBA=1
export A_MAMBA_EXPAND=2
export A_MAMBA_DSTATE=16
export A_MAMBA_DCONV=4

python3 inference.py --in_wav=$1 --out_wav=restored.wav