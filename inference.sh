#!/bin/sh

export WANDB_MODE=offline
export A_WIN_PARTS=250
export A_FEATURE_DIM=256
export A_LAYERS=6
export A_BANDWIDTH_D=80
export A_BANDWIDTH_N=39

python3 inference.py --in_wav=$1 --out_wav=restored.wav