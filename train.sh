#!/bin/sh

export WANDB_MODE=offline
export A_FEATURE_DIM=256
export A_LAYERS=6
export A_BANDWIDTH_D=80
export A_BANDWIDTH_N=40
export A_WIN_PARTS=240

python3 train.py --conf_dir=configs/apollo.yaml