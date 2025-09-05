
```sh
export WANDB_MODE=offline
export A_WIN_PARTS=250
export A_FEATURE_DIM=256
export A_LAYERS=6
export A_BANDWIDTH_D=80
export A_BANDWIDTH_N=39

Model:
  | Name          | Type                        | Params | Mode
----------------------------------------------------------------------
0 | audio_model   | Apollo                      | 18.5 M | train
1 | discriminator | MultiFrequencyDiscriminator | 2.8 M  | train
2 | metrics       | MultiSrcNegSDR              | 0      | train
----------------------------------------------------------------------
21.3 M    Trainable params
0         Non-trainable params
21.3 M    Total params
85.280    Total estimated model params size (MB)
638       Modules in train mode
0         Modules in eval mode
```

```
export A_FEATURE_DIM=256
export A_LAYERS=6
export A_BANDWIDTH_D=80
export A_BANDWIDTH_N=40
export A_WIN_PARTS=240

```