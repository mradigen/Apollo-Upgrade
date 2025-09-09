
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

Model:
  | Name          | Type                        | Params | Mode
----------------------------------------------------------------------
0 | audio_model   | Apollo                      | 18.7 M | train
1 | discriminator | MultiFrequencyDiscriminator | 2.8 M  | train
2 | metrics       | MultiSrcNegSDR              | 0      | train
----------------------------------------------------------------------
21.4 M    Trainable params
0         Non-trainable params
21.4 M    Total params
85.734    Total estimated model params size (MB)
645       Modules in train mode
0         Modules in eval mode
```

6 hours for 


200 epochs:
wandb:   train_loss_d_step ▄█▄▆▅▇▆▄▄▅▇▅▄▆▄▅▁▄▄▇▁▂▃▆▇▅▅▆▆▄█▄▇▆▄▅▁▃▆▆
wandb:  train_loss_g_epoch ▅▅▇█▅▆█▆▅▃▃▆▃▄▇▁▅█▄▆
wandb:   train_loss_g_step ▄▅▄▄▅▄▄▄▅▄▆▃▄▇▄▆▂█▆▃▇▅▂▅▇▂▃▃▄▃▂▁▂▁▃▆▄▆▄▄
wandb: trainer/global_step ▁▁▂▂▂▂▃▃▃▃▃▃▃▄▄▄▄▅▅▅▅▅▅▅▆▆▆▆▆▆▆▇▇▇▇▇▇███
wandb:            val_loss ▇▄▅▆▄▅▃▁██▁▇▅▄▅▆▃▃▅▅
wandb:       val_pit_sisnr ▃▂▅▄▃▅▄▆█▁▁█▂▄▅▄▃▆▆▄▄
wandb:
wandb: Run summary:
wandb:               epoch 185
wandb:       learning_rate 0.00015
wandb:                  lr 0.00015
wandb:  train_loss_d_epoch 0.49714
wandb:   train_loss_d_step 0.4942
wandb:  train_loss_g_epoch 0.5018
wandb:   train_loss_g_step 0.58148
wandb: trainer/global_step 64249
wandb:            val_loss -14.82044
wandb:       val_pit_sisnr 14.82044
