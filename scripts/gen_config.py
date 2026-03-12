#!/usr/bin/env python3
"""
Generate training config YAML for Apollo.
Extracted from kaggle_notebook.ipynb Cell 4.

Usage:
    python scripts/gen_config.py --out /path/to/config.yaml [options]
"""
import argparse
import os
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="config.yaml", help="Output config path")
    parser.add_argument("--exp_dir", default="./Exps", help="Experiment output dir")
    parser.add_argument("--exp_name", default="Apollo")
    parser.add_argument("--dataset_path", required=True, help="Path to HDF5 dataset dir")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--num_samples", type=int, default=8000)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--feature_dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--use_mamba", type=int, default=1)
    parser.add_argument("--mamba_expand", type=int, default=2)
    parser.add_argument("--mamba_dstate", type=int, default=16)
    parser.add_argument("--mamba_dconv", type=int, default=4)
    parser.add_argument("--bandwidth_d", type=int, default=80)
    parser.add_argument("--bandwidth_n", type=int, default=40)
    parser.add_argument("--win_parts", type=int, default=240)
    # GPU devices: comma-separated ints, e.g. "0" or "0,1"
    parser.add_argument("--devices", default="0", help="GPU devices, e.g. '0' or '0,1'")
    parser.add_argument("--wandb_offline", action="store_true", default=False)
    parser.add_argument("--precision", default="16-mixed", help="Trainer precision")
    args = parser.parse_args()

    devices = [int(d.strip()) for d in args.devices.split(",")]
    multi_gpu = len(devices) > 1

    ckpt_dir = os.path.join(args.exp_dir, args.exp_name, "checkpoints")
    log_dir = os.path.join(args.exp_dir, args.exp_name, "logs")

    config = {
        "exp": {"dir": args.exp_dir, "name": args.exp_name},
        "compile": True,  # torch.compile — ~20-30% speedup, first epoch slower
        "seed": 614020,
        "datas": {
            "_target_": "look2hear.datas.MusdbMoisesdbDataModule",
            "train_dir": args.dataset_path,
            "eval_dir": os.path.join(args.dataset_path, "eval"),
            "codec_type": "mp3",
            "codec_options": {
                "bitrate": "random",
                "compression": "random",
                "complexity": "random",
                "vbr": "random",
            },
            "sr": 44100,
            "segments": 3,
            "num_stems": 8,
            "snr_range": [-10, 10],
            "num_samples": args.num_samples,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "persistent_workers": True,  # keep workers alive between epochs
        },
        "model": {
            "_target_": "look2hear.models.apollo.Apollo",
            "sr": 44100,
            "win": 20,
            "feature_dim": args.feature_dim,
            "layer": args.layers,
        },
        "discriminator": {
            "_target_": "look2hear.discriminators.frequencydis.MultiFrequencyDiscriminator",
            "nch": 2,
            "window": [32, 64, 128, 256, 512, 1024, 2048],
        },
        "optimizer_g": {"_target_": "torch.optim.AdamW", "lr": 0.001, "weight_decay": 0.01},
        "optimizer_d": {
            "_target_": "torch.optim.AdamW",
            "lr": 0.0001,
            "weight_decay": 0.01,
            "betas": [0.5, 0.99],
        },
        "scheduler_g": {
            "_target_": "torch.optim.lr_scheduler.StepLR",
            "step_size": 2,
            "gamma": 0.98,
        },
        "scheduler_d": {
            "_target_": "torch.optim.lr_scheduler.StepLR",
            "step_size": 2,
            "gamma": 0.98,
        },
        "loss_g": {"_target_": "look2hear.losses.gan_losses.MultiFrequencyGenLoss", "eps": 1e-8},
        "loss_d": {"_target_": "look2hear.losses.gan_losses.MultiFrequencyDisLoss", "eps": 1e-8},
        "metrics": {"_target_": "look2hear.losses.MultiSrcNegSDR", "sdr_type": "sisdr"},
        "system": {"_target_": "look2hear.system.audio_litmodule.AudioLightningModule"},
        "early_stopping": {
            "_target_": "pytorch_lightning.callbacks.EarlyStopping",
            "monitor": "val_loss",
            "patience": 20,
            "mode": "min",
            "verbose": True,
        },
        "checkpoint": {
            "_target_": "pytorch_lightning.callbacks.ModelCheckpoint",
            "dirpath": ckpt_dir,
            "monitor": "val_loss",
            "mode": "min",
            "verbose": True,
            "save_top_k": 5,
            "save_last": True,
            "filename": "{epoch}-{val_loss:.4f}",
        },
        "logger": {
            "_target_": "pytorch_lightning.loggers.WandbLogger",
            "name": args.exp_name,
            "save_dir": log_dir,
            "offline": args.wandb_offline,
            "project": "Audio-Restoration",
        },
        "trainer": {
            "_target_": "pytorch_lightning.Trainer",
            "devices": devices,
            "max_epochs": args.max_epochs,
            "sync_batchnorm": multi_gpu,
            "default_root_dir": os.path.join(args.exp_dir, args.exp_name),
            "accelerator": "cuda",
            "precision": args.precision,
            "limit_train_batches": 1.0,
            "fast_dev_run": False,
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"Config written to {args.out}")
    print(f"  devices={devices}, batch_size={args.batch_size}, max_epochs={args.max_epochs}")
    if multi_gpu:
        print(f"  multi-GPU: sync_batchnorm=True, DDP will be used")


if __name__ == "__main__":
    main()
