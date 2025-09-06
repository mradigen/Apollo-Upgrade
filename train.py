###
# Author: Kai Li
# Date: 2024-01-22 01:16:22
# Email: lk21@mails.tsinghua.edu.cn
# LastEditTime: 2024-01-24 00:05:10
###
import json
from typing import Any, Dict, List, Optional, Tuple
import os
import glob
from omegaconf import OmegaConf
import argparse
import pytorch_lightning as pl
import torch
torch.set_float32_matmul_precision("highest")
import hydra
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
# from pytorch_lightning.loggers import Logger
from omegaconf import DictConfig
import look2hear.system
import look2hear.datas
import look2hear.losses
from look2hear.utils import RankedLogger, instantiate, print_only
import warnings
warnings.filterwarnings("ignore")



def _find_resume_checkpoint(exp_dir: str) -> Optional[str]:
    """Try to locate a suitable checkpoint to resume from.

    Priority order:
    1. best_k_models.json (first key assumed best)
    2. checkpoint containing 'best'
    3. checkpoint containing 'last'
    4. most recently modified .ckpt file
    """
    # 1. Use recorded best_k_models.json if it exists
    best_json = os.path.join(exp_dir, "best_k_models.json")
    if os.path.isfile(best_json):
        try:
            with open(best_json) as f:
                data = json.load(f)
            if isinstance(data, dict) and data:
                # keys are checkpoint paths
                # choose the one with best value depending on min/max not stored; assume first is best
                # but safer: pick key whose file exists
                existing = [k for k in data.keys() if os.path.isfile(k)]
                if existing:
                    return existing[0]
        except Exception:
            pass

    # 2-4. Scan checkpoints directory
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        return None
    ckpts = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
    if not ckpts:
        return None
    # prefer filenames containing 'best'
    best = [c for c in ckpts if "best" in os.path.basename(c).lower()]
    if best:
        return best[0]
    last = [c for c in ckpts if "last" in os.path.basename(c).lower()]
    if last:
        return last[0]
    # fallback: newest by mtime
    return max(ckpts, key=os.path.getmtime)


def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)
    
    # instantiate datamodule
    print_only(f"Instantiating datamodule <{cfg.datas._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datas)
    
    # instantiate model
    print_only(f"Instantiating AudioNet <{cfg.model._target_}>")
    model: torch.nn.Module = hydra.utils.instantiate(cfg.model)
    print_only(f"Instantiating Discriminator <{cfg.discriminator._target_}>")
    discriminator: torch.nn.Module = hydra.utils.instantiate(cfg.discriminator)
    
    # instantiate optimizer
    print_only(f"Instantiating optimizer <{cfg.optimizer_g._target_}>")
    optimizer_g: torch.optim = hydra.utils.instantiate(cfg.optimizer_g, params=model.parameters())
    optimizer_d: torch.optim = hydra.utils.instantiate(cfg.optimizer_d, params=discriminator.parameters())
    # optimizer: torch.optim = torch.optim.Adam(model.parameters(), lr=cfg.optimizer.lr)

    # instantiate scheduler
    print_only(f"Instantiating scheduler <{cfg.scheduler_g._target_}>")
    scheduler_g: torch.optim.lr_scheduler = hydra.utils.instantiate(cfg.scheduler_g, optimizer=optimizer_g)
    scheduler_d: torch.optim.lr_scheduler = hydra.utils.instantiate(cfg.scheduler_d, optimizer=optimizer_d)
        
    # instantiate loss
    print_only(f"Instantiating loss <{cfg.loss_g._target_}>")
    loss_g: torch.nn.Module = hydra.utils.instantiate(cfg.loss_g)
    loss_d: torch.nn.Module = hydra.utils.instantiate(cfg.loss_d)
    losses = {
        "g": loss_g,
        "d": loss_d
    }
    
    # instantiate metrics
    print_only(f"Instantiating metrics <{cfg.metrics._target_}>")
    metrics: torch.nn.Module = hydra.utils.instantiate(cfg.metrics)
    # instantiate system
    print_only(f"Instantiating system <{cfg.system._target_}>")
    system: LightningModule = hydra.utils.instantiate(
        cfg.system,
        model=model,
        discriminator=discriminator,
        loss_func=losses,
        metrics=metrics,
        optimizer=[optimizer_g, optimizer_d],
        scheduler=[scheduler_g, scheduler_d]
    )
    
    # instantiate callbacks
    callbacks: List[Callback] = []
    if cfg.get("early_stopping"):
        print_only(f"Instantiating early_stopping <{cfg.early_stopping._target_}>")
        callbacks.append(hydra.utils.instantiate(cfg.early_stopping))
    if cfg.get("checkpoint"):
        print_only(f"Instantiating checkpoint <{cfg.checkpoint._target_}>")
        checkpoint: pl.callbacks.ModelCheckpoint = hydra.utils.instantiate(cfg.checkpoint)
        callbacks.append(checkpoint)
        
    # instantiate logger
    print_only(f"Instantiating logger <{cfg.logger._target_}>")
    os.makedirs(os.path.join(cfg.exp.dir, cfg.exp.name, "logs"), exist_ok=True)
    logger = hydra.utils.instantiate(cfg.logger)
    
    # instantiate trainer
    print_only(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
        strategy=DDPStrategy(find_unused_parameters=True),
    )
    
    # Auto-resume logic
    ckpt_path = None
    if getattr(cfg, "resume", True):
        exp_dir = os.path.join(cfg.exp.dir, cfg.exp.name)
        ckpt_path = _find_resume_checkpoint(exp_dir)
        if ckpt_path:
            print_only(f"Resuming training from checkpoint: {ckpt_path}")
        else:
            print_only("No existing checkpoint found. Starting fresh training.")

    trainer.fit(system, datamodule=datamodule, ckpt_path=ckpt_path)
    print_only("Training finished!")
    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with open(os.path.join(cfg.exp.dir, cfg.exp.name, "best_k_models.json"), "w") as f:
        json.dump(best_k, f, indent=0)

    state_dict = torch.load(checkpoint.best_model_path, weights_only=False)
    system.load_state_dict(state_dict=state_dict["state_dict"])
    system.cpu()

    to_save = system.audio_model.serialize()
    torch.save(to_save, os.path.join(cfg.exp.dir, cfg.exp.name, "best_model.pth"))
    import wandb
    if wandb.run:
        print_only("Closing wandb!")
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conf_dir",
        default="local/conf.yml",
        help="Path to experiment config file",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable auto-resume from last/best checkpoint (enabled by default)",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.conf_dir)
    # Inject resume flag (default True unless --no-resume passed)
    cfg.resume = not args.no_resume

    os.makedirs(os.path.join(cfg.exp.dir, cfg.exp.name), exist_ok=True)
    # Save (possibly updated) config to experiment folder
    OmegaConf.save(cfg, os.path.join(cfg.exp.dir, cfg.exp.name, "config.yaml"))

    train(cfg)
    
