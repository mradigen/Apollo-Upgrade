###
# Author: Kai Li
# Date: 2024-01-22 01:16:22
# Email: lk21@mails.tsinghua.edu.cn
# LastEditTime: 2024-01-24 00:05:10
###
import json
import shutil
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
    """Locate the checkpoint we should resume from.

    Rules (in order):
    1. If ``best_model_checkpoint.pth`` (a copied "best" .ckpt) exists, use it.
    2. Use ``best_k_models.json`` selecting the true best according to saved metric values.
    3. Otherwise, pick the *best* .ckpt inside ``checkpoints/`` by parsing metric from filename
       (supports pattern ``val_loss=...``) or presence of "best" in name.
    4. Fallback to ``last*.ckpt`` then most recent .ckpt by modification time.
    """

    # 1. Stable copied best checkpoint
    stable_best = os.path.join(exp_dir, "best_model_checkpoint.pth")
    if os.path.isfile(stable_best):
        return stable_best

    mode = "min"

    # 2. best_k_models.json
    best_json = os.path.join(exp_dir, "best_k_models.json")
    if os.path.isfile(best_json):
        try:
            with open(best_json) as f:
                data = json.load(f)
            if isinstance(data, dict) and data:
                # filter existing
                items = [(p, v) for p, v in data.items() if os.path.isfile(p)]
                if items:
                    # choose best based on mode
                    reverse = mode == "max"
                    items.sort(key=lambda x: x[1], reverse=reverse)
                    return items[0][0]
        except Exception:
            pass

    # 3. Scan checkpoints directory and infer best
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        return None
    ckpts = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
    if not ckpts:
        return None

    # Prefer explicit 'best'
    best_named = [c for c in ckpts if "best" in os.path.basename(c).lower()]
    if best_named:
        # if multiple, pick newest
        return max(best_named, key=os.path.getmtime)

    # Parse metric from filename pattern epoch=E-val_loss=VAL.ckpt
    parsed = []  # (path, metric value)
    for c in ckpts:
        base = os.path.basename(c)
        if "val_loss=" in base:
            try:
                part = base.split("val_loss=")[1]
                num_str = part.split(".ckpt")[0]
                value = float(num_str)
                parsed.append((c, value))
            except Exception:
                continue
    if parsed:
        reverse = mode == "max"
        parsed.sort(key=lambda x: x[1], reverse=reverse)
        return parsed[0][0]

    # 4. last*.ckpt
    last = [c for c in ckpts if "last" in os.path.basename(c).lower()]
    if last:
        return max(last, key=os.path.getmtime)

    # Fallback newest by mtime
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

    # Callback to persist best_k_models.json after every validation so mid-run interruption keeps latest info
    class BestKModelsWriter(pl.Callback):
        def __init__(self, ckpt_cb: Optional[pl.callbacks.ModelCheckpoint], out_dir: str):
            self.ckpt_cb = ckpt_cb
            self.out_path = os.path.join(out_dir, "best_k_models.json")
            self.best_copy_path = os.path.join(out_dir, "best_model_checkpoint.pth")

        def _write(self):
            if not self.ckpt_cb:
                return
            # best_k_models maps path -> metric tensor
            try:
                best_k = {k: (v.item() if hasattr(v, "item") else float(v)) for k, v in self.ckpt_cb.best_k_models.items()}
            except Exception:
                # Fallback: cast directly
                best_k = {k: float(v) for k, v in getattr(self.ckpt_cb, "best_k_models", {}).items()}
            os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
            with open(self.out_path, "w") as f:
                json.dump(best_k, f, indent=0)
            # Also maintain a stable copy/symlink of current best checkpoint for easy resume
            if best_k:
                # Determine current best according to callback's mode
                mode = getattr(self.ckpt_cb, "mode", "min")
                reverse = mode == "max"
                items = sorted(best_k.items(), key=lambda x: x[1], reverse=reverse)
                best_path = items[0][0]
                if os.path.isfile(best_path):
                    try:
                        # Copy only if different (by name)
                        if not os.path.exists(self.best_copy_path) or os.path.getmtime(best_path) > os.path.getmtime(self.best_copy_path):
                            shutil.copy2(best_path, self.best_copy_path)
                    except Exception:
                        pass

        def on_validation_end(self, trainer, pl_module):  # after each validation epoch
            if getattr(trainer, "is_global_zero", True):
                self._write()

        def on_train_end(self, trainer, pl_module):  # final safeguard
            if getattr(trainer, "is_global_zero", True):
                self._write()

    # Always add writer (only acts if checkpoint callback present)
    callbacks.append(BestKModelsWriter(checkpoint if 'checkpoint' in locals() else None,
                                       os.path.join(cfg.exp.dir, cfg.exp.name)))
        
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
    
