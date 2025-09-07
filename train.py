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



def _find_resume_checkpoint(
    exp_dir: str,
    monitor_mode: Optional[str] = "min",
    preference: str = "auto",
) -> Optional[str]:
    """Locate a suitable checkpoint to resume from with robust heuristics.

    preference:
      - 'auto' (default): favor newest last*.ckpt for continuity; use best only if clearly newer.
      - 'last': force newest last*.ckpt.
      - 'best': force best according to metric (best_k_models.json or *best* filename).

    Strategy order (auto):
      1. Scan checkpoints.
      2. Identify newest last*.ckpt (continuity candidate).
      3. If best_k_models.json exists and references existing paths, compute best metric.
         Use it only if no last exists or best is newer/equal mtime.
      4. Else fallback to newest last*.ckpt.
      5. Else explicit *best* filenames.
      6. Else highest epoch=E-... number.
      7. Else newest by mtime.
    """

    def _select_best(d: Dict[str, float]) -> Optional[str]:
        if not d:
            return None
        reverse = (monitor_mode == "max")
        sorted_items = sorted(d.items(), key=lambda kv: kv[1], reverse=reverse)
        for path, _ in sorted_items:
            if os.path.isfile(path):
                return path
        return None

    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        return None
    ckpts = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
    if not ckpts:
        return None

    def _is_last(name: str) -> bool:
        base = os.path.basename(name).lower()
        return base.startswith("last") or base == "last.ckpt"

    last_ckpts = [c for c in ckpts if _is_last(c)]
    newest_last = max(last_ckpts, key=os.path.getmtime) if last_ckpts else None

    best_json = os.path.join(exp_dir, "best_k_models.json")
    best_from_json = None
    if os.path.isfile(best_json):
        try:
            with open(best_json) as f:
                data = json.load(f)
            if isinstance(data, dict) and data:
                parsed = {}
                for k, v in data.items():
                    if os.path.isfile(k):
                        try:
                            parsed[k] = float(v)
                        except Exception:
                            continue
                best_from_json = _select_best(parsed)
        except Exception:
            best_from_json = None

    # preference handling shortcuts
    if preference == "last" and newest_last:
        return newest_last
    if preference == "best" and best_from_json:
        return best_from_json

    # auto logic
    if preference == "auto":
        if best_from_json and newest_last:
            # compare mtimes; pick best only if its file is not older (avoid rewinding)
            if os.path.getmtime(best_from_json) >= os.path.getmtime(newest_last):
                return best_from_json
            return newest_last
        if best_from_json:
            return best_from_json
        if newest_last:
            return newest_last

    # Named best ckpts (rare if not using explicit naming)
    best_named = [c for c in ckpts if "best" in os.path.basename(c).lower()]
    if best_named:
        return max(best_named, key=os.path.getmtime)

    # Try epoch pattern
    def _epoch_num(path: str) -> Optional[int]:
        base = os.path.basename(path)
        if "epoch=" in base:
            try:
                seg = base.split("epoch=")[1]
                num = seg.split("-")[0]
                return int(num)
            except Exception:
                return None
        return None
    with_epochs = [(c, _epoch_num(c)) for c in ckpts]
    valid_epochs = [c for c, e in with_epochs if e is not None]
    if valid_epochs:
        return max(valid_epochs, key=lambda p: _epoch_num(p))

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

        def on_validation_end(self, trainer, pl_module):  # after each validation epoch
            if getattr(trainer, "is_global_zero", True):
                self._write()

        def on_train_end(self, trainer, pl_module):  # final safeguard
            if getattr(trainer, "is_global_zero", True):
                self._write()

    # Always add writer (only acts if checkpoint callback present)
    callbacks.append(BestKModelsWriter(checkpoint if 'checkpoint' in locals() else None,
                                       os.path.join(cfg.exp.dir, cfg.exp.name)))

    # Callback to export serialized audio model whenever a new best checkpoint is achieved
    class ExportBestSerialized(pl.Callback):
        def __init__(self, ckpt_cb: Optional[pl.callbacks.ModelCheckpoint], out_dir: str, monitor_mode: str = "min"):
            self.ckpt_cb = ckpt_cb
            self.out_dir = out_dir
            self.monitor_mode = monitor_mode
            self.best_metric: Optional[float] = None

        def _is_improvement(self, current: float) -> bool:
            if self.best_metric is None:
                return True
            if self.monitor_mode == "min":
                return current < self.best_metric
            return current > self.best_metric

        def on_validation_end(self, trainer, pl_module):
            if not self.ckpt_cb:
                return
            if not getattr(trainer, "is_global_zero", True):
                return
            # Determine current best metric from callback
            try:
                # best_model_score is a tensor
                score = self.ckpt_cb.best_model_score
                if score is None:
                    return
                metric_val = float(score.item() if hasattr(score, "item") else score)
            except Exception:
                return
            if self._is_improvement(metric_val):
                # Update best and export
                self.best_metric = metric_val
                try:
                    # Load the best checkpoint path's state dict to ensure consistent export
                    best_path = self.ckpt_cb.best_model_path
                    if best_path and os.path.isfile(best_path):
                        state_dict = torch.load(best_path, weights_only=False)
                        # Temporarily move to CPU for serialization if needed
                        pl_module.load_state_dict(state_dict['state_dict'])
                        pl_module.cpu()
                        serialized = pl_module.audio_model.serialize()
                        out_file = os.path.join(self.out_dir, "best_model_checkpoint.pth")
                        torch.save(serialized, out_file)
                        print_only(f"Exported new best serialized model to {out_file} (metric={metric_val:.6f})")
                except Exception as e:
                    print_only(f"Failed to export best model: {e}")

    monitor_mode = getattr(cfg.checkpoint, 'mode', 'min') if cfg.get('checkpoint') else 'min'
    callbacks.append(ExportBestSerialized(checkpoint if 'checkpoint' in locals() else None,
                                          os.path.join(cfg.exp.dir, cfg.exp.name),
                                          monitor_mode))
        
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
        monitor_mode = None
        if cfg.get("checkpoint") and hasattr(cfg.checkpoint, "mode"):
            monitor_mode = cfg.checkpoint.mode
        preference = getattr(cfg, "resume_preference", "auto")
        ckpt_path = _find_resume_checkpoint(exp_dir, monitor_mode=monitor_mode, preference=preference)
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

