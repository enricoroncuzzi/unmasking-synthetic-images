"""
Training script for the MoE gating network (Phase 3).

Loads 5 frozen expert checkpoints, trains only the gating network
using the merged MoE dataset. Supports 4 gating strategies via Hydra.

Usage:
    python training/train_moe.py                              # default: logit
    python training/train_moe.py moe=embedding                # embedding gating
    python training/train_moe.py moe=attention training.lr=5e-5
"""

import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger

import hydra
from omegaconf import DictConfig

# PyTorch 2.6 changed torch.load default to weights_only=True, breaking omegaconf
# objects saved via save_hyperparameters(). Patch to restore legacy behavior for
# trusted local checkpoints.
_orig_torch_load = torch.load
torch.load = lambda *a, **kw: _orig_torch_load(*a, **{**kw, "weights_only": False})

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.dataset import MoEDataModule
from data.transforms import get_train_transforms, get_val_transforms
from models.moe import MoEModel, resolve_checkpoint_paths


class MoELitModule(pl.LightningModule):
    """
    Lightning wrapper for MoE gating network training.

    Experts are frozen inside MoEModel — only the gating network
    parameters are passed to the optimizer.

    Logs per-step and per-epoch: train_loss, train_acc, val_loss, val_acc.
    Alpha statistics (mean per expert) are logged at validation time
    to monitor gating behavior during training.
    """

    def __init__(self, cfg: DictConfig, checkpoint_paths: dict[str, str]):
        super().__init__()
        self.save_hyperparameters(ignore=["checkpoint_paths"])
        self.cfg = cfg

        self.model = MoEModel(
            checkpoint_paths=checkpoint_paths,
            gating_strategy=cfg.moe.gating_strategy,
        )

        self.criterion = nn.CrossEntropyLoss()

        # Accumulate alphas during validation for epoch-level logging
        self._val_alphas = []

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        patches, labels = batch
        alphas, logits = self.model(patches)
        loss = self.criterion(logits, labels)
        acc = (logits.argmax(dim=1) == labels).float().mean()

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        patches, labels = batch
        alphas, logits = self.model(patches)
        loss = self.criterion(logits, labels)
        acc = (logits.argmax(dim=1) == labels).float().mean()

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

        # Accumulate alphas for epoch-level mean logging
        self._val_alphas.append(alphas.detach().cpu())

        return loss

    def on_validation_epoch_end(self):
        """
        Logs the mean alpha per expert over the full validation set.

        This is the primary signal for monitoring gating behavior:
        - Balanced alphas (~0.2 each) suggest the gating is not specializing
        - A dominant expert (alpha >> 0.2) suggests strong routing
        - Evolution across epochs reveals when the gating starts to specialize
        """
        if not self._val_alphas:
            return

        all_alphas = torch.cat(self._val_alphas, dim=0)  # [N, num_experts]
        mean_alphas = all_alphas.mean(dim=0)              # [num_experts]

        expert_names = self.model.expert_names
        for i, name in enumerate(expert_names):
            self.log(f"alpha_mean/{name}", mean_alphas[i], on_epoch=True)

        self._val_alphas.clear()

    def configure_optimizers(self):
        # Only gating network parameters — experts are frozen
        optimizer = optim.Adam(
            self.model.trainable_parameters(),
            lr=self.cfg.training.lr,
        )

        if self.cfg.training.scheduler == "reduce_on_plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                patience=self.cfg.training.lr_patience,
                factor=0.1,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                },
            }

        elif self.cfg.training.scheduler == "cosine_warm_restarts":
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=5, T_mult=2
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}

        return optimizer


@hydra.main(config_path="../configs", config_name="config_moe", version_base=None)
def main(cfg: DictConfig):

    # Reproducibility
    pl.seed_everything(cfg.training.seed, workers=True)

    # Hydra changes cwd — resolve all paths relative to project root
    orig_cwd = hydra.utils.get_original_cwd()
    dataset_root = os.path.join(orig_cwd, cfg.dataset_root)
    manifests_dir = os.path.join(orig_cwd, cfg.manifests_dir)
    checkpoints_dir = os.path.join(orig_cwd, cfg.checkpoints_dir)

    # Resolve expert checkpoint paths (glob patterns)
    checkpoint_paths = resolve_checkpoint_paths(checkpoints_dir)

    # Transforms — use expert resize from moe config
    train_transform = get_train_transforms(
        resize=cfg.moe.resize,
        aug_prob=cfg.data.aug_prob,
        min_downscaling=cfg.data.min_downscaling,
    )
    val_transform = get_val_transforms(resize=cfg.moe.resize)

    # DataModule
    dm = MoEDataModule(
        manifests_dir=manifests_dir,
        dataset_root=dataset_root,
        train_transform=train_transform,
        val_transform=val_transform,
        patch_size=cfg.data.patch_size,
        num_patches=cfg.data.num_patches,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
    )

    # Model
    model = MoELitModule(cfg=cfg, checkpoint_paths=checkpoint_paths)

    # Logger
    mlflow_logger = MLFlowLogger(
        experiment_name="moe_training",
        tracking_uri=os.path.join(orig_cwd, "experiments", "mlruns"),
        run_name=cfg.moe.gating_strategy,
    )

    # Callbacks
    ckpt_dir = os.path.join(orig_cwd, "checkpoints", "moe", cfg.moe.gating_strategy)

    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best-{epoch}-{val_loss:.4f}-{val_acc:.4f}",
    )

    last_ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="last",
        save_last=True,
        every_n_epochs=1,
    )

    early_stop_cb = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=cfg.training.es_patience,
    )

    # Tensor Core optimization
    torch.set_float32_matmul_precision("medium")

    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        logger=mlflow_logger,
        callbacks=[checkpoint_cb, last_ckpt_cb, early_stop_cb],
        deterministic=cfg.training.deterministic,
        precision=cfg.training.precision,
        gradient_clip_val=1.0,
        log_every_n_steps=1,
    )

    last_ckpt_path = os.path.join(ckpt_dir, "last.ckpt")
    resume_from = last_ckpt_path if os.path.exists(last_ckpt_path) else None
    if resume_from:
        print(f"[INFO] Resuming from checkpoint: {resume_from}")

    trainer.fit(model, datamodule=dm, ckpt_path=resume_from)

    print(f"\n[INFO] Best model: {checkpoint_cb.best_model_path}")
    print(f"[INFO] Best val_loss: {checkpoint_cb.best_model_score:.4f}")


if __name__ == "__main__":
    main()