"""
Training script for expert binary classifiers.

Uses PyTorch Lightning for training, Hydra for configuration,
and MLflow for experiment tracking.

Usage:
    python training/train_expert.py                          # default: sd15
    python training/train_expert.py expert=flux              # train FLUX expert
    python training/train_expert.py expert=sd21 training.lr=1e-3  # override LR
"""

import os
import hydra
from omegaconf import DictConfig

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger

import torch
import torch.nn as nn
import torch.optim as optim

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.dataset import ExpertDataModule
from data.transforms import get_train_transforms, get_val_transforms
from models.expert import ExpertModel


class ExpertLitModule(pl.LightningModule):
    """
    Lightning wrapper for expert training.

    Handles training/validation step logic, loss computation,
    metric logging, and optimizer/scheduler configuration.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()

        self.model = ExpertModel(num_classes=cfg.model.num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.cfg = cfg

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        patches, labels = batch
        logits = self(patches)
        loss = self.criterion(logits, labels)

        acc = (logits.argmax(dim=1) == labels).float().mean()

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        patches, labels = batch
        logits = self(patches)
        loss = self.criterion(logits, labels)

        acc = (logits.argmax(dim=1) == labels).float().mean()

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.cfg.training.lr)

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


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):

    # Reproducibility
    pl.seed_everything(cfg.training.seed, workers=True)

    # Hydra changes working directory — resolve paths relative to original cwd
    orig_cwd = hydra.utils.get_original_cwd()
    dataset_root = os.path.join(orig_cwd, cfg.dataset_root)
    manifests_dir = os.path.join(orig_cwd, cfg.manifests_dir)

    # Transforms
    train_transform = get_train_transforms(
        resize=cfg.expert.resize,
        aug_prob=cfg.data.aug_prob,
        min_downscaling=cfg.data.min_downscaling,
    )
    val_transform = get_val_transforms(resize=cfg.expert.resize)

    # DataModule
    dm = ExpertDataModule(
        train_manifest=os.path.join(manifests_dir, f"{cfg.expert.name}_train.csv"),
        val_manifest=os.path.join(manifests_dir, f"{cfg.expert.name}_val.csv"),
        test_manifest=os.path.join(manifests_dir, f"{cfg.expert.name}_test.csv"),
        dataset_root=dataset_root,
        train_transform=train_transform,
        val_transform=val_transform,
        patch_size=cfg.data.patch_size,
        num_patches=cfg.data.num_patches,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
    )

    # Model
    model = ExpertLitModule(cfg)

    # Logger
    mlflow_logger = MLFlowLogger(
        experiment_name="expert_training",
        tracking_uri=os.path.join(orig_cwd, "experiments", "mlruns"),
        run_name=cfg.expert.name,
    )

    # Callbacks
    checkpoint_cb = ModelCheckpoint(
        dirpath=os.path.join(orig_cwd, "checkpoints", cfg.expert.name),
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best-{epoch}-{val_loss:.4f}-{val_acc:.4f}",
    )

    early_stop_cb = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=cfg.training.es_patience,
    )

    # Enable Tensor Core optimization on NVIDIA GPUs
    torch.set_float32_matmul_precision('medium')

    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        logger=mlflow_logger,
        callbacks=[checkpoint_cb, early_stop_cb],
        deterministic=True,
        log_every_n_steps=10,
    )

    # Train
    trainer.fit(model, datamodule=dm)

    # Print best model path
    print(f"\n[INFO] Best model: {checkpoint_cb.best_model_path}")
    print(f"[INFO] Best val_loss: {checkpoint_cb.best_model_score:.4f}")


if __name__ == "__main__":
    main()