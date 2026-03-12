"""
LightningDataModule for expert training.

Handles manifest loading, dataset creation, and DataLoader setup.
Each dataset item is a (real_image, laundered_image) pair from which
random patches are extracted for training.
"""

import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pytorch_lightning as pl


class PairedPatchDataset(Dataset):
    """
    Manifest-based dataset that loads (real, laundered) image pairs
    and extracts random patches from each.

    Each __getitem__ returns (patches, labels) where:
        patches: [2 * num_patches, 3, patch_size, patch_size]
        labels:  [2 * num_patches] — 0 for real, 1 for synthetic
    """

    def __init__(
        self,
        manifest_path: str,
        dataset_root: str,
        transform=None,
        patch_size: int = 256,
        num_patches: int = 5,
    ):
        self.dataset_root = dataset_root
        self.transform = transform
        self.patch_size = patch_size
        self.num_patches = num_patches

        df = pd.read_csv(manifest_path)
        required_cols = {"real_path", "ai_path"}
        if not required_cols.issubset(set(df.columns)):
            raise ValueError(
                f"Manifest must contain columns: {required_cols}. "
                f"Found: {list(df.columns)}"
            )

        self.pairs = list(zip(df["real_path"].astype(str), df["ai_path"].astype(str)))

        if len(self.pairs) == 0:
            raise RuntimeError(f"Manifest is empty: {manifest_path}")

    def __len__(self):
        return len(self.pairs)

    def _load_image(self, rel_path: str) -> np.ndarray:
        """Load image as numpy array (HWC, uint8)."""
        abs_path = os.path.join(self.dataset_root, rel_path)
        try:
            img = Image.open(abs_path).convert("RGB")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Image not found: {abs_path}\n"
                f"Check that dataset_root points to the folder containing "
                f"DATASET_IMGREAL/, DATASET_IMGLAUND_SD15/, etc."
            ) from None

        return np.array(img)

    def _apply_transform(self, img: np.ndarray) -> torch.Tensor:
        """Apply Albumentations transform or fallback to raw tensor."""
        if self.transform is not None:
            return self.transform(image=img)["image"]

        # Fallback: HWC uint8 → CHW float32, no normalization
        return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

    def _random_patch(self, img: torch.Tensor) -> torch.Tensor:
        """
        Extract a random patch from a CHW tensor.
        If image is smaller than patch_size, pad with zeros.
        """
        _, H, W = img.shape
        ps = self.patch_size

        if H < ps or W < ps:
            padded = torch.zeros(3, max(H, ps), max(W, ps), dtype=img.dtype)
            padded[:, :H, :W] = img
            img = padded
            _, H, W = img.shape

        top = random.randint(0, H - ps)
        left = random.randint(0, W - ps)
        return img[:, top:top + ps, left:left + ps]

    def __getitem__(self, idx):
        real_rel, ai_rel = self.pairs[idx]

        # Load as numpy (HWC) → transform → tensor (CHW)
        real_img = self._apply_transform(self._load_image(real_rel))
        ai_img = self._apply_transform(self._load_image(ai_rel))

        patches = []
        labels = []

        for _ in range(self.num_patches):
            patches.append(self._random_patch(real_img))
            labels.append(0)

        for _ in range(self.num_patches):
            patches.append(self._random_patch(ai_img))
            labels.append(1)

        patches = torch.stack(patches)  # [2*num_patches, 3, ps, ps]
        labels = torch.tensor(labels, dtype=torch.long)

        return patches, labels


def patch_collate_fn(batch):
    """
    Collate function that concatenates patches across batch items.

    Input:  list of (patches, labels) from each dataset item
            patches: [2*num_patches, 3, ps, ps]
            labels:  [2*num_patches]

    Output: (all_patches, all_labels)
            all_patches: [B * 2 * num_patches, 3, ps, ps]
            all_labels:  [B * 2 * num_patches]
    """
    patches = torch.cat([b[0] for b in batch], dim=0)
    labels = torch.cat([b[1] for b in batch], dim=0)
    return patches, labels


class ExpertDataModule(pl.LightningDataModule):
    """
    LightningDataModule for training a single expert classifier.

    Handles manifest paths, dataset creation, and DataLoader configuration.
    The Trainer calls train_dataloader() and val_dataloader() automatically.
    """

    def __init__(
        self,
        train_manifest: str,
        val_manifest: str,
        test_manifest: str,
        dataset_root: str,
        train_transform=None,
        val_transform=None,
        patch_size: int = 256,
        num_patches: int = 5,
        batch_size: int = 16,
        num_workers: int = 4,
    ):
        super().__init__()
        self.train_manifest = train_manifest
        self.val_manifest = val_manifest
        self.test_manifest = test_manifest
        self.dataset_root = dataset_root
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        """Called by Lightning before training/validation/testing."""
        if stage == "fit" or stage is None:
            self.train_dataset = PairedPatchDataset(
                manifest_path=self.train_manifest,
                dataset_root=self.dataset_root,
                transform=self.train_transform,
                patch_size=self.patch_size,
                num_patches=self.num_patches,
            )
            self.val_dataset = PairedPatchDataset(
                manifest_path=self.val_manifest,
                dataset_root=self.dataset_root,
                transform=self.val_transform,
                patch_size=self.patch_size,
                num_patches=self.num_patches,
            )

        if stage == "test" or stage is None:
            self.test_dataset = PairedPatchDataset(
                manifest_path=self.test_manifest,
                dataset_root=self.dataset_root,
                transform=self.val_transform,
                patch_size=self.patch_size,
                num_patches=self.num_patches,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=patch_collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=patch_collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=patch_collate_fn,
        )