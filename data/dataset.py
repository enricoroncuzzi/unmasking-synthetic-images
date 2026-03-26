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
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
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
    
     
 
EXPERT_NAMES = ["sd15", "sd21", "sdxlbase", "sd35", "flux"]
 
 
class MoEFlatDataset(Dataset):
    """
    Flat dataset for MoE gating network training.
 
    Merges the per-expert manifest CSVs that were generated in Phase 2
    (seed=42, same real-image split across all experts) into a single
    unified dataset. Each item is a single image path with a binary label.
 
    Unlike PairedPatchDataset, items here are individual images rather
    than (real, synthetic) pairs. This is intentional: the gating network
    sees a single patch per forward pass and must route it correctly.
 
    The same real images appear once per expert manifest (5 experts), so
    the merged dataset contains duplicates on the real side. This is
    handled by WeightedRandomSampler in MoEDataModule, which ensures the
    DataLoader sees an equal number of real and synthetic samples per batch
    regardless of raw class frequencies.
 
    Args:
        manifest_paths : list of CSV paths, one per expert per split
        dataset_root   : root directory containing DATASET_IMGREAL/, etc.
        transform      : Albumentations transform (or None)
        patch_size     : size of the random square patch to extract
        num_patches    : number of patches returned per image
    """
 
    def __init__(
        self,
        manifest_paths: list[str],
        dataset_root: str,
        transform=None,
        patch_size: int = 256,
        num_patches: int = 5,
    ):
        self.dataset_root = dataset_root
        self.transform = transform
        self.patch_size = patch_size
        self.num_patches = num_patches
 
        # Merge all manifests: collect (image_path, label) rows
        # real_path → label 0, ai_path → label 1
        records: list[tuple[str, int]] = []
        for csv_path in manifest_paths:
            df = pd.read_csv(csv_path)
            required = {"real_path", "ai_path"}
            if not required.issubset(set(df.columns)):
                raise ValueError(
                    f"Manifest {csv_path} must contain columns: {required}. "
                    f"Found: {list(df.columns)}"
                )
            for _, row in df.iterrows():
                records.append((str(row["real_path"]), 0))
                records.append((str(row["ai_path"]), 1))
 
        if not records:
            raise RuntimeError("All manifests are empty — nothing to load.")
 
        self.records = records
 
        # Pre-compute per-item labels for WeightedRandomSampler
        self.labels = [label for _, label in self.records]
 
    def __len__(self) -> int:
        return len(self.records)
 
    def get_labels(self) -> list[int]:
        """
        Returns the label for every item in the dataset.
        Used by MoEDataModule to build the WeightedRandomSampler.
        """
        return self.labels
 
    def _load_image(self, rel_path: str) -> np.ndarray:
        abs_path = os.path.join(self.dataset_root, rel_path)
        try:
            img = Image.open(abs_path).convert("RGB")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Image not found: {abs_path}\n"
                f"Check dataset_root: {self.dataset_root}"
            ) from None
        return np.array(img)
 
    def _apply_transform(self, img: np.ndarray) -> torch.Tensor:
        if self.transform is not None:
            return self.transform(image=img)["image"]
        return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
 
    def _random_patch(self, img: torch.Tensor) -> torch.Tensor:
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
 
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            patches : [num_patches, 3, patch_size, patch_size]
            labels  : [num_patches]  — same label repeated (all same image)
        """
        rel_path, label = self.records[idx]
        img = self._apply_transform(self._load_image(rel_path))
 
        patches = torch.stack([self._random_patch(img) for _ in range(self.num_patches)])
        labels = torch.full((self.num_patches,), label, dtype=torch.long)
 
        return patches, labels
 
 
def moe_collate_fn(batch):
    """
    Collate function for MoEFlatDataset.
 
    Input:  list of (patches, labels)
            patches : [num_patches, 3, ps, ps]
            labels  : [num_patches]
 
    Output: (all_patches, all_labels)
            all_patches : [B * num_patches, 3, ps, ps]
            all_labels  : [B * num_patches]
 
    Note: unlike Phase 2's patch_collate_fn (factor 2*num_patches per item),
    here each item contributes num_patches rows — the real/synthetic split
    is handled at the dataset level, not within a single item.
    """
    patches = torch.cat([b[0] for b in batch], dim=0)
    labels = torch.cat([b[1] for b in batch], dim=0)
    return patches, labels
 
 
def _build_weighted_sampler(dataset: MoEFlatDataset) -> WeightedRandomSampler:
    """
    Builds a WeightedRandomSampler that balances real (0) and synthetic (1)
    samples regardless of their raw frequencies in the merged dataset.
 
    The merged dataset contains 5× more real entries than synthetic for each
    expert (because the same 800 real images appear in all 5 expert manifests),
    so without resampling the gating network would see a skewed distribution.
 
    Strategy: assign weight = 1 / class_count to each sample, so the
    effective sampling probability is uniform across classes.
    """
    labels = dataset.get_labels()
    label_tensor = torch.tensor(labels, dtype=torch.long)
 
    class_counts = torch.bincount(label_tensor)           # [num_classes]
    class_weights = 1.0 / class_counts.float()            # inverse frequency
    sample_weights = class_weights[label_tensor]          # per-sample weight
 
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
 
 
def _collect_manifests(manifests_dir: str, split: str) -> list[str]:
    """
    Collects the manifest CSV paths for all 5 experts for a given split.
 
    Expected naming convention (same as Phase 2):
        {manifests_dir}/{expert_name}_{split}.csv
    where split ∈ {"train", "val", "test"}.
 
    Args:
        manifests_dir : directory containing all Phase 2 manifest CSVs
        split         : "train", "val", or "test"
 
    Returns:
        List of 5 absolute CSV paths, one per expert.
 
    Raises:
        FileNotFoundError if any expected manifest is missing.
    """
    paths = []
    for name in EXPERT_NAMES:
        path = os.path.join(manifests_dir, f"{name}_{split}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing manifest for expert '{name}' split '{split}': {path}\n"
                f"Run data/prepare_manifests.py (Phase 2) before Phase 3 training."
            )
        paths.append(path)
    return paths
 
 
class MoEDataModule(pl.LightningDataModule):
    """
    LightningDataModule for training the MoE gating network.
 
    Reuses the per-expert manifest CSVs generated in Phase 2 (seed=42).
    Merges all 5 expert sub-datasets into a single flat dataset and applies
    WeightedRandomSampler to maintain real/synthetic balance across batches.
 
    Key design decisions:
    - Manifests are not regenerated: the same seed=42 splits are reused
      to prevent data leakage between Phase 2 (expert training) and
      Phase 3 (gating network training).
    - Items are individual images (not pairs): each __getitem__ returns
      num_patches patches from a single image with a shared binary label.
    - Real/synthetic balance: WeightedRandomSampler compensates for the
      5× over-representation of real images in the merged manifest.
    - Val/test use a standard DataLoader (no sampler) for deterministic
      evaluation metrics.
 
    Args:
        manifests_dir  : directory containing Phase 2 manifest CSVs
        dataset_root   : root directory containing DATASET_IMGREAL/, etc.
        train_transform: Albumentations transform for training
        val_transform  : Albumentations transform for val/test
        patch_size     : size of random square patches (default: 256)
        num_patches    : patches extracted per image per __getitem__ (default: 5)
        batch_size     : DataLoader batch size (default: 16)
        num_workers    : DataLoader worker count (default: 4)
    """
 
    def __init__(
        self,
        manifests_dir: str,
        dataset_root: str,
        train_transform=None,
        val_transform=None,
        patch_size: int = 256,
        num_patches: int = 5,
        batch_size: int = 16,
        num_workers: int = 4,
    ):
        super().__init__()
        self.manifests_dir = manifests_dir
        self.dataset_root = dataset_root
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.batch_size = batch_size
        self.num_workers = num_workers
 
    def setup(self, stage=None):
        """
        Instantiates datasets for each stage.
        Called by Lightning before fit/validate/test.
        """
        if stage == "fit" or stage is None:
            self.train_dataset = MoEFlatDataset(
                manifest_paths=_collect_manifests(self.manifests_dir, "train"),
                dataset_root=self.dataset_root,
                transform=self.train_transform,
                patch_size=self.patch_size,
                num_patches=self.num_patches,
            )
            self.val_dataset = MoEFlatDataset(
                manifest_paths=_collect_manifests(self.manifests_dir, "val"),
                dataset_root=self.dataset_root,
                transform=self.val_transform,
                patch_size=self.patch_size,
                num_patches=self.num_patches,
            )
 
        if stage == "test" or stage is None:
            self.test_dataset = MoEFlatDataset(
                manifest_paths=_collect_manifests(self.manifests_dir, "test"),
                dataset_root=self.dataset_root,
                transform=self.val_transform,
                patch_size=self.patch_size,
                num_patches=self.num_patches,
            )
 
    def train_dataloader(self) -> DataLoader:
        sampler = _build_weighted_sampler(self.train_dataset)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,          # shuffle=True incompatible with sampler
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=moe_collate_fn,
        )
 
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=moe_collate_fn,
        )
 
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=moe_collate_fn,
        )
