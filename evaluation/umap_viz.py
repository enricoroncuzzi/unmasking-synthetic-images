"""
UMAP 2D embedding visualization on expert ResNet50 models.

Extracts 2048-dim embeddings from each expert's penultimate layer and reduces
them to 2D with UMAP. Produces two types of visualization:

  Type 1 — Per-expert (in-distribution):
    Each expert on its own test set. 2 classes: Real vs Synthetic.
    Shows whether the expert has learned a discriminative feature space.

  Type 2 — Cross-expert:
    Expert SD15 on all 5 test manifests. 6 classes: Real + 5 SD variants.
    Shows how well one expert separates different generative sources.

Produces:
    assets/umap_experts_grid.png    — 1×5 scatter grid (one per expert)
    assets/umap_cross_expert.png    — single scatter, 6 classes

Usage:
    python evaluation/umap_viz.py
    python evaluation/umap_viz.py --device cuda --cross_expert sd15
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import seaborn as sns
import torch
import umap
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import PairedPatchDataset, patch_collate_fn
from data.transforms import get_val_transforms
from models.moe import EXPERT_NAMES, _load_expert, resolve_checkpoint_paths

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_DATASET_ROOT    = os.path.join(PROJECT_ROOT, "dataset")
DEFAULT_MANIFESTS_DIR   = os.path.join(PROJECT_ROOT, "manifests")
DEFAULT_CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
DEFAULT_OUTPUT_DIR      = os.path.join(PROJECT_ROOT, "assets")


# ── Embedding extraction ──────────────────────────────────────────────────────

def extract_embeddings(
    expert: torch.nn.Module,
    manifest_path: str,
    dataset_root: str,
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Runs the expert's penultimate layer (get_embedding) on the full manifest.

    Returns:
        embeddings : [N, 2048] float32 numpy array
        labels     : [N] int numpy array (0=real, 1=synthetic)
    """
    transform = get_val_transforms(resize=512)
    dataset = PairedPatchDataset(
        manifest_path=manifest_path,
        dataset_root=dataset_root,
        transform=transform,
        patch_size=256,
        num_patches=5,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=patch_collate_fn,
    )

    all_emb    = []
    all_labels = []

    expert.eval()
    with torch.no_grad():
        for patches, labels in loader:
            emb = expert.get_embedding(patches.to(device))  # [B, 2048]
            all_emb.append(emb.cpu())
            all_labels.append(labels.cpu())

    embeddings = torch.cat(all_emb).numpy()
    labels     = torch.cat(all_labels).numpy()
    return embeddings, labels


def reduce_umap(
    embeddings: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    """Reduces [N, 2048] embeddings to [N, 2] with UMAP."""
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    return reducer.fit_transform(embeddings)


# ── Plot helpers ──────────────────────────────────────────────────────────────

def plot_umap_grid(
    umap_data: dict[str, tuple[np.ndarray, np.ndarray]],
    output_dir: str,
) -> None:
    """
    Type 1: 1×5 grid of scatter plots, one per expert (in-distribution).
    Each plot has 2 classes: Real (blue) vs Synthetic (orange).

    umap_data : {variant: (embeddings_2d [N,2], labels [N])}
    """
    n = len(EXPERT_NAMES)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))

    palette = {"Real": "#4C72B0", "Synthetic": "#DD8452"}

    for ax, variant in zip(axes, EXPERT_NAMES):
        emb_2d, labels = umap_data[variant]
        label_names = np.where(labels == 0, "Real", "Synthetic")

        for cls, color in palette.items():
            mask = label_names == cls
            ax.scatter(
                emb_2d[mask, 0], emb_2d[mask, 1],
                c=color, label=cls, s=8, alpha=0.5, linewidths=0,
            )

        ax.set_title(variant.upper(), fontsize=12, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        if ax == axes[0]:
            ax.legend(fontsize=9, markerscale=2, loc="best")

    fig.suptitle(
        "UMAP 2D — Expert Embeddings (In-Distribution Test Set)",
        fontsize=14,
    )
    plt.tight_layout()

    out_path = os.path.join(output_dir, "umap_experts_grid.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_umap_cross_expert(
    embeddings_2d: np.ndarray,
    variant_labels: np.ndarray,
    binary_labels: np.ndarray,
    cross_expert_name: str,
    output_dir: str,
) -> None:
    """
    Type 2: single scatter plot with 6 classes.
    Real images from all manifests share label "Real".
    Synthetic images are labelled by their SD variant.

    variant_labels : string array, e.g. ["Real", "SD15", "SD21", ...]
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    unique_labels = ["Real"] + [n.upper() for n in EXPERT_NAMES]
    palette = sns.color_palette("husl", len(unique_labels))
    color_map = dict(zip(unique_labels, palette))

    for lbl in unique_labels:
        mask = variant_labels == lbl
        if mask.sum() == 0:
            continue
        ax.scatter(
            embeddings_2d[mask, 0], embeddings_2d[mask, 1],
            c=[color_map[lbl]], label=lbl,
            s=10, alpha=0.5, linewidths=0,
        )

    ax.set_title(
        f"UMAP 2D — {cross_expert_name.upper()} Expert, All SD Variants\n"
        f"(6 classes: Real + 5 SD sources)",
        fontsize=13,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=10, markerscale=2, loc="best", title="Source")
    sns.despine(left=True, bottom=True)

    out_path = os.path.join(output_dir, "umap_cross_expert.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="UMAP 2D visualization of expert ResNet50 embeddings."
    )
    parser.add_argument("--dataset_root",    default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--manifests_dir",   default=DEFAULT_MANIFESTS_DIR)
    parser.add_argument("--checkpoints_dir", default=DEFAULT_CHECKPOINTS_DIR)
    parser.add_argument("--output_dir",      default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--device",
        default=(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        ),
    )
    parser.add_argument("--batch_size",  type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--cross_expert", default="sd35",
        choices=EXPERT_NAMES,
        help="Which expert to use for the cross-distribution UMAP (default: sd35)",
    )
    parser.add_argument(
        "--n_neighbors", type=int, default=15,
        help="UMAP n_neighbors parameter (default: 15)",
    )
    parser.add_argument(
        "--min_dist", type=float, default=0.1,
        help="UMAP min_dist parameter (default: 0.1)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pl.seed_everything(42)

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  UMAP Visualization ")
    print(f"  device        : {device}")
    print(f"  cross_expert  : {args.cross_expert}")
    print(f"  n_neighbors   : {args.n_neighbors}  min_dist: {args.min_dist}")
    print(f"{'='*60}\n")

    ckpt_paths = resolve_checkpoint_paths(args.checkpoints_dir)

    # ── Type 1: per-expert in-distribution ───────────────────────────────────
    print("=== Type 1: per-expert in-distribution ===")
    umap_data = {}   # {variant: (embeddings_2d, labels)}

    for variant in EXPERT_NAMES:
        print(f"\n[{variant}]")
        print(f"  Loading expert ...", end=" ", flush=True)
        expert = _load_expert(ckpt_paths[variant], device)
        print("OK")

        manifest = os.path.join(args.manifests_dir, f"{variant}_test.csv")
        print(f"  Extracting embeddings ...", end=" ", flush=True)
        embeddings, labels = extract_embeddings(
            expert=expert,
            manifest_path=manifest,
            dataset_root=args.dataset_root,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        print(f"OK  [{embeddings.shape[0]} points, 2048-dim]")

        print(f"  UMAP reduction ...", end=" ", flush=True)
        embeddings_2d = reduce_umap(
            embeddings,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
        )
        print("OK")

        umap_data[variant] = (embeddings_2d, labels)

        del expert
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\nPlotting Type 1 grid...")
    plot_umap_grid(umap_data, args.output_dir)

    # ── Type 2: cross-expert, all variants ───────────────────────────────────
    print(f"\n=== Type 2: {args.cross_expert} expert, all SD variants ===")
    print(f"  Loading expert {args.cross_expert} ...", end=" ", flush=True)
    expert = _load_expert(ckpt_paths[args.cross_expert], device)
    print("OK")

    all_emb    = []
    all_vlabels = []   # string variant label
    all_blabels = []   # binary label (0/1)

    for variant in EXPERT_NAMES:
        manifest = os.path.join(args.manifests_dir, f"{variant}_test.csv")
        print(f"  Extracting [{variant}] ...", end=" ", flush=True)
        embeddings, labels = extract_embeddings(
            expert=expert,
            manifest_path=manifest,
            dataset_root=args.dataset_root,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        print(f"OK  [{embeddings.shape[0]} points]")

        # Real images → "Real", synthetic → variant name (uppercase)
        variant_label = np.where(labels == 0, "Real", variant.upper())
        all_emb.append(embeddings)
        all_vlabels.append(variant_label)
        all_blabels.append(labels)

    del expert
    if device.type == "cuda":
        torch.cuda.empty_cache()

    all_emb_cat     = np.concatenate(all_emb)
    all_vlabels_cat = np.concatenate(all_vlabels)
    all_blabels_cat = np.concatenate(all_blabels)

    # Deduplicate real images: each manifest contributes the same 500 real patches.
    # Keep only the first occurrence of each "Real" point to avoid overplotting.
    real_mask  = all_blabels_cat == 0
    synt_mask  = all_blabels_cat == 1

    # Take real points from first manifest only (first 500 real patches)
    real_indices = np.where(real_mask)[0][:500]
    synt_indices = np.where(synt_mask)[0]

    keep = np.concatenate([real_indices, synt_indices])
    keep.sort()

    emb_cross   = all_emb_cat[keep]
    vlabels_cross = all_vlabels_cat[keep]
    blabels_cross = all_blabels_cat[keep]

    print(f"\n  UMAP reduction on {emb_cross.shape[0]} points ...", end=" ", flush=True)
    emb_cross_2d = reduce_umap(
        emb_cross,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
    )
    print("OK")

    print("Plotting Type 2 cross-expert...")
    plot_umap_cross_expert(
        embeddings_2d=emb_cross_2d,
        variant_labels=vlabels_cross,
        binary_labels=blabels_cross,
        cross_expert_name=args.cross_expert,
        output_dir=args.output_dir,
    )

    print("\nTest complete.\n")


if __name__ == "__main__":
    main()
