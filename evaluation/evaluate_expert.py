"""
Expert evaluation on test set.

Evaluates each of the 5 ResNet50 experts both in-distribution (expert tested on
its own SD variant) and cross-distribution (expert tested on all other variants).

Produces:
    results/roc_experts_in_dist.png     — ROC curves overlay (in-distribution)
    results/cross_dist_heatmap.png      — 5×5 Balanced Accuracy heatmap
    results/expert_results.json         — full metrics dict for T12 aggregation

Usage:
    python evaluation/evaluate_expert.py
    python evaluation/evaluate_expert.py --dataset_root /path/to/dataset --device cuda
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import PairedPatchDataset, patch_collate_fn
from data.transforms import get_val_transforms
from models.moe import EXPERT_NAMES, _load_expert, resolve_checkpoint_paths


# ── Defaults ─────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_DATASET_ROOT = os.path.join(PROJECT_ROOT, "dataset")
DEFAULT_MANIFESTS_DIR = os.path.join(PROJECT_ROOT, "manifests")
DEFAULT_CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results")


# ── Core evaluation ───────────────────────────────────────────────────────────

def evaluate_on_manifest(
    expert: torch.nn.Module,
    manifest_path: str,
    dataset_root: str,
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 4,
    patch_size: int = 256,
    num_patches: int = 5,
) -> dict:
    """
    Runs inference of a single expert on the given manifest's test split.

    Evaluation is patch-by-patch (consistent with training): each image produces
    num_patches independent patches, each treated as a separate sample.

    Args:
        expert        : ExpertModel in eval mode
        manifest_path : path to test CSV (columns: real_path, ai_path)
        dataset_root  : root directory of the image dataset
        device        : inference device
        batch_size    : DataLoader batch size
        num_workers   : DataLoader worker processes
        patch_size    : patch side length (default 256)
        num_patches   : patches per image (default 5)

    Returns:
        dict with keys: auc, ba, precision, recall, f1
        plus raw arrays: y_true, y_prob (for ROC curve plotting)
    """
    transform = get_val_transforms(resize=512)

    dataset = PairedPatchDataset(
        manifest_path=manifest_path,
        dataset_root=dataset_root,
        transform=transform,
        patch_size=patch_size,
        num_patches=num_patches,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=patch_collate_fn,
    )

    all_probs = []
    all_labels = []

    expert.eval()
    with torch.no_grad():
        for patches, labels in loader:
            patches = patches.to(device)
            logits = expert(patches)                      # [B, 2]
            probs = F.softmax(logits, dim=1)[:, 1]        # prob(synthetic)
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

    y_prob = torch.cat(all_probs).numpy()
    y_true = torch.cat(all_labels).numpy()
    y_pred = (y_prob > 0.5).astype(int)

    metrics = {
        "auc":       float(roc_auc_score(y_true, y_prob)),
        "ba":        float(balanced_accuracy_score(y_true, y_pred) * 100),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        # Raw arrays kept for ROC plotting — not serialised to JSON
        "_y_true":   y_true,
        "_y_prob":   y_prob,
    }
    return metrics


# ── Plot helpers ──────────────────────────────────────────────────────────────

def plot_roc_curves(results: dict, output_dir: str) -> None:
    """
    Plot 1: overlaid ROC curves for each expert (in-distribution only).
    Each expert is evaluated on its own test manifest (diagonal of cross-dist matrix).
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    palette = sns.color_palette("husl", len(EXPERT_NAMES))

    for color, name in zip(palette, EXPERT_NAMES):
        m = results[name][name]
        fpr, tpr, _ = roc_curve(m["_y_true"], m["_y_prob"])
        auc = m["auc"]
        label = f"{name.upper()} (AUC = {auc:.3f})"
        ax.plot(fpr, tpr, color=color, lw=2, label=label)

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — In-Distribution (Expert on Own Test Set)", fontsize=13)
    ax.legend(fontsize=10, loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    sns.despine()

    out_path = os.path.join(output_dir, "roc_experts_in_dist.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_cross_dist_heatmap(results: dict, output_dir: str) -> None:
    """
    Plot 2: 5×5 cross-distribution heatmap of Balanced Accuracy.
    Rows = expert (model), Columns = test variant (data).
    Diagonal = in-distribution; off-diagonal = cross-distribution.
    """
    ba_matrix = np.zeros((len(EXPERT_NAMES), len(EXPERT_NAMES)))
    for i, expert_name in enumerate(EXPERT_NAMES):
        for j, test_variant in enumerate(EXPERT_NAMES):
            ba_matrix[i, j] = results[expert_name][test_variant]["ba"]

    labels = [n.upper() for n in EXPERT_NAMES]
    fig, ax = plt.subplots(figsize=(7, 6))

    mask = None  # no masking — show full matrix
    sns.heatmap(
        ba_matrix,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        xticklabels=labels,
        yticklabels=labels,
        vmin=40,
        vmax=100,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_xlabel("Test Variant", fontsize=12)
    ax.set_ylabel("Expert Model", fontsize=12)
    ax.set_title("Cross-Distribution Balanced Accuracy (%)", fontsize=13)
    # Highlight diagonal (in-distribution) with a box
    for k in range(len(EXPERT_NAMES)):
        ax.add_patch(plt.Rectangle((k, k), 1, 1, fill=False, edgecolor="blue", lw=2.5))

    out_path = os.path.join(output_dir, "cross_dist_heatmap.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate expert ResNet50 models on the test set."
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pl.seed_everything(42)

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Expert Evaluation ")
    print(f"  device        : {device}")
    print(f"  dataset_root  : {args.dataset_root}")
    print(f"  manifests_dir : {args.manifests_dir}")
    print(f"{'='*60}\n")

    # ── Load all 5 experts ────────────────────────────────────────────────────
    ckpt_paths = resolve_checkpoint_paths(args.checkpoints_dir)
    experts = {}
    print("Loading experts...")
    for name in EXPERT_NAMES:
        print(f"  {name} ...", end=" ", flush=True)
        experts[name] = _load_expert(ckpt_paths[name], device)
        print("OK")

    # ── Evaluate: expert × test_variant ──────────────────────────────────────
    results = {}   # results[expert_name][test_variant] = metrics dict
    print("\nEvaluating (expert × test variant)...")

    for expert_name in EXPERT_NAMES:
        results[expert_name] = {}
        for test_variant in EXPERT_NAMES:
            tag = "in-dist " if expert_name == test_variant else "cross   "
            print(f"  [{tag}] {expert_name} → {test_variant} ...", end=" ", flush=True)

            manifest = os.path.join(args.manifests_dir, f"{test_variant}_test.csv")
            metrics = evaluate_on_manifest(
                expert=experts[expert_name],
                manifest_path=manifest,
                dataset_root=args.dataset_root,
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            results[expert_name][test_variant] = metrics
            print(f"AUC={metrics['auc']:.3f}  BA={metrics['ba']:.1f}%")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print(f"{'Expert':<12} {'Test':<12} {'AUC':>6} {'BA%':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}  Dist")
    print("─" * 70)
    for expert_name in EXPERT_NAMES:
        for test_variant in EXPERT_NAMES:
            m = results[expert_name][test_variant]
            dist = "IN " if expert_name == test_variant else "   "
            print(
                f"{expert_name:<12} {test_variant:<12} "
                f"{m['auc']:>6.3f} {m['ba']:>6.1f} "
                f"{m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f}  {dist}"
            )
    print("─" * 70)

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    plot_roc_curves(results, args.output_dir)
    plot_cross_dist_heatmap(results, args.output_dir)

    # ── Serialise results to JSON (strip raw arrays) ──────────────────────────
    serialisable = {}
    for expert_name in EXPERT_NAMES:
        serialisable[expert_name] = {}
        for test_variant in EXPERT_NAMES:
            m = results[expert_name][test_variant]
            serialisable[expert_name][test_variant] = {
                k: v for k, v in m.items() if not k.startswith("_")
            }

    json_path = os.path.join(args.output_dir, "expert_results.json")
    with open(json_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"  Saved: {json_path}")

    print("\nTest complete.\n")

if __name__ == "__main__":
    main()
