"""
MoE evaluation on test set.

Evaluates all 4 gating strategies (logit, embedding, image, attention) on the
merged test set and compares them against the 5 individual experts.

Also performs per-variant alpha analysis: for each strategy, measures how the
gating weights distribute across experts when the input belongs to a specific
SD variant — revealing whether the gating correctly attributes the source.

Produces:
    results/confusion_matrices_moe.png   — 2×2 confusion matrix grid
    results/alpha_heatmap.png            — alpha attribution per variant (4 subplots)
    results/ba_comparison.png            — bar chart: experts vs MoE strategies
    results/moe_results.json             — full metrics + alphas dict for T12

Usage:
    python evaluation/evaluate_moe.py
    python evaluation/evaluate_moe.py --device cuda --num_workers 2
"""

import argparse
import json
import os
import sys
import glob as glob_module

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import PairedPatchDataset, patch_collate_fn
from data.transforms import get_val_transforms
from models.gating import build_gating
from models.moe import EXPERT_NAMES, MoEModel, _resolve_checkpoint, resolve_checkpoint_paths

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_DATASET_ROOT    = os.path.join(PROJECT_ROOT, "dataset")
DEFAULT_MANIFESTS_DIR   = os.path.join(PROJECT_ROOT, "manifests")
DEFAULT_CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
DEFAULT_OUTPUT_DIR      = os.path.join(PROJECT_ROOT, "results")

MOE_STRATEGIES = ["logit", "embedding", "image", "attention"]


# ── Model loading ─────────────────────────────────────────────────────────────

def load_moe_model(
    strategy: str,
    expert_ckpt_paths: dict,
    moe_ckpt_path: str,
    device: torch.device,
) -> MoEModel:
    """
    Builds a MoEModel and loads the trained gating weights from a Lightning checkpoint.

    The MoEModel constructor loads frozen expert weights from their own checkpoints.
    Then only the gating network state is overwritten from the MoE checkpoint
    (keys prefixed 'model.gating.' in the Lightning state_dict).

    Args:
        strategy         : gating strategy name
        expert_ckpt_paths: dict expert_name → checkpoint path (or glob)
        moe_ckpt_path    : path (or glob) to MoE Lightning checkpoint
        device           : target device

    Returns:
        MoEModel in eval mode on the target device
    """
    model = MoEModel(
        checkpoint_paths=expert_ckpt_paths,
        gating_strategy=strategy,
    )
    model.to(device)

    resolved = _resolve_checkpoint(moe_ckpt_path)
    moe_ckpt = torch.load(resolved, map_location=device, weights_only=False)

    gating_state = {
        k.removeprefix("model.gating."): v
        for k, v in moe_ckpt["state_dict"].items()
        if k.startswith("model.gating.")
    }

    if not gating_state:
        raise RuntimeError(
            f"Could not extract gating weights from {resolved}.\n"
            f"Expected keys with prefix 'model.gating.' — got: "
            f"{[k for k in moe_ckpt['state_dict'].keys() if 'gating' in k][:5]}"
        )

    model.gating.load_state_dict(gating_state)
    model.eval()
    return model


# ── Core evaluation ───────────────────────────────────────────────────────────

def evaluate_on_manifest(
    model: MoEModel,
    manifest_path: str,
    dataset_root: str,
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 4,
) -> dict:
    """
    Runs MoE inference on a single test manifest.

    Returns metrics dict with raw arrays (_y_true, _y_prob, _alphas) for
    downstream plotting (prefixed _ are stripped before JSON serialisation).
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

    all_probs   = []
    all_labels  = []
    all_alphas  = []

    with torch.no_grad():
        for patches, labels in loader:
            patches = patches.to(device)
            alphas, logits = model(patches)              # [B,5], [B,2]
            probs = F.softmax(logits, dim=1)[:, 1]       # prob(synthetic)
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())
            all_alphas.append(alphas.cpu())

    y_prob   = torch.cat(all_probs).numpy()
    y_true   = torch.cat(all_labels).numpy()
    alphas_np = torch.cat(all_alphas).numpy()           # [N, 5]
    y_pred   = (y_prob > 0.5).astype(int)

    metrics = {
        "auc":       float(roc_auc_score(y_true, y_prob)),
        "ba":        float(balanced_accuracy_score(y_true, y_pred) * 100),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "_y_true":   y_true,
        "_y_prob":   y_prob,
        "_alphas":   alphas_np,
    }
    return metrics


def compute_alpha_matrix(
    model: MoEModel,
    manifests_dir: str,
    dataset_root: str,
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 4,
) -> np.ndarray:
    """
    Computes mean gating weights (alphas) per SD variant on synthetic images only.

    For each test manifest (one per SD variant), runs inference on the synthetic
    images and averages the alpha vectors. Returns a 5×5 matrix where:
        rows    = test variant (what SD variant generated the image)
        columns = expert (gating weight assigned to that expert)

    A strong diagonal means the gating correctly routes each variant to its
    corresponding expert — good attribution.

    Returns:
        alpha_matrix : [5, 5] numpy array of mean alphas per variant
    """
    alpha_matrix = np.zeros((len(EXPERT_NAMES), len(EXPERT_NAMES)))
    transform = get_val_transforms(resize=512)

    for i, variant in enumerate(EXPERT_NAMES):
        manifest_path = os.path.join(manifests_dir, f"{variant}_test.csv")
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

        variant_alphas = []
        all_labels_list = []
        with torch.no_grad():
            for patches, labels in loader:
                patches = patches.to(device)
                alphas, _ = model(patches)
                variant_alphas.append(alphas.cpu())
                all_labels_list.append(labels.cpu())

        alphas_cat = torch.cat(variant_alphas).numpy()    # [N, 5]
        labels_cat = torch.cat(all_labels_list).numpy()  # [N]

        # Average alphas on synthetic images only (label == 1)
        synthetic_mask = labels_cat == 1
        if synthetic_mask.sum() > 0:
            alpha_matrix[i] = alphas_cat[synthetic_mask].mean(axis=0)
        else:
            alpha_matrix[i] = alphas_cat.mean(axis=0)

    return alpha_matrix


# ── Plot helpers ──────────────────────────────────────────────────────────────

def plot_confusion_matrices(results: dict, output_dir: str) -> None:
    """
    Plot 1: 2×2 grid of confusion matrices, one per MoE strategy.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()

    for ax, strategy in zip(axes, MOE_STRATEGIES):
        m = results[strategy]
        y_true = m["_y_true"]
        y_pred = (m["_y_prob"] > 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Real", "Synthetic"],
            yticklabels=["Real", "Synthetic"],
            ax=ax,
        )
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("True", fontsize=11)
        ba = m["ba"]
        auc = m["auc"]
        ax.set_title(f"MoE-{strategy.capitalize()}  (BA={ba:.1f}%  AUC={auc:.3f})", fontsize=11)

    plt.suptitle("Confusion Matrices — MoE Strategies", fontsize=14, y=1.01)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "confusion_matrices_moe.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_alpha_heatmaps(alpha_matrices: dict, output_dir: str) -> None:
    """
    Plot 2: 1×4 subplot with alpha attribution heatmaps per strategy.
    Rows = test variant (input SD source), Columns = expert gating weight.
    Strong diagonal = correct attribution.
    """
    labels = [n.upper() for n in EXPERT_NAMES]
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    for ax, strategy in zip(axes, MOE_STRATEGIES):
        mat = alpha_matrices[strategy]
        sns.heatmap(
            mat,
            annot=True,
            fmt=".2f",
            cmap="YlOrRd",
            vmin=0.0,
            vmax=1.0,
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            linewidths=0.4,
            cbar=ax == axes[-1],
        )
        ax.set_xlabel("Expert Weight (α)", fontsize=10)
        ax.set_ylabel("Test Variant" if ax == axes[0] else "", fontsize=10)
        ax.set_title(f"MoE-{strategy.capitalize()}", fontsize=11)
        # Highlight diagonal
        for k in range(len(EXPERT_NAMES)):
            ax.add_patch(plt.Rectangle((k, k), 1, 1, fill=False, edgecolor="blue", lw=2))

    plt.suptitle("Alpha Attribution Heatmaps (synthetic images only)", fontsize=13, y=1.02)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "alpha_heatmap.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_ba_comparison(
    results: dict,
    expert_results_path: str,
    output_dir: str,
) -> None:
    """
    Plot 3: bar chart comparing cross-distribution average BA of each expert
    against overall BA of each MoE strategy on the merged test set.

    Using cross-distribution BA for experts (mean over all off-diagonal cells)
    puts both axes on the same footing: experts evaluated on unseen variants,
    MoE evaluated on the full merged set. This is the fair comparison that
    motivates the MoE architecture.
    """
    from matplotlib.patches import Patch

    names      = []
    ba_values  = []
    colors     = []

    if os.path.exists(expert_results_path):
        with open(expert_results_path) as f:
            expert_results = json.load(f)
        for name in EXPERT_NAMES:
            # Mean BA across all OTHER variants (off-diagonal of cross-dist matrix)
            cross_bas = [
                expert_results[name][variant]["ba"]
                for variant in EXPERT_NAMES
                if variant != name
            ]
            ba = float(np.mean(cross_bas))
            names.append(name.upper())
            ba_values.append(ba)
            colors.append("#4C72B0")
    else:
        print("  [WARN] expert_results.json not found — skipping expert bars")

    for strategy in MOE_STRATEGIES:
        names.append(f"MoE-{strategy.capitalize()}")
        ba_values.append(results[strategy]["ba"])
        colors.append("#DD8452")

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(names, ba_values, color=colors, edgecolor="white", width=0.6)

    for bar, val in zip(bars, ba_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.axhline(90, color="gray", linestyle="--", lw=1.2)
    ax.set_ylim(40, 105)
    ax.set_ylabel("Balanced Accuracy (%)", fontsize=12)
    ax.set_title(
        "Balanced Accuracy: Expert Cross-Distribution Average vs MoE on Merged Test Set",
        fontsize=12,
    )
    ax.tick_params(axis="x", rotation=20)

    legend_handles = [
        Patch(facecolor="#4C72B0", label="Expert (avg cross-distribution BA)"),
        Patch(facecolor="#DD8452", label="MoE (merged test set BA)"),
    ]
    ax.legend(handles=legend_handles, fontsize=10)
    sns.despine()

    out_path = os.path.join(output_dir, "ba_comparison.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate MoE gating strategies on the test set."
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
    print(f"  MoE Evaluation ")
    print(f"  device        : {device}")
    print(f"  dataset_root  : {args.dataset_root}")
    print(f"  manifests_dir : {args.manifests_dir}")
    print(f"{'='*60}\n")

    expert_ckpt_paths = resolve_checkpoint_paths(args.checkpoints_dir)

    # ── Evaluate all 4 strategies ─────────────────────────────────────────────
    results      = {}   # results[strategy] = merged metrics
    alpha_matrices = {} # alpha_matrices[strategy] = [5, 5] array

    for strategy in MOE_STRATEGIES:
        moe_ckpt_glob = os.path.join(
            args.checkpoints_dir, "moe", strategy, "best-*.ckpt"
        )
        print(f"Loading MoE-{strategy} ...", end=" ", flush=True)
        model = load_moe_model(
            strategy=strategy,
            expert_ckpt_paths=expert_ckpt_paths,
            moe_ckpt_path=moe_ckpt_glob,
            device=device,
        )
        print("OK")

        # Evaluate on all 5 manifests merged (like the flat test set)
        all_probs  = []
        all_labels = []
        print(f"  Inference on test manifests...")
        for variant in EXPERT_NAMES:
            manifest = os.path.join(args.manifests_dir, f"{variant}_test.csv")
            m = evaluate_on_manifest(
                model=model,
                manifest_path=manifest,
                dataset_root=args.dataset_root,
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            all_probs.append(m["_y_prob"])
            all_labels.append(m["_y_true"])
            print(f"    {variant}: AUC={m['auc']:.3f}  BA={m['ba']:.1f}%")

        # Aggregate across all variants → overall metrics
        y_prob_all  = np.concatenate(all_probs)
        y_true_all  = np.concatenate(all_labels)
        y_pred_all  = (y_prob_all > 0.5).astype(int)

        results[strategy] = {
            "auc":       float(roc_auc_score(y_true_all, y_prob_all)),
            "ba":        float(balanced_accuracy_score(y_true_all, y_pred_all) * 100),
            "precision": float(precision_score(y_true_all, y_pred_all, zero_division=0)),
            "recall":    float(recall_score(y_true_all, y_pred_all, zero_division=0)),
            "f1":        float(f1_score(y_true_all, y_pred_all, zero_division=0)),
            # Keep raw for confusion matrix
            "_y_true": y_true_all,
            "_y_prob":  y_prob_all,
        }
        print(
            f"  [{strategy}] AUC={results[strategy]['auc']:.3f}  "
            f"BA={results[strategy]['ba']:.1f}%  "
            f"F1={results[strategy]['f1']:.3f}\n"
        )

        # Alpha attribution matrix (synthetic images only, per variant)
        print(f"  Computing alpha attribution matrix...")
        alpha_matrices[strategy] = compute_alpha_matrix(
            model=model,
            manifests_dir=args.manifests_dir,
            dataset_root=args.dataset_root,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        # Free model memory before loading next
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print(f"{'Strategy':<14} {'AUC':>6} {'BA%':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}")
    print("─" * 60)
    for strategy in MOE_STRATEGIES:
        m = results[strategy]
        print(
            f"{'MoE-'+strategy:<14} "
            f"{m['auc']:>6.3f} {m['ba']:>6.1f} "
            f"{m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f}"
        )
    print("─" * 60)

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    plot_confusion_matrices(results, args.output_dir)
    plot_alpha_heatmaps(alpha_matrices, args.output_dir)

    expert_json = os.path.join(args.output_dir, "expert_results.json")
    plot_ba_comparison(results, expert_json, args.output_dir)

    # ── Serialise JSON ────────────────────────────────────────────────────────
    serialisable = {}
    for strategy in MOE_STRATEGIES:
        m = results[strategy]
        serialisable[strategy] = {
            k: v for k, v in m.items() if not k.startswith("_")
        }
        serialisable[strategy]["alphas"] = {
            variant: alpha_matrices[strategy][i].tolist()
            for i, variant in enumerate(EXPERT_NAMES)
        }

    json_path = os.path.join(args.output_dir, "moe_results.json")
    with open(json_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"  Saved: {json_path}")

    print("\nTest complete.\n")


if __name__ == "__main__":
    main()
