"""
Grad-CAM visualization on expert ResNet50 models.

For each of the 5 experts, selects the first 3 real + 3 synthetic images from
the corresponding test manifest and generates Grad-CAM overlays on layer4[-1].
Target class is always 1 (synthetic) — shows where the model detects
'syntheticness', even on real images.

Produces:
    assets/gradcam_{variant}.png   — 2×3 grid per expert (5 files)
    assets/gradcam_summary.png     — 2×5 summary (1 pair per expert, for README)

Usage:
    python evaluation/gradcam.py
    python evaluation/gradcam.py --device cuda --num_images 3
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.moe import EXPERT_NAMES, _load_expert, resolve_checkpoint_paths

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_DATASET_ROOT    = os.path.join(PROJECT_ROOT, "dataset")
DEFAULT_MANIFESTS_DIR   = os.path.join(PROJECT_ROOT, "manifests")
DEFAULT_CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
DEFAULT_OUTPUT_DIR      = os.path.join(PROJECT_ROOT, "assets")

# ImageNet normalization (must match training transforms)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DISPLAY_SIZE = 256   # resize for Grad-CAM and display


# ── Image helpers ─────────────────────────────────────────────────────────────

def load_and_resize(abs_path: str, size: int = DISPLAY_SIZE) -> np.ndarray:
    """
    Loads an image and resizes to (size, size).
    Returns HWC float32 numpy array in [0, 1].
    """
    img = Image.open(abs_path).convert("RGB").resize((size, size), Image.BILINEAR)
    return np.array(img, dtype=np.float32) / 255.0


def to_tensor(img_01: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Converts HWC float32 [0,1] to normalised CHW tensor [1, 3, H, W].
    Applies ImageNet normalisation to match training preprocessing.
    """
    normalised = (img_01 - IMAGENET_MEAN) / IMAGENET_STD          # HWC
    tensor = torch.from_numpy(normalised).permute(2, 0, 1).float() # CHW
    return tensor.unsqueeze(0).to(device)                          # [1,3,H,W]


# ── Grad-CAM for a single expert ──────────────────────────────────────────────

def generate_gradcam_overlays(
    expert: torch.nn.Module,
    image_paths: list[tuple[str, int]],   # (abs_path, label)
    device: torch.device,
) -> list[tuple[np.ndarray, np.ndarray, int]]:
    """
    Generates Grad-CAM overlays for a list of images.

    Target class is always 1 (synthetic) — consistent across real and synthetic
    images so the heatmap always shows where the model looks for synthetic artifacts.

    Args:
        expert      : ExpertModel in eval mode
        image_paths : list of (absolute_path, label) tuples
        device      : inference device

    Returns:
        List of (original_img_01, cam_overlay, label) tuples
        original_img_01 : HWC float32 [0,1]
        cam_overlay     : HWC uint8 [0,255] — original + heatmap blended
        label           : 0=real, 1=synthetic
    """
    target_layers = [expert.backbone.layer4[-1]]
    # Use the backbone directly — Grad-CAM needs the model that produces
    # the feature maps at the target layer
    cam = GradCAM(model=expert, target_layers=target_layers)
    targets = [ClassifierOutputTarget(1)]   # always target synthetic class

    results = []
    for abs_path, label in image_paths:
        img_01 = load_and_resize(abs_path)
        input_tensor = to_tensor(img_01, device)

        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)  # [1, H, W]

        overlay = show_cam_on_image(img_01, grayscale_cam[0], use_rgb=True)
        results.append((img_01, overlay, label))

    return results


# ── Plot helpers ──────────────────────────────────────────────────────────────

def plot_expert_gradcam(
    overlays: list[tuple[np.ndarray, np.ndarray, int]],
    variant: str,
    output_dir: str,
) -> None:
    """
    2×3 grid for a single expert: row 0 = real images, row 1 = synthetic.
    Each cell shows the Grad-CAM overlay.
    """
    real_overlays = [(img, ov) for img, ov, lbl in overlays if lbl == 0]
    synt_overlays = [(img, ov) for img, ov, lbl in overlays if lbl == 1]

    n = max(len(real_overlays), len(synt_overlays))
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))

    row_labels = ["Real", "Synthetic"]
    for row, (row_label, row_data) in enumerate(
        zip(row_labels, [real_overlays, synt_overlays])
    ):
        for col, (img_01, overlay) in enumerate(row_data):
            ax = axes[row][col]
            ax.imshow(overlay)
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(row_label, fontsize=12, fontweight="bold")
            ax.set_title(f"{row_label} {col+1}", fontsize=9)

    fig.suptitle(
        f"Grad-CAM — Expert ResNet50-{variant.upper()}\n"
        f"(target class: synthetic — layer4[-1])",
        fontsize=13,
    )
    plt.tight_layout()

    out_path = os.path.join(output_dir, f"gradcam_{variant}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_gradcam_summary(
    all_overlays: dict[str, list],
    output_dir: str,
) -> None:
    """
    2×5 summary plot: one real + one synthetic example per expert.
    Row 0 = real, Row 1 = synthetic. Designed for README/Medium.
    """
    fig, axes = plt.subplots(2, len(EXPERT_NAMES), figsize=(4 * len(EXPERT_NAMES), 8))

    for col, variant in enumerate(EXPERT_NAMES):
        overlays = all_overlays[variant]
        real_data = [(img, ov) for img, ov, lbl in overlays if lbl == 0]
        synt_data = [(img, ov) for img, ov, lbl in overlays if lbl == 1]

        for row, (row_label, row_data) in enumerate(
            zip(["Real", "Synthetic"], [real_data, synt_data])
        ):
            ax = axes[row][col]
            if row_data:
                ax.imshow(row_data[0][1])   # first image of this row/expert
            ax.axis("off")
            if row == 0:
                ax.set_title(variant.upper(), fontsize=11, fontweight="bold")
            if col == 0:
                ax.set_ylabel(row_label, fontsize=11, fontweight="bold")

    fig.suptitle(
        "Grad-CAM Summary — Where Each Expert Detects Synthetic Artifacts",
        fontsize=13,
    )
    plt.tight_layout()

    out_path = os.path.join(output_dir, "gradcam_summary.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Manifest helpers ──────────────────────────────────────────────────────────

def select_images_from_manifest(
    manifest_path: str,
    dataset_root: str,
    num_images: int = 3,
) -> list[tuple[str, int]]:
    """
    Selects the first `num_images` real + `num_images` synthetic image paths
    from a manifest CSV. Deterministic (no shuffling).

    Returns list of (absolute_path, label) tuples,
    real images first then synthetic.
    """
    df = pd.read_csv(manifest_path)
    selected = []

    for _, row in df.head(num_images).iterrows():
        real_abs = os.path.join(dataset_root, str(row["real_path"]))
        selected.append((real_abs, 0))

    for _, row in df.head(num_images).iterrows():
        ai_abs = os.path.join(dataset_root, str(row["ai_path"]))
        selected.append((ai_abs, 1))

    return selected


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM visualizations for expert ResNet50 models."
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
    parser.add_argument(
        "--num_images", type=int, default=3,
        help="Number of real and synthetic images per expert (default: 3)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Grad-CAM Visualization ")
    print(f"  device      : {device}")
    print(f"  num_images  : {args.num_images} real + {args.num_images} synthetic per expert")
    print(f"{'='*60}\n")

    ckpt_paths  = resolve_checkpoint_paths(args.checkpoints_dir)
    all_overlays = {}

    for variant in EXPERT_NAMES:
        print(f"[{variant}]")
        print(f"  Loading expert ...", end=" ", flush=True)
        expert = _load_expert(ckpt_paths[variant], device)
        print("OK")

        manifest_path = os.path.join(args.manifests_dir, f"{variant}_test.csv")
        image_paths = select_images_from_manifest(
            manifest_path=manifest_path,
            dataset_root=args.dataset_root,
            num_images=args.num_images,
        )

        print(f"  Generating Grad-CAM ({len(image_paths)} images) ...", end=" ", flush=True)
        overlays = generate_gradcam_overlays(
            expert=expert,
            image_paths=image_paths,
            device=device,
        )
        print("OK")

        all_overlays[variant] = overlays
        plot_expert_gradcam(overlays, variant, args.output_dir)

        del expert
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\nGenerating summary plot...")
    plot_gradcam_summary(all_overlays, args.output_dir)

    print("\nT10 complete.\n")


if __name__ == "__main__":
    main()
