"""
MoE inference pipeline for the demo.

Shared between demo/app.py (Gradio) and demo/api.py (FastAPI).
Loads 5 frozen expert ResNet50 checkpoints + a trained gating network from
HuggingFace Hub (enricoroncuzzi/unmasking-synthetic-images-models) or
from local checkpoints/ if present.

Usage:
    from demo.pipeline import MoEPipeline

    pipeline = MoEPipeline(device="cpu", strategy="logit")

    result = pipeline.predict(pil_image)
    # {"prediction": "synthetic", "confidence": 0.97,
    #  "alpha_weights": {"sd15": 0.02, ...}, "attributed_source": "flux"}

    cam_img = pipeline.gradcam(pil_image)
    # PIL.Image with Grad-CAM overlay on the attributed expert's layer4[-1]
"""

import fnmatch
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Project root: demo/ is one level below the repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from models.moe import EXPERT_NAMES, MoEModel, _resolve_checkpoint  # noqa: E402

HF_REPO_ID = "enricoroncuzzi/unmasking-synthetic-images-models"

# Gating strategy shipped in v1. Logit-only: ~1K params, best attribution,
# CPU-friendly on HF Spaces. Embedding can be re-enabled by adding it here.
_SUPPORTED_STRATEGIES = ("logit",)

# ImageNet normalization — must match training preprocessing in data/transforms.py
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ── HuggingFace Hub helpers ────────────────────────────────────────────────────

def _list_hf_files(repo_id: str) -> list:
    """Returns the full list of files in an HF model repo."""
    from huggingface_hub import list_repo_files
    return list(list_repo_files(repo_id))


def _hf_resolve_and_download(repo_id: str, pattern: str, all_files: list) -> str:
    """
    Finds a single file in an HF repo matching a glob pattern and downloads it.

    Args:
        repo_id   : HuggingFace repo ID
        pattern   : glob pattern relative to repo root, e.g. "experts/sd15/best-*.ckpt"
        all_files : pre-fetched list of all repo files (avoid repeated API calls)

    Returns:
        Local cache path (from huggingface_hub.hf_hub_download)

    Raises:
        FileNotFoundError if no match
        RuntimeError if more than one match
    """
    from huggingface_hub import hf_hub_download

    matches = [f for f in all_files if fnmatch.fnmatch(f, pattern)]
    if len(matches) == 0:
        raise FileNotFoundError(
            f"No file in {repo_id} matching pattern: {pattern}"
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Ambiguous pattern '{pattern}' matched {len(matches)} files in {repo_id}: {matches}"
        )
    return hf_hub_download(repo_id, matches[0])


# ── Checkpoint resolution ──────────────────────────────────────────────────────

def _resolve_checkpoint_paths(
    checkpoints_dir: Optional[str],
    strategies: Tuple[str, ...],
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Builds expert_ckpt_paths and moe_ckpt_paths dicts.

    Local mode: uses glob patterns like checkpoints/experts/sd15/best-*.ckpt
    HF Hub mode: downloads each file and returns local cache paths.

    Args:
        checkpoints_dir : path to local checkpoints/ root, or None
        strategies      : gating strategies to load MoE checkpoints for

    Returns:
        (expert_ckpt_paths, moe_ckpt_paths) — both dicts map name → path string
    """
    local_dir = checkpoints_dir or str(_REPO_ROOT / "checkpoints")
    use_local = os.path.isdir(local_dir)

    if use_local:
        expert_paths = {
            name: os.path.join(local_dir, "experts", name, "best-*.ckpt")
            for name in EXPERT_NAMES
        }
        moe_paths = {
            strategy: os.path.join(local_dir, "moe", strategy, "best-*.ckpt")
            for strategy in strategies
        }
        return expert_paths, moe_paths

    # HF Hub: fetch file listing once, then download each checkpoint
    print(f"Downloading checkpoints from {HF_REPO_ID} ...")
    all_files = _list_hf_files(HF_REPO_ID)

    expert_paths: Dict[str, str] = {}
    for name in EXPERT_NAMES:
        print(f"  Expert [{name}] ...", end=" ", flush=True)
        expert_paths[name] = _hf_resolve_and_download(
            HF_REPO_ID, f"experts/{name}/best-*.ckpt", all_files
        )
        print("OK")

    moe_paths: Dict[str, str] = {}
    for strategy in strategies:
        print(f"  MoE [{strategy}] ...", end=" ", flush=True)
        moe_paths[strategy] = _hf_resolve_and_download(
            HF_REPO_ID, f"moe/{strategy}/best-*.ckpt", all_files
        )
        print("OK")

    return expert_paths, moe_paths


def _load_gating_weights(model: MoEModel, moe_ckpt_path: str, device: torch.device) -> None:
    """
    Loads trained gating weights from a Lightning MoE checkpoint into model.gating.

    Lightning saves the full module under 'state_dict' with prefix 'model.gating.'
    (MoELitModule has self.model = MoEModel and MoEModel has self.gating = ...).
    """
    resolved = _resolve_checkpoint(moe_ckpt_path)
    ckpt = torch.load(resolved, map_location=device, weights_only=False)

    gating_state = {
        k.removeprefix("model.gating."): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model.gating.")
    }

    if not gating_state:
        raise RuntimeError(
            f"No gating weights found in checkpoint: {resolved}\n"
            f"Expected keys prefixed 'model.gating.' — got: "
            f"{[k for k in ckpt['state_dict'] if 'gating' in k][:5]}"
        )

    model.gating.load_state_dict(gating_state)


# ── Image preprocessing ────────────────────────────────────────────────────────

def _preprocess(image: Image.Image) -> torch.Tensor:
    """
    Converts a PIL Image to a normalized [1, 3, 256, 256] float32 tensor.

    Resize short side to 288 (preserving aspect ratio), then center-crop 256.
    This mirrors the training distribution more closely than a direct resize
    to 256×256: training sampled 256 patches from 512px+ images, so the model
    expects near-native pixel statistics rather than squashed content.
    """
    img = image.convert("RGB")

    w, h = img.size
    short = min(w, h)
    new_w = round(w * 288 / short)
    new_h = round(h * 288 / short)
    img = img.resize((new_w, new_h), Image.BILINEAR)

    left = (new_w - 256) // 2
    top = (new_h - 256) // 2
    img = img.crop((left, top, left + 256, top + 256))

    arr = np.array(img, dtype=np.float32) / 255.0        # HWC [0, 1]
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD          # HWC normalized
    tensor = torch.from_numpy(arr).permute(2, 0, 1)       # CHW
    return tensor.unsqueeze(0)                             # [1, 3, 256, 256]


# ── Main pipeline class ────────────────────────────────────────────────────────

class MoEPipeline:
    """
    Encapsulates the full MoE inference pipeline for single-image demo use.

    Loads 5 frozen expert ResNet50 checkpoints and a trained gating network.
    Experts are loaded once and reused across calls. 
    Supports predict() for classification + attribution, and gradcam() for visual explanation.

    Args:
        device          : "cpu" | "cuda" | "auto". Auto selects cuda if available.
        strategy        : "logit" | "embedding". Logit is faster and has better
                          attribution (use as default on HF Spaces CPU tier).
        checkpoints_dir : if set and the directory exists, loads from local disk.
                          Otherwise downloads from HuggingFace Hub.
    """

    def __init__(
        self,
        device: str = "cpu",
        strategy: str = "logit",
        checkpoints_dir: Optional[str] = None,
    ):
        if strategy not in _SUPPORTED_STRATEGIES:
            raise ValueError(
                f"Unsupported strategy '{strategy}'. Choose from: {_SUPPORTED_STRATEGIES}"
            )
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.strategy = strategy

        expert_ckpt_paths, moe_ckpt_paths = _resolve_checkpoint_paths(
            checkpoints_dir, strategies=(strategy,)
        )

        print(f"Loading MoEModel [strategy={strategy}] on {self.device} ...")
        self._model = MoEModel(
            checkpoint_paths=expert_ckpt_paths,
            gating_strategy=strategy,
        )
        _load_gating_weights(self._model, moe_ckpt_paths[strategy], self.device)
        self._model.to(self.device)
        self._model.eval()
        print("Model ready.")

    # ── Public API ─────────────────────────────────────────────────────────────

    def predict(self, image: Image.Image) -> dict:
        """
        Full MoE inference on a single PIL image.

        Steps:
            1. Resize to 256×256 and ImageNet-normalize (no random patching)
            2. Forward pass through MoEModel — experts run under torch.no_grad()
            3. Softmax on final logits → confidence
            4. argmax(alphas) → attributed source (synthetic images only)

        Returns:
            {
                "prediction": "real" | "synthetic",
                "confidence": float,          # in [0, 1], for the predicted class
                "alpha_weights": {            # gating weights per expert
                    "sd15": float,
                    "sd21": float,
                    "sdxlbase": float,
                    "sd35": float,
                    "flux": float,
                },
                "attributed_source": str | None,  # expert name with max alpha, None if real
            }
        """
        x = _preprocess(image).to(self.device)  # [1, 3, 256, 256]

        with torch.no_grad():
            alphas, logits = self._model(x)  # [1, 5], [1, 2]

        probs = F.softmax(logits, dim=1)               # [1, 2]
        pred_class = int(probs[0].argmax().item())     # 0=real, 1=synthetic
        synthetic_prob = probs[0, 1].item()

        prediction = "synthetic" if pred_class == 1 else "real"
        confidence = synthetic_prob if pred_class == 1 else 1.0 - synthetic_prob

        alpha_np = alphas[0].detach().cpu().numpy()
        alpha_weights = {name: float(alpha_np[i]) for i, name in enumerate(EXPERT_NAMES)}

        attributed_source = EXPERT_NAMES[int(alpha_np.argmax())] if prediction == "synthetic" else None

        return {
            "prediction": prediction,
            "confidence": round(float(confidence), 4),
            "alpha_weights": alpha_weights,
            "attributed_source": attributed_source,
        }

    def gradcam(self, image: Image.Image, expert_name: Optional[str] = None) -> Image.Image:
        """
        Grad-CAM heatmap overlay for the attributed expert (or a specified expert).

        If expert_name is None, runs predict() to find the expert with the highest
        alpha weight and uses that expert for visualization.

        Target class is always 1 (synthetic), consistent with evaluation results
        in Phase 4 — this shows where each expert looks for synthetic artifacts.

        Args:
            image       : input PIL image (any size; resized internally to 256×256)
            expert_name : one of EXPERT_NAMES, or None to use the attributed expert

        Returns:
            PIL.Image (256×256 RGB) with the Grad-CAM heatmap blended over the image
        """
        if expert_name is not None and expert_name not in EXPERT_NAMES:
            raise ValueError(f"Unknown expert '{expert_name}'. Available: {EXPERT_NAMES}")

        if expert_name is None:
            result = self.predict(image)
            # For synthetic images use attributed source; for real use argmax of alphas
            if result["attributed_source"] is not None:
                expert_name = result["attributed_source"]
            else:
                expert_name = max(
                    EXPERT_NAMES,
                    key=lambda n: result["alpha_weights"][n],
                )

        expert_idx = EXPERT_NAMES.index(expert_name)
        expert = self._model.experts[expert_idx]
        target_layers = [expert.backbone.layer4[-1]]

        # Prepare numpy image [0,1] for overlay and normalized tensor for GradCAM.
        # Match the same resize-short-side + center-crop used by _preprocess.
        img = image.convert("RGB")
        w, h = img.size
        short = min(w, h)
        new_w = round(w * 288 / short)
        new_h = round(h * 288 / short)
        img = img.resize((new_w, new_h), Image.BILINEAR)
        left = (new_w - 256) // 2
        top = (new_h - 256) // 2
        img = img.crop((left, top, left + 256, top + 256))
        img_np = np.array(img, dtype=np.float32) / 255.0       # HWC [0,1]

        arr = (img_np - _IMAGENET_MEAN) / _IMAGENET_STD
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
        # requires_grad=True is necessary so that backward() can propagate through
        # the frozen expert (whose parameters have requires_grad=False). The gradient
        # w.r.t. activations is computed via the input's computation graph.
        tensor.requires_grad_(True)

        cam = GradCAM(model=expert, target_layers=target_layers)
        targets = [ClassifierOutputTarget(1)]  # always target synthetic class

        grayscale_cam = cam(input_tensor=tensor, targets=targets)  # [1, H, W]
        overlay = show_cam_on_image(img_np, grayscale_cam[0], use_rgb=True)  # HWC uint8

        return Image.fromarray(overlay)
