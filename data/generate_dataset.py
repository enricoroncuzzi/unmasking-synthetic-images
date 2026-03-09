"""
generate_dataset.py
-------------------
Forensic image laundering pipeline for synthetic image detection research.

For each real image in REAL_DIR, applies a lightweight img2img pass
(VAE roundtrip + strength=0.05 denoising) using multiple Stable Diffusion
variants to simulate real-world AI laundering techniques.

Supported models:
    - sd15   : Stable Diffusion 1.5  (512px)
    - sd21   : Stable Diffusion 2.1  (768px)
    - sdxlbase: Stable Diffusion XL Base (1024px)
    - sd35   : Stable Diffusion 3.5 Medium (512px)
    - flux   : FLUX.1-schnell (768px)

Usage:
    python generate_dataset.py --model sd15
    python generate_dataset.py --model flux --hf_token YOUR_TOKEN
    python generate_dataset.py --model all --hf_token YOUR_TOKEN
    python generate_dataset.py --model sd15 --dry_run        # test: 10 images, CPU
    python generate_dataset.py --model sd15 --batch_size 4   # custom batch size

Output:
    DATASET_IMGLAUND_<MODEL>/  — laundered images
    file_mappings.json         — maps each real image to its laundered versions
"""

import argparse
import gc
import json
import os
from pathlib import Path

import torch
from PIL import Image
from huggingface_hub import login, snapshot_download

# ── Constants ─────────────────────────────────────────────────────────────────

REAL_DIR = Path("DATASET_IMGREAL")
MAPPING_PATH = Path("file_mappings.json")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LAUNDERING_STRENGTH = 0.05
GUIDANCE_SCALE = 1.0
SAVE_EVERY = 50
DEFAULT_BATCH_SIZE = 8


MODEL_CONFIGS = {
    "sd15": {
        "out_dir": "DATASET_IMGLAUND_SD15",
        "suffix": "__sd15.jpg",
        "mapping_key": "sd15",
        "resolution": 512,
    },
    "sd21": {
        "out_dir": "DATASET_IMGLAUND_SD21",
        "suffix": "__sd21.jpg",
        "mapping_key": "sd21",
        "resolution": 768,
    },
    "sdxlbase": {
        "out_dir": "DATASET_IMGLAUND_SDXLBASE",
        "suffix": "__sdxlbase.jpg",
        "mapping_key": "sdxlbase",
        "resolution": 1024,
    },
    "sd35": {
        "out_dir": "DATASET_IMGLAUND_SD35",
        "suffix": "__sd35.jpg",
        "mapping_key": "sd35",
        "resolution": 512,
    },
    "flux": {
        "out_dir": "DATASET_IMGLAUND_FLUX",
        "suffix": "__flux.jpg",
        "mapping_key": "flux",
        "resolution": 768,
    },
}

# ── Pipeline loaders ───────────────────────────────────────────────────────────

def load_sd15():
    from diffusers import StableDiffusionImg2ImgPipeline
    print("Loading SD 1.5 pipeline...")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=dtype(),
        safety_checker=None,
    ).to(device())
    pipe.set_progress_bar_config(disable=True)
    return pipe


def load_sd21():
    from diffusers import StableDiffusionImg2ImgPipeline
    print("Loading SD 2.1 pipeline...")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "Manojb/stable-diffusion-2-1-base",
        torch_dtype=dtype(),
        safety_checker=None,
    ).to(device())
    pipe.set_progress_bar_config(disable=True)
    return pipe


def load_sdxlbase():
    from diffusers import StableDiffusionXLImg2ImgPipeline, AutoencoderKL
    print("Loading SDXL Base pipeline with fp16-fix VAE...")
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix",
        torch_dtype=dtype(),
    )
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=dtype(),
        vae=vae,
    ).to(device())
    pipe.set_progress_bar_config(disable=True)
    return pipe


def load_sd35():
    from diffusers import StableDiffusion3Img2ImgPipeline
    print("Loading SD 3.5 Medium pipeline...")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium",
        torch_dtype=dtype(),
    )
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)
    return pipe


def load_flux():
    from diffusers import FluxImg2ImgPipeline
    print("Loading FLUX.1-schnell pipeline...")
    pipe = FluxImg2ImgPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype=dtype(),
    )
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)
    return pipe


LOADERS = {
    "sd15": load_sd15,
    "sd21": load_sd21,
    "sdxlbase": load_sdxlbase,
    "sd35": load_sd35,
    "flux": load_flux,
}

# ── Inference ──────────────────────────────────────────────────────────────────

def launder_batch(pipe, imgs: list, model_name: str, resolution: int) -> list:
    """
    Apply laundering pass to a batch of images.
    All images are resized to (resolution, resolution) before inference.
    Returns a list of PIL images.
    """
    imgs_resized = [
        img.convert("RGB").resize((resolution, resolution), Image.LANCZOS)
        for img in imgs
    ]
    kwargs = dict(
        prompt=[""] * len(imgs_resized),
        image=imgs_resized,
        strength=LAUNDERING_STRENGTH,
        guidance_scale=GUIDANCE_SCALE,
        num_inference_steps=20,
    )
    if model_name == "flux":
        kwargs["num_inference_steps"] = 4
        kwargs["guidance_scale"] = 0.0
    return pipe(**kwargs).images

# ── Utilities ──────────────────────────────────────────────────────────────────

def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

def dtype():
    return torch.float16 if torch.cuda.is_available() else torch.float32

def load_mappings() -> dict:
    if MAPPING_PATH.exists():
        with open(MAPPING_PATH, "r") as f:
            return json.load(f)
    return {}


def save_mappings(mappings: dict):
    with open(MAPPING_PATH, "w") as f:
        json.dump(mappings, f, indent=2)


def free_memory(pipe):
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        print(f"VRAM after cleanup: {allocated:.2f} GB allocated")

def chunked(lst: list, n: int):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# ── Main loop ──────────────────────────────────────────────────────────────────

def run_laundering(model_name: str, batch_size: int = DEFAULT_BATCH_SIZE, dry_run: bool = False):
    config = MODEL_CONFIGS[model_name]
    out_dir = Path(config["out_dir"])
    out_dir.mkdir(exist_ok=True, parents=True)

    real_files = sorted([
        p for p in REAL_DIR.iterdir()
        if p.suffix.lower() in IMAGE_EXTS
    ])

    if dry_run:
        real_files = real_files[:10]
        print(f"[DRY RUN] Processing first {len(real_files)} images on {device()}.")
    else:
        print(f"Found {len(real_files)} real images in '{REAL_DIR}'.")

    print(f"Batch size: {batch_size} | Device: {device()}")

    mappings = load_mappings()
    pipe = LOADERS[model_name]()

    processed = 0
    for batch_paths in chunked(real_files, batch_size):
        to_process = []
        for img_path in batch_paths:
            out_path = out_dir / f"{img_path.stem}{config['suffix']}"
            if not out_path.exists():
                to_process.append(img_path)

        if not to_process:
            processed += len(batch_paths)
            continue

        try:
            imgs = [Image.open(p).convert("RGB") for p in to_process]
            results = launder_batch(pipe, imgs, model_name, config["resolution"])

            for img_path, img_out in zip(to_process, results):
                out_path = out_dir / f"{img_path.stem}{config['suffix']}"
                img_out.save(out_path, quality=95)

                rel_name = img_path.name
                if rel_name not in mappings:
                    mappings[rel_name] = {}
                mappings[rel_name][config["mapping_key"]] = str(out_path)

        except Exception as e:
            print(f"[ERROR] batch starting at {to_process[0].name}: {e}")

        processed += len(batch_paths)

        if processed % SAVE_EVERY == 0:
            save_mappings(mappings)
            print(f"[{processed}/{len(real_files)}] Checkpoint saved.")

    save_mappings(mappings)
    free_memory(pipe)
    print(f"Done. Laundered images saved to '{out_dir}'.")

# ── Entry point ────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Forensic image laundering pipeline using Stable Diffusion variants."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_CONFIGS.keys()) + ["all"],
        help="Model to use for laundering. Use 'all' to run all models sequentially.",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="HuggingFace token for gated models (required for FLUX and SD3.5).",
    )
    parser.add_argument(
        "--real_dir",
        type=str,
        default=None,
        help="Path to local real images directory. If not set, downloads from HuggingFace.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Number of images per batch for GPU inference (default: {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--hf_workers",
        type=int,
        default=2,
        help="Number of parallel workers for HuggingFace dataset download (default: 2, increase for faster download on stable connections).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Process only the first 10 images to verify the pipeline works correctly.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.real_dir:
        REAL_DIR = Path(args.real_dir)
    else:
        print("Downloading real images from HuggingFace...")
        snapshot_download(
            repo_id="enricoroncuzzi/unmasking-synthetic-images-dataset",
            repo_type="dataset",
            local_dir="./data",
            allow_patterns=["DATASET_IMGREAL/*"],
            max_workers=args.hf_workers,
        )
        REAL_DIR = Path("./data/DATASET_IMGREAL")

    if not REAL_DIR.exists():
        raise FileNotFoundError(f"Real images directory not found: {REAL_DIR}")

    # HuggingFace login for gated models
    if args.hf_token:
        login(token=args.hf_token)
    elif args.model in ("flux", "sd35", "all"):
        token = os.environ.get("HF_TOKEN")
        if token:
            login(token=token)
        else:
            print("Warning: HF_TOKEN not set. Gated models (Flux, SD3.5) may fail.")

    models_to_run = list(MODEL_CONFIGS.keys()) if args.model == "all" else [args.model]

    for model_name in models_to_run:
        print(f"\n{'='*50}")
        print(f"Running laundering: {model_name.upper()}")
        print(f"{'='*50}")
        run_laundering(model_name, batch_size=args.batch_size, dry_run=args.dry_run)

    print("\nAll laundering complete.")
    print(f"Mappings saved to: {MAPPING_PATH}")
