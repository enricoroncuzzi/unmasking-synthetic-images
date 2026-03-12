"""
Downloads the dataset from HuggingFace to a local directory.

This is a one-time data preparation step. Run this before training.
After downloading, set DATASET_ROOT to the output directory.

Authentication (required by HuggingFace rate limits):
    Option 1: export HF_TOKEN=hf_your_token
    Option 2: python data/download_dataset.py --token hf_your_token
    Option 3: huggingface-cli login (interactive, saves token locally)

Usage:
    python data/download_dataset.py                            # default: ./dataset/
    python data/download_dataset.py --output /workspace/data   # custom path (e.g., AWS, RunPod)
    python data/download_dataset.py --token hf_xxx             # explicit token
"""
import os
import argparse
from huggingface_hub import snapshot_download

HF_REPO_ID = "enricoroncuzzi/unmasking-synthetic-images-dataset"


def main():
    parser = argparse.ArgumentParser(
        description="Download dataset from HuggingFace"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dataset",
        help="Local directory to download into (default: dataset/)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token. Also reads HF_TOKEN env var if not provided."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Parallel download workers (default: 2, increase on fast connections)",
    )

    args = parser.parse_args()

    print(f"[INFO] Downloading {HF_REPO_ID} → {args.output}")

    os.makedirs(args.output, exist_ok=True)

    path = snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        local_dir=args.output,
        max_workers=args.workers,
        token=args.token or os.environ.get("HF_TOKEN"),

    )

    print(f"[INFO] Dataset downloaded to: {path}")
    print(f"[INFO] Set DATASET_ROOT to this path before training:")
    print(f"       export DATASET_ROOT={path}")


if __name__ == "__main__":
    main()