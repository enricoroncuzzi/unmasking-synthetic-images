"""
Generates train/val/test CSV manifests from file_mappings.json.

Each expert gets 3 CSV files (train, val, test) with columns:
    real_path, ai_path

All experts share the same image split (same real images in train/val/test)
to prevent data leakage when training the MoE gating network later.

Assumes the dataset has been downloaded locally via download_dataset.py.

Usage:
    python data/prepare_manifests.py                                        # default: dataset/file_mappings.json
    python data/prepare_manifests.py --mappings path/to/file_mappings.json  # custom path
    python data/prepare_manifests.py --output manifests/ --seed 42
"""

import json
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


# The 5 expert keys matching file_mappings.json
EXPERT_KEYS = ["sd15", "sd21", "sdxlbase", "sd35", "flux"]

# Default split ratios
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10


def load_mappings(mappings_path: str) -> dict:
    """Load and validate file_mappings.json."""
    with open(mappings_path, "r") as f:
        mappings = json.load(f)

    if len(mappings) == 0:
        raise ValueError(f"file_mappings.json is empty: {mappings_path}")

    # Validate structure: every entry must have 'real' + all expert keys
    first_key = next(iter(mappings))
    entry = mappings[first_key]
    required_keys = {"real"} | set(EXPERT_KEYS)
    missing = required_keys - set(entry.keys())
    if missing:
        raise ValueError(
            f"file_mappings.json missing keys: {missing}. "
            f"Found: {list(entry.keys())}. "
            f"Expected: {sorted(required_keys)}"
        )

    return mappings


def split_indices(n: int, seed: int) -> dict:
    """
    Split n indices into train/val/test using fixed seed.

    Returns dict with keys 'train', 'val', 'test', each containing
    a sorted numpy array of indices.
    """
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)

    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    # test gets the remainder (handles rounding)

    splits = {
        "train": np.sort(indices[:n_train]),
        "val": np.sort(indices[n_train : n_train + n_val]),
        "test": np.sort(indices[n_train + n_val :]),
    }

    # Sanity check: no overlap, full coverage
    all_indices = np.concatenate(list(splits.values()))
    assert len(all_indices) == n, f"Split lost indices: {len(all_indices)} vs {n}"
    assert len(set(all_indices)) == n, "Split has duplicate indices"

    return splits


def generate_manifests(mappings: dict, splits: dict, output_dir: str) -> None:
    """
    Generate CSV manifests for each expert and split.

    Creates files like:
        output_dir/sd15_train.csv
        output_dir/sd15_val.csv
        output_dir/sd15_test.csv
        ...
    """
    # Convert mappings to ordered list (dict preserves insertion order in Python 3.7+)
    image_keys = list(mappings.keys())

    os.makedirs(output_dir, exist_ok=True)

    total_files = 0
    for expert in EXPERT_KEYS:
        for split_name, indices in splits.items():
            rows = []
            for idx in indices:
                entry = mappings[image_keys[idx]]
                rows.append(
                    {"real_path": entry["real"], "ai_path": entry[expert]}
                )

            df = pd.DataFrame(rows)
            filename = f"{expert}_{split_name}.csv"
            filepath = os.path.join(output_dir, filename)
            df.to_csv(filepath, index=False)
            total_files += 1

    return total_files


def print_summary(mappings: dict, splits: dict, output_dir: str) -> None:
    """Print a summary of the generated manifests."""
    n = len(mappings)
    print(f"\n{'='*50}")
    print(f"Manifest Generation Summary")
    print(f"{'='*50}")
    print(f"Total images:    {n}")
    print(f"Experts:         {', '.join(EXPERT_KEYS)}")
    print(f"Split (seed 42): train={len(splits['train'])} | val={len(splits['val'])} | test={len(splits['test'])}")
    print(f"Output dir:      {output_dir}")
    print(f"Files created:   {len(EXPERT_KEYS) * len(splits)} CSVs")
    print()

    # Show file list
    for expert in EXPERT_KEYS:
        for split_name in ["train", "val", "test"]:
            filepath = os.path.join(output_dir, f"{expert}_{split_name}.csv")
            df = pd.read_csv(filepath)
            print(f"  {expert}_{split_name}.csv  →  {len(df)} pairs")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Generate train/val/test CSV manifests from file_mappings.json"
    )
    parser.add_argument(
        "--mappings",
        type=str,
        default="dataset/file_mappings.json",
        help="Path to file_mappings.json (default: dataset/file_mappings.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="manifests",
        help="Output directory for CSV files (default: manifests/)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits (default: 42)",
    )

    args = parser.parse_args()

    # Resolve mappings path (local file or HuggingFace download)
    mappings_path = args.mappings

    # Load and validate
    mappings = load_mappings(mappings_path)
    print(f"[INFO] Loaded {len(mappings)} image entries from {mappings_path}")

    # Split
    splits = split_indices(n=len(mappings), seed=args.seed)

    # Generate CSVs
    generate_manifests(mappings, splits, args.output)

    # Summary
    print_summary(mappings, splits, args.output)

    print("Done.")


if __name__ == "__main__":
    main()
