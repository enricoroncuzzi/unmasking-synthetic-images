"""
— Example Images Gallery curation script.

Run from repo root:
    .venv/bin/python3 demo/examples/_curate.py

Selects 10 images (2 real + 5 synthetic + 3 ambiguous), resizes to ≤512×512,
and saves them to demo/examples/ with the naming convention required by T17.
"""

import csv
import sys
from pathlib import Path

from PIL import Image

# ── repo root / sys.path ──────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from demo.pipeline import MoEPipeline  # noqa: E402

# ── paths ─────────────────────────────────────────────────────────────────────
DATASET_ROOT = REPO_ROOT / "dataset"
MANIFESTS    = REPO_ROOT / "manifests"
OUT_DIR      = Path(__file__).resolve().parent   # demo/examples/

VARIANTS = {
    "sd15":     ("sd15_test.csv",      "sd15"),
    "sd21":     ("sd21_test.csv",      "sd21"),
    "sdxl":     ("sdxlbase_test.csv",  "sdxlbase"),
    "sd35":     ("sd35_test.csv",      "sd35"),
    "flux":     ("flux_test.csv",      "flux"),
}


def _read_csv(csv_path: Path, column: str, limit: int) -> list[Path]:
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(DATASET_ROOT / row[column])
            if len(rows) >= limit:
                break
    return rows


def _resize_save(src: Path, dest: Path, max_side: int = 512, quality: int = 90) -> None:
    img = Image.open(src).convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        ratio = max_side / max(w, h)
        img = img.resize((round(w * ratio), round(h * ratio)), Image.LANCZOS)
    img.save(dest, format="JPEG", quality=quality, optimize=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading MoEPipeline …")
    pipeline = MoEPipeline(device="cpu", strategy="logit")
    print()

    real_images: list[dict] = []
    synthetic_images: dict[str, list[dict]] = {k: [] for k in VARIANTS}
    ambiguous_candidates: list[dict] = []

    # ── sweep synthetic variants (first 30 from each test CSV) ────────────────
    for variant_key, (csv_name, expert_name) in VARIANTS.items():
        csv_path = MANIFESTS / csv_name
        ai_paths = _read_csv(csv_path, "ai_path", 30)
        print(f"[{variant_key}] sweeping {len(ai_paths)} images …")

        for path in ai_paths:
            if not path.exists():
                continue
            img = Image.open(path).convert("RGB")
            result = pipeline.predict(img)

            entry = {
                "path": path,
                "result": result,
                "variant_key": variant_key,
                "expert_name": expert_name,
            }

            # Primary: correct synthetic prediction with high alpha on right expert
            if result["prediction"] == "synthetic":
                alpha_score = result["alpha_weights"][expert_name]
                entry["score"] = alpha_score
                synthetic_images[variant_key].append(entry)

            # Ambiguous: correct prediction but low confidence, or split alphas
            conf = result["confidence"]
            alphas = list(result["alpha_weights"].values())
            alphas_sorted = sorted(alphas, reverse=True)
            alpha_split = (alphas_sorted[0] - alphas_sorted[1]) < 0.15  # two experts close

            if (result["prediction"] == "synthetic"
                    and 0.60 <= conf <= 0.80) or (
                    result["prediction"] == "synthetic" and alpha_split and conf > 0.60):
                ambiguous_candidates.append(entry)

    # ── sweep real images (first 30 from any test CSV) ────────────────────────
    first_csv = MANIFESTS / "sd15_test.csv"
    real_paths = _read_csv(first_csv, "real_path", 30)
    # Deduplicate (all variants share the same real split)
    seen: set[Path] = set()
    unique_real_paths = []
    for p in real_paths:
        if p not in seen:
            seen.add(p)
            unique_real_paths.append(p)

    print(f"[real] sweeping {len(unique_real_paths)} images …")
    for path in unique_real_paths:
        if not path.exists():
            continue
        img = Image.open(path).convert("RGB")
        result = pipeline.predict(img)

        entry = {
            "path": path,
            "result": result,
            "score": result["confidence"] if result["prediction"] == "real" else 0.0,
        }
        real_images.append(entry)

        # Real images can also be ambiguous candidates
        conf = result["confidence"]
        if result["prediction"] == "real" and 0.60 <= conf <= 0.80:
            ambiguous_candidates.append(entry)

    print()

    # ── select best candidates ─────────────────────────────────────────────────

    # 2 real: highest confidence with prediction == "real"
    real_selected = sorted(
        [e for e in real_images if e["result"]["prediction"] == "real"],
        key=lambda e: e["score"],
        reverse=True,
    )[:2]

    if len(real_selected) < 2:
        print(f"WARNING: only {len(real_selected)} real images passed (need 2)")

    # 5 synthetic: 1 per variant, highest alpha on correct expert
    synth_selected: list[dict] = []
    for variant_key in VARIANTS:
        pool = sorted(synthetic_images[variant_key], key=lambda e: e["score"], reverse=True)
        if pool:
            synth_selected.append(pool[0])
        else:
            print(f"WARNING: no correct synthetic predictions found for {variant_key}")

    # 3 ambiguous
    # Sort by confidence ascending (most uncertain first), deduplicate paths
    seen_paths: set[Path] = set()
    unique_ambiguous: list[dict] = []
    for e in sorted(ambiguous_candidates, key=lambda e: e["result"]["confidence"]):
        if e["path"] not in seen_paths:
            seen_paths.add(e["path"])
            unique_ambiguous.append(e)
        if len(unique_ambiguous) == 3:
            break

    # Fallback: if not enough ambiguous with strict criteria, relax to any
    # correct-but-not-top prediction from the synthetic sweep
    if len(unique_ambiguous) < 3:
        print("Not enough strict ambiguous — relaxing to any correct synthetic with conf < 0.90")
        for variant_key in VARIANTS:
            pool = sorted(
                [e for e in synthetic_images[variant_key]
                 if e["result"]["confidence"] < 0.90
                 and e["path"] not in seen_paths],
                key=lambda e: e["result"]["confidence"],
            )
            for e in pool:
                if e["path"] not in seen_paths:
                    seen_paths.add(e["path"])
                    unique_ambiguous.append(e)
                if len(unique_ambiguous) == 3:
                    break
            if len(unique_ambiguous) == 3:
                break

    if len(unique_ambiguous) < 3:
        # Last resort: take lowest-confidence correct predictions overall
        print("Still short on ambiguous — using lowest-confidence correct predictions")
        all_correct = [
            e for variant_entries in synthetic_images.values()
            for e in variant_entries
            if e["path"] not in seen_paths
        ] + [
            e for e in real_images
            if e["result"]["prediction"] == "real" and e["path"] not in seen_paths
        ]
        all_correct_sorted = sorted(all_correct, key=lambda e: e["result"]["confidence"])
        for e in all_correct_sorted:
            if e["path"] not in seen_paths:
                seen_paths.add(e["path"])
                unique_ambiguous.append(e)
            if len(unique_ambiguous) == 3:
                break

    # ── print summary ─────────────────────────────────────────────────────────
    print("=== SELECTED IMAGES ===")
    print("\n-- Real --")
    for e in real_selected:
        print(f"  {e['path'].name}  conf={e['result']['confidence']:.3f}")

    print("\n-- Synthetic --")
    for e in synth_selected:
        print(f"  {e['path'].name}  variant={e['variant_key']}  "
              f"alpha_{e['expert_name']}={e['result']['alpha_weights'][e['expert_name']]:.3f}  "
              f"attributed={e['result']['attributed_source']}")

    print("\n-- Ambiguous --")
    for e in unique_ambiguous:
        print(f"  {e['path'].name}  pred={e['result']['prediction']}  "
              f"conf={e['result']['confidence']:.3f}  alphas={e['result']['alpha_weights']}")

    # ── save ─────────────────────────────────────────────────────────────────
    print("\n=== SAVING ===")
    saved: list[Path] = []

    for idx, e in enumerate(real_selected, start=1):
        dest = OUT_DIR / f"real_portrait_{idx:02d}.jpg"
        _resize_save(e["path"], dest)
        size = dest.stat().st_size // 1024
        print(f"  {dest.name}  ({size} KB)")
        saved.append(dest)

    variant_name_map = {
        "sd15":  "sd15",
        "sd21":  "sd21",
        "sdxl":  "sdxl",
        "sd35":  "sd35",
        "flux":  "flux",
    }
    for e in synth_selected:
        short_name = variant_name_map[e["variant_key"]]
        dest = OUT_DIR / f"{short_name}_laundered_01.jpg"
        _resize_save(e["path"], dest)
        size = dest.stat().st_size // 1024
        print(f"  {dest.name}  ({size} KB)")
        saved.append(dest)

    for idx, e in enumerate(unique_ambiguous, start=1):
        dest = OUT_DIR / f"ambiguous_{idx:02d}.jpg"
        _resize_save(e["path"], dest)
        size = dest.stat().st_size // 1024
        print(f"  {dest.name}  ({size} KB)")
        saved.append(dest)

    print(f"\nTotal saved: {len(saved)} images")

    # ── verify acceptance criteria ────────────────────────────────────────────
    print("\n=== VERIFICATION ===")
    all_ok = True

    jpegs = list(OUT_DIR.glob("*.jpg"))
    if len(jpegs) != 10:
        print(f"FAIL: expected 10 images, found {len(jpegs)}")
        all_ok = False
    else:
        print("OK: 10 images present")

    for p in jpegs:
        img = Image.open(p)
        w, h = img.size
        kb = p.stat().st_size // 1024
        if max(w, h) > 512:
            print(f"FAIL: {p.name} is {w}x{h} (exceeds 512)")
            all_ok = False
        if kb > 200:
            print(f"FAIL: {p.name} is {kb} KB (exceeds 200 KB)")
            all_ok = False

    if all_ok:
        print("OK: all images ≤512×512 and ≤200KB")
    print("DONE." if all_ok else "Some checks failed — review output above.")


if __name__ == "__main__":
    main()
