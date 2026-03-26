import time
import torch
from data.dataset import MoEFlatDataset, moe_collate_fn, _collect_manifests
from data.transforms import get_val_transforms
from models.moe import MoEModel, resolve_checkpoint_paths
from torch.utils.data import DataLoader

# ─── CONFIG ──────────────────────────────────────────────────────────────────
DATASET_ROOT   = "dataset"
MANIFESTS_DIR  = "manifests"
CHECKPOINTS_DIR = "checkpoints"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {DEVICE}")
print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")


# ─── LOAD REAL MODEL ─────────────────────────────────────────────────────────
print("\nLoading real expert checkpoints...")
ckpt_paths = resolve_checkpoint_paths(CHECKPOINTS_DIR)
model = MoEModel(
    checkpoint_paths=ckpt_paths,
    gating_strategy="logit"
).to(DEVICE)
model.eval()

total_params = sum(p.numel() for p in model.parameters())
print(f"Model loaded — total params: {total_params/1e6:.1f}M")
print(f"VRAM after model load: {torch.cuda.memory_allocated()/1e9:.2f}GB")

# ─── GPU OOM TEST ─────────────────────────────────────────────────────────────
print("\n--- GPU batch_size OOM test (real checkpoints, num_patches=5) ---")

transform = get_val_transforms(resize=512)
paths = _collect_manifests(MANIFESTS_DIR, "train")
ds = MoEFlatDataset(
    paths, DATASET_ROOT,
    transform=transform,
    patch_size=256,
    num_patches=5
)

for batch_size in [32, 64, 128, 256, 512]:
    try:
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=4,
            pin_memory=True,
            collate_fn=moe_collate_fn
        )
        torch.cuda.reset_peak_memory_stats()
        batch = next(iter(loader))
        images = batch[0].to(DEVICE)

        with torch.no_grad():
            _ = model(images)

        mem_alloc = torch.cuda.memory_allocated() / 1e9
        mem_peak  = torch.cuda.max_memory_allocated() / 1e9
        print(f"batch_size={batch_size:5d}: OK — alloc={mem_alloc:.2f}GB  peak={mem_peak:.2f}GB")

        del images, loader
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    except RuntimeError:
        mem_peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"batch_size={batch_size:5d}: OOM — peak before crash={mem_peak:.2f}GB")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        break

# ─── DATALOADER BENCHMARK (con batch_size ottimale trovato sopra) ─────────────
print("\n--- DataLoader benchmark (usa il batch_size massimo sicuro trovato sopra) ---")
SAFE_BATCH_SIZE = 512  # aggiorna manualmente dopo aver visto l'OOM test
NUM_BATCHES = 20

for num_workers in [8, 16, 24,48]:
    loader = DataLoader(
        ds,
        batch_size=SAFE_BATCH_SIZE,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=moe_collate_fn
    )
    # warmup
    for i, batch in enumerate(loader):
        if i == 3: break

    start = time.time()
    for i, batch in enumerate(loader):
        if i == NUM_BATCHES: break
    elapsed = time.time() - start
    print(f"num_workers={num_workers:3d}: {NUM_BATCHES/elapsed:.2f} batches/sec — {elapsed:.1f}s for {NUM_BATCHES} batches")

# ─── STREAM PARALLELISM BENCHMARK ────────────────────────────────────────────
print("\n--- Expert forward pass: sequential vs parallel streams (vari batch_size) ---")

NUM_FORWARD = 20

for bench_bs in [16, 32, 64, 128, 256, 512]:
    loader_bench = DataLoader(
        ds,
        batch_size=bench_bs,
        num_workers=16,
        pin_memory=True,
        collate_fn=moe_collate_fn
    )
    images = next(iter(loader_bench))[0].to(DEVICE)

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(images)
    torch.cuda.synchronize()

    # Parallel
    start = time.time()
    with torch.no_grad():
        for _ in range(NUM_FORWARD):
            _ = model(images)
    torch.cuda.synchronize()
    parallel_time = time.time() - start

    # Warmup per sequential
    model._streams = None
    with torch.no_grad():
        for _ in range(3):
            _ = model(images)
    torch.cuda.synchronize()

    # Benchmark sequential
    start = time.time()
    with torch.no_grad():
        for _ in range(NUM_FORWARD):
            _ = model(images)
    torch.cuda.synchronize()
    sequential_time = time.time() - start

    # Restore
    model._streams = [torch.cuda.Stream() for _ in model.experts]

    speedup = sequential_time / parallel_time
    print(f"batch_size={bench_bs:5d}: parallel={parallel_time/NUM_FORWARD*1000:.1f}ms  sequential={sequential_time/NUM_FORWARD*1000:.1f}ms  speedup={speedup:.2f}x")
