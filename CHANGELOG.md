# Changelog

## [0.1.0] - 2026-03-09

### Added
- `data/generate_dataset.py`: forensic img2img laundering pipeline
- `requirements.txt`: minimal direct dependencies
- Dataset published on HuggingFace: `enricoroncuzzi/unmasking-synthetic-images-dataset`
  - 6000 images: 1000 real + 5000 laundered across SD1.5, SD2.1, SDXL Base, SD3.5, FLUX
  - `file_mappings.json`: maps each real image to its 5 laundered counterparts

## [0.2.0] - 2026-03-14

### Added
- `data/download_dataset.py`: downloads dataset from HuggingFace to local disk
- `data/prepare_manifests.py`: generates 15 train/val/test CSV manifests (5 experts × 3 splits, seed=42)
- `data/dataset.py`: manifest-based DataModule with patch extraction (256×256, 5 patches/image)
- `data/transforms.py`: Albumentations augmentation pipeline with ImageNet normalization
- `models/expert.py`: ResNet50 binary classifier with embedding extraction
- `training/train_expert.py`: PyTorch Lightning training script with Hydra + MLflow
- `configs/`: Hydra config files for all 5 experts and global training parameters
- `scripts/train_all_experts.sh`: sequential training script for all 5 experts

### Results
- Trained 5 ResNet50 expert classifiers on RTX 4000 Ada (RunPod)
- val_acc: sd15=1.000, flux=1.000, sd21=0.975, sdxlbase=0.968, sd35=0.966

## [0.3.0] - 2026-03-26

### Added
- `models/gating.py`: 4 gating network architectures
  - `LogitGating`: MLP over concatenated expert logits (~965 params)
  - `EmbeddingGating`: MLP over concatenated expert embeddings, 10240→1024→256→5 (~10.5M params)
  - `ImageGating`: lightweight CNN on raw input patch (~20k params)
  - `AttentionGating`: self-attention over 5 expert logit tokens (~300 params)
- `models/moe.py`: `MoEModel` — loads 5 frozen expert checkpoints + trainable gating network; parallel expert forward pass via per-expert CUDA streams
- `training/train_moe.py`: Lightning training loop for the gating network; logs per-expert alpha means at each validation epoch
- `data/dataset.py`: `MoEFlatDataset`, `MoEDataModule`, `WeightedRandomSampler` for balanced real/synthetic sampling across merged expert manifests
- `configs/config_moe.yaml` + `configs/moe/`: Hydra configs for all 4 gating strategies
- `scripts/train_all_moe.sh`: trains all 4 strategies sequentially
- `profiling.py`: GPU OOM test and throughput benchmark (batch size sweep, num_workers sweep, stream parallelism speedup)

### Results
- Trained 4 MoE gating strategies on RTX 4000 Ada (RunPod), batch_size=512, bf16-mixed

| Strategy  | val_acc | val_loss | Best Epoch |
|-----------|---------|----------|------------|
| logit     | 0.9590  | 0.0994   | 51         |
| embedding | 0.9648  | 0.0954   | 43         |
| image     | 0.7472  | 0.8310   | 18         |
| attention | 0.7314  | 0.9552   | 97         |

## [0.3.1] - 2026-03-26

### Changed
- Reorganized checkpoint directory structure: expert checkpoints moved from `checkpoints/<name>/` to `checkpoints/experts/<name>/`
- Updated `models/moe.py` (`resolve_checkpoint_paths`, docstrings) to reflect new path layout
- Mirrored reorganization on HuggingFace model repo (`enricoroncuzzi/unmasking-synthetic-images-models`): expert folders now live under `experts/`, gating under `moe/`