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