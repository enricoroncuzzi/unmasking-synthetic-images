# Unmasking Synthetic Images

> A Mixture of Experts framework for forensic detection and attribution of AI-generated images across 5 Stable Diffusion variants.

## Results

| Expert | val_acc | val_loss | Best Epoch |
|--------|---------|----------|------------|
| SD 1.5 | 1.0000 | 0.0006 | 30 |
| SD 2.1 | 0.9750 | 0.0626 | 49 |
| SDXL Base | 0.9680 | 0.0775 | 20 |
| SD 3.5 | 0.9660 | 0.0872 | 46 |
| FLUX.1 | 1.0000 | 0.0018 | 86 |

## Dataset

6000 images — 1000 real + 5000 laundered across 5 Stable Diffusion variants using img2img at strength=0.05.  
Available on HuggingFace: [enricoroncuzzi/unmasking-synthetic-images-dataset](https://huggingface.co/datasets/enricoroncuzzi/unmasking-synthetic-images-dataset)

## Quickstart
```bash
pip install -r requirements.txt

# Download dataset (use --workers 1 to avoid HuggingFace rate limiting)
python data/download_dataset.py --token YOUR_HF_TOKEN --workers 1

# Generate train/val/test manifests
python data/prepare_manifests.py

# Train all 5 experts sequentially
bash train_all_experts.sh
```

## Architecture

- **Expert Networks**: 5 ResNet50 classifiers (pretrained on ImageNet1K, fine-tuned for binary forensic classification), each specialized on one SD variant
- **Gating Network**: dynamically weights expert outputs for detection and attribution (Phase 3)

## Stack

PyTorch · PyTorch Lightning · Hydra · MLflow · Albumentations · HuggingFace · AWS EC2 · RunPod

## Roadmap

- [x] Phase 1 — Forensic dataset generation (6000 images, 5 SD variants)
- [x] Phase 2 — ResNet50 expert training (val_acc 0.966–1.000)
- [ ] Phase 3 — Mixture of Experts with 4 gating strategies
- [ ] Phase 4 — Evaluation, Grad-CAM, UMAP visualization
- [ ] Phase 5 — Gradio demo + FastAPI on HuggingFace Spaces
- [ ] Phase 6 — Docker, pytest, packaging