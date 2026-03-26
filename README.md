# Unmasking Synthetic Images

> A Mixture of Experts framework for forensic detection and attribution of AI-generated images across 5 Stable Diffusion variants.

## Results

### Phase 2 — Expert classifiers

| Expert    | val_acc | val_loss | Best Epoch |
|-----------|---------|----------|------------|
| SD 1.5    | 1.0000  | 0.0006   | 30         |
| SD 2.1    | 0.9750  | 0.0626   | 49         |
| SDXL Base | 0.9680  | 0.0775   | 20         |
| SD 3.5    | 0.9660  | 0.0872   | 46         |
| FLUX.1    | 1.0000  | 0.0018   | 86         |

### Phase 3 — MoE gating strategies

| Strategy  | val_acc | val_loss | Best Epoch |
|-----------|---------|----------|------------|
| logit     | 0.9590  | 0.0994   | 51         |
| embedding | 0.9648  | 0.0954   | 43         |
| image     | 0.7472  | 0.8310   | 18         |
| attention | 0.7314  | 0.9552   | 97         |

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
bash scripts/train_all_experts.sh

# Train all 4 MoE gating strategies sequentially (requires expert checkpoints)
bash scripts/train_all_moe.sh
```

## Architecture

- **Expert Networks**: 5 ResNet50 classifiers (pretrained on ImageNet1K, fine-tuned for binary forensic classification), each specialized on one SD variant
- **Gating Network**: 4 strategies for weighting expert outputs — `logit` (MLP on expert logits, ~965 params), `embedding` (MLP on expert embeddings, ~10.5M params), `image` (CNN on raw patch, ~20k params), `attention` (self-attention over expert logit tokens, ~300 params)
- **MoE Forward Pass**: all 5 experts run in parallel on separate CUDA streams; only the gating network is trained

## Stack

PyTorch · PyTorch Lightning · Hydra · MLflow · Albumentations · HuggingFace · AWS EC2 · RunPod

## Articles

- [Building a Forensic AI Dataset Across 5 Stable Diffusion Variants (Phase 1)](https://medium.com/@enricoroncuzzi/part-1-building-a-forensic-ai-dataset-across-5-stable-diffusion-variants-sd1-5-to-flux-dfd39f5b50d1)
- [Training Five Specialized Detectors on a Forensic Image Dataset (Phase 2)](https://medium.com/@enricoroncuzzi/phase-2-training-five-specialized-detectors-on-a-forensic-image-dataset-267f59841940)

## Roadmap

- [x] Phase 1 — Forensic dataset generation (6000 images, 5 SD variants)
- [x] Phase 2 — ResNet50 expert training (val_acc 0.966–1.000)
- [x] Phase 3 — Mixture of Experts with 4 gating strategies
- [ ] Phase 4 — Evaluation, Grad-CAM, UMAP visualization
- [ ] Phase 5 — Gradio demo + FastAPI on HuggingFace Spaces
- [ ] Phase 6 — Docker, pytest, packaging