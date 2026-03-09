# Unmasking Synthetic Images

> A Mixture of Experts framework for synthetic image detection and attribution.

## Dataset

6000 images — 1000 real + 5000 laundered across 5 Stable Diffusion variants.  
Available on HuggingFace: [enricoroncuzzi/unmasking-synthetic-images-dataset](https://huggingface.co/datasets/enricoroncuzzi/unmasking-synthetic-images-dataset)

## Quickstart
```bash
pip install -r requirements.txt
python data/generate_dataset.py --dry_run
```

## Roadmap

- [x] Phase 1 — Dataset generation
- [ ] Phase 2 — ResNet50 expert training
- [ ] Phase 3 — Mixture of Experts
- [ ] Phase 4 — Evaluation & visualization
- [ ] Phase 5 — Gradio demo