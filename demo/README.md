---
title: Unmasking Synthetic Images Demo
colorFrom: gray
colorTo: red
sdk: gradio
sdk_version: "5.25.0"
app_file: app.py
pinned: false
license: mit
models:
  - enricoroncuzzi/unmasking-synthetic-images-models
---

# Unmasking Synthetic Images — Forensic MoE Demo

Drop in any image and find out whether it's real or AI-generated — and if it's fake, which model made it.

Five ResNet50 detectors, one trained per Stable Diffusion variant (SD 1.5, SD 2.1, SDXL, SD 3.5, FLUX), feed into a small gating network that picks the right one for each image. The bar
chart shows which generator's fingerprint dominated. The Grad-CAM heatmap shows where the chosen expert was looking when it made the call.

## What you see in the demo

- **Prediction** — *real* or *synthetic*, with confidence.
- **α bar chart** — the gating weights per expert. A high α on expert *E* means the image carries artifacts characteristic of variant *E*.
- **Grad-CAM heatmap** — where the attributed expert's attention concentrated, typically on facial regions where the VAE roundtrip at `strength=0.05` introduces sub-pixel artifacts.

## Why this gating strategy

The demo runs **MoE-Logit** (~1K trainable params), one of four gating strategies trained in the project:

| Strategy | Balanced Accuracy | AUC-ROC | Attribution |
|---|---|---|---|
| MoE-Logit | 94.1% | 0.986 | Clear diagonal — each variant routes to its specialist expert |
| MoE-Embedding | 95.2% | 0.985 | Loses the diagonal — routes most inputs to the SD1.5 expert |

MoE-Embedding wins on raw detection by 1 point, but collapses routing into a non-diagnostic pattern. MoE-Logit hits 94.1% BA across all 5 variants *while preserving the attribution
signal* — which is what makes the α bar chart meaningful.

## Links

[GitHub repo](https://github.com/enricoroncuzzi/unmasking-synthetic-images) · [Model checkpoints](https://huggingface.co/enricoroncuzzi/unmasking-synthetic-images-models)
