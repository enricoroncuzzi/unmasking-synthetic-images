"""
Gradio demo app — Unmasking Synthetic Images.

Entry point for HuggingFace Spaces (app_file: demo/app.py).
Wraps MoEPipeline for single-image forensic classification and attribution.

HF Space: https://huggingface.co/spaces/enricoroncuzzi/unmasking-synthetic-images-demo
"""

import os
import sys
from pathlib import Path
from typing import Optional

# Redirect HuggingFace cache to the Storage Bucket mounted at /data.
# Bucket: enricoroncuzzi/unmasking-synthetic-images-demo-storage (mounted in Space settings).
# On cold start after sleep, checkpoints are already on disk — no re-download.
# Falls back silently to the default ~/.cache/huggingface when running locally.
if Path("/data").exists():
    os.environ["HF_HOME"] = "/data/.cache/huggingface"
    # Redirect PyTorch Hub cache (resnet50 pretrained weights) to persistent storage.
    # Without this, torchvision re-downloads ~98 MB on every cold start.
    os.environ["TORCH_HOME"] = "/data/.cache/torch"

# Ensure repo root is on sys.path so "from demo.pipeline import ..." resolves
# correctly whether this file is run as "python demo/app.py" (HF Spaces style)
# or as "python -m demo.app" from the repo root.
_DEMO_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _DEMO_DIR.parent
# _REPO_ROOT: needed locally so models/ is importable (repo_root/models/)
# _DEMO_DIR:  needed on HF Space so pipeline.py is importable (Space root = _DEMO_DIR)
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_DEMO_DIR))

import gradio as gr
import matplotlib

matplotlib.use("Agg")  # non-interactive backend, required in server environments
import matplotlib.pyplot as plt

from pipeline import MoEPipeline

# ── Theme & CSS ────────────────────────────────────────────────────────────────

_THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.red,
    secondary_hue=gr.themes.colors.neutral,
    neutral_hue=gr.themes.colors.neutral,
    font=[gr.themes.GoogleFont("Space Mono"), "ui-monospace", "monospace"],
).set(
    body_background_fill="#0d0d0d",
    body_background_fill_dark="#0d0d0d",
    block_background_fill="#141414",
    block_background_fill_dark="#141414",
    block_border_color="#252525",
    block_border_color_dark="#252525",
    block_border_width="1px",
    input_background_fill="#1a1a1a",
    input_background_fill_dark="#1a1a1a",
    button_primary_background_fill="#991b1b",
    button_primary_background_fill_hover="#7f1d1d",
    button_primary_background_fill_dark="#991b1b",
    button_primary_background_fill_hover_dark="#7f1d1d",
    button_primary_text_color="white",
    button_primary_text_color_dark="white",
    body_text_color="#d4d4d4",
    body_text_color_dark="#d4d4d4",
    body_text_color_subdued="#737373",
    body_text_color_subdued_dark="#737373",
    block_label_text_color="#555555",
    block_label_text_color_dark="#555555",
    block_title_text_color="#d4d4d4",
    block_title_text_color_dark="#d4d4d4",
    background_fill_secondary="#141414",
    background_fill_secondary_dark="#141414",
)

_CSS = """
footer { display: none !important; }

.gradio-container h1 {
    font-size: 1.0rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: #cc2020 !important;
    border-bottom: 1px solid #252525;
    padding-bottom: 0.6rem;
    margin-bottom: 0.4rem;
}

.prose p, .prose li { color: #9ca3af !important; font-size: 0.83rem !important; }
.prose strong { color: #d4d4d4 !important; }
.prose a { color: #cc2020 !important; }
.prose a:hover { color: #ef4444 !important; }

button.primary {
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    transition: box-shadow 0.2s ease !important;
    box-shadow: 0 0 10px rgba(153, 27, 27, 0.35) !important;
}
button.primary:hover {
    box-shadow: 0 0 20px rgba(153, 27, 27, 0.65) !important;
}
"""

# ── Constants ──────────────────────────────────────────────────────────────────

_EXPERT_LABELS = {
    "sd15":     "Stable Diffusion 1.5",
    "sd21":     "Stable Diffusion 2.1",
    "sdxlbase": "SDXL Base",
    "sd35":     "Stable Diffusion 3.5",
    "flux":     "FLUX.1",
}

_STRATEGY = "logit"

_EXAMPLES_DIR = _DEMO_DIR / "examples"

_TITLE = "Unmasking Synthetic Images"

_DESCRIPTION = """
Drop in any image and find out whether it's real or AI-generated — and if it's fake, which model made it.

Five ResNet50 detectors, one trained per Stable Diffusion variant (SD 1.5, SD 2.1, SDXL, SD 3.5, FLUX), feed into a small gating network that picks the right one for each image. The bar
chart shows which generator's fingerprint dominated. The heatmap shows where the expert was looking.

94.1% balanced accuracy on the held-out test set.

For more information check my [GitHub](https://github.com/enricoroncuzzi/unmasking-synthetic-images)

For the full story follow the [Medium series](https://medium.com/@enricoroncuzzi/list/unmasking-synthetic-images-moe-detection-and-attribution-4c63d2f4f4d0)

"""

# ── Pipeline preload ───────────────────────────────────────────────────────────
# Loaded at import time so HF Spaces pays the cold-start cost during boot,
# not on the first user click.
PIPELINE = MoEPipeline(device="cpu", strategy=_STRATEGY)


# ── Visualization helpers ──────────────────────────────────────────────────────

def _alpha_bar_chart(
    alpha_weights: dict,
    attributed_source: Optional[str],
) -> plt.Figure:
    """
    Horizontal bar chart of per-expert gating weights.

    The attributed expert bar is highlighted in red; others in steel blue.
    """
    names  = list(alpha_weights.keys())
    values = [alpha_weights[n] for n in names]
    labels = [_EXPERT_LABELS.get(n, n) for n in names]
    colors = [
        "#cc2020" if n == attributed_source else "#2d5a8a"
        for n in names
    ]

    fig, ax = plt.subplots(figsize=(6, 2.8), facecolor="#141414")
    ax.set_facecolor("#141414")
    bars = ax.barh(labels, values, color=colors, edgecolor="none", height=0.55)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Gating weight (α)", fontsize=9, color="#737373")
    ax.tick_params(axis="y", labelsize=9, colors="#9ca3af")
    ax.tick_params(axis="x", labelsize=8, colors="#555555")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color("#252525")

    if attributed_source:
        title = f"Attributed to: {_EXPERT_LABELS.get(attributed_source, attributed_source)}"
    else:
        title = "Expert routing weights"
    ax.set_title(title, fontsize=10, pad=6, color="#d4d4d4")

    for bar, val in zip(bars, values):
        x_pos = val + 0.02 if val < 0.88 else val - 0.06
        ax.text(
            x_pos,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1%}",
            va="center",
            fontsize=8,
            color="#d4d4d4",
        )

    fig.tight_layout()
    return fig


# ── Gradio callback ────────────────────────────────────────────────────────────

def analyze(image):
    """
    Main inference callback wired to the Gradio interface.

    Args:
        image : PIL.Image from gr.Image input (None if no image uploaded)

    Returns:
        label_out  : dict for gr.Label  e.g. {"synthetic": 0.97}
        alpha_plot : matplotlib Figure  for gr.Plot
        gradcam_out: PIL.Image          for gr.Image
    """
    if image is None:
        return None, None, None

    result = PIPELINE.predict(image)
    gradcam_img = PIPELINE.gradcam(image)
    alpha_fig = _alpha_bar_chart(result["alpha_weights"], result["attributed_source"])

    label_out = {result["prediction"]: result["confidence"]}
    return label_out, alpha_fig, gradcam_img


# ── Example images ─────────────────────────────────────────────────────────────

def _load_examples() -> list:
    """Returns a list of [image_path] rows for gr.Examples, if available."""
    if not _EXAMPLES_DIR.is_dir():
        return []
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    paths = sorted(
        p for p in _EXAMPLES_DIR.iterdir()
        if p.suffix.lower() in exts
    )
    return [[str(p)] for p in paths]


# ── UI layout ──────────────────────────────────────────────────────────────────

def build_demo() -> gr.Blocks:
    examples = _load_examples()

    with gr.Blocks(title=_TITLE, theme=_THEME, css=_CSS) as demo:

        gr.Markdown(f"# {_TITLE}")
        gr.Markdown(_DESCRIPTION)

        with gr.Row():
            # ── Left column: input ───────────────────────────────────────────
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type="pil",
                    label="Upload image (drag & drop or click)",
                )
                analyze_btn = gr.Button("Analyze", variant="primary")

                if examples:
                    gr.Examples(
                        examples=examples,
                        inputs=[image_input],
                        label="Example images",
                    )

            # ── Right column: outputs ────────────────────────────────────────
            with gr.Column(scale=1):
                label_output = gr.Label(
                    num_top_classes=2,
                    label="Prediction",
                )
                alpha_plot = gr.Plot(
                    label="Expert routing weights (α)",
                )
                gradcam_output = gr.Image(
                    type="pil",
                    label="Grad-CAM — where the attributed expert looks",
                )

        # ── Wiring ──────────────────────────────────────────────────────────
        analyze_btn.click(
            fn=analyze,
            inputs=[image_input],
            outputs=[label_output, alpha_plot, gradcam_output],
        )

        # Also trigger on image upload for snappier UX
        image_input.upload(
            fn=analyze,
            inputs=[image_input],
            outputs=[label_output, alpha_plot, gradcam_output],
        )

        gr.Markdown(
            "---\n"
            "*Weights: [HuggingFace Hub](https://huggingface.co/enricoroncuzzi/unmasking-synthetic-images-models) · "
            "MoE-Logit · CPU inference*"
        )

    return demo


demo = build_demo()

if __name__ == "__main__":
    demo.launch()
