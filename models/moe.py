"""
MoE model for Phase 3: frozen expert ensemble + gating network.

The MoEModel loads 5 pre-trained ResNet50 expert checkpoints (frozen),
runs each forward pass to extract logits and embeddings, then delegates
the final classification to a gating network.

Only the gating network is trained. Experts are never updated.

Usage:
    from models.moe import MoEModel

    model = MoEModel(
        checkpoint_paths={
            "sd15":      "checkpoints/sd15/best-epoch=30-....ckpt",
            "sd21":      "checkpoints/sd21/best-epoch=49-....ckpt",
            "sdxl_base": "checkpoints/sdxl_base/best-epoch=20-....ckpt",
            "sd35":      "checkpoints/sd35/best-epoch=46-....ckpt",
            "flux":      "checkpoints/flux/best-epoch=86-....ckpt",
        },
        gating_strategy="embedding",   # "logit" | "embedding" | "image" | "attention"
    )

    alphas, logits = model(patches)    # patches: [B, 3, 256, 256]
"""

import os
import glob

import torch
import torch.nn as nn
from omegaconf import DictConfig

from models.expert import ExpertModel
from models.gating import build_gating, GATING_STRATEGIES


# Canonical expert order — must match EXPERT_NAMES in data/dataset.py
EXPERT_NAMES = ["sd15", "sd21", "sdxlbase", "sd35", "flux"]


def _resolve_checkpoint(ckpt_path: str) -> str:
    """
    Resolves a checkpoint path that may contain a glob wildcard.

    Allows configs to specify  checkpoints/sd15/best-*.ckpt  instead
    of the full timestamped filename, which changes between runs.

    Args:
        ckpt_path : exact path or glob pattern

    Returns:
        Resolved absolute path

    Raises:
        FileNotFoundError if no match is found
        RuntimeError if the pattern matches more than one file
    """
    if "*" not in ckpt_path and "?" not in ckpt_path:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        return ckpt_path

    matches = glob.glob(ckpt_path)
    if len(matches) == 0:
        raise FileNotFoundError(
            f"No checkpoint found matching pattern: {ckpt_path}"
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Ambiguous checkpoint pattern — {len(matches)} files matched: "
            f"{ckpt_path}\nMatches: {matches}"
        )
    return matches[0]


def _load_expert(ckpt_path: str, device: torch.device) -> ExpertModel:
    """
    Loads an ExpertModel from a Lightning checkpoint.

    Lightning saves the full LightningModule state dict under the key
    'state_dict', with parameter names prefixed by 'model.' (the
    attribute name in ExpertLitModule). This function strips that prefix
    and loads only the ExpertModel weights.

    Args:
        ckpt_path : path to .ckpt file (Lightning format)
        device    : target device

    Returns:
        ExpertModel in eval mode with weights loaded, on the target device
    """
    resolved = _resolve_checkpoint(ckpt_path)
    checkpoint = torch.load(resolved, map_location=device, weights_only=False)

    state_dict = checkpoint["state_dict"]

    # Strip 'model.' prefix added by ExpertLitModule (self.model = ExpertModel(...))
    expert_state = {
        k.removeprefix("model."): v
        for k, v in state_dict.items()
        if k.startswith("model.")
    }

    if not expert_state:
        raise RuntimeError(
            f"Could not extract expert weights from checkpoint: {resolved}\n"
            f"Expected keys prefixed with 'model.' in state_dict. "
            f"Found keys: {list(state_dict.keys())[:5]}"
        )

    expert = ExpertModel(num_classes=2)
    expert.load_state_dict(expert_state)
    expert.to(device)
    expert.eval()

    return expert


class MoEModel(nn.Module):
    """
    Mixture of Experts model for synthetic image detection and attribution.

    Wraps 5 frozen ResNet50 expert classifiers and a trainable gating network.
    During the forward pass, all 5 experts process the input in parallel to
    produce logits and embeddings. The gating network combines these outputs
    into a final binary prediction and a set of alpha weights.

    The alpha weights are the attribution signal: a high alpha for expert E
    on a given image suggests the image carries artifacts characteristic of
    the SD variant that E was trained on.

    Args:
        checkpoint_paths : dict mapping expert name → checkpoint path (or glob)
                           Keys must match EXPERT_NAMES order, but can be a
                           subset — missing experts raise FileNotFoundError.
        gating_strategy  : one of "logit", "embedding", "image", "attention"
        gating_kwargs    : additional kwargs forwarded to build_gating()
                           (e.g. num_experts, embedding_dim)
    """

    def __init__(
        self,
        checkpoint_paths: dict[str, str],
        gating_strategy: str,
        **gating_kwargs,
    ):
        super().__init__()

        self.expert_names = EXPERT_NAMES
        self.num_experts = len(EXPERT_NAMES)
        self.gating_strategy = gating_strategy

        # --- Load and freeze experts ---
        # Stored as a ModuleList so Lightning/optimizer sees them,
        # but no gradients will flow through them (requires_grad=False).
        device = torch.device("cpu")  # moved to GPU by Lightning Trainer
        experts = []
        for name in EXPERT_NAMES:
            if name not in checkpoint_paths:
                raise KeyError(
                    f"Missing checkpoint for expert '{name}'. "
                    f"Provided keys: {list(checkpoint_paths.keys())}"
                )
            expert = _load_expert(checkpoint_paths[name], device)
            for param in expert.parameters():
                param.requires_grad = False
            experts.append(expert)

        self.experts = nn.ModuleList(experts)

        # --- Per-expert CUDA streams (reused across forward calls) ---
        self._streams = [torch.cuda.Stream() for _ in self.experts] if torch.cuda.is_available() else None

        # --- Gating network (trainable) ---
        kwargs = {"num_experts": self.num_experts}
        if gating_strategy == "embedding":
            kwargs["embedding_dim"] = 2048
        kwargs.update(gating_kwargs)

        self.gating = build_gating(gating_strategy, **kwargs)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the full MoE pipeline.

        All experts run concurrently on separate CUDA streams (no_grad).
        Only the gating network participates in gradient computation.

        Args:
            x : [batch, 3, 256, 256] — input image patches

        Returns:
            alphas       : [batch, num_experts] — gating weights, sum to 1
            final_logits : [batch, 2]           — weighted binary prediction
        """
        results = [None] * len(self.experts)

        with torch.no_grad():
            if self._streams is not None:
                # CUDA available: launch all experts on separate streams
                # so the GPU can overlap their execution concurrently.
                # Each stream must wait for the default stream (which produced x)
                # before consuming x — otherwise reads of x are a data race.
                default_stream = torch.cuda.current_stream()
                for i, (expert, stream) in enumerate(zip(self.experts, self._streams)):
                    with torch.cuda.stream(stream):
                        stream.wait_stream(default_stream)
                        emb = expert.get_embedding(x)
                        logits = expert.backbone.fc(emb)
                        results[i] = (logits, emb)
                # Wait for all streams to finish before reading results
                torch.cuda.synchronize()
            else:
                # CPU / MPS fallback: sequential execution
                for i, expert in enumerate(self.experts):
                    emb = expert.get_embedding(x)
                    logits = expert.backbone.fc(emb)
                    results[i] = (logits, emb)

        logits_list     = [r[0] for r in results]
        embeddings_list = [r[1] for r in results]

        alphas, final_logits = self.gating(logits_list, embeddings_list, x)
        return alphas, final_logits


    def trainable_parameters(self):
        """
        Returns only the gating network parameters.
        Convenience method for passing to the optimizer in train_moe.py.
        """
        return self.gating.parameters()


def resolve_checkpoint_paths(checkpoints_dir: str) -> dict[str, str]:
    """
    Builds the checkpoint_paths dict from a checkpoints/ directory,
    using glob patterns to avoid hardcoding filenames.

    Expected directory structure (produced by Phase 2 training):
        checkpoints/
            sd15/best-*.ckpt
            sd21/best-*.ckpt
            sdxl_base/best-*.ckpt
            sd35/best-*.ckpt
            flux/best-*.ckpt

    Args:
        checkpoints_dir : path to the checkpoints/ root directory

    Returns:
        dict mapping expert name → glob pattern string
        (resolved lazily by _load_expert → _resolve_checkpoint)
    """
    paths = {}
    for name in EXPERT_NAMES:
        pattern = os.path.join(checkpoints_dir, name, "best-*.ckpt")
        paths[name] = pattern
    return paths