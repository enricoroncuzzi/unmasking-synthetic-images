"""
Gating network architectures for the MoE framework.

Four strategies, each accepting pre-extracted expert outputs and
returning (alphas, final_logits):

    - LogitGating      : routes on concatenated expert logits (10-dim)
    - EmbeddingGating  : routes on concatenated expert embeddings (10240-dim)
    - ImageGating      : routes on raw input image patch (3×256×256)
    - AttentionGating  : self-attention over 5 expert logit tokens (NEW)

All classes share the same call signature:

    alphas, logits = gating(logits_list, embeddings_list, image)

where:
    logits_list     : list of 5 tensors, each [batch, 2]
    embeddings_list : list of 5 tensors, each [batch, 2048]
    image           : tensor [batch, 3, 256, 256]

Unused arguments are accepted but ignored, so the MoE training loop
can call every gating strategy identically without branching.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


def _weighted_logits(
    alphas: torch.Tensor,
    logits_list: List[torch.Tensor],
) -> torch.Tensor:
    """
    Combines expert logits using gating weights.

    Args:
        alphas      : [batch, num_experts] — softmax weights from gating network
        logits_list : list of num_experts tensors, each [batch, 2]

    Returns:
        Weighted sum of logits: [batch, 2]
    """
    # Stack to [batch, num_experts, 2], then weight and sum over expert dim
    stacked = torch.stack(logits_list, dim=1)          # [batch, E, 2]
    alphas_expanded = alphas.unsqueeze(-1)              # [batch, E, 1]
    return (alphas_expanded * stacked).sum(dim=1)       # [batch, 2]


class LogitGating(nn.Module):
    """
    Routes based on concatenated expert logits.

    Input:  5 × [batch, 2] → concat → [batch, 10]
    Hidden: FC(10 → 64) + ReLU
    Output: FC(64 → 5)   + Softmax → alphas [batch, 5]

    Rationale: simplest possible gating — uses only the final binary
    confidence of each expert. Fast and interpretable; the gating
    learns linear and non-linear combinations of expert decisions.
    ~965 trainable parameters.
    """

    def __init__(self, num_experts: int = 5):
        super().__init__()
        self.num_experts = num_experts
        input_dim = num_experts * 2  # 5 experts × 2 logits = 10

        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_experts),
        )

    def forward(
        self,
        logits_list: List[torch.Tensor],
        embeddings_list: List[torch.Tensor],
        image: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            logits_list     : list of 5 tensors [batch, 2]
            embeddings_list : ignored
            image           : ignored

        Returns:
            alphas  : [batch, 5]  — gating weights (sum to 1)
            logits  : [batch, 2]  — weighted combination of expert logits
        """
        x = torch.cat(logits_list, dim=1)          # [batch, 10]
        raw = self.net(x)                          # [batch, 5]
        alphas = F.softmax(raw, dim=1)             # [batch, 5]
        final_logits = _weighted_logits(alphas, logits_list)
        return alphas, final_logits


class EmbeddingGating(nn.Module):
    """
    Routes based on concatenated expert embeddings (2048-dim each).

    Input:  5 × [batch, 2048] → concat → [batch, 10240]
    Stage 1: BatchNorm(10240) → FC(10240 → 1024) → ReLU
    Stage 2: FC(1024 → 256)  → ReLU
    Stage 3: FC(256 → 5)     → Softmax → alphas [batch, 5]
    Combination: weighted mean of embeddings → FC(2048 → 2) → final logits

    Rationale: the richest gating signal — full ResNet50 feature
    representations from each expert. The progressive compression
    (10240 → 1024 → 256 → 5) avoids information bottlenecks.
    BatchNorm at input stabilises training given scale differences
    between expert embedding distributions. ~10.5M trainable parameters.
    """

    def __init__(self, num_experts: int = 5, embedding_dim: int = 2048):
        super().__init__()
        self.num_experts = num_experts
        self.embedding_dim = embedding_dim
        input_dim = num_experts * embedding_dim  # 5 × 2048 = 10240

        self.routing_net = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, num_experts),
        )

        # Produces final logits from the alpha-weighted embedding combination
        self.classifier = nn.Linear(embedding_dim, 2)

    def forward(
        self,
        logits_list: List[torch.Tensor],
        embeddings_list: List[torch.Tensor],
        image: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            logits_list     : list of 5 tensors [batch, 2]  — used for weighted logits fallback
            embeddings_list : list of 5 tensors [batch, 2048] — primary gating input
            image           : ignored

        Returns:
            alphas  : [batch, 5]  — gating weights (sum to 1)
            logits  : [batch, 2]  — from alpha-weighted embedding combination
        """
        x = torch.cat(embeddings_list, dim=1)      # [batch, 10240]
        raw = self.routing_net(x)                  # [batch, 5]
        alphas = F.softmax(raw, dim=1)             # [batch, 5]

        # Weighted combination of embeddings → linear classifier
        stacked_emb = torch.stack(embeddings_list, dim=1)   # [batch, E, 2048]
        alphas_exp = alphas.unsqueeze(-1)                    # [batch, E, 1]
        weighted_emb = (alphas_exp * stacked_emb).sum(dim=1) # [batch, 2048]
        final_logits = self.classifier(weighted_emb)         # [batch, 2]

        return alphas, final_logits


class ImageGating(nn.Module):
    """
    Routes based on the raw input image patch.

    Input:  [batch, 3, 256, 256]
    CNN:    Conv(3→32, 3×3) → ReLU → MaxPool(2)
            Conv(32→64, 3×3) → ReLU → AdaptiveAvgPool(1×1)
    MLP:    FC(64 → 32) → ReLU → FC(32 → 5) → Softmax → alphas [batch, 5]
    Final:  alphas applied to expert logits → weighted sum [batch, 2]

    Rationale: routing decision is independent of expert outputs —
    the gating network inspects the image directly and decides
    which expert to trust before seeing any expert prediction.
    Lightweight (~20k params); tends toward a dominant expert
    (good for detection, limits attribution sensitivity).
    """

    def __init__(self, num_experts: int = 5):
        super().__init__()
        self.num_experts = num_experts

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.mlp = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_experts),
        )

    def forward(
        self,
        logits_list: List[torch.Tensor],
        embeddings_list: List[torch.Tensor],
        image: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            logits_list     : list of 5 tensors [batch, 2] — used for final weighted sum
            embeddings_list : ignored
            image           : [batch, 3, 256, 256] — primary gating input

        Returns:
            alphas  : [batch, 5]  — gating weights (sum to 1)
            logits  : [batch, 2]  — weighted combination of expert logits
        """
        features = self.cnn(image)                  # [batch, 64, 1, 1]
        features = features.flatten(1)              # [batch, 64]
        raw = self.mlp(features)                    # [batch, 5]
        alphas = F.softmax(raw, dim=1)              # [batch, 5]
        final_logits = _weighted_logits(alphas, logits_list)
        return alphas, final_logits


class AttentionGating(nn.Module):
    """
    Self-attention over expert logit tokens.

    Each expert's 2-dim logit vector is treated as a token.
    Self-attention (1 head, d_model=2) allows the gating network to
    learn pairwise interactions between experts: if expert A and B agree
    but expert C disagrees, the attention weights reflect that tension.

    Input:  5 × [batch, 2] → stack → [batch, 5, 2]  (seq_len=5, d=2)
    Attn:   nn.MultiheadAttention(embed_dim=2, num_heads=1, batch_first=True)
    Pool:   mean over sequence → [batch, 2]
    Output: FC(2 → 5) + Softmax → alphas [batch, 5]
    Final:  alphas applied to expert logits → weighted sum [batch, 2]

    Rationale: standard logit-based gating treats each expert independently.
    Self-attention captures inter-expert agreement/disagreement as a
    routing signal. ~300 trainable parameters — deliberately minimal
    so that the attention mechanism itself carries the inductive bias,
    not a large MLP. Novel contribution relative to the thesis.

    Note on dimensionality: d_model=2 is very low. If training is
    unstable, the fallback is to project tokens to a higher dim
    (e.g. FC(2→16) before attention) without changing the architecture
    concept. This is documented in Phase 3 open questions.
    """

    def __init__(self, num_experts: int = 5, d_model: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.d_model = d_model

        # num_heads=1 required when embed_dim=2 (must divide evenly)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=1,
            batch_first=True,
            dropout=0.0,
        )

        self.fc_out = nn.Linear(d_model, num_experts)

    def forward(
        self,
        logits_list: List[torch.Tensor],
        embeddings_list: List[torch.Tensor],
        image: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            logits_list     : list of 5 tensors [batch, 2] — tokenised as sequence
            embeddings_list : ignored
            image           : ignored

        Returns:
            alphas  : [batch, 5]  — gating weights (sum to 1)
            logits  : [batch, 2]  — weighted combination of expert logits
        """
        # Stack expert logits as a sequence of tokens
        tokens = torch.stack(logits_list, dim=1)    # [batch, 5, 2]

        # Self-attention: each expert token attends to all others
        attended, _ = self.attention(tokens, tokens, tokens)  # [batch, 5, 2]

        # Mean pooling over expert sequence → single vector
        pooled = attended.mean(dim=1)               # [batch, 2]

        raw = self.fc_out(pooled)                   # [batch, 5]
        alphas = F.softmax(raw, dim=1)              # [batch, 5]
        final_logits = _weighted_logits(alphas, logits_list)
        return alphas, final_logits


GATING_STRATEGIES = {
    "logit": LogitGating,
    "embedding": EmbeddingGating,
    "image": ImageGating,
    "attention": AttentionGating,
}


def build_gating(strategy: str, **kwargs) -> nn.Module:
    """
    Instantiates a gating network by name.

    Args:
        strategy : one of "logit", "embedding", "image", "attention"
        **kwargs : forwarded to the gating class constructor

    Returns:
        Instantiated gating nn.Module

    Example:
        gating = build_gating("embedding", num_experts=5)
    """
    if strategy not in GATING_STRATEGIES:
        raise ValueError(
            f"Unknown gating strategy '{strategy}'. "
            f"Available: {list(GATING_STRATEGIES.keys())}"
        )
    return GATING_STRATEGIES[strategy](**kwargs)