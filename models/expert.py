"""
Expert model for binary classification (real vs synthetic).

ResNet50 pretrained on ImageNet with final FC layer replaced
for 2-class output. Exposes embeddings for MoE gating (Phase 3).
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ExpertModel(nn.Module):
    """
    ResNet50-based binary classifier.

    Forward pass returns logits [batch, 2].
    get_embedding() returns the 2048-dim feature vector before FC.
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()

        self.backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1
        )

        # Replace final FC: 1000 (ImageNet) → 2 (real vs synthetic)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits [batch, num_classes]."""
        return self.backbone(x)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns 2048-dim embedding before the final FC layer.
        Used by the MoE embedding-based gating network (Phase 3).
        """
        # ResNet50 structure: conv1 → bn1 → relu → maxpool → layer1-4 → avgpool → fc
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)  # [batch, 2048]

        return x