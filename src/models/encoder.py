"""ResNet50 encoder — frozen, FC stripped, outputs pooled 2048-d vector.

See PLAN.md § Encoder."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


class Encoder(nn.Module):
    """Wraps torchvision ResNet50 (ImageNet pretrained, frozen) minus its FC."""

    def __init__(self) -> None:
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        # Strip final FC — resnet.avgpool still produces the 2048-d pooled vector.
        modules = list(resnet.children())[:-1]
        self.features = nn.Sequential(*modules)
        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, 3, 224, 224) -> (B, 2048)."""
        with torch.no_grad():
            feats = self.features(images)
        return feats.flatten(1)  # (B, 2048, 1, 1) -> (B, 2048)
