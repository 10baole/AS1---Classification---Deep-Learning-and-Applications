from __future__ import annotations

import torch.nn as nn
from torchvision.models import ViT_B_16_Weights, vit_b_16


class ViTClassifier(nn.Module):
    def __init__(self, num_classes: int, freeze_backbone: bool = False, dropout: float = 0.1):
        super().__init__()
        self.backbone = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        in_features = self.backbone.heads.head.in_features
        self.backbone.heads = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_features // 2, num_classes),
        )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x, return_features: bool = False):
        features = self.backbone(x)
        logits = self.classifier(features)
        if return_features:
            return logits, features
        return logits
