from __future__ import annotations

import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


class ResNet50Classifier(nn.Module):
    def __init__(self, num_classes: int, freeze_backbone: bool = False, dropout: float = 0.1):
        super().__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
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



if __name__ == "__main__":
    model = ResNet50Classifier(num_classes=10)
