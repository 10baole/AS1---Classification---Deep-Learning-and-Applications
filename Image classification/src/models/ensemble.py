from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, ViT_B_16_Weights, resnet50, vit_b_16


class EnsembleClassifier(nn.Module):
    """Ensemble of ViT-B/16 (Transformer) and ResNet-50 (CNN).

    Fusion strategies:
        - "concat": concatenate features, then classify
        - "weighted": learnable scalar weight to blend logits
    """

    def __init__(
        self,
        num_classes: int,
        fusion: str = "concat",
        freeze_backbones: bool = False,
        dropout: float = 0.1,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.fusion = fusion

        # --- ViT backbone (768-d) ---
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        vit_dim = self.vit.heads.head.in_features  # 768
        self.vit.heads = nn.Identity()

        # --- ResNet-50 backbone (2048-d) ---
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        resnet_dim = self.resnet.fc.in_features  # 2048
        self.resnet.fc = nn.Identity()

        if freeze_backbones:
            for p in self.vit.parameters():
                p.requires_grad = False
            for p in self.resnet.parameters():
                p.requires_grad = False

        # --- Project both branches to the same hidden_dim ---
        self.vit_proj = nn.Sequential(
            nn.Linear(vit_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.resnet_proj = nn.Sequential(
            nn.Linear(resnet_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # --- Fusion heads ---
        if fusion == "concat":
            self.classifier = nn.Linear(hidden_dim * 2, num_classes)
        elif fusion == "weighted":
            self.vit_head = nn.Linear(hidden_dim, num_classes)
            self.resnet_head = nn.Linear(hidden_dim, num_classes)
            self.alpha_logit = nn.Parameter(torch.zeros(1))
        else:
            raise ValueError(f"Unknown fusion strategy: {fusion}")

    def forward(self, x, return_features: bool = False):
        vit_raw = self.vit(x)        # (B, 768)
        resnet_raw = self.resnet(x)   # (B, 2048)

        vit_feat = self.vit_proj(vit_raw)       # (B, hidden_dim)
        resnet_feat = self.resnet_proj(resnet_raw)  # (B, hidden_dim)

        if self.fusion == "concat":
            combined = torch.cat([vit_feat, resnet_feat], dim=1)
            logits = self.classifier(combined)
        elif self.fusion == "weighted":
            alpha = torch.sigmoid(self.alpha_logit)
            logits = alpha * self.vit_head(vit_feat) + (1 - alpha) * self.resnet_head(resnet_feat)

        if return_features:
            return logits, torch.cat([vit_feat, resnet_feat], dim=1)
        return logits


if __name__ == "__main__":
    model = EnsembleClassifier(num_classes=67, fusion="concat")
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print(f"Output shape: {out.shape}")
