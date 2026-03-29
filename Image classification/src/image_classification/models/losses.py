from __future__ import annotations

from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                alpha_t = self.alpha.to(logits.device)[targets]
            else:
                alpha_t = self.alpha
            focal_loss = alpha_t * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


def build_focal_alpha_from_dataset(dataset):
    labels = [label for _, label in dataset.samples]
    class_counts = Counter(labels)
    num_classes = len(dataset.classes)
    counts = torch.tensor([class_counts[i] for i in range(num_classes)], dtype=torch.float)
    alpha = 1.0 / counts
    alpha = alpha / alpha.sum() * num_classes
    return alpha
