from __future__ import annotations

import torch.optim as optim


def build_layerwise_lr_optimizer(model, head_lr: float = 3e-4, layer_decay: float = 0.8, weight_decay: float = 0.05):
    param_groups = []

    blocks = list(model.backbone.encoder.layers)
    n_blocks = len(blocks)

    param_groups.append(
        {
            "params": model.backbone.conv_proj.parameters(),
            "lr": head_lr * (layer_decay ** (n_blocks + 1)),
        }
    )
    param_groups.append(
        {
            "params": [model.backbone.class_token],
            "lr": head_lr * (layer_decay ** (n_blocks + 1)),
        }
    )
    param_groups.append(
        {
            "params": [model.backbone.encoder.pos_embedding],
            "lr": head_lr * (layer_decay ** (n_blocks + 1)),
        }
    )

    for i, block in enumerate(blocks):
        depth_from_head = n_blocks - 1 - i
        lr = head_lr * (layer_decay ** (depth_from_head + 1))
        param_groups.append({"params": block.parameters(), "lr": lr})

    param_groups.append({"params": model.classifier.parameters(), "lr": head_lr})
    return optim.AdamW(param_groups, weight_decay=weight_decay)


def build_ensemble_optimizer(model, head_lr: float = 3e-4, backbone_lr: float = 1e-5, weight_decay: float = 0.05):
    """Two-group optimizer for the ensemble: low LR for backbones, high LR for fusion heads."""
    backbone_params = list(model.vit.parameters()) + list(model.resnet.parameters())
    head_params = [
        p for name, p in model.named_parameters()
        if not name.startswith("vit.") and not name.startswith("resnet.")
    ]

    param_groups = [
        {"params": backbone_params, "lr": backbone_lr},
        {"params": head_params, "lr": head_lr},
    ]
    return optim.AdamW(param_groups, weight_decay=weight_decay)
