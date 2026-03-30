from __future__ import annotations

import math
import types

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def denormalize_image(img_tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    img = img_tensor.detach().cpu().clone()
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]
    return img.clamp(0, 1)


@torch.no_grad()
def predict_one(model, image_tensor, classes):
    model.eval()
    logits = model(image_tensor)
    pred = logits.argmax(dim=1).item()
    probs = torch.softmax(logits, dim=1)[0]
    conf = probs[pred].item()
    return pred, classes[pred], conf


def patch_torchvision_vit_attention(model):
    for block in model.backbone.encoder.layers:
        def wrapped_forward(self, query, key, value, **kwargs):
            kwargs["need_weights"] = True
            kwargs["average_attn_weights"] = False
            out, attn_weights = nn.MultiheadAttention.forward(
                self,
                query,
                key,
                value,
                **kwargs,
            )
            self.last_attn_map = attn_weights.detach()
            return out, attn_weights

        block.self_attention.forward = types.MethodType(wrapped_forward, block.self_attention)
    return model


def compute_attention_rollout(attn_maps, start_layer=0):
    device = attn_maps[0].device
    batch_size = attn_maps[0].shape[0]
    tokens = attn_maps[0].shape[-1]

    eye = torch.eye(tokens, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    result = eye

    for attn in attn_maps[start_layer:]:
        attn = attn.mean(dim=1)
        attn = attn + eye
        attn = attn / attn.sum(dim=-1, keepdim=True)
        result = attn @ result
    return result


@torch.no_grad()
def get_attention_rollout(model, image_tensor, start_layer=0):
    model.eval()
    _ = model(image_tensor)
    attn_maps = []
    for block in model.backbone.encoder.layers:
        if hasattr(block.self_attention, "last_attn_map"):
            attn_maps.append(block.self_attention.last_attn_map)
    if len(attn_maps) == 0:
        raise ValueError("No attention maps found.")
    return compute_attention_rollout(attn_maps, start_layer=start_layer)


def show_attention_map(image_tensor, rollout, title="Attention Rollout"):
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("show_attention_map requires opencv-python.") from exc

    image = denormalize_image(image_tensor)
    image_np = image.permute(1, 2, 0).numpy()

    mask = rollout[0, 0, 1:]
    grid_size = int(math.sqrt(mask.shape[0]))
    mask = mask.reshape(grid_size, grid_size).detach().cpu().numpy()
    mask = cv2.resize(mask, (image_np.shape[1], image_np.shape[0]))
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)

    plt.figure(figsize=(6, 6))
    plt.imshow(image_np)
    plt.imshow(mask, cmap="jet", alpha=0.45)
    plt.title(title)
    plt.axis("off")
    plt.show()
