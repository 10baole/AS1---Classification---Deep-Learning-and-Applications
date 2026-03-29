from __future__ import annotations

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from image_classification.viz.attention import denormalize_image


class ViTGradCAMWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        vit = self.model.backbone
        x = vit._process_input(x)
        n = x.shape[0]
        batch_class_token = vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = vit.encoder(x)
        cls_token = x[:, 0]
        logits = self.model.classifier(cls_token)
        return logits


def reshape_transform_vit_torchvision(tensor, height=14, width=14):
    tensor = tensor[:, 1:, :]
    tensor = tensor.reshape(tensor.size(0), height, width, tensor.size(2))
    tensor = tensor.permute(0, 3, 1, 2)
    return tensor


def show_gradcam_vit(model, image_tensor, target_category=None, class_name=None):
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("show_gradcam_vit requires grad-cam.") from exc

    model.eval()
    cam_model = ViTGradCAMWrapper(model).to(image_tensor.device)
    target_layers = [cam_model.model.backbone.encoder.ln]

    cam = GradCAM(
        model=cam_model,
        target_layers=target_layers,
        reshape_transform=reshape_transform_vit_torchvision,
    )

    if target_category is None:
        with torch.no_grad():
            logits = cam_model(image_tensor)
            target_category = logits.argmax(dim=1).item()

    targets = [ClassifierOutputTarget(target_category)]
    grayscale_cam = cam(input_tensor=image_tensor, targets=targets)[0]

    rgb_img = denormalize_image(image_tensor[0]).permute(1, 2, 0).numpy()
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    plt.figure(figsize=(6, 6))
    plt.imshow(visualization)
    plt.title(f"Grad-CAM | {class_name if class_name is not None else target_category}")
    plt.axis("off")
    plt.show()
