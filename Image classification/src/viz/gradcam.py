from __future__ import annotations

import matplotlib.pyplot as plt
import torch

from src.viz.attention import denormalize_image


def show_gradcam_resnet(model, image_tensor, target_category=None, class_name=None):
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("show_gradcam_resnet requires grad-cam.") from exc

    model.eval()
    target_layers = [model.backbone.layer4[-1]]

    cam = GradCAM(model=model, target_layers=target_layers)

    if target_category is None:
        with torch.no_grad():
            logits = model(image_tensor)
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
