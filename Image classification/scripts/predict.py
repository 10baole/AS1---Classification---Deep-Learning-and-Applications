from __future__ import annotations

import argparse
from pathlib import Path
import sys

from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from image_classification.config import load_config
from image_classification.data.datasets import get_dataloaders
from image_classification.data.transforms import get_transforms
from image_classification.models.vit import ViTClassifier
from image_classification.utils.paths import resolve_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/infer_vit.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    root = Path.cwd()
    config = load_config(args.config)
    processed_dir = resolve_path(root, config["paths"]["processed_dir"])
    data_bundle = get_dataloaders(
        data_root=processed_dir,
        batch_size=1,
        image_size=int(config["data"]["image_size"]),
        use_aug=False,
        use_oversampler=False,
        num_workers=0,
    )
    classes = data_bundle["train_ds"].classes

    checkpoint = args.checkpoint or config["inference"]["checkpoint"]
    checkpoint = resolve_path(root, checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViTClassifier(num_classes=len(classes), freeze_backbone=False, dropout=0.1).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    _, eval_tf = get_transforms(image_size=int(config["data"]["image_size"]), use_aug=False)
    image = Image.open(Path(args.image)).convert("RGB")
    image_tensor = eval_tf(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = logits.argmax(dim=1).item()

    print(f"pred={classes[pred_idx]} confidence={probs[pred_idx].item():.4f}")


if __name__ == "__main__":
    main()
