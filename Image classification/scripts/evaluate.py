from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from image_classification.config import load_config
from image_classification.data.datasets import get_dataloaders
from image_classification.engine.evaluator import evaluate
from image_classification.models.vit import ViTClassifier
from image_classification.utils.paths import resolve_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_vit.yaml")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    root = Path.cwd()
    config = load_config(args.config)
    processed_dir = resolve_path(root, config["paths"]["processed_dir"])
    data_bundle = get_dataloaders(
        data_root=processed_dir,
        batch_size=int(config["train"]["batch_size"]),
        image_size=int(config["data"]["image_size"]),
        use_aug=False,
        use_oversampler=False,
        num_workers=int(config["data"]["num_workers"]),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViTClassifier(
        num_classes=len(data_bundle["train_ds"].classes),
        freeze_backbone=False,
        dropout=float(config["train"]["dropout"]),
    ).to(device)
    model.load_state_dict(torch.load(Path(args.checkpoint), map_location=device))
    criterion = nn.CrossEntropyLoss()

    test_loss, test_acc, test_f1, _, _ = evaluate(
        model,
        data_bundle["test_loader"],
        criterion,
        device,
    )
    print(f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} test_f1={test_f1:.4f}")


if __name__ == "__main__":
    main()
