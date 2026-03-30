from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from src.data.datasets import get_dataloaders
from src.engine.callbacks import EarlyStopping
from src.engine.evaluator import evaluate
from src.models.losses import FocalLoss, build_focal_alpha_from_dataset
from src.models.optim import build_ensemble_optimizer, build_layerwise_lr_optimizer
from src.models.vit import ViTClassifier
from src.models.resnet_50 import ResNet50Classifier
from src.models.ensemble import EnsembleClassifier


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    progress_bar = tqdm(loader, desc="Training", leave=False)

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())
        avg_loss = running_loss / len(all_labels)
        progress_bar.set_postfix(loss=f"{avg_loss:.4f}")

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average="macro")
    return epoch_loss, epoch_acc, epoch_f1


def run_experiment(config: dict, processed_dir: Path, device: torch.device):
    train_cfg = config["train"]
    data_cfg = config["data"]
    paths_cfg = config["paths"]

    models_dir = Path(paths_cfg["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    exp_name = train_cfg["experiment_name"]
    save_path = models_dir / f"{exp_name}_best.pth"

    data_bundle = get_dataloaders(
        data_root=processed_dir,
        batch_size=int(train_cfg["batch_size"]),
        image_size=int(data_cfg["image_size"]),
        use_aug=bool(train_cfg["use_aug"]),
        use_oversampler=bool(train_cfg["use_oversampler"]),
        num_workers=int(data_cfg["num_workers"]),
    )

    train_ds = data_bundle["train_ds"]
    num_classes = len(train_ds.classes)
    model_type = train_cfg.get("model_type", "vit")

    if model_type == "vit":
        model = ViTClassifier(
            num_classes=num_classes,
            freeze_backbone=bool(train_cfg["freeze_backbone"]),
            dropout=float(train_cfg["dropout"]),
        ).to(device)
    elif model_type == "resnet50":
        model = ResNet50Classifier(
            num_classes=num_classes,
            freeze_backbone=bool(train_cfg["freeze_backbone"]),
            dropout=float(train_cfg["dropout"]),
        ).to(device)
    elif model_type == "ensemble":
        model = EnsembleClassifier(
            num_classes=num_classes,
            fusion=train_cfg.get("fusion", "concat"),
            freeze_backbones=bool(train_cfg["freeze_backbone"]),
            dropout=float(train_cfg["dropout"]),
        ).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    if bool(train_cfg["use_focal_loss"]):
        alpha = build_focal_alpha_from_dataset(train_ds)
        criterion = FocalLoss(alpha=alpha, gamma=float(train_cfg["focal_gamma"]), reduction="mean")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=float(train_cfg["label_smoothing"]))

    optimizer_mode = train_cfg["optimizer_mode"]
    if model_type == "ensemble" and optimizer_mode == "ensemble":
        optimizer = build_ensemble_optimizer(
            model,
            head_lr=float(train_cfg["head_lr"]),
            backbone_lr=float(train_cfg.get("backbone_lr", 1e-5)),
            weight_decay=float(train_cfg["weight_decay"]),
        )
    elif optimizer_mode == "freeze":
        classifier_params = model.classifier.parameters() if hasattr(model, "classifier") else model.parameters()
        optimizer = optim.AdamW(classifier_params, lr=1e-3, weight_decay=1e-4)
    elif optimizer_mode == "full":
        optimizer = optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.05)
    elif optimizer_mode == "layerwise":
        optimizer = build_layerwise_lr_optimizer(
            model,
            head_lr=float(train_cfg["head_lr"]),
            layer_decay=float(train_cfg["layer_decay"]),
            weight_decay=float(train_cfg["weight_decay"]),
        )
    else:
        raise ValueError(f"Unknown optimizer_mode: {optimizer_mode}")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(train_cfg["epochs"]),
        eta_min=1e-6,
    )
    early_stopper = EarlyStopping(
        patience=int(train_cfg["patience"]),
        mode="max",
        min_delta=float(train_cfg["min_delta"]),
        save_path=str(save_path),
    )

    history = {
        "train_loss": [],
        "train_acc": [],
        "train_f1": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "time": [],
    }

    print("=" * 90)
    print("Experiment      :", exp_name)
    print("Use Aug         :", train_cfg["use_aug"])
    print("Oversampler     :", train_cfg["use_oversampler"])
    print("Focal Loss      :", train_cfg["use_focal_loss"])
    print("Freeze Backbone :", train_cfg["freeze_backbone"])
    print("Optimizer Mode  :", train_cfg["optimizer_mode"])
    print("=" * 90)

    for epoch in range(int(train_cfg["epochs"])):
        print(f"\nEpoch {epoch + 1}/{train_cfg['epochs']}")
        start = time.time()
        train_loss, train_acc, train_f1 = train_one_epoch(
            model,
            data_bundle["train_loader"],
            criterion,
            optimizer,
            device,
        )
        val_loss, val_acc, val_f1, _, _ = evaluate(
            model,
            data_bundle["val_loader"],
            criterion,
            device,
        )
        elapsed = time.time() - start
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_f1"].append(train_f1)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["time"].append(elapsed)

        print(
            f"[{exp_name}] Epoch {epoch + 1}/{train_cfg['epochs']} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_f1={train_f1:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f} | "
            f"time={elapsed:.1f}s"
        )

        early_stopper(val_acc, model)
        scheduler.step()
        if early_stopper.early_stop:
            print(f"[{exp_name}] Early stopping triggered.")
            break

    if save_path.exists():
        model.load_state_dict(torch.load(save_path, map_location=device))

    test_loss, test_acc, test_f1, y_true, y_pred = evaluate(
        model,
        data_bundle["test_loader"],
        criterion,
        device,
    )

    reports_dir = Path(paths_cfg["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(reports_dir / f"{exp_name}_history.csv", index=False)

    return {
        "exp_name": exp_name,
        "model": model,
        "model_path": str(save_path),
        "use_aug": train_cfg["use_aug"],
        "use_oversampler": train_cfg["use_oversampler"],
        "use_focal_loss": train_cfg["use_focal_loss"],
        "focal_gamma": train_cfg["focal_gamma"],
        "freeze_backbone": train_cfg["freeze_backbone"],
        "optimizer_mode": train_cfg["optimizer_mode"],
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_f1": test_f1,
        "history": history,
        "y_true": y_true,
        "y_pred": y_pred,
        "classes": train_ds.classes,
        "test_loader": data_bundle["test_loader"],
        "test_ds": data_bundle["test_ds"],
    }
