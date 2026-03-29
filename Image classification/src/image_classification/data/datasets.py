from __future__ import annotations

from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder

from .transforms import get_transforms


def build_weighted_sampler(dataset: ImageFolder):
    labels = [label for _, label in dataset.samples]
    class_counts = Counter(labels)
    class_weights = {cls_idx: 1.0 / count for cls_idx, count in class_counts.items()}
    sample_weights = torch.DoubleTensor([class_weights[label] for label in labels])
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler, class_counts, class_weights


def get_dataloaders(
    data_root: str | Path,
    batch_size: int = 32,
    image_size: int = 224,
    use_aug: bool = False,
    use_oversampler: bool = False,
    num_workers: int = 2,
):
    data_root = Path(data_root)
    train_tf, eval_tf = get_transforms(image_size=image_size, use_aug=use_aug)

    train_ds = ImageFolder(data_root / "train", transform=train_tf)
    val_ds = ImageFolder(data_root / "val", transform=eval_tf)
    test_ds = ImageFolder(data_root / "test", transform=eval_tf)

    train_sampler = None
    train_class_counts = None
    train_class_weights = None

    if use_oversampler:
        train_sampler, train_class_counts, train_class_weights = build_weighted_sampler(train_ds)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return {
        "train_ds": train_ds,
        "val_ds": val_ds,
        "test_ds": test_ds,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "train_class_counts": train_class_counts,
        "train_class_weights": train_class_weights,
    }
