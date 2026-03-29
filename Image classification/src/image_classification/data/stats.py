from __future__ import annotations

import os
from pathlib import Path

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")


def count_all_images(root: Path) -> int:
    total = 0
    for _, _, files in os.walk(root):
        total += sum(1 for f in files if f.lower().endswith(IMAGE_EXTS))
    return total


def get_class_counts(images_root: Path) -> dict[str, int]:
    class_counts: dict[str, int] = {}
    for class_name in sorted(os.listdir(images_root)):
        class_path = images_root / class_name
        if class_path.is_dir():
            count = sum(
                1 for f in os.listdir(class_path)
                if f.lower().endswith(IMAGE_EXTS)
            )
            class_counts[class_name] = count
    return class_counts
