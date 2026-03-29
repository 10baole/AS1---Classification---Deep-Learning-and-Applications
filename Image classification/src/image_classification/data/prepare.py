from __future__ import annotations

import os
import shutil
from pathlib import Path

from .manifests import iter_split_entries


def materialize_split(
    split_file: Path,
    images_root: Path,
    output_root: Path,
    mode: str = "copy",
) -> tuple[int, int]:
    created = 0
    skipped = 0

    split_name = split_file.stem
    split_root = output_root / split_name
    split_root.mkdir(parents=True, exist_ok=True)

    for entry in iter_split_entries(split_file):
        src_path = images_root / entry["class_name"] / entry["file_name"]
        dst_dir = split_root / entry["class_name"]
        dst_path = dst_dir / entry["file_name"]

        if not src_path.exists():
            skipped += 1
            continue

        dst_dir.mkdir(parents=True, exist_ok=True)

        if dst_path.exists():
            skipped += 1
            continue

        if mode == "copy":
            shutil.copy2(src_path, dst_path)
        elif mode == "hardlink":
            os.link(src_path, dst_path)
        elif mode == "symlink":
            os.symlink(src_path.resolve(), dst_path)
        else:
            raise ValueError(f"Unsupported split materialization mode: {mode}")
        created += 1

    return created, skipped


def prepare_splits(
    images_root: Path,
    split_dir: Path,
    output_root: Path,
    mode: str = "copy",
    clean: bool = False,
) -> dict[str, tuple[int, int]]:
    if clean and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    results = {}
    for split_name in ("train", "val", "test"):
        split_file = split_dir / f"{split_name}.txt"
        results[split_name] = materialize_split(
            split_file=split_file,
            images_root=images_root,
            output_root=output_root,
            mode=mode,
        )
    return results
