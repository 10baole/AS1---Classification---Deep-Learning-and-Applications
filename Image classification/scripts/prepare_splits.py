from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data.prepare import prepare_splits
from src.utils.paths import resolve_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    root = Path.cwd()
    config = load_config(args.config)
    paths_cfg = config["paths"]
    data_cfg = config["data"]

    images_root = resolve_path(root, paths_cfg["raw_images_dir"])
    split_dir = resolve_path(root, paths_cfg["split_dir"])
    output_root = resolve_path(root, paths_cfg["processed_dir"])

    results = prepare_splits(
        images_root=images_root,
        split_dir=split_dir,
        output_root=output_root,
        mode=data_cfg.get("split_mode", "symlink"),
        clean=args.clean,
    )

    print(f"Prepared splits in: {output_root}")
    for split_name, (created, skipped) in results.items():
        print(f"{split_name}: created={created}, skipped={skipped}")


if __name__ == "__main__":
    main()
