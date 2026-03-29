from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from image_classification.config import load_config
from image_classification.engine.trainer import run_experiment
from image_classification.utils.paths import resolve_path
from image_classification.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_vit.yaml")
    args = parser.parse_args()

    root = Path.cwd()
    config = load_config(args.config)
    set_seed(int(config["project"]["seed"]))

    processed_dir = resolve_path(root, config["paths"]["processed_dir"])
    if not processed_dir.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {processed_dir}. Run scripts/prepare_splits.py first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    result = run_experiment(config=config, processed_dir=processed_dir, device=device)
    print(
        f"{result['exp_name']} | "
        f"test_loss={result['test_loss']:.4f} | "
        f"test_acc={result['test_acc']:.4f} | "
        f"test_f1={result['test_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
