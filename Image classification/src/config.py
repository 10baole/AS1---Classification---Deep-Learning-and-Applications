from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Override values take priority."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | Path, base_path: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML config, optionally merging on top of a base config.

    If *base_path* is provided, the base config is loaded first and the
    model-specific config is deep-merged on top of it.
    """
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {path}")

    if base_path is not None:
        base = Path(base_path)
        with base.open("r", encoding="utf-8") as f:
            base_config = yaml.safe_load(f)
        if not isinstance(base_config, dict):
            raise ValueError(f"Base config must be a mapping: {base}")
        config = _deep_merge(base_config, config)

    return config
