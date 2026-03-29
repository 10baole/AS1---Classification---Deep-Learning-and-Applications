from __future__ import annotations

from pathlib import Path


def iter_split_entries(split_file: Path):
    with split_file.open("r", encoding="utf-8") as f:
        for line in f:
            rel_path = line.strip()
            if not rel_path:
                continue
            parts = Path(rel_path).parts
            if len(parts) < 3:
                continue
            yield {
                "split": parts[0],
                "class_name": parts[1],
                "file_name": parts[-1],
                "relative_path": rel_path,
            }
