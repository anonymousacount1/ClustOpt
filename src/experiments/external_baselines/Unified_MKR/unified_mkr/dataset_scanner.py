"""Discover dataset folders under a subfamily root and attach split metadata.

A dataset folder is any directory containing ``data_data.csv``. The split
assignment CSV (``experiment_splits/dataset_split_assignments.csv``) supplies
``split_id`` / family / subfamily / difficulty / cluster_count, keyed by
``dataset_id`` (the folder name).
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class SplitInfo:
    split_id: Optional[int]
    family: str = ""
    subfamily: str = ""
    difficulty: str = ""
    cluster_count: Optional[int] = None
    n_points: Optional[int] = None


def discover_dataset_dirs(subfamily_root: str | Path) -> List[Path]:
    """All folders under ``subfamily_root`` that contain ``data_data.csv``."""
    root = Path(subfamily_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Subfamily root not found: {root}")
    found = []
    for csv_path in root.rglob("data_data.csv"):
        found.append(csv_path.parent)
    return sorted(set(found), key=lambda p: p.name)


def _to_int(value: str) -> Optional[int]:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def load_split_assignments(csv_path: str | Path) -> Dict[str, SplitInfo]:
    """Map dataset_id -> SplitInfo from the split assignment CSV."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Split assignment CSV not found: {path}")
    out: Dict[str, SplitInfo] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            ds_id = row.get("dataset_id")
            if not ds_id:
                continue
            out[ds_id] = SplitInfo(
                split_id=_to_int(row.get("split_id", "")),
                family=row.get("family", "") or row.get("family_id", ""),
                subfamily=row.get("subfamily", "") or row.get("subfamily_id", ""),
                difficulty=row.get("difficulty", ""),
                cluster_count=_to_int(row.get("cluster_count", "")),
                n_points=_to_int(row.get("n_points", "")),
            )
    return out


def split_info_for(
    dataset_id: str, assignments: Dict[str, SplitInfo]
) -> SplitInfo:
    return assignments.get(dataset_id, SplitInfo(split_id=None))
