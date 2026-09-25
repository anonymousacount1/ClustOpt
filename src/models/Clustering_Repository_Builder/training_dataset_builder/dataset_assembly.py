"""Pure functions used by build_training_dataset.py.

Each dataset on disk has three feature rows (one per view: x_only, y_only,
xy_2d) and three utility_vector.json files (one per view). Merging means:
take every feature row, find the matching utility vector by view, and append
the 60 utility__<metric> scores. Helper columns from metric_utilities.csv are
intentionally ignored — only the final per-metric utility score is kept, as
that is the regression target.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd

VIEW_IDS = ("x_only", "y_only", "xy_2d")


@dataclass
class MissingArtifact:
    family: str
    subfamily: str
    dataset_id: str
    view_id: str
    reason: str


def iter_datasets(analyzed_data_dir: Path) -> Iterator[tuple[str, str, Path]]:
    """Yield (family, subfamily, dataset_dir) for every dataset on disk.

    A dataset is recognized by the presence of features/features_records.csv.
    Skips the output folder itself and any other non-family directories.
    """
    for fam_dir in sorted(analyzed_data_dir.iterdir()):
        if not fam_dir.is_dir():
            continue
        sub_root = fam_dir / "subfamilies"
        if not sub_root.is_dir():
            continue
        for sub_dir in sorted(sub_root.iterdir()):
            if not sub_dir.is_dir():
                continue
            for ds_dir in sorted(sub_dir.iterdir()):
                if not ds_dir.is_dir():
                    continue
                if not (ds_dir / "features" / "features_records.csv").is_file():
                    continue
                yield fam_dir.name, sub_dir.name, ds_dir


def load_features(dataset_dir: Path) -> pd.DataFrame:
    return pd.read_csv(dataset_dir / "features" / "features_records.csv")


def load_utilities(dataset_dir: Path, view_id: str) -> dict | None:
    """Return the 60-key utilities dict for a given view, or None if missing."""
    p = dataset_dir / "utility" / view_id / "utility_vector.json"
    if not p.is_file():
        return None
    with p.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("utilities")


def assemble_per_dataset(
    family: str,
    subfamily: str,
    dataset_dir: Path,
    missing: list[MissingArtifact],
) -> list[dict]:
    """Build merged (feature + utility) rows for one dataset — one per view."""
    try:
        feats = load_features(dataset_dir)
    except Exception as e:
        missing.append(
            MissingArtifact(family, subfamily, dataset_dir.name, "*",
                            f"features read failed: {e!r}")
        )
        return []

    rows: list[dict] = []
    for _, row in feats.iterrows():
        view_id = str(row["view_mode"])
        utilities = load_utilities(dataset_dir, view_id)
        if utilities is None:
            missing.append(
                MissingArtifact(family, subfamily, dataset_dir.name, view_id,
                                "utility_vector.json missing")
            )
            continue
        merged = row.to_dict()
        merged.update(utilities)
        rows.append(merged)
    return rows


def atomic_write_csv(df: pd.DataFrame, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(suffix=".tmp.csv", dir=str(target.parent))
    os.close(fd)
    tmp = Path(tmp_path)
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def atomic_write_json(obj: dict, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(suffix=".tmp.json", dir=str(target.parent))
    os.close(fd)
    tmp = Path(tmp_path)
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, default=str)
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
