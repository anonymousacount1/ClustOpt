"""Build a dataset-level index table from the analyzed-data repository.

Walks the canonical layout::

    <analyzed_data_root>/<family>/subfamilies/<subfamily>/<dataset_folder>/

and reads each dataset's ``data_metadata.json`` (``tags`` block) to produce one
row per *raw dataset* (not per view). The resulting DataFrame is the input to
the stratified splitter.

The walk is intentionally tolerant: folders without ``data_data.csv`` /
``data_metadata.json`` are skipped, and metadata fields fall back to path-based
inference when absent so a partial repository never crashes the index build.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

import pandas as pd

INDEX_COLUMNS: List[str] = [
    "dataset_id",
    "dataset_dir",
    "family",
    "family_id",
    "subfamily",
    "subfamily_id",
    "specific_subfamily",
    "difficulty",
    "cluster_count",
    "n_points",
]


@dataclass
class IndexBuildStats:
    n_datasets: int = 0
    n_skipped_no_data: int = 0
    n_metadata_errors: int = 0


def _read_json(path: Path) -> Optional[dict]:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def _iter_subfamily_dirs(analyzed_data_root: Path) -> Iterator[Path]:
    """Yield every ``.../<family>/subfamilies/<subfamily>`` directory."""
    for family_dir in sorted(p for p in analyzed_data_root.iterdir() if p.is_dir()):
        subfamilies_root = family_dir / "subfamilies"
        if not subfamilies_root.is_dir():
            continue
        for subfamily_dir in sorted(p for p in subfamilies_root.iterdir() if p.is_dir()):
            yield subfamily_dir


def _dataset_row(dataset_dir: Path) -> Optional[dict]:
    """Build one index row from a dataset folder, or None if not a dataset."""
    if not (dataset_dir / "data_data.csv").is_file():
        return None

    metadata = _read_json(dataset_dir / "data_metadata.json") or {}
    tags = metadata.get("tags") if isinstance(metadata.get("tags"), dict) else {}

    # Path-based fallbacks: .../<family>/subfamilies/<subfamily>/<dataset>
    parts = dataset_dir.resolve().parts
    family_from_path = subfamily_from_path = None
    if "subfamilies" in parts:
        idx = parts.index("subfamilies")
        if idx - 1 >= 0:
            family_from_path = parts[idx - 1]
        if idx + 1 < len(parts):
            subfamily_from_path = parts[idx + 1]

    family_id = tags.get("family_id") or family_from_path
    family_name = tags.get("family_name") or family_id
    subfamily_id = tags.get("subfamily_id") or subfamily_from_path

    cluster_count = tags.get("cluster_count")
    if cluster_count is None:
        sgt = metadata.get("structural_ground_truth")
        if isinstance(sgt, dict):
            if isinstance(sgt.get("clusters"), list):
                cluster_count = len(sgt["clusters"])
            elif sgt.get("n_clusters") is not None:
                cluster_count = sgt.get("n_clusters")

    n_points = tags.get("n_points_requested") or tags.get("n_points")

    return {
        "dataset_id": dataset_dir.name,
        "dataset_dir": str(dataset_dir.resolve()),
        "family": family_name,
        "family_id": family_id,
        "subfamily": subfamily_id,
        "subfamily_id": subfamily_id,
        "specific_subfamily": subfamily_id,
        "difficulty": tags.get("difficulty"),
        "cluster_count": cluster_count,
        "n_points": n_points,
    }


def build_dataset_index(
    analyzed_data_root: str | Path,
    *,
    progress_every: int = 2000,
    verbose: bool = True,
) -> tuple[pd.DataFrame, IndexBuildStats]:
    """Scan the repository and return ``(index_df, stats)``.

    One row per raw dataset, columns = :data:`INDEX_COLUMNS`. Datasets are
    de-duplicated by ``dataset_id`` (first occurrence wins) so the three views
    can never be split apart downstream.
    """
    analyzed_data_root = Path(analyzed_data_root)
    if not analyzed_data_root.is_dir():
        raise FileNotFoundError(f"analyzed-data root not found: {analyzed_data_root}")

    rows: List[dict] = []
    stats = IndexBuildStats()
    seen_ids: set[str] = set()

    for subfamily_dir in _iter_subfamily_dirs(analyzed_data_root):
        for entry in sorted(p for p in subfamily_dir.iterdir() if p.is_dir()):
            row = _dataset_row(entry)
            if row is None:
                stats.n_skipped_no_data += 1
                continue
            if row["dataset_id"] in seen_ids:
                continue
            seen_ids.add(row["dataset_id"])
            rows.append(row)
            stats.n_datasets += 1
            if verbose and stats.n_datasets % progress_every == 0:
                print(f"[index] scanned {stats.n_datasets} datasets...", flush=True)

    df = pd.DataFrame(rows, columns=INDEX_COLUMNS)
    if verbose:
        print(f"[index] done: {len(df)} datasets indexed.", flush=True)
    return df, stats
