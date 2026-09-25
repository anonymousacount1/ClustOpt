"""Load a single dataset folder for utility generation.

This is intentionally minimal: we only need the raw points and the ground-truth
cluster labels. Feature records (``features/features_records.csv``) are NOT
loaded here — they will be merged with the utility outputs in a later phase.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class LoadedDatasetForUtility:
    dataset_dir: Path
    dataset_id: str
    X: np.ndarray                 # shape (n, n_features), no label column
    feature_names: list[str]      # CSV columns kept as X (label column excluded)
    y_true: np.ndarray            # shape (n,) ground-truth labels
    metadata: dict
    structural_ground_truth: Optional[dict]

    family_id: Optional[str] = None
    family_name: Optional[str] = None
    subfamily_id: Optional[str] = None
    specific_subfamily: Optional[str] = None
    cluster_count_from_generator: Optional[int] = None

    @property
    def n_points(self) -> int:
        return int(self.X.shape[0])


def _read_json(path: Path) -> Optional[dict]:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _label_column_name(df: pd.DataFrame) -> Optional[str]:
    for c in ("cluster_id", "label", "labels", "y_true"):
        if c in df.columns:
            return c
    return None


def _infer_family_subfamily(dataset_dir: Path) -> tuple[Optional[str], Optional[str]]:
    parts = dataset_dir.resolve().parts
    if "subfamilies" not in parts:
        return None, None
    idx = parts.index("subfamilies")
    family_id = parts[idx - 1] if idx - 1 >= 0 else None
    subfamily_id = parts[idx + 1] if idx + 1 < len(parts) else None
    return family_id, subfamily_id


def load_dataset_for_utility(dataset_dir: Path) -> LoadedDatasetForUtility:
    dataset_dir = Path(dataset_dir)

    csv_path = dataset_dir / "data_data.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing data_data.csv in {dataset_dir}")

    df = pd.read_csv(csv_path)
    label_col = _label_column_name(df)
    if label_col is None:
        raise ValueError(
            f"Could not find a label column ('cluster_id'/'label'/'labels'/'y_true') "
            f"in {csv_path}"
        )
    y_true = df[label_col].to_numpy()

    metadata = _read_json(dataset_dir / "data_metadata.json") or {}
    structural_gt = _read_json(dataset_dir / "data_structural_ground_truth.json")

    feat_names_meta = None
    if isinstance(metadata.get("feature_names"), list):
        feat_names_meta = [str(n) for n in metadata["feature_names"]]

    if feat_names_meta:
        feature_cols = [c for c in feat_names_meta if c in df.columns]
        if not feature_cols:
            feature_cols = [c for c in df.columns if c != label_col]
    else:
        feature_cols = [c for c in df.columns if c != label_col]

    if not feature_cols:
        raise ValueError(f"No usable feature columns in {csv_path}")

    X = df[feature_cols].to_numpy(dtype=float, na_value=np.nan)
    if X.ndim != 2 or X.shape[1] == 0:
        raise ValueError(
            f"Dataset {dataset_dir} has unexpected feature matrix shape {X.shape}"
        )

    family_id_path, subfamily_id_path = _infer_family_subfamily(dataset_dir)
    family_id = metadata.get("family_id") or family_id_path
    family_name = metadata.get("family_name") or family_id
    subfamily_id = metadata.get("subfamily_id") or subfamily_id_path

    cluster_count = None
    if isinstance(metadata.get("n_clusters"), int):
        cluster_count = int(metadata["n_clusters"])
    elif structural_gt is not None:
        if isinstance(structural_gt.get("clusters"), list):
            cluster_count = len(structural_gt["clusters"])
        elif structural_gt.get("n_clusters") is not None:
            try:
                cluster_count = int(structural_gt["n_clusters"])
            except (ValueError, TypeError):
                cluster_count = None

    return LoadedDatasetForUtility(
        dataset_dir=dataset_dir,
        dataset_id=dataset_dir.name,
        X=X,
        feature_names=feature_cols,
        y_true=np.asarray(y_true),
        metadata=metadata,
        structural_ground_truth=structural_gt,
        family_id=family_id,
        family_name=family_name,
        subfamily_id=subfamily_id,
        specific_subfamily=subfamily_id,
        cluster_count_from_generator=cluster_count,
    )
