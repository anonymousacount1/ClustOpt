"""
data_loader.py
==============

Load the unified training dataset and align it to the *exact* MLP schema, then
attach the experiment split assignment to every record.

The KNN reuses, without inference:

  * the MLP meta-feature columns and their order (CSV order, minus id/target),
  * the utility metric columns and their order (``utility__`` prefix, CSV order),
  * the record / dataset ids, the view naming, and the split assignments.

The loader aborts with a clear message if the feature or metric order cannot be
aligned deterministically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from . import config as C


@dataclass
class UnifiedData:
    """Everything the engine needs, with metadata kept separate from features."""

    feature_columns: List[str]
    target_columns: List[str]
    metric_names: List[str]                    # target_columns without the prefix
    X: np.ndarray                              # [N, 250] float64 (raw, uncleaned)
    Y: np.ndarray                              # [N, 60]  float64 (raw utilities)
    dataset_id: np.ndarray                     # [N] str
    record_id: np.ndarray                      # [N] str
    view: np.ndarray                           # [N] str  (x_only / y_only / xy_2d)
    split_id: np.ndarray                       # [N] int
    family: np.ndarray                         # [N] str
    subfamily: np.ndarray                      # [N] str
    difficulty: np.ndarray                     # [N] str
    report: Dict = field(default_factory=dict)

    @property
    def n_records(self) -> int:
        return int(self.X.shape[0])

    def view_mask(self, view: str) -> np.ndarray:
        return self.view == view

    def split_mask(self, split_ids) -> np.ndarray:
        return np.isin(self.split_id, np.asarray(list(split_ids), dtype=int))


def _detect_columns(all_columns: List[str]):
    id_set = set(C.ID_COLUMNS)
    target_columns = [c for c in all_columns if c.startswith(C.TARGET_PREFIX)]
    feature_columns = [
        c for c in all_columns if c not in id_set and c not in set(target_columns)
    ]
    return feature_columns, target_columns


def load_unified_data(
    csv_path: Path = C.UNIFIED_CSV,
    split_csv: Path = C.SPLIT_ASSIGNMENTS_CSV,
    limit_rows: int | None = None,
) -> UnifiedData:
    read_kwargs = {"nrows": int(limit_rows)} if limit_rows else {}
    df = pd.read_csv(csv_path, **read_kwargs)
    all_columns = list(df.columns)

    feature_columns, target_columns = _detect_columns(all_columns)

    # ---- deterministic-alignment guards ---------------------------------- #
    if len(feature_columns) != C.N_FEATURE_COLUMNS:
        raise ValueError(
            f"Detected {len(feature_columns)} feature columns, expected "
            f"{C.N_FEATURE_COLUMNS}. Cannot align KNN features to the MLP schema."
        )
    if len(target_columns) != C.N_TARGET_COLUMNS:
        raise ValueError(
            f"Detected {len(target_columns)} utility columns, expected "
            f"{C.N_TARGET_COLUMNS}. Cannot align KNN metrics to the MLP schema."
        )
    for req in (C.DATASET_ID_COLUMN, C.RECORD_ID_COLUMN, C.VIEW_COLUMN):
        if req not in df.columns:
            raise ValueError(f"Required id column '{req}' missing from CSV.")

    # Coerce feature / target columns to numeric (non-numeric -> NaN, handled
    # by downstream median imputation — never dropped silently).
    for col in feature_columns + target_columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ---- attach split assignment (dataset_id -> split_id) ----------------- #
    assign = pd.read_csv(split_csv, usecols=["dataset_id", "split_id"])
    id_to_split = {
        str(k): int(v) for k, v in zip(assign["dataset_id"], assign["split_id"])
    }
    ds_ids = df[C.DATASET_ID_COLUMN].astype(str).to_numpy()
    unknown = sorted({k for k in ds_ids if k not in id_to_split})
    if unknown:
        raise ValueError(
            f"{len(unknown)} dataset_id(s) have no split assignment "
            f"(e.g. {unknown[:5]}). Aborting to avoid silent leakage."
        )
    split_id = np.array([id_to_split[k] for k in ds_ids], dtype=int)

    # ---- metric names (strip the prefix, keep order) --------------------- #
    metric_names = [c[len(C.TARGET_PREFIX):] for c in target_columns]

    X = df[feature_columns].to_numpy(dtype=np.float64)
    Y = df[target_columns].to_numpy(dtype=np.float64)

    def _meta(col: str) -> np.ndarray:
        return df[col].astype(str).to_numpy() if col in df.columns else np.array(
            ["<missing>"] * len(df)
        )

    report = {
        "csv_path": str(csv_path),
        "split_csv": str(split_csv),
        "n_records": int(len(df)),
        "n_datasets": int(df[C.DATASET_ID_COLUMN].nunique()),
        "n_features": len(feature_columns),
        "n_targets": len(target_columns),
        "n_unique_record_id": int(df[C.RECORD_ID_COLUMN].nunique()),
        "record_id_unique": bool(df[C.RECORD_ID_COLUMN].nunique() == len(df)),
        "views": {v: int((df[C.VIEW_COLUMN] == v).sum()) for v in C.VIEWS},
        "unexpected_views": sorted(
            set(df[C.VIEW_COLUMN].astype(str)) - set(C.VIEWS)
        ),
        "split_counts": {
            int(s): int((split_id == s).sum()) for s in sorted(set(split_id))
        },
        "feature_nan_cells": int(np.isnan(X).sum()),
        "target_nan_cells": int(np.isnan(Y).sum()),
        "target_min": float(np.nanmin(Y)),
        "target_max": float(np.nanmax(Y)),
    }

    return UnifiedData(
        feature_columns=feature_columns,
        target_columns=target_columns,
        metric_names=metric_names,
        X=X,
        Y=Y,
        dataset_id=ds_ids,
        record_id=df[C.RECORD_ID_COLUMN].astype(str).to_numpy(),
        view=df[C.VIEW_COLUMN].astype(str).to_numpy(),
        split_id=split_id,
        family=_meta(C.FAMILY_COLUMN),
        subfamily=_meta(C.SUBFAMILY_COLUMN),
        difficulty=_meta(C.DIFFICULTY_COLUMN),
        report=report,
    )
