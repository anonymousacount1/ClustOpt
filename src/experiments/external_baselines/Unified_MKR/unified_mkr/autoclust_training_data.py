"""Training-data assembly for the derived AutoClust repositories.

Two targets, mirroring AutoClust (Experiments/RelatedWork/AutoClust.py):
* algorithm labels: per record, the algorithm of the max-ARI configuration
  (AutoClust picks the best-ARI algorithm of the most-similar dataset);
* ARI regression rows: one per (record, config) valid evaluation, with the CVI
  vector as input and ARI as target (AutoClust's MLP CVI -> ARI surrogate).
"""
from __future__ import annotations

from typing import List

import pandas as pd

_MF_NON_FEATURE = {
    "record_id", "dataset_id", "view_type", "split_id",
    "ml2dac_status", "ml2dac_runtime_sec", "ml2dac_error",
    "autoclust_status", "autoclust_runtime_sec", "autoclust_error",
}

ALGO_LABEL_COLUMNS = [
    "record_id", "split_id", "family", "subfamily", "view_type",
    "best_algorithm", "best_config_id", "best_ari",
]


def autoclust_feature_columns(metafeatures: pd.DataFrame) -> List[str]:
    """MeanShift landmarking CVI columns (autoclust_*), excluding bookkeeping."""
    return [
        c for c in metafeatures.columns
        if c.startswith("autoclust_") and c not in _MF_NON_FEATURE
    ]


def build_algorithm_labels(
    train_evaluations: pd.DataFrame, records: pd.DataFrame
) -> pd.DataFrame:
    """best_algorithm = algorithm of the max-ARI valid config for each record."""
    valid = train_evaluations[train_evaluations["valid_result"] == True]  # noqa: E712
    valid = valid.dropna(subset=["ari"])
    # deterministic argmax: highest ARI, then config_id tiebreak
    valid = valid.sort_values(["record_id", "ari", "config_id"],
                              ascending=[True, False, True], kind="mergesort")
    best = valid.groupby("record_id", as_index=False).first()
    meta = records[["record_id", "split_id", "family", "subfamily", "view_type"]].drop_duplicates("record_id")
    out = best.merge(meta, on="record_id", how="left", suffixes=("", "_rec"))
    out = out.rename(columns={"algorithm": "best_algorithm", "config_id": "best_config_id", "ari": "best_ari"})
    # split_id present in both; prefer records' split_id
    if "split_id_rec" in out.columns:
        out["split_id"] = out["split_id_rec"]
        out = out.drop(columns=[c for c in out.columns if c.endswith("_rec")])
    return out[ALGO_LABEL_COLUMNS]


def build_ari_regression_dataset(
    train_evaluations: pd.DataFrame, records: pd.DataFrame, candidate_ids: List[str]
) -> tuple[pd.DataFrame, dict]:
    """One row per valid (record, config); CVI candidate columns + ARI target."""
    n_before = int(len(train_evaluations))
    df = train_evaluations[train_evaluations["valid_result"] == True].dropna(subset=["ari"])  # noqa: E712
    n_after = int(len(df))
    meta = records[["record_id", "family", "subfamily"]].drop_duplicates("record_id")
    keep = ["record_id", "split_id", "view_type", "config_id", "algorithm", "ari"] + \
           [m for m in candidate_ids if m in df.columns]
    out = df[keep].merge(meta, on="record_id", how="left")
    cols = ["record_id", "split_id", "family", "subfamily", "view_type",
            "config_id", "algorithm", "ari"] + [m for m in candidate_ids if m in df.columns]
    out = out[cols]
    nan_rate = {m: float(out[m].isna().mean()) for m in candidate_ids if m in out.columns}
    info = {
        "n_rows_before": n_before,
        "n_rows_after": n_after,
        "n_rows_dropped_invalid_or_nan_ari": n_before - n_after,
        "nan_rate_by_feature": nan_rate,
    }
    return out, info
