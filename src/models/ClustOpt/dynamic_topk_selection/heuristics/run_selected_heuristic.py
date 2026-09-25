"""Apply the selected Dynamic Top-K heuristic to every dataset-view record.

Reuses the Phase-1 extracted utility vectors (``records_utility_vectors.parquet``
+ ``records_utility_summary.parquet``) -- no rescanning of the raw repository.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from selected_dynamic_topk import (
    ALPHA, HEURISTIC_NAME, MAX_K, SOFT_CAP_BASE, clean_utilities, select_batch,
)

META = ["record_id", "dataset_id", "family_id", "subfamily_id", "view_type"]


def run(phase1_dir: str) -> pd.DataFrame:
    """Return the per-record assignment table for the selected heuristic."""
    summary = pd.read_parquet(
        os.path.join(phase1_dir, "records_utility_summary.parquet"),
        columns=[c for c in META + ["n_metrics", "status"]])
    wide = pd.read_parquet(
        os.path.join(phase1_dir, "records_utility_vectors.parquet"))

    df = summary.merge(wide, on="record_id", how="left")
    metric_cols = [c for c in wide.columns if c.startswith("metric_")]
    metric_names = [c[len("metric_"):] for c in metric_cols]
    values = df[metric_cols].to_numpy(dtype=float)

    selected_k, k_rel_raw, u_max = select_batch(values)

    # cleaned matrix + per-row descending order for metric extraction
    u_clean = clean_utilities(values)
    order = np.argsort(-u_clean, axis=1, kind="stable")     # N x M
    names_arr = np.array(metric_names)

    sel_names, sel_utils, sel_ranks = [], [], []
    for i in range(len(df)):
        k = int(selected_k[i])
        idx = order[i, :k]
        sel_names.append(list(names_arr[idx]))
        sel_utils.append([float(values[i, j]) for j in idx])
        sel_ranks.append(list(range(1, k + 1)))

    out = pd.DataFrame({
        "record_id": df["record_id"],
        "dataset_id": df["dataset_id"],
        "family": df["family_id"],
        "subfamily": df["subfamily_id"],
        "view_type": df["view_type"],
        "n_metrics": df["n_metrics"].astype(int),
        "u_max": u_max,
        "k_rel_raw": k_rel_raw.astype(int),
        "selected_k": selected_k.astype(int),
        "selected_metric_names": sel_names,
        "selected_metric_utilities": sel_utils,
        "selected_metric_ranks": sel_ranks,
        "heuristic_name": HEURISTIC_NAME,
        "alpha": ALPHA,
        "soft_cap_base": SOFT_CAP_BASE,
        "max_k": MAX_K,
    })
    return out


def save(df: pd.DataFrame, tables_dir: str):
    os.makedirs(tables_dir, exist_ok=True)
    df.to_parquet(os.path.join(tables_dir,
                               "selected_heuristic_assignments.parquet"),
                  index=False)
    # compact, human-readable CSV (list columns flattened to strings)
    compact = df.drop(columns=["selected_metric_utilities",
                               "selected_metric_ranks"]).copy()
    compact["selected_metric_names"] = df["selected_metric_names"].apply(
        lambda xs: "|".join(xs))
    compact.to_csv(os.path.join(tables_dir, "selected_heuristic_summary.csv"),
                   index=False)
