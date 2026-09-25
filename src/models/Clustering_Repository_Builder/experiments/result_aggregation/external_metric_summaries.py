"""Focused external-metric summaries (per method and per group).

The grouped statistic block in :mod:`aggregators` already carries mean external
metrics; this module provides a compact, metric-centric table (mean + std of
every external metric per method) that is convenient for reporting/plots.
"""
from __future__ import annotations

import pandas as pd

EXTERNAL_METRIC_COLS = [
    "ari", "nmi", "ami", "v_measure", "homogeneity", "completeness",
    "fowlkes_mallows", "purity",
]


def external_metric_table(raw: pd.DataFrame, *, group_col: str = "method_name") -> pd.DataFrame:
    if raw.empty or group_col not in raw.columns:
        return pd.DataFrame()
    success = raw[raw["status"] == "success"].copy()
    if success.empty:
        return pd.DataFrame()
    for c in EXTERNAL_METRIC_COLS:
        success[c] = pd.to_numeric(success.get(c), errors="coerce")
    agg_spec = {}
    for c in EXTERNAL_METRIC_COLS:
        agg_spec[f"mean_{c}"] = (c, "mean")
        agg_spec[f"std_{c}"] = (c, "std")
    out = success.groupby(group_col).agg(n=("ari", "size"), **agg_spec).reset_index()
    return out.sort_values("mean_ari", ascending=False).reset_index(drop=True)
