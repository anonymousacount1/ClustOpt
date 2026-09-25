"""Metric-selection analysis for the regressor (dynamic) methods.

Aggregates which metrics the regressor selected, how often, their average
weight, the family each metric belongs to, and the weight entropy. Helps show
the model selects different metrics for different data types.
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional

import pandas as pd

try:
    from models.metric_utility_mlp.head_groups import METRIC_HEAD_GROUPS
    _METRIC_TO_FAMILY = {
        m: group for group, metrics in METRIC_HEAD_GROUPS.items() for m in metrics
    }
except Exception:  # pragma: no cover - keep aggregation working without the MLP pkg
    _METRIC_TO_FAMILY = {}


def _iter_selected(raw: pd.DataFrame):
    """Yield (row, {metric: weight}) for regressor-group successful runs."""
    reg = raw[(raw["status"] == "success") & (raw["method_group"] == "regressor")]
    for _, row in reg.iterrows():
        weights_raw = row.get("selected_metric_weights")
        if not weights_raw:
            continue
        try:
            weights = json.loads(weights_raw) if isinstance(weights_raw, str) else dict(weights_raw)
        except Exception:
            continue
        if weights:
            yield row, weights


def metric_frequency(raw: pd.DataFrame, *, by: Optional[List[str]] = None) -> pd.DataFrame:
    """Selected-metric frequency + mean weight, optionally grouped by columns."""
    by = by or []
    records: List[Dict] = []
    for row, weights in _iter_selected(raw):
        group_vals = {c: row.get(c) for c in by}
        for metric, weight in weights.items():
            records.append({
                **group_vals,
                "metric": metric,
                "metric_family": _METRIC_TO_FAMILY.get(metric, "other"),
                "weight": float(weight),
            })
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    group_cols = by + ["metric", "metric_family"]
    agg = df.groupby(group_cols).agg(
        selection_count=("weight", "size"),
        mean_weight=("weight", "mean"),
        total_weight=("weight", "sum"),
    ).reset_index()
    # Top-metric frequency: count of times each metric was the max-weight one.
    return agg.sort_values(by + ["selection_count"], ascending=[True] * len(by) + [False]
                           ).reset_index(drop=True)


def metric_family_frequency(raw: pd.DataFrame, *, by: Optional[List[str]] = None) -> pd.DataFrame:
    by = by or []
    records: List[Dict] = []
    for row, weights in _iter_selected(raw):
        group_vals = {c: row.get(c) for c in by}
        for metric, weight in weights.items():
            records.append({
                **group_vals,
                "metric_family": _METRIC_TO_FAMILY.get(metric, "other"),
                "weight": float(weight),
            })
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    return df.groupby(by + ["metric_family"]).agg(
        selection_count=("weight", "size"),
        mean_weight=("weight", "mean"),
    ).reset_index()


def top_metric_frequency(raw: pd.DataFrame, *, by: Optional[List[str]] = None) -> pd.DataFrame:
    """How often each metric was the single highest-weight (top) metric."""
    by = by or []
    reg = raw[(raw["status"] == "success") & (raw["method_group"] == "regressor")]
    reg = reg[reg["top_metric"].notna()]
    if reg.empty:
        return pd.DataFrame()
    return reg.groupby(by + ["top_metric"]).size().rename("top_count").reset_index()


def weight_entropy_summary(raw: pd.DataFrame, *, by: Optional[List[str]] = None) -> pd.DataFrame:
    by = by or ["method_name"]
    reg = raw[(raw["status"] == "success") & (raw["method_group"] == "regressor")].copy()
    if reg.empty:
        return pd.DataFrame()
    reg["_ent"] = pd.to_numeric(reg["metric_weight_entropy"], errors="coerce")
    return reg.groupby(by).agg(
        mean_weight_entropy=("_ent", "mean"),
        median_weight_entropy=("_ent", "median"),
        n=("_ent", "size"),
    ).reset_index()
