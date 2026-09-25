"""Grouped win-rate, grouped K-accuracy, and Top-K regressor comparison.

These extend the existing overall analyses with structural groupings
(family / subfamily / cluster_count / difficulty) while preserving the same
pairing-by-dataset_id and epsilon conventions.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from . import aggregators
from .method_registry import TOPK_ORDER as _TOPK_ORDER
from .winrate_analysis import (
    BASELINE_METHODS, EPSILON, REGRESSOR_METHODS, baselines_present,
    regressors_present, winrate_row,
)

REGRESSOR_METHODS = tuple(REGRESSOR_METHODS)


# --------------------------------------------------------------------------- #
# Grouped win-rate
# --------------------------------------------------------------------------- #
def grouped_winrate(
    dataset_level: pd.DataFrame,
    group_cols: List[str],
    *,
    methods: Optional[List[str]] = None,
    baselines: Optional[List[str]] = None,
    epsilon: float = EPSILON,
) -> pd.DataFrame:
    """Win/tie/loss per (group x method x baseline), pairing within each group."""
    if dataset_level.empty:
        return pd.DataFrame()
    if methods is None:
        methods = regressors_present(dataset_level)
    if baselines is None:
        baselines = baselines_present(dataset_level)
    cols = [c for c in group_cols if c in dataset_level.columns]
    if not cols:
        # Fall back to ungrouped.
        rows = []
        present = set(dataset_level["method_name"].unique())
        for m in methods:
            if m not in present:
                continue
            for b in baselines:
                if b not in present or b == m:
                    continue
                rows.append(winrate_row(dataset_level, m, b, epsilon))
        return pd.DataFrame(rows)

    out = []
    for key, g in dataset_level.groupby(cols, dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        present = set(g["method_name"].unique())
        for m in methods:
            if m not in present:
                continue
            for b in baselines:
                if b not in present or b == m:
                    continue
                row = {c: v for c, v in zip(cols, key_tuple)}
                row.update(winrate_row(g, m, b, epsilon))
                out.append(row)
    df = pd.DataFrame(out)
    if df.empty:
        return df
    sort_cols = cols + ["baseline", "method"]
    return df.sort_values([c for c in sort_cols if c in df.columns]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Grouped K-accuracy
# --------------------------------------------------------------------------- #
def k_accuracy_by(raw: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """Per (group x method) K accuracy / error / over-under rates."""
    if raw.empty:
        return pd.DataFrame()
    cols = [c for c in group_cols if c in raw.columns]
    success = raw[raw["status"] == "success"].copy()
    if success.empty or not cols:
        return pd.DataFrame()
    out = []
    for key, g in success.groupby(cols, dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        signed = pd.to_numeric(g["k_signed_error"], errors="coerce").dropna()
        absr = pd.to_numeric(g["k_abs_error"], errors="coerce").dropna()
        correct = g["k_correct"].dropna().astype(bool)
        n = int(signed.size)
        row = {c: v for c, v in zip(cols, key_tuple)}
        row.update({
            "n": n,
            "k_accuracy": float(correct.mean()) if len(correct) else float("nan"),
            "mean_k_abs_error": float(absr.mean()) if len(absr) else float("nan"),
            "median_k_abs_error": float(absr.median()) if len(absr) else float("nan"),
            "mean_k_signed_error": float(signed.mean()) if n else float("nan"),
            "overcluster_rate": float((signed > 0).sum() / n) if n else float("nan"),
            "undercluster_rate": float((signed < 0).sum() / n) if n else float("nan"),
            "exact_k_rate": float((signed == 0).sum() / n) if n else float("nan"),
        })
        out.append(row)
    df = pd.DataFrame(out)
    sort_cols = [c for c in cols if c != "method_name"] + (
        ["method_name"] if "method_name" in cols else [])
    return df.sort_values(sort_cols).reset_index(drop=True) if sort_cols else df


# --------------------------------------------------------------------------- #
# Top-K regressor comparison  (_TOPK_ORDER imported from method_registry)
# --------------------------------------------------------------------------- #
def topk_regressor_comparison(
    raw_with_trace: pd.DataFrame, *, by: Optional[List[str]] = None
) -> pd.DataFrame:
    """Compact comparison of the four Top-K regressor settings.

    Answers: is Top-1 too brittle, Top-10 too broad, Top-3/5 the sweet spot?
    """
    by = by or []
    cols = [c for c in by if c in raw_with_trace.columns]
    reg = raw_with_trace[raw_with_trace["method_name"].isin(_TOPK_ORDER)].copy()
    if reg.empty:
        return pd.DataFrame()
    # Lean on the shared summary block, then add entropy + top_k.
    summ = aggregators.summarize_by(reg, cols + ["method_name"])
    if summ.empty:
        return pd.DataFrame()
    # Weight entropy per (group x method).
    ent_cols = cols + ["method_name"]
    succ = reg[reg["status"] == "success"].copy()
    succ["_ent"] = pd.to_numeric(succ.get("metric_weight_entropy"), errors="coerce")
    ent = succ.groupby(ent_cols, dropna=False)["_ent"].mean().rename(
        "mean_weight_entropy").reset_index()
    summ = summ.merge(ent, on=ent_cols, how="left")
    summ["top_k"] = summ["method_name"].map(_TOPK_ORDER)
    keep = cols + [
        "method_name", "top_k", "count", "mean_ari", "median_ari", "mean_nmi",
        "mean_ami", "k_accuracy", "mean_runtime_sec", "mean_ari_gap_to_trace_best",
        "selected_is_trace_best_rate", "mean_weight_entropy",
    ]
    keep = [c for c in keep if c in summ.columns]
    out = summ[keep].copy()
    sort_cols = cols + ["top_k"]
    return out.sort_values([c for c in sort_cols if c in out.columns]).reset_index(drop=True)
