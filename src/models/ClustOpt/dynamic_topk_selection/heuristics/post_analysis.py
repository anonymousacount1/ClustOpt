"""Post-analysis of the selected-heuristic assignments: K distributions,
grouped statistics, and output validation (sections 8 & 9)."""

from __future__ import annotations

import numpy as np
import pandas as pd

MAX_K = 10


def k_stats(k: pd.Series) -> dict:
    """Summary statistics for a vector of selected K."""
    k = k.dropna().astype(int)
    n = len(k)
    out = {"n": int(n)}
    if n == 0:
        return out
    out["mean_K"] = float(k.mean())
    out["median_K"] = float(k.median())
    out["mode_K"] = int(k.mode().iloc[0])
    out["std_K"] = float(k.std(ddof=0))
    out["P_K1"] = float((k == 1).mean())
    out["P_2_5"] = float(k.between(2, 5).mean())
    out["P_gt5"] = float((k > 5).mean())
    out["P_eq10"] = float((k == 10).mean())
    return out


def global_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """K = 1..10 with count and percent."""
    k = df["selected_k"].astype(int)
    n = len(k)
    rows = []
    for kk in range(1, MAX_K + 1):
        c = int((k == kk).sum())
        rows.append({"K": kk, "count": c, "percent": 100.0 * c / n if n else 0.0})
    return pd.DataFrame(rows)


def grouped_summary(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Per-group K statistics + per-K percentages."""
    rows = []
    for g, sub in df.groupby(by):
        rec = {by: g}
        rec.update(k_stats(sub["selected_k"]))
        kk = sub["selected_k"].astype(int)
        for v in range(1, MAX_K + 1):
            rec[f"pct_K{v}"] = float((kk == v).mean() * 100)
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(by).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Validation (section 9)                                                      #
# --------------------------------------------------------------------------- #
def validate(df: pd.DataFrame, n_expected: int) -> dict:
    k = df["selected_k"]
    names = df["selected_metric_names"]
    utils = df["selected_metric_utilities"]

    is_int = bool(np.all([float(x).is_integer() for x in k.to_numpy()]))
    in_range = bool(((k >= 1) & (k <= MAX_K)).all())
    count_matches = bool((names.apply(len) == k).all())

    def _desc(xs):
        a = np.asarray(xs, dtype=float)
        return bool(np.all(np.diff(a) <= 1e-9))   # non-increasing

    sorted_desc = bool(utils.apply(_desc).all())
    no_nan = bool(utils.apply(
        lambda xs: not np.any(np.isnan(np.asarray(xs, dtype=float)))).all())

    rep = {
        "heuristic_ran_on_all_records": bool(len(df) == n_expected),
        "n_records": int(len(df)),
        "n_records_expected": int(n_expected),
        "no_records_dropped": bool(len(df) == n_expected),
        "selected_k_all_integer": is_int,
        "selected_k_in_range_1_10": in_range,
        "selected_count_equals_k": count_matches,
        "selected_metrics_sorted_desc": sorted_desc,
        "no_selected_metric_nan": no_nan,
        "record_count_matches_phase1": bool(len(df) == n_expected),
        "duplicate_record_ids": int(df["record_id"].duplicated().sum()),
    }
    rep["all_checks_pass"] = bool(
        rep["heuristic_ran_on_all_records"] and rep["no_records_dropped"]
        and is_int and in_range and count_matches and sorted_desc and no_nan
        and rep["record_count_matches_phase1"]
        and rep["duplicate_record_ids"] == 0)
    return rep
