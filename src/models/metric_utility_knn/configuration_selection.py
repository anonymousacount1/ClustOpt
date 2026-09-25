"""
configuration_selection.py
=========================

Four-group composite ranking (plan §19) and the one-standard-error selection
rule (plan §20), operating on the pooled (``level == 'ALL'``) LOSO fold metrics.

Composite scoring is done *per fold*: within a fold the 96 configurations are
ranked on each metric, converted to a direction-aware percentile in [0, 1],
averaged within each of the four groups, and the four group scores are averaged
equally into one composite score.  The per-configuration mean and standard error
of that composite across the 15 folds drive the one-SE selection.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from . import config as C

CONFIG_KEYS = ["config_id", "n_neighbors", "scaler", "distance", "neighbor_weighting"]


def _metric_percentile(values: pd.Series, higher_is_better: bool) -> pd.Series:
    """Direction-aware percentile in [0, 1]; NaN -> neutral 0.5."""
    r = values.rank(pct=True, ascending=higher_is_better, method="average")
    return r.fillna(0.5)


def compute_fold_composite(fold_metrics_all: pd.DataFrame) -> pd.DataFrame:
    """Per-(fold, config) group scores + overall composite score."""
    records: List[dict] = []
    for fold, sub in fold_metrics_all.groupby("fold"):
        sub = sub.reset_index(drop=True)
        group_score: Dict[str, pd.Series] = {}
        for gname, metrics in C.RANKING_GROUPS.items():
            parts = []
            for metric, higher in metrics.items():
                if metric not in sub.columns:
                    raise KeyError(f"Ranking metric '{metric}' missing from metrics.")
                parts.append(_metric_percentile(sub[metric], higher))
            group_score[gname] = pd.concat(parts, axis=1).mean(axis=1)
        composite = pd.concat(group_score.values(), axis=1).mean(axis=1)
        for i in range(len(sub)):
            rec = {"fold": int(fold)}
            for k in CONFIG_KEYS:
                rec[k] = sub.loc[i, k]
            for gname in C.RANKING_GROUPS:
                rec[f"score_{gname}"] = float(group_score[gname].iloc[i])
            rec["composite_score"] = float(composite.iloc[i])
            records.append(rec)
    return pd.DataFrame(records)


def aggregate_composite(fold_composite: pd.DataFrame,
                        fold_metrics_all: pd.DataFrame) -> pd.DataFrame:
    """Per-config mean/std/SE of the composite across folds (+ mean query time)."""
    n_folds = fold_composite["fold"].nunique()
    qt = (fold_metrics_all.groupby("config_id")["query_time_sec"].mean()
          .rename("mean_query_time_sec"))

    grp = fold_composite.groupby(CONFIG_KEYS, as_index=False)
    agg = grp["composite_score"].agg(
        mean_composite_score="mean", std_composite_score="std",
        worst_fold_score="min", best_fold_score="max",
    )
    for gname in C.RANKING_GROUPS:
        agg[f"mean_score_{gname}"] = (
            grp[f"score_{gname}"].mean()[f"score_{gname}"].values
        )
    agg["std_composite_score"] = agg["std_composite_score"].fillna(0.0)
    agg["standard_error"] = agg["std_composite_score"] / np.sqrt(max(n_folds, 1))
    agg = agg.merge(qt, on="config_id", how="left")
    agg = agg.sort_values("mean_composite_score", ascending=False,
                          ignore_index=True)
    agg.insert(0, "rank", np.arange(1, len(agg) + 1))
    return agg


def _simplicity_key(row) -> Tuple[int, int, int]:
    return (
        0 if row["scaler"] == "standard" else 1,
        0 if row["distance"] == "euclidean" else 1,
        0 if row["neighbor_weighting"] == "uniform" else 1,
    )


def apply_one_se_rule(ranking: pd.DataFrame) -> Dict:
    """Select within one SE of the best mean composite (plan §20)."""
    best = ranking.iloc[0]
    best_mean = float(best["mean_composite_score"])
    best_se = float(best["standard_error"])
    threshold = best_mean - best_se

    candidates = ranking[ranking["mean_composite_score"] >= threshold].copy()
    # 1) smallest n_neighbors; 2) lower query runtime; 3) simplicity preferences.
    candidates["_simp"] = candidates.apply(_simplicity_key, axis=1)
    candidates = candidates.sort_values(
        by=["n_neighbors", "mean_query_time_sec", "_simp"],
        ascending=[True, True, True], kind="stable",
    )
    selected = candidates.iloc[0]

    def _cfg(row) -> Dict:
        return {
            "config_id": row["config_id"],
            "n_neighbors": int(row["n_neighbors"]),
            "scaler": row["scaler"],
            "distance": row["distance"],
            "neighbor_weighting": row["neighbor_weighting"],
            "mean_composite_score": float(row["mean_composite_score"]),
            "std_composite_score": float(row["std_composite_score"]),
            "standard_error": float(row["standard_error"]),
            "worst_fold_score": float(row["worst_fold_score"]),
            "best_fold_score": float(row["best_fold_score"]),
        }

    return {
        "raw_best_configuration": _cfg(best),
        "one_se_selected_configuration": _cfg(selected),
        "one_se_threshold": threshold,
        "n_candidates_within_one_se": int(len(candidates)),
        "candidate_config_ids": candidates["config_id"].tolist(),
        "selection_rule": (
            "within one SE of best mean composite; then smallest n_neighbors, "
            "then lower query time, then simplicity (standard/euclidean/uniform)"
        ),
    }


def selected_knn_config(selection: Dict) -> C.KnnConfig:
    c = selection["one_se_selected_configuration"]
    return C.KnnConfig(
        n_neighbors=int(c["n_neighbors"]), scaler=c["scaler"],
        distance=c["distance"], neighbor_weighting=c["neighbor_weighting"],
    )
