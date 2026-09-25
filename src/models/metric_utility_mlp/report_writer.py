"""
report_writer.py
================

Markdown report generation for per-fold results and the overall run summary.
Pure string building - no side effects beyond returning the markdown text
(callers persist via :mod:`artifact_writer`).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def _fmt(v) -> str:
    if isinstance(v, float):
        if np.isnan(v):
            return "nan"
        return f"{v:.4f}"
    return str(v)


def _metric_table(metrics: Dict[str, float], keys: List[str]) -> str:
    lines = ["| metric | value |", "| --- | --- |"]
    for k in keys:
        if k in metrics:
            lines.append(f"| {k} | {_fmt(metrics[k])} |")
    return "\n".join(lines)


def _df_table(df: pd.DataFrame, max_rows: int = 15, float_fmt: str = "{:.4f}") -> str:
    if df is None or df.empty:
        return "_(none)_"
    sub = df.head(max_rows).copy()
    for c in sub.columns:
        if pd.api.types.is_float_dtype(sub[c]):
            sub[c] = sub[c].map(lambda x: float_fmt.format(x) if pd.notnull(x) else "nan")
    header = "| " + " | ".join(str(c) for c in sub.columns) + " |"
    sep = "| " + " | ".join("---" for _ in sub.columns) + " |"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in sub.itertuples(index=False)]
    return "\n".join([header, sep, *rows])


REGRESSION_KEYS = ["mse", "rmse", "mae", "r2", "macro_mae", "macro_rmse", "macro_r2"]
TOP1_KEYS = ["top1_accuracy", "top1_rank_distance_mean", "top1_utility_error_mean"]
TOPK_KEYS = [
    "top3_overlap", "top3_ordered_accuracy", "top3_utility_mae",
    "top5_overlap", "top5_ordered_accuracy", "top5_utility_mae",
    "top10_overlap", "top10_ordered_accuracy", "top10_utility_mae",
]
RANK_KEYS = [
    "spearman_mean", "spearman_std", "kendall_mean", "pairwise_ranking_accuracy",
    "rank_displacement_mean", "rank_displacement_median", "rank_displacement_max",
]
NDCG_KEYS = ["ndcg@3", "ndcg@5", "ndcg@10", "ndcg@all"]


def write_fold_report(
    fold: int,
    train_result,
    metrics: Dict[str, float],
    per_metric: pd.DataFrame,
    grouped: Dict[str, pd.DataFrame],
    split_summary: Dict,
) -> str:
    parts: List[str] = []
    parts.append(f"# Fold {fold} — Test Report\n")
    parts.append("## Training\n")
    parts.append(
        f"- Best epoch: **{train_result.best_epoch}**\n"
        f"- Best validation combined loss: **{_fmt(train_result.best_val_loss)}**\n"
        f"- Stopped epoch: {train_result.stopped_epoch}\n"
        f"- Train / val / test rows: "
        f"{split_summary['n_rows']['train']} / {split_summary['n_rows']['val']} / "
        f"{split_summary['n_rows']['test']}\n"
        f"- Train / val / test groups: "
        f"{split_summary['n_groups']['train']} / {split_summary['n_groups']['val']} / "
        f"{split_summary['n_groups']['test']}\n"
    )
    parts.append("\n## Regression metrics\n" + _metric_table(metrics, REGRESSION_KEYS))
    parts.append("\n## Top-1 metrics\n" + _metric_table(metrics, TOP1_KEYS))
    parts.append("\n## Top-K metrics\n" + _metric_table(metrics, TOPK_KEYS))
    parts.append("\n## Ranking metrics\n" + _metric_table(metrics, RANK_KEYS))
    parts.append("\n## NDCG\n" + _metric_table(metrics, NDCG_KEYS))

    parts.append("\n## Worst 15 metrics by MAE\n" + _df_table(per_metric, 15))

    if grouped:
        parts.append("\n## Grouped analysis\n")
        for logical, gdf in grouped.items():
            parts.append(f"\n### By {logical}\n" + _df_table(gdf, 20))
    return "\n".join(parts) + "\n"


def write_run_summary(
    run_name: str,
    fold_results: pd.DataFrame,
    overall: pd.DataFrame,
    worst_metrics: pd.DataFrame,
    worst_groups: Dict[str, pd.DataFrame],
    config_dict: Dict,
) -> str:
    parts: List[str] = []
    parts.append(f"# Run Summary — {run_name}\n")
    parts.append(
        f"- Model: **{config_dict['model']['type']}**, "
        f"loss mode: **{config_dict['loss']['mode']}**, "
        f"folds: **{len(fold_results)}**\n"
    )

    parts.append("\n## Mean ± std across folds\n")
    parts.append(_df_table(overall, max_rows=len(overall)))

    parts.append("\n## Per-fold comparison\n")
    compare_cols = [
        "fold", "best_epoch", "best_val_loss", "mae", "rmse", "r2",
        "top1_accuracy", "top3_overlap", "top5_overlap", "top10_overlap",
        "spearman_mean", "kendall_mean", "ndcg@10", "ndcg@all",
    ]
    cols = [c for c in compare_cols if c in fold_results.columns]
    parts.append(_df_table(fold_results[cols], max_rows=len(fold_results)))

    parts.append("\n## Overall ranking quality\n" + _metric_table(
        _mean_dict(fold_results, RANK_KEYS + NDCG_KEYS), RANK_KEYS + NDCG_KEYS))
    parts.append("\n## Overall Top-K quality\n" + _metric_table(
        _mean_dict(fold_results, TOPK_KEYS + ["top1_accuracy"]), ["top1_accuracy"] + TOPK_KEYS))

    parts.append("\n## Worst metrics (mean MAE across folds)\n" + _df_table(worst_metrics, 15))

    if worst_groups:
        parts.append("\n## Worst groups\n")
        for logical, gdf in worst_groups.items():
            parts.append(f"\n### Worst {logical} (by MAE)\n" + _df_table(gdf, 10))

    parts.append("\n## Recommendations\n")
    parts.append(_recommendations(fold_results))
    return "\n".join(parts) + "\n"


def _mean_dict(fold_results: pd.DataFrame, keys: List[str]) -> Dict[str, float]:
    return {k: float(fold_results[k].mean()) for k in keys if k in fold_results.columns}


def _recommendations(fold_results: pd.DataFrame) -> str:
    recs: List[str] = []
    targets = {
        "mae": (0.04, 0.08, "lower is better"),
        "top1_accuracy": (0.55, 0.75, "higher is better"),
        "top10_overlap": (0.85, 0.95, "higher is better"),
        "ndcg@10": (0.85, 1.0, "higher is better"),
        "spearman_mean": (0.65, 0.85, "higher is better"),
    }
    for key, (lo, hi, direction) in targets.items():
        if key not in fold_results.columns:
            continue
        val = float(fold_results[key].mean())
        if "lower" in direction:
            status = "within target" if val <= hi else "above target (worse)"
        else:
            status = "within/above target" if val >= lo else "below target (worse)"
        recs.append(f"- `{key}` = {_fmt(val)} (target {lo}–{hi}, {direction}): {status}.")
    recs.append(
        "- If ranking metrics lag regression metrics, consider raising "
        "`pairwise_rank_weight`; if top-K selection lags, raise `topk_mse_weight`."
    )
    return "\n".join(recs)
