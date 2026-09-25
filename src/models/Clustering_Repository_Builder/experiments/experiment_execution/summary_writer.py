"""Subfamily/split execution summaries.

These summarise one subfamily/split execution; they do not replace the
per-dataset raw outputs (which live under each dataset's ``experiments/``
folder). Saved under ``<subfamily_dir>/experiment_execution_summaries/split_XX/``.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .config import ExperimentExecutionConfig
from .method_runner import RunRecord
from .output_writer import (
    ensure_dir, write_csv_atomic, write_json_atomic, write_text_atomic,
)

SUMMARY_FILES: List[str] = [
    "execution_summary.csv",            # raw: one row per dataset x view x method
    "execution_summary_all_views.csv",  # per-method mean ARI over all views
    "execution_summary_best_view.csv",  # per-method best-view mean/median/std ARI
    "execution_summary.json",
    "execution_report.md",
    "experiment_failures.csv",
    "checkpoint.json",
]


def summary_dir(cfg: ExperimentExecutionConfig) -> Path:
    return cfg.subfamily_dir / "experiment_execution_summaries" / cfg.split_dir_name()


# Phase-D extra columns (from RunRecord.extra). Appended after the canonical
# columns; absent/empty for legacy runs so the existing schema is preserved.
_EXTRA_COLUMNS: List[str] = [
    "selected_metric_count", "weighting_mode", "dynamic_k_source",
    "utility_source", "metric_ranking_source", "early_stopping_enabled",
    "optuna_trials_requested", "optuna_trials_completed", "fallback_used",
]


def _records_frame(records: List[RunRecord]) -> pd.DataFrame:
    rows = []
    for r in records:
        row = {
            "dataset_id": r.dataset_id,
            "split_id": r.split_id,
            "family": r.family,
            "subfamily": r.subfamily,
            "difficulty": r.difficulty,
            "cluster_count": r.cluster_count,
            "view_id": r.view_id,
            "method_name": r.method_name,
            "method_group": r.method_group,
            "status": r.status,
            "ari": r.ari,
            "selected_k": r.selected_k,
            "true_k": r.true_k,
            "top_metric": r.top_metric,
            "runtime_sec": r.runtime_sec,
            "failure_reason": (r.failure_reason or "").splitlines()[0] if r.failure_reason else "",
        }
        extra = r.extra or {}
        for col in _EXTRA_COLUMNS:
            row[col] = extra.get(col)
        rows.append(row)
    return pd.DataFrame(rows)


def _all_views_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-method mean ARI over ALL views (one row per method).

    Columns: ``method_name, all_views_mean_ari, n_rows`` — matches the canonical
    ``execution_summary_all_views.csv`` schema. Computed over successful rows.
    """
    cols = ["method_name", "all_views_mean_ari", "n_rows"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    ok = df[df["status"] == "success"].copy()
    ok["ari"] = pd.to_numeric(ok["ari"], errors="coerce")
    ok = ok.dropna(subset=["ari"])
    if ok.empty:
        return pd.DataFrame(columns=cols)
    out = (ok.groupby("method_name")
           .agg(all_views_mean_ari=("ari", "mean"), n_rows=("ari", "size"))
           .reset_index()
           .sort_values("all_views_mean_ari", ascending=False, ignore_index=True))
    return out


def _best_view_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-method best-view ARI stats (one row per method).

    For each (method, dataset) the view with the highest ARI is taken; then per
    method we report mean / median / std over datasets plus the per-view counts
    of which view won. Columns match the canonical
    ``execution_summary_best_view.csv`` schema:
    ``method_name, best_view_mean_ari, median_best_view_ari, std_best_view_ari,
    n_datasets, x_only_best, y_only_best, xy_2d_best``.
    """
    cols = ["method_name", "best_view_mean_ari", "median_best_view_ari",
            "std_best_view_ari", "n_datasets",
            "x_only_best", "y_only_best", "xy_2d_best"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    ok = df[df["status"] == "success"].copy()
    ok["ari"] = pd.to_numeric(ok["ari"], errors="coerce")
    ok = ok.dropna(subset=["ari"])
    if ok.empty:
        return pd.DataFrame(columns=cols)
    best = ok.loc[ok.groupby(["method_name", "dataset_id"])["ari"].idxmax()]
    rows = []
    for method, g in best.groupby("method_name"):
        vc = g["view_id"].value_counts()
        rows.append({
            "method_name": method,
            "best_view_mean_ari": float(g["ari"].mean()),
            "median_best_view_ari": float(g["ari"].median()),
            "std_best_view_ari": float(g["ari"].std()),
            "n_datasets": int(g["dataset_id"].nunique()),
            "x_only_best": int(vc.get("x_only", 0)),
            "y_only_best": int(vc.get("y_only", 0)),
            "xy_2d_best": int(vc.get("xy_2d", 0)),
        })
    out = pd.DataFrame(rows, columns=cols)
    return out.sort_values("best_view_mean_ari", ascending=False, ignore_index=True)


def write_execution_summaries(
    *,
    cfg: ExperimentExecutionConfig,
    records: List[RunRecord],
    tracker_summary: Dict[str, object],
    family_id,
    subfamily_id,
) -> Path:
    out = summary_dir(cfg)
    ensure_dir(out)

    df = _records_frame(records)
    # Raw per-(dataset, view, method) records, plus the two per-method
    # aggregations (all-views mean and best-view mean/median/std).
    write_csv_atomic(out / "execution_summary.csv", df)
    write_csv_atomic(out / "execution_summary_all_views.csv", _all_views_summary(df))
    write_csv_atomic(out / "execution_summary_best_view.csv", _best_view_summary(df))

    failures = df[df["status"] == "failed"] if not df.empty else df
    write_csv_atomic(out / "experiment_failures.csv", failures)

    payload = {
        "config": cfg.to_summary_dict(),
        "family_id": family_id,
        "subfamily_id": subfamily_id,
        "summary": tracker_summary,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json_atomic(out / "execution_summary.json", payload)
    write_json_atomic(out / "checkpoint.json", {
        "config": cfg.to_summary_dict(),
        "summary": tracker_summary,
        "n_records": len(records),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    })
    write_text_atomic(out / "execution_report.md", _render_report_md(
        cfg, df, tracker_summary, family_id, subfamily_id
    ))
    return out


def write_checkpoint(*, cfg: ExperimentExecutionConfig, records: List[RunRecord],
                     tracker_summary: Dict[str, object]) -> None:
    out = summary_dir(cfg)
    ensure_dir(out)
    write_json_atomic(out / "checkpoint.json", {
        "config": cfg.to_summary_dict(),
        "summary": tracker_summary,
        "n_records": len(records),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    })


def _best_view_method_summary(ok: pd.DataFrame) -> pd.DataFrame:
    """Best-View Mean ARI per method: for each (method, dataset) take the view
    with the highest ARI, then average those per-dataset best values over
    datasets. This is the PRIMARY comparison metric (matches how external
    AutoClustering frameworks report a single best record per dataset).
    """
    s = ok.copy()
    s["ari"] = pd.to_numeric(s["ari"], errors="coerce")
    s = s.dropna(subset=["ari"])
    if s.empty:
        return pd.DataFrame(columns=["method_name", "best_view_mean_ari",
                                     "n_datasets"])
    best_idx = s.groupby(["method_name", "dataset_id"])["ari"].idxmax()
    best = s.loc[best_idx]
    out = (best.groupby("method_name")
           .agg(best_view_mean_ari=("ari", "mean"),
                n_datasets=("dataset_id", "nunique"))
           .reset_index()
           .sort_values("best_view_mean_ari", ascending=False))
    return out


def _render_report_md(cfg, df: pd.DataFrame, summary, family_id, subfamily_id) -> str:
    lines = ["# Experiment Execution Report\n"]
    lines.append(f"- Family: **{family_id}** | Subfamily: **{subfamily_id}**")
    lines.append(f"- Split id: **{cfg.split_id}**")
    lines.append(f"- Datasets processed: **{summary.get('datasets_processed')}** "
                 f"(fatal={summary.get('datasets_fatal')})")
    lines.append(f"- Runs: ok={summary.get('runs_success')} "
                 f"failed={summary.get('runs_failed')} skipped={summary.get('runs_skipped')} "
                 f"/ expected={summary.get('expected_runs')}")
    lines.append(f"- Runtime: {summary.get('runtime_total_str')}")
    lines.append("")
    if not df.empty:
        ok = df[df["status"] == "success"]
        if not ok.empty and "ari" in ok.columns:
            # PRIMARY: Best-View Mean ARI.
            bv = _best_view_method_summary(ok)
            lines.append("## Best-View Mean ARI by method (PRIMARY)\n")
            lines.append(
                "> Best-View Mean ARI = for each (dataset, method) take the view "
                "with the highest ARI, then average over datasets. This is the "
                "headline metric for comparison with external AutoClustering "
                "frameworks (AutoML4Clust / ML2DAC / cSmartML). The All-Views "
                "table below is the legacy mean over all dataset x view rows and "
                "understates performance.\n")
            lines.append("| method | best_view_mean_ari | n_datasets |")
            lines.append("| --- | --- | --- |")
            for _, r in bv.iterrows():
                lines.append(f"| {r['method_name']} | "
                             f"{float(r['best_view_mean_ari']):.4f} | "
                             f"{int(r['n_datasets'])} |")
            lines.append("")

            # SECONDARY (legacy): mean ARI over all views.
            lines.append("## All-Views Mean ARI by method (secondary, legacy)\n")
            grp = ok.groupby("method_name")["ari"].mean().sort_values(ascending=False)
            lines.append("| method | all_views_mean_ari | n |")
            lines.append("| --- | --- | --- |")
            counts = ok.groupby("method_name").size()
            for method, val in grp.items():
                lines.append(f"| {method} | {val:.4f} | {int(counts.get(method, 0))} |")
            lines.append("")
    return "\n".join(lines) + "\n"
