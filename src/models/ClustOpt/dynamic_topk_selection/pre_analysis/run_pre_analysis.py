"""Orchestrator for the Dynamic Top-K metric-selection pre-analysis.

End-to-end: scan every utility vector -> per-vector statistics -> simulate
candidate dynamic-K heuristics -> aggregate -> validate -> plots + reports.

This is *pre-analysis only*. It reads existing ClustOpt utility outputs and
never modifies ClustOpt behaviour or any model code.

Usage (from this directory)
---------------------------
    python run_pre_analysis.py                  # full run, auto-timestamped out dir
    python run_pre_analysis.py --workers 8
    python run_pre_analysis.py --limit 600      # quick smoke test
    python run_pre_analysis.py --out-dir <dir>  # resume into an existing run

Outputs land under
``<repo>/results_analysis/dynamic_topk_utility_pre_analysis_YYYYMMDD_HHMMSS``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys

import pandas as pd

# allow running as a plain script (sibling-module imports)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from heuristic_candidates import heuristic_columns  # noqa: E402
import plotting  # noqa: E402
import reporting  # noqa: E402
from scan_utility_vectors import ScanPaths, build_task_index, scan  # noqa: E402

META_COLS = ["record_id", "dataset_id", "family", "family_id", "subfamily",
             "subfamily_id", "view_type", "split_id", "difficulty",
             "cluster_count", "n_points", "utility_file_path"]


def default_repo_root() -> str:
    # .../models/ClustOpt/dynamic_topk_selection/pre_analysis -> repo root
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "..", "..", ".."))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", default=default_repo_root())
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--limit", type=int, default=None,
                    help="cap number of records (smoke testing)")
    ap.add_argument("--out-dir", default=None,
                    help="resume into an existing output dir instead of new one")
    ap.add_argument("--shard-size", type=int, default=4000)
    args = ap.parse_args()

    paths = ScanPaths(repo_root=args.repo_root)
    out_root = os.path.join(args.repo_root, "results_analysis")
    if args.out_dir:
        out_dir = args.out_dir
    else:
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(out_root, f"dynamic_topk_utility_pre_analysis_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[run] output dir: {out_dir}", flush=True)

    # ---- scan -------------------------------------------------------------
    shard_dir = os.path.join(out_dir, "_scan_shards")
    n_index = len(build_task_index(paths))
    summary_df, wide_df = scan(paths, args.workers, shard_dir,
                               shard_size=args.shard_size, limit=args.limit)
    print(f"[run] scanned {len(summary_df)} records "
          f"({(summary_df['status'] == 'ok').sum()} ok)", flush=True)

    # ---- persist record tables -------------------------------------------
    summary_df.to_parquet(os.path.join(out_dir, "records_utility_summary.parquet"),
                          index=False)
    wide_df.to_parquet(os.path.join(out_dir, "records_utility_vectors.parquet"),
                       index=False)
    keep = [c for c in META_COLS if c in summary_df.columns]
    hcols = heuristic_columns()
    raw = [f"{c}_raw" for c in hcols if f"{c}_raw" in summary_df.columns]
    summary_df[keep + [c for c in hcols if c in summary_df.columns] + raw].to_parquet(
        os.path.join(out_dir, "heuristic_k_assignments.parquet"), index=False)

    df_ok = summary_df[summary_df["status"] == "ok"].copy()

    # ---- validation -------------------------------------------------------
    val = reporting.validation(summary_df, n_index,
                               os.path.join(out_dir, "validation"))
    print(f"[run] validation: ok={val['n_ok']} missing={val['n_missing']} "
          f"malformed={val['n_malformed']} n_metrics={val['n_metrics_value_counts']}",
          flush=True)

    # ---- aggregation tables ----------------------------------------------
    gsum = reporting.global_summary(df_ok)
    fam = reporting.family_summary(df_ok)
    sub = reporting.subfamily_summary(df_ok)
    view = reporting.view_summary(df_ok)
    metric = reporting.metric_summary(wide_df, summary_df)
    dist = reporting.heuristic_distribution_table(df_ok)

    gsum.to_csv(os.path.join(out_dir, "global_summary.csv"), index=False)
    fam.to_csv(os.path.join(out_dir, "family_summary.csv"), index=False)
    sub.to_csv(os.path.join(out_dir, "subfamily_summary.csv"), index=False)
    view.to_csv(os.path.join(out_dir, "view_summary.csv"), index=False)
    metric.to_csv(os.path.join(out_dir, "metric_summary.csv"), index=False)
    dist.to_csv(os.path.join(out_dir, "heuristic_k_distribution.csv"), index=False)

    # ---- reports + plots --------------------------------------------------
    bell = reporting.write_reports(out_dir, val, dist, gsum, fam, view, df_ok)
    made = plotting.make_all(out_dir, summary_df, wide_df, dist)
    print(f"[run] plots written: {len(made)}", flush=True)
    print(f"[run] bell-shaped candidates: {bell}", flush=True)
    print(f"[run] DONE -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
