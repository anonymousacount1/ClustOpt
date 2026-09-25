"""Phase-D aggregation across all subfamilies (best-view protocol).

Reads every subfamily's ``execution_summary.csv`` for the Phase-D split folder
and produces the standard ClustOpt rollups: overall / family / subfamily / view
method summaries, K-accuracy, runtime, win-rates, the dynamic-K selected-metric
count distribution, plus a markdown report and a validation JSON.

Best-view protocol: a dataset's score for a method is the **max ARI over its
three views** (x_only / y_only / xy_2d).

Run from the repo root::

    python -m models.Clustering_Repository_Builder.experiments.experiment_execution.aggregate_phaseD \
        --repo-root . --split-id 1
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

PHASED_SUFFIX = "phaseD_no_earlystop_50trials_dynamic_ablation"

# The 28 Phase-D methods (folder names under each dataset's split_01/).
PHASED_METHODS = [
    "regressor_top1_raw", "regressor_top3_raw", "regressor_top5_raw",
    "regressor_top10_raw", "regressor_top_dynamic_realK_raw",
    "regressor_top_dynamic_predictK_raw", "regressor_top1_softmax_t05",
    "regressor_top3_softmax_t05", "regressor_top5_softmax_t05",
    "regressor_top10_softmax_t05", "regressor_top_dynamic_realK_softmax_t05",
    "regressor_top_dynamic_predictK_softmax_t05",
    "real_top1_raw", "real_top3_raw", "real_top5_raw", "real_top10_raw",
    "real_top_dynamic_raw", "real_top1_softmax_t05", "real_top3_softmax_t05",
    "real_top5_softmax_t05", "real_top10_softmax_t05", "real_top_dynamic_softmax_t05",
    "all_metrics_uniform", "calinski_harabasz_single", "davies_bouldin_single",
    "dbcv_single", "silhouette_single", "uniform_classic_cvi",
]
_VIEWS = ("x_only", "y_only", "xy_2d")


def _ext(path: str) -> str:
    s = os.path.abspath(path)
    if os.name == "nt" and not s.startswith("\\\\?\\"):
        s = "\\\\?\\" + s
    return s


def _read_json_ext(path: str):
    try:
        with open(_ext(path), "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def collect_from_split01(repo: Path, dataset_list_file: str) -> pd.DataFrame:
    """Read Phase-D per-(dataset, method, view) results straight from each
    dataset's merged ``experiments/split_01`` folder (long-path-safe). This is
    independent of the subfamily execution summaries."""
    assign = pd.read_csv(
        repo / "results_analysis" / "clustering_repository" / "analyzed_data"
        / "experiment_splits" / "dataset_split_assignments.csv")
    meta = {r["dataset_id"]: (r.get("family"), r.get("subfamily"),
                              r.get("cluster_count"))
            for _, r in assign.iterrows()}

    with open(dataset_list_file, "r", encoding="utf-8") as fh:
        dataset_dirs = [ln.strip() for ln in fh if ln.strip()]

    rows = []
    for i, ds in enumerate(dataset_dirs, 1):
        did = os.path.basename(ds)
        fam, sub, ck = meta.get(did, (None, None, None))
        base = os.path.join(str(repo), ds, "experiments", "split_01")
        for m in PHASED_METHODS:
            for v in _VIEWS:
                st = _read_json_ext(os.path.join(base, m, v, "status.json"))
                if not st:
                    continue
                md = _read_json_ext(os.path.join(base, m, v, "phaseD_metadata.json")) or {}
                rows.append({
                    "dataset_id": did, "split_id": 1, "family": fam,
                    "subfamily": sub,
                    "cluster_count": ck, "view_id": v, "method_name": m,
                    "status": st.get("status"), "ari": st.get("ari"),
                    "selected_k": st.get("selected_k"), "true_k": st.get("true_k"),
                    "runtime_sec": st.get("runtime_sec"),
                    "selected_metric_count": md.get("selected_metric_count"),
                    "early_stopping_enabled": md.get("early_stopping_enabled"),
                    "optuna_trials_requested": md.get("optuna_trials_requested"),
                    "optuna_trials_completed": md.get("optuna_trials_completed"),
                    "fallback_used": md.get("fallback_used"),
                })
        if i % 200 == 0:
            print(f"  [collect] {i}/{len(dataset_dirs)} datasets", flush=True)
    return pd.DataFrame(rows)


def collect_summaries(analyzed: Path, split_dir_name: str) -> pd.DataFrame:
    frames = []
    pattern = f"*/subfamilies/*/experiment_execution_summaries/{split_dir_name}/execution_summary.csv"
    for csv in analyzed.glob(pattern):
        try:
            df = pd.read_csv(csv)
            if not df.empty:
                frames.append(df)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def best_view(df_ok: pd.DataFrame) -> pd.DataFrame:
    df_ok = df_ok.copy()
    df_ok["ari"] = pd.to_numeric(df_ok["ari"], errors="coerce")
    df_ok = df_ok.dropna(subset=["ari"])
    idx = df_ok.groupby(["method_name", "dataset_id"])["ari"].idxmax()
    return df_ok.loc[idx]


def _method_summary(best: pd.DataFrame, by: List[str]) -> pd.DataFrame:
    g = best.groupby(by).agg(
        best_view_mean_ari=("ari", "mean"),
        median_best_view_ari=("ari", "median"),
        std_best_view_ari=("ari", "std"),
        n_datasets=("dataset_id", "nunique"),
    ).reset_index()
    return g.sort_values(by[:-1] + ["best_view_mean_ari"] if len(by) > 1
                         else "best_view_mean_ari", ascending=False)


def k_accuracy(best: pd.DataFrame) -> pd.DataFrame:
    b = best.copy()
    b["selected_k"] = pd.to_numeric(b["selected_k"], errors="coerce")
    b["true_k"] = pd.to_numeric(b["true_k"], errors="coerce")
    b = b.dropna(subset=["selected_k", "true_k"])
    b["k_correct"] = (b["selected_k"] == b["true_k"]).astype(float)
    b["k_abs_error"] = (b["selected_k"] - b["true_k"]).abs()
    return (b.groupby("method_name")
            .agg(k_accuracy=("k_correct", "mean"),
                 mean_k_abs_error=("k_abs_error", "mean"),
                 n=("dataset_id", "nunique"))
            .reset_index().sort_values("k_accuracy", ascending=False))


def win_rates(best: pd.DataFrame) -> pd.DataFrame:
    # per dataset, the method with the highest best-view ARI wins (ties -> all share)
    wins = {}
    for ds, g in best.groupby("dataset_id"):
        mx = g["ari"].max()
        winners = g.loc[g["ari"] >= mx - 1e-12, "method_name"].unique()
        for m in winners:
            wins[m] = wins.get(m, 0) + 1.0 / len(winners)
    n_ds = best["dataset_id"].nunique()
    rows = [{"method_name": m, "wins": w, "win_rate": w / n_ds}
            for m, w in wins.items()]
    return pd.DataFrame(rows).sort_values("win_rate", ascending=False)


def dynamic_k_distribution(best: pd.DataFrame) -> pd.DataFrame:
    if "selected_metric_count" not in best.columns:
        return pd.DataFrame()
    dyn = best[best["method_name"].str.contains("dynamic")].copy()
    dyn["selected_metric_count"] = pd.to_numeric(
        dyn["selected_metric_count"], errors="coerce")
    dyn = dyn.dropna(subset=["selected_metric_count"])
    rows = []
    for m, g in dyn.groupby("method_name"):
        vc = g["selected_metric_count"].astype(int).value_counts().sort_index()
        rows.append({"method_name": m,
                     "mean_selected_metric_count": float(g["selected_metric_count"].mean()),
                     "n": int(g["dataset_id"].nunique()),
                     **{f"K={k}": int(v) for k, v in vc.items()}})
    return pd.DataFrame(rows)


def validate(df: pd.DataFrame, best: pd.DataFrame, split_id: int) -> dict:
    fams = df["family"].nunique() if "family" in df else 0
    subs = df["subfamily"].nunique() if "subfamily" in df else 0
    methods = sorted(df["method_name"].unique().tolist())
    es_col = df.get("early_stopping_enabled")
    trials_col = pd.to_numeric(df.get("optuna_trials_requested"), errors="coerce")
    smc = pd.to_numeric(df.get("selected_metric_count"), errors="coerce")
    dyn = df[df["method_name"].str.contains("dynamic")] if "method_name" in df else df
    dyn_smc = pd.to_numeric(dyn.get("selected_metric_count"), errors="coerce").dropna()
    return {
        "n_families": int(fams),
        "n_subfamilies": int(subs),
        "n_methods": len(methods),
        "methods": methods,
        "all_28_methods_present": bool(len(methods) == 28),
        "only_split_1": bool(
            set(pd.to_numeric(df["split_id"], errors="coerce").dropna().unique().tolist())
            <= {float(split_id)}) if "split_id" in df.columns else None,
        "views_present": sorted(df["view_id"].unique().tolist()) if "view_id" in df else [],
        "early_stopping_any_enabled": bool(es_col.fillna(False).any()) if es_col is not None else None,
        "trials_requested_all_50": bool((trials_col.dropna() == 50).all()) if trials_col is not None else None,
        "dynamic_selected_metric_count_in_1_10": bool(
            ((dyn_smc >= 1) & (dyn_smc <= 10)).all()) if len(dyn_smc) else None,
        "n_best_view_datasets": int(best["dataset_id"].nunique()) if not best.empty else 0,
    }


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", required=True)
    p.add_argument("--split-id", type=int, default=1)
    p.add_argument("--split-dir-suffix", default=PHASED_SUFFIX)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--from-split01", default=None,
                   help="Read per-dataset results directly from each dataset's "
                        "merged split_01 folder (dataset-list file), instead of "
                        "the subfamily execution summaries.")
    args = p.parse_args(argv)

    repo = Path(args.repo_root).resolve()
    analyzed = repo / "results_analysis" / "clustering_repository" / "analyzed_data"
    split_dir_name = f"split_{int(args.split_id):02d}_{args.split_dir_suffix}"
    out_dir = Path(args.out_dir) if args.out_dir else (
        repo / "results_analysis" / "clustering_repository"
        / f"phaseD_split{int(args.split_id):02d}_aggregation")
    for d in ("tables", "reports", "validation"):
        (out_dir / d).mkdir(parents=True, exist_ok=True)

    if args.from_split01:
        df = collect_from_split01(repo, args.from_split01)
    else:
        df = collect_summaries(analyzed, split_dir_name)
    if df.empty:
        print(f"[aggregate] no results found")
        return 1
    ok = df[df["status"] == "success"].copy()
    best = best_view(ok) if not ok.empty else pd.DataFrame()

    t = out_dir / "tables"
    overall = _method_summary(best, ["method_name"])
    overall.to_csv(t / "overall_method_summary.csv", index=False)
    _method_summary(best, ["family", "method_name"]).to_csv(
        t / "family_method_summary.csv", index=False)
    _method_summary(best, ["subfamily", "method_name"]).to_csv(
        t / "subfamily_method_summary.csv", index=False)
    # view summary: per (method, view) mean ARI over all-views rows
    ok["ari"] = pd.to_numeric(ok["ari"], errors="coerce")
    (ok.groupby(["method_name", "view_id"]).agg(
        mean_ari=("ari", "mean"), n=("dataset_id", "nunique")).reset_index()
     ).to_csv(t / "view_summary.csv", index=False)
    kacc = k_accuracy(best); kacc.to_csv(t / "k_accuracy_summary.csv", index=False)
    (best.groupby("method_name")["runtime_sec"].mean().reset_index()
     .rename(columns={"runtime_sec": "mean_runtime_sec"})
     .sort_values("mean_runtime_sec")).to_csv(t / "runtime_summary.csv", index=False)
    wr = win_rates(best); wr.to_csv(t / "win_rates.csv", index=False)
    dynk = dynamic_k_distribution(best)
    if not dynk.empty:
        dynk.to_csv(t / "dynamic_k_distribution.csv", index=False)
    best.to_csv(t / "best_view_records.csv", index=False)

    val = validate(df, best, args.split_id)
    with open(out_dir / "validation" / "phaseD_aggregation_validation.json",
              "w", encoding="utf-8") as fh:
        json.dump(val, fh, indent=2)

    # report
    lines = ["# Phase-D Aggregation Report", "",
             f"- Split: {args.split_id} | suffix: `{args.split_dir_suffix}`",
             f"- Families: {val['n_families']} | Subfamilies: {val['n_subfamilies']} "
             f"| Methods: {val['n_methods']} | Best-view datasets: {val['n_best_view_datasets']}",
             f"- Early stopping any enabled: {val['early_stopping_any_enabled']} | "
             f"trials all 50: {val['trials_requested_all_50']} | "
             f"dynamic K in [1,10]: {val['dynamic_selected_metric_count_in_1_10']}",
             "", "## Overall best-view mean ARI by method", "",
             "| method | best_view_mean_ari | median | std | n |", "|---|---|---|---|---|"]
    for _, r in overall.iterrows():
        lines.append(f"| {r['method_name']} | {r['best_view_mean_ari']:.4f} | "
                     f"{r['median_best_view_ari']:.4f} | {r['std_best_view_ari']:.4f} | "
                     f"{int(r['n_datasets'])} |")
    with open(out_dir / "reports" / "PHASED_AGGREGATION_REPORT.md", "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"[aggregate] {val['n_methods']} methods, {val['n_subfamilies']} subfamilies, "
          f"{val['n_best_view_datasets']} datasets -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
