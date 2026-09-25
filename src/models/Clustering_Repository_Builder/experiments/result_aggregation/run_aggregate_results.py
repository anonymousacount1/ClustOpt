"""CLI: aggregate raw experiment results into research summaries.

Independent of execution -- reruns cheaply without repeating Optuna/ClustOpt
searches (it only *reads* existing experiment outputs). Produces a structured
run directory with raw tables, standard + cross summaries, ARI trace-best /
opportunity-gap analysis, grouped win-rates, K analyses, metric-selection
analyses, optional paired statistical tests, formatted Excel workbooks, plots
and Markdown reports.

Run as a module::

    python -m models.Clustering_Repository_Builder.experiments.result_aggregation.run_aggregate_results \
        --analyzed-data-root <PATH> \
        --split-assignments <PATH>/dataset_split_assignments.csv \
        --split-ids 1
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd

from . import (
    aggregators, excel_export, grouped_analysis, k_analysis,
    metric_selection_analysis, plotting, report_writer, statistical_tests,
    trace_best_gap,
)
from .dataset_level_selector import build_dataset_level
from .external_metric_summaries import external_metric_table
from .method_registry import METHOD_NAMES
from .raw_result_collector import collect_raw_results
from .winrate_analysis import winrate_summary

# Default output root (used when --output-dir is not provided).
_DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[4]
    / "results_analysis" / "clustering_repository" / "experiments"
)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Aggregate AutoClustering experiment results.")
    p.add_argument("--analyzed-data-root", required=True, type=str)
    p.add_argument("--split-assignments", required=True, type=str)
    p.add_argument("--split-ids", nargs="+", type=int, required=True)
    p.add_argument("--methods", nargs="+", type=str, default=None,
                   help="Subset of method folder names to aggregate (phase-tagged). "
                        "Default: all registered methods. Use this to separate the "
                        "phase-A (regressors vs fixed-CVI baselines) and phase-B "
                        "(regressors vs external baselines) comparisons.")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Defaults to results_analysis/clustering_repository/experiments.")
    p.add_argument("--run-name", type=str, default="aggregation")
    p.add_argument("--collect-workers", type=int, default=None,
                   help="Thread-pool size for the two I/O-bound file-reading "
                        "stages (raw collection + trace-best). Default: serial "
                        "(unchanged). Set >1 to parallelize the many small "
                        "long-path reads; output is identical to serial.")
    p.add_argument("--include-missing", action="store_true",
                   help="Include 'missing' rows (runs with no status.json).")
    p.add_argument("--no-excel", action="store_true", help="Skip Excel workbooks.")
    p.add_argument("--no-plots", action="store_true", help="Skip plot generation.")
    p.add_argument("--no-statistical-tests", action="store_true",
                   help="Skip paired statistical tests.")
    return p


def _slug(name: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z]+", "_", str(name)).strip("_").lower()
    return s or "unknown"


def _ext(path: Path | str) -> str:
    r"""Long-path-safe absolute path string (``\\?\`` prefix on Windows).

    The deep ``cross_analysis/family_specific/<family>/...`` CSV paths routinely
    exceed ``MAX_PATH`` (260) once the run dir + timestamp are included, and this
    machine has long paths disabled -- so every write must go through this.
    """
    s = os.path.abspath(str(path))
    if sys.platform == "win32" and not s.startswith("\\\\?\\"):
        s = "\\\\?\\" + s
    return s


def _mkdirs(path: Path) -> None:
    os.makedirs(_ext(path), exist_ok=True)


def _write_text(path: Path, text: str) -> None:
    """Long-path-safe text write."""
    _mkdirs(path.parent)
    with open(_ext(path), "w", encoding="utf-8") as fh:
        fh.write(text)


def _w(run_dir: Path, rel: str, df: pd.DataFrame) -> None:
    """Write a DataFrame as CSV under the run dir (parents auto-created)."""
    if df is None:
        return
    path = run_dir / rel
    _mkdirs(path.parent)
    index = isinstance(df.index, pd.MultiIndex) or df.index.name is not None
    df.to_csv(_ext(path), index=index)


def run(args: argparse.Namespace) -> int:
    assignments = pd.read_csv(args.split_assignments)
    output_dir = Path(args.output_dir) if args.output_dir else _DEFAULT_OUTPUT_DIR
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"{timestamp}__{args.run_name}"
    _mkdirs(run_dir)

    print("=" * 70, flush=True)
    print("AutoClustering experiment — enhanced result aggregation", flush=True)
    print("=" * 70, flush=True)
    selected_methods = list(args.methods) if args.methods else list(METHOD_NAMES)
    print(f"Split ids: {args.split_ids}", flush=True)
    print(f"Analyzed data root: {args.analyzed_data_root}", flush=True)
    print(f"Output run dir: {run_dir}", flush=True)
    print(f"Methods ({len(selected_methods)}): {selected_methods}", flush=True)
    print(f"Include missing: {bool(args.include_missing)}", flush=True)
    print(flush=True)

    # ---- [1/12] split assignments -------------------------------------- #
    print("[1/12] Loading split assignments...", flush=True)
    wanted = set(int(s) for s in args.split_ids)
    n_assigned = int(assignments["split_id"].astype("Int64").isin(wanted).sum())
    print(f"        {n_assigned} datasets in requested split(s).", flush=True)

    # ---- [2/12] raw results -------------------------------------------- #
    print("[2/12] Collecting raw results...", flush=True)
    raw = collect_raw_results(
        assignments=assignments, split_ids=args.split_ids,
        method_names=selected_methods,
        include_missing=bool(args.include_missing),
        max_workers=args.collect_workers,
    )
    _w(run_dir, "raw/raw_results.csv", raw)
    _w(run_dir, "raw_results.csv", raw)  # legacy mirror
    if raw.empty:
        print("[warn] no raw results found for the requested split(s). "
              "Run experiments first.", flush=True)

    # ---- [3/12] trace-best ARI gap ------------------------------------- #
    print("[3/12] Computing trace-best ARI gap...", flush=True)
    trace_table, trace_warnings = trace_best_gap.compute_trace_best_table(
        raw, max_workers=args.collect_workers)
    raw_t = trace_best_gap.merge_into_raw(raw, trace_table)
    _w(run_dir, "trace_best_gap/raw_trace_best_gap.csv", raw_t)

    # Best-view frame: one row per (dataset, method) = the highest-ARI view.
    # This is the canonical frame for every reported *mean* (mean best view).
    # ``raw_t`` (all three views) is kept only for the raw dump, the per-view
    # summary, and metric-selection frequency (counts, not averages).
    bv = aggregators.collapse_best_view(raw_t)

    # ---- [4/12] dataset-level best-view -------------------------------- #
    print("[4/12] Building dataset-level best-view table...", flush=True)
    dataset_level = build_dataset_level(raw)
    _w(run_dir, "raw/dataset_level_results.csv", dataset_level)
    _w(run_dir, "dataset_level_results.csv", dataset_level)  # legacy mirror
    ds_trace = trace_best_gap.build_dataset_level_trace_best(dataset_level, trace_table)
    _w(run_dir, "trace_best_gap/dataset_level_trace_best_gap.csv", ds_trace)

    # ---- [5/12] standard summaries ------------------------------------- #
    print("[5/12] Writing standard summaries...", flush=True)
    summaries = aggregators.standard_summaries(bv, raw_t)
    for name, df in summaries.items():
        _w(run_dir, f"standard_summaries/{name}.csv", df)
        _w(run_dir, f"{name}.csv", df)  # legacy mirror
    ext_table = external_metric_table(bv)
    _w(run_dir, "standard_summaries/external_metric_table.csv", ext_table)
    _w(run_dir, "external_metric_table.csv", ext_table)

    # trace-best grouped summaries (best-view frame: gap of the best view).
    tb_summary = trace_best_gap.trace_best_summary_by(bv, [])
    gap_by_method = trace_best_gap.trace_best_summary_by(bv, ["method_name"])
    gap_by_family = trace_best_gap.trace_best_summary_by(bv, ["family"])
    gap_by_family_method = trace_best_gap.trace_best_summary_by(bv, ["family", "method_name"])
    _w(run_dir, "trace_best_gap/trace_best_gap_summary.csv", tb_summary)
    _w(run_dir, "trace_best_gap/trace_best_gap_by_method.csv", gap_by_method)
    _w(run_dir, "trace_best_gap/trace_best_gap_by_family.csv", gap_by_family)
    _w(run_dir, "trace_best_gap/trace_best_gap_by_subfamily.csv",
       trace_best_gap.trace_best_summary_by(bv, ["subfamily"]))
    _w(run_dir, "trace_best_gap/trace_best_gap_by_cluster_count.csv",
       trace_best_gap.trace_best_summary_by(bv, ["cluster_count"]))
    _w(run_dir, "trace_best_gap/trace_best_gap_by_family_cluster_count.csv",
       trace_best_gap.trace_best_summary_by(bv, ["family", "cluster_count"]))
    _w(run_dir, "trace_best_gap/trace_best_gap_by_subfamily_cluster_count.csv",
       trace_best_gap.trace_best_summary_by(bv, ["subfamily", "cluster_count"]))

    # ---- [6/12] global cross analyses ---------------------------------- #
    print("[6/12] Computing cross analyses...", flush=True)
    # Family x view summary intentionally keeps all views (it is *about* views).
    fam_cc = aggregators.summarize_by(bv, ["family", "cluster_count"])
    fam_cc_gap = trace_best_gap.trace_best_summary_by(bv, ["family", "cluster_count"])
    cross = {
        "global_family_x_cluster_count_summary": fam_cc,
        "global_family_x_difficulty_summary": aggregators.summarize_by(bv, ["family", "difficulty"]),
        "global_family_x_view_summary": aggregators.summarize_by(raw_t, ["family", "view_id"]),
        "global_method_x_cluster_count_summary": aggregators.summarize_by(bv, ["method_name", "cluster_count"]),
        "global_method_x_difficulty_summary": aggregators.summarize_by(bv, ["method_name", "difficulty"]),
        "global_family_x_cluster_count_x_method_summary":
            aggregators.summarize_by(bv, ["family", "cluster_count", "method_name"]),
        "global_family_x_difficulty_x_method_summary":
            aggregators.summarize_by(bv, ["family", "difficulty", "method_name"]),
    }
    for name, df in cross.items():
        _w(run_dir, f"cross_analysis/{name}.csv", df)
    topk = grouped_analysis.topk_regressor_comparison(bv)
    _w(run_dir, "cross_analysis/topk_regressor_comparison.csv", topk)
    _w(run_dir, "cross_analysis/topk_regressor_comparison_by_family.csv",
       grouped_analysis.topk_regressor_comparison(bv, by=["family"]))
    _w(run_dir, "cross_analysis/topk_regressor_comparison_by_subfamily.csv",
       grouped_analysis.topk_regressor_comparison(bv, by=["subfamily"]))
    _w(run_dir, "cross_analysis/topk_regressor_comparison_by_cluster_count.csv",
       grouped_analysis.topk_regressor_comparison(bv, by=["cluster_count"]))

    # ---- [7/12] family-specific analyses (global + per-family) --------- #
    print("[7/12] Computing family-specific analyses (global + per-family)...", flush=True)
    families = sorted([f for f in bv["family"].dropna().unique()]) if not bv.empty else []
    n_fam = len(families)
    print(f"[analysis 1/{n_fam + 1}] Global analysis (done above)", flush=True)
    family_gap_summaries = {}
    family_winrates = {}
    family_top_metrics = {}
    for fi, family in enumerate(families, start=1):
        print(f"[family {fi}/{n_fam}] {family}", flush=True)
        fam = bv[bv["family"] == family]               # best-view frame (means)
        fam_raw = raw_t[raw_t["family"] == family]      # all-views (metric counts)
        dl_fam = dataset_level[dataset_level["family"] == family]
        slug = _slug(family)
        base = f"cross_analysis/family_specific/{slug}"
        print(f"  datasets: {fam['dataset_id'].nunique()} | "
              f"methods: {fam['method_name'].nunique()} | writing summaries...", flush=True)
        _w(run_dir, f"{base}/subfamily_x_cluster_count_summary.csv",
           aggregators.summarize_by(fam, ["subfamily", "cluster_count"]))
        _w(run_dir, f"{base}/subfamily_x_difficulty_summary.csv",
           aggregators.summarize_by(fam, ["subfamily", "difficulty"]))
        _w(run_dir, f"{base}/subfamily_x_method_summary.csv",
           aggregators.summarize_by(fam, ["subfamily", "method_name"]))
        _w(run_dir, f"{base}/subfamily_x_cluster_count_x_method_summary.csv",
           aggregators.summarize_by(fam, ["subfamily", "cluster_count", "method_name"]))
        _w(run_dir, f"{base}/subfamily_x_difficulty_x_method_summary.csv",
           aggregators.summarize_by(fam, ["subfamily", "difficulty", "method_name"]))
        _w(run_dir, f"{base}/cluster_count_x_method_summary.csv",
           aggregators.summarize_by(fam, ["cluster_count", "method_name"]))
        _w(run_dir, f"{base}/difficulty_x_method_summary.csv",
           aggregators.summarize_by(fam, ["difficulty", "method_name"]))
        _w(run_dir, f"{base}/view_x_method_summary.csv",
           aggregators.summarize_by(fam_raw, ["view_id", "method_name"]))
        _w(run_dir, f"{base}/algorithm_x_method_summary.csv",
           aggregators.summarize_by(fam, ["selected_algorithm", "method_name"]))
        _w(run_dir, f"{base}/trace_best_gap_by_subfamily_cluster_count.csv",
           trace_best_gap.trace_best_summary_by(fam, ["subfamily", "cluster_count"]))
        _w(run_dir, f"{base}/winrate_by_subfamily.csv",
           grouped_analysis.grouped_winrate(dl_fam, ["subfamily"]))
        _w(run_dir, f"{base}/k_accuracy_by_subfamily.csv",
           grouped_analysis.k_accuracy_by(fam, ["subfamily", "method_name"]))

        # Family-level metric-selection interpretability (counts -> all views).
        mbase = f"metric_selection/family_specific/{slug}"
        _w(run_dir, f"{mbase}/top_selected_metrics.csv",
           metric_selection_analysis.metric_frequency(fam_raw))
        _w(run_dir, f"{mbase}/top_metric_frequency.csv",
           metric_selection_analysis.top_metric_frequency(fam_raw))
        _w(run_dir, f"{mbase}/metric_family_frequency.csv",
           metric_selection_analysis.metric_family_frequency(fam_raw))
        _w(run_dir, f"{mbase}/metric_weight_entropy.csv",
           metric_selection_analysis.weight_entropy_summary(fam_raw))

        family_gap_summaries[family] = trace_best_gap.trace_best_summary_by(fam, ["method_name"])
        family_winrates[family] = grouped_analysis.grouped_winrate(dl_fam, [])
        family_top_metrics[family] = metric_selection_analysis.metric_frequency(fam_raw)

    # ---- [8/12] win-rate analyses -------------------------------------- #
    print("[8/12] Computing win-rate analyses...", flush=True)
    winrate = winrate_summary(dataset_level)
    _w(run_dir, "winrates/winrate_summary.csv", winrate)
    _w(run_dir, "winrate_summary.csv", winrate)  # legacy mirror
    _w(run_dir, "winrates/winrate_by_family.csv", grouped_analysis.grouped_winrate(dataset_level, ["family"]))
    _w(run_dir, "winrates/winrate_by_subfamily.csv", grouped_analysis.grouped_winrate(dataset_level, ["subfamily"]))
    _w(run_dir, "winrates/winrate_by_cluster_count.csv", grouped_analysis.grouped_winrate(dataset_level, ["cluster_count"]))
    _w(run_dir, "winrates/winrate_by_family_cluster_count.csv",
       grouped_analysis.grouped_winrate(dataset_level, ["family", "cluster_count"]))
    _w(run_dir, "winrates/winrate_by_subfamily_cluster_count.csv",
       grouped_analysis.grouped_winrate(dataset_level, ["subfamily", "cluster_count"]))
    _w(run_dir, "winrates/winrate_by_difficulty.csv", grouped_analysis.grouped_winrate(dataset_level, ["difficulty"]))

    # ---- [9/12] K analyses --------------------------------------------- #
    print("[9/12] Computing K analyses...", flush=True)
    k_acc = k_analysis.k_accuracy_summary(bv)
    _w(run_dir, "k_analysis/k_accuracy_summary.csv", k_acc)
    _w(run_dir, "k_accuracy_summary.csv", k_acc)  # legacy mirror
    cms = k_analysis.all_k_confusion_matrices(bv)
    for sub in ("k_analysis/k_confusion_matrices", "k_confusion_matrices"):
        cm_dir = run_dir / sub
        _mkdirs(cm_dir)
        for method, cm in cms.items():
            cm.to_csv(_ext(cm_dir / f"{method}.csv"))
    _w(run_dir, "k_analysis/k_accuracy_by_family.csv",
       grouped_analysis.k_accuracy_by(bv, ["family", "method_name"]))
    _w(run_dir, "k_analysis/k_accuracy_by_subfamily.csv",
       grouped_analysis.k_accuracy_by(bv, ["subfamily", "method_name"]))
    _w(run_dir, "k_analysis/k_accuracy_by_family_cluster_count.csv",
       grouped_analysis.k_accuracy_by(bv, ["family", "cluster_count", "method_name"]))

    # ---- [10/12] metric-selection analyses ----------------------------- #
    print("[10/12] Computing metric-selection analyses...", flush=True)
    metric_freq = metric_selection_analysis.metric_frequency(raw_t)
    _w(run_dir, "metric_selection/metric_selection_frequency.csv", metric_freq)
    _w(run_dir, "metric_selection_frequency.csv", metric_freq)  # legacy mirror
    msf = {
        "metric_family_frequency": metric_selection_analysis.metric_family_frequency(raw_t),
        "metric_selection_frequency_by_family": metric_selection_analysis.metric_frequency(raw_t, by=["family"]),
        "metric_selection_frequency_by_subfamily": metric_selection_analysis.metric_frequency(raw_t, by=["subfamily"]),
        "top_metric_frequency": metric_selection_analysis.top_metric_frequency(raw_t),
        "top_metric_frequency_by_family": metric_selection_analysis.top_metric_frequency(raw_t, by=["family"]),
        "top_metric_frequency_by_subfamily": metric_selection_analysis.top_metric_frequency(raw_t, by=["subfamily"]),
        "metric_family_frequency_by_family": metric_selection_analysis.metric_family_frequency(raw_t, by=["family"]),
        "metric_family_frequency_by_subfamily": metric_selection_analysis.metric_family_frequency(raw_t, by=["subfamily"]),
        "metric_selection_by_family_method": metric_selection_analysis.metric_frequency(raw_t, by=["family", "method_name"]),
        "metric_selection_by_subfamily_method": metric_selection_analysis.metric_frequency(raw_t, by=["subfamily", "method_name"]),
        "metric_weight_entropy_summary": metric_selection_analysis.weight_entropy_summary(raw_t),
    }
    for name, df in msf.items():
        _w(run_dir, f"metric_selection/{name}.csv", df)
    _w(run_dir, "metric_family_frequency.csv", msf["metric_family_frequency"])
    _w(run_dir, "metric_selection_frequency_by_family.csv", msf["metric_selection_frequency_by_family"])
    _w(run_dir, "metric_weight_entropy_summary.csv", msf["metric_weight_entropy_summary"])

    # ---- statistical tests (optional) ---------------------------------- #
    if not args.no_statistical_tests:
        print("        Computing paired statistical tests "
              f"(scipy={'yes' if statistical_tests.scipy_available() else 'no'})...", flush=True)
        _w(run_dir, "statistical_tests/paired_significance_tests.csv",
           statistical_tests.paired_significance(dataset_level))
        _w(run_dir, "statistical_tests/paired_significance_by_family.csv",
           statistical_tests.paired_significance(dataset_level, group_cols=["family"]))

    # ---- [11/12] plots ------------------------------------------------- #
    saved_plots: List[str] = []
    if not args.no_plots:
        print("[11/12] Generating plots...", flush=True)
        plots_dir = run_dir / "plots"
        saved_plots += plotting.generate_all_plots(
            plots_dir=plots_dir,
            overall_summary=summaries.get("overall_method_summary", pd.DataFrame()),
            family_summary=summaries.get("family_method_summary", pd.DataFrame()),
            winrate_df=winrate, k_confusion=cms, metric_freq=metric_freq,
            best_view_raw=bv,
        )
        saved_plots += plotting.generate_extended_plots(
            plots_dir=plots_dir, gap_by_method=gap_by_method,
            gap_by_family_method=gap_by_family_method,
            family_cluster_ari=fam_cc, family_cluster_gap=fam_cc_gap, topk_df=topk,
        )
    else:
        print("[11/12] Skipping plots (--no-plots).", flush=True)

    # ---- [12/12] Excel workbooks + reports ----------------------------- #
    print("[12/12] Writing Excel workbooks and reports...", flush=True)
    xl_written: List[str] = []
    if not args.no_excel and excel_export.available():
        xl_dir = run_dir / "formatted_excel"
        ov = summaries.get("overall_method_summary", pd.DataFrame())
        fam_sum = summaries.get("family_method_summary", pd.DataFrame())
        sub_sum = summaries.get("subfamily_method_summary", pd.DataFrame())
        diff_sum = summaries.get("difficulty_method_summary", pd.DataFrame())
        cc_sum = summaries.get("cluster_count_method_summary", pd.DataFrame())
        view_sum = summaries.get("view_method_summary", pd.DataFrame())
        algo_sum = summaries.get("algorithm_method_summary", pd.DataFrame())
        xl_written.append(excel_export.write_workbook(xl_dir / "standard_summaries.xlsx", [
            ("overall", ov, [], True), ("family", fam_sum, ["family"], True),
            ("subfamily", sub_sum, ["subfamily"], True),
            ("difficulty", diff_sum, ["difficulty"], True),
            ("cluster_count", cc_sum, ["cluster_count"], True),
            ("view", view_sum, ["view_id"], True),
            ("algorithm", algo_sum, ["selected_algorithm"], True),
            ("external_metrics", ext_table, [], True),
        ]))
        xl_written.append(excel_export.write_workbook(xl_dir / "cross_analysis.xlsx", [
            ("family_x_cluster_count", cross["global_family_x_cluster_count_summary"], ["family", "cluster_count"], True),
            ("family_x_difficulty", cross["global_family_x_difficulty_summary"], ["family", "difficulty"], True),
            ("family_x_view", cross["global_family_x_view_summary"], ["family", "view_id"], True),
            ("method_x_cluster_count", cross["global_method_x_cluster_count_summary"], ["cluster_count"], True),
            ("method_x_difficulty", cross["global_method_x_difficulty_summary"], ["difficulty"], True),
            ("topk_regressor", topk, [], True),
        ]))
        xl_written.append(excel_export.write_workbook(xl_dir / "trace_best_gap.xlsx", [
            ("summary", tb_summary, [], False),
            ("by_method", gap_by_method, [], True),
            ("by_family", gap_by_family, ["family"], False),
            ("by_subfamily", trace_best_gap.trace_best_summary_by(bv, ["subfamily"]), ["subfamily"], False),
            ("by_cluster_count", trace_best_gap.trace_best_summary_by(bv, ["cluster_count"]), ["cluster_count"], False),
            ("by_family_cluster_count", fam_cc_gap, ["family", "cluster_count"], False),
        ]))
        xl_written.append(excel_export.write_workbook(xl_dir / "winrates.xlsx", [
            ("winrate_summary", winrate, ["baseline"], True),
            ("by_family", grouped_analysis.grouped_winrate(dataset_level, ["family"]), ["family", "baseline"], True),
            ("by_subfamily", grouped_analysis.grouped_winrate(dataset_level, ["subfamily"]), ["subfamily", "baseline"], True),
            ("by_cluster_count", grouped_analysis.grouped_winrate(dataset_level, ["cluster_count"]), ["cluster_count", "baseline"], True),
            ("by_difficulty", grouped_analysis.grouped_winrate(dataset_level, ["difficulty"]), ["difficulty", "baseline"], True),
        ]))
        xl_written.append(excel_export.write_workbook(xl_dir / "k_analysis.xlsx", [
            ("k_accuracy_summary", k_acc, [], True),
            ("by_family", grouped_analysis.k_accuracy_by(bv, ["family", "method_name"]), ["family"], True),
            ("by_subfamily", grouped_analysis.k_accuracy_by(bv, ["subfamily", "method_name"]), ["subfamily"], True),
        ]))
        xl_written.append(excel_export.write_workbook(xl_dir / "metric_selection.xlsx", [
            ("metric_frequency", metric_freq, [], False),
            ("metric_family_frequency", msf["metric_family_frequency"], [], False),
            ("top_metric_frequency", msf["top_metric_frequency"], [], False),
            ("weight_entropy", msf["metric_weight_entropy_summary"], [], False),
        ]))
        xl_written.append(excel_export.write_workbook(xl_dir / "full_analysis_workbook.xlsx", [
            ("overall_method_summary", ov, [], True),
            ("family_method_summary", fam_sum, ["family"], True),
            ("subfamily_method_summary", sub_sum, ["subfamily"], True),
            ("cluster_count_method", cc_sum, ["cluster_count"], True),
            ("winrate_summary", winrate, ["baseline"], True),
            ("winrate_by_family", grouped_analysis.grouped_winrate(dataset_level, ["family"]), ["family", "baseline"], True),
            ("k_accuracy_summary", k_acc, [], True),
            ("trace_best_gap_summary", tb_summary, [], False),
            ("trace_best_gap_by_family", gap_by_family, ["family"], False),
            ("family_x_cluster_count", cross["global_family_x_cluster_count_summary"], ["family", "cluster_count"], True),
            ("topk_regressor", topk, [], True),
            ("metric_selection_frequency", metric_freq, [], False),
            ("metric_family_frequency", msf["metric_family_frequency"], [], False),
        ]))
        xl_written = [x for x in xl_written if x]
    elif not args.no_excel:
        print("[warn] openpyxl not available; skipping Excel workbooks.", flush=True)

    # Reports.
    reports_dir = run_dir / "reports"
    _mkdirs(reports_dir)
    # 'overall_method_summary' is now the best-view summary (the only mean).
    overall_summary = summaries.get("overall_method_summary", pd.DataFrame())
    report_md = report_writer.render_aggregation_report(
        run_name=args.run_name, split_ids=args.split_ids, raw=raw_t,
        overall=overall_summary,
        winrate=winrate, k_accuracy=k_acc, config=vars(args),
        trace_best_summary=tb_summary, gap_by_method=gap_by_method, topk=topk,
        warnings=trace_warnings,
    )
    _write_text(run_dir / "aggregation_report.md", report_md)
    _write_text(reports_dir / "recommendations.md",
                report_writer.render_recommendations(overall_summary, winrate))
    _write_text(reports_dir / "global_analysis_report.md",
                report_writer.render_global_analysis_report(
                    overall=overall_summary,
                    family_summary=summaries.get("family_method_summary", pd.DataFrame()),
                    gap_by_method=gap_by_method, winrate=winrate, topk=topk))
    fam_reports_dir = reports_dir / "family_reports"
    _mkdirs(fam_reports_dir)
    fam_method_summary = summaries.get("family_method_summary", pd.DataFrame())
    for family in families:
        fm = (fam_method_summary[fam_method_summary["family"] == family]
              if not fam_method_summary.empty else pd.DataFrame())
        _write_text(fam_reports_dir / f"{_slug(family)}.md",
                    report_writer.render_family_report(
                        family=family, family_methods=fm,
                        gap_by_method_family=family_gap_summaries.get(family),
                        k_by_family=None,
                        top_metrics=family_top_metrics.get(family),
                        winrate_family=family_winrates.get(family)))

    # Config snapshot.
    config_payload = {
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "run_dir": str(run_dir),
        "timestamp": timestamp,
        "n_raw_rows": int(len(raw)),
        "n_dataset_level_rows": int(len(dataset_level)),
        "n_trace_rows": int(len(trace_table)),
        "families": list(families),
        "trace_warnings": trace_warnings,
        "scipy_available": statistical_tests.scipy_available(),
        "excel_written": [Path(x).name for x in xl_written] if not args.no_excel else [],
    }
    _write_text(run_dir / "aggregation_config.json",
                json.dumps(config_payload, indent=2, default=str))

    print(flush=True)
    print("Aggregation complete.", flush=True)
    print(f"Raw rows: {len(raw)}", flush=True)
    print(f"Dataset-level rows: {len(dataset_level)}", flush=True)
    print(f"Trace-best rows: {len(trace_table)}", flush=True)
    print(f"Families analyzed: {n_fam}", flush=True)
    print(f"Excel workbooks: {len(xl_written) if not args.no_excel else 0}", flush=True)
    print(f"Plots: {len(saved_plots)}", flush=True)
    print(f"Trace warnings: {trace_warnings or 'none'}", flush=True)
    print(f"Output: {run_dir}", flush=True)
    return 0


def main(argv: List[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
