"""Markdown report writer for the aggregation step."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict

import pandas as pd


def _fmt(v, nd: int = 4) -> str:
    try:
        f = float(v)
        if f != f:  # NaN
            return "nan"
        return f"{f:.{nd}f}"
    except Exception:
        return str(v)


def _best_row(df, col: str, ascending: bool):
    if df is None or getattr(df, "empty", True) or col not in df.columns:
        return None
    d = df.copy()
    d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna(subset=[col])
    if d.empty:
        return None
    return d.sort_values(col, ascending=ascending).iloc[0]


def _render_overall_ari_table(lines, summary_df, ari_col: str, title: str):
    """Append a 'mean best-view ARI by method' style table reading ``ari_col``."""
    if summary_df is None or getattr(summary_df, "empty", True) \
            or ari_col not in summary_df.columns:
        return
    lines.append(f"## {title}\n")
    lines.append("| method | n | success_rate | mean_best_view_ari | median_ari | "
                 "mean_nmi | mean_runtime_s | k_accuracy |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    ov = summary_df.sort_values(ari_col, ascending=False)
    for _, r in ov.iterrows():
        lines.append(
            f"| {r['method_name']} | {int(r['count'])} | "
            f"{_fmt(r.get('success_rate'),3)} | {_fmt(r[ari_col])} | "
            f"{_fmt(r.get('median_ari'))} | {_fmt(r.get('mean_nmi'))} | "
            f"{_fmt(r.get('mean_runtime_sec'),2)} | {_fmt(r.get('k_accuracy'),3)} |"
        )
    lines.append("")


def render_aggregation_report(
    *,
    run_name: str,
    split_ids,
    raw: pd.DataFrame,
    overall: pd.DataFrame,
    winrate: pd.DataFrame,
    k_accuracy: pd.DataFrame,
    config: Dict,
    trace_best_summary=None,
    gap_by_method=None,
    topk=None,
    warnings: Dict = None,
    overall_best_view: pd.DataFrame = None,
) -> str:
    lines = ["# AutoClustering Experiment — Aggregation Report\n"]
    lines.append(f"- Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"- Run: **{run_name}**")
    lines.append(f"- Split ids: {list(split_ids)}")
    lines.append(f"- Raw rows: **{len(raw)}**")
    if not raw.empty:
        n_datasets = raw["dataset_id"].nunique()
        n_success = int((raw["status"] == "success").sum())
        lines.append(f"- Datasets: **{n_datasets}** | successful runs: "
                     f"**{n_success}/{len(raw)}**")
    lines.append("")
    lines.append("> **Headline metric = Mean Best-View ARI.** For each "
                 "dataset x method the view (x_only / y_only / xy_2d) with the "
                 "highest ARI is selected, then averaged over datasets. The same "
                 "best-view collapse is applied identically to every method, so "
                 "the comparison stays fair; it matches how external "
                 "AutoClustering frameworks report a single best record per "
                 "dataset. Per-view numbers remain in `raw_results.csv` and the "
                 "per-view summary.")
    lines.append("")

    if not overall.empty:
        _render_overall_ari_table(
            lines, overall, "mean_ari",
            "Overall: Mean Best-View ARI by method")

    if not winrate.empty:
        lines.append("## Win rates (regressor vs baseline)\n")
        lines.append("| method | baseline | n | win | tie | loss | mean_ari_gain |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for _, r in winrate.iterrows():
            lines.append(
                f"| {r['method']} | {r['baseline']} | {int(r['n_paired'])} | "
                f"{_fmt(r['win_rate'],3)} | {_fmt(r['tie_rate'],3)} | "
                f"{_fmt(r['loss_rate'],3)} | {_fmt(r['mean_ari_gain'])} |"
            )
        lines.append("")

    if not k_accuracy.empty:
        lines.append("## K-selection accuracy by method\n")
        lines.append("| method | n | k_accuracy | mean_k_abs_err | "
                     "overcluster | undercluster | exact |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for _, r in k_accuracy.iterrows():
            lines.append(
                f"| {r['method_name']} | {int(r['n'])} | {_fmt(r['k_accuracy'],3)} | "
                f"{_fmt(r['mean_k_abs_error'],3)} | {_fmt(r['overcluster_rate'],3)} | "
                f"{_fmt(r['undercluster_rate'],3)} | {_fmt(r['exact_k_rate'],3)} |"
            )
        lines.append("")

    # Headline winners (best-view mean is the only headline metric).
    lines.append("## Headline winners\n")
    headline_overall = overall
    headline_col = "mean_ari"
    headline_label = "Mean Best-View ARI"
    best_ari = _best_row(headline_overall, headline_col, ascending=False)
    if best_ari is not None:
        lines.append(f"- Best {headline_label}: **{best_ari['method_name']}** "
                     f"({_fmt(best_ari[headline_col])}).")
    best_k = _best_row(overall, "k_accuracy", ascending=False)
    if best_k is not None:
        lines.append(f"- Best K accuracy: **{best_k['method_name']}** "
                     f"({_fmt(best_k['k_accuracy'],3)}).")
    if gap_by_method is not None and not getattr(gap_by_method, "empty", True):
        best_gap = _best_row(gap_by_method, "mean_ari_gap_to_trace_best", ascending=True)
        if best_gap is not None:
            lines.append(f"- Smallest trace-best opportunity gap: "
                         f"**{best_gap['method_name']}** "
                         f"(mean gap {_fmt(best_gap['mean_ari_gap_to_trace_best'])}).")
    if topk is not None and not getattr(topk, "empty", True):
        bt = _best_row(topk[topk.get("method_name").notna()] if "method_name" in topk else topk,
                       "mean_ari", ascending=False)
        if bt is not None and "method_name" in topk:
            lines.append(f"- Best Top-K regressor variant: **{bt['method_name']}** "
                         f"(mean ARI {_fmt(bt['mean_ari'])}).")
    # Best baseline (non-regressor), on the primary metric.
    if headline_overall is not None and not getattr(headline_overall, "empty", True) \
            and "method_name" in headline_overall.columns:
        base = headline_overall[
            ~headline_overall["method_name"].astype(str).str.startswith("regressor")]
        bb = _best_row(base, headline_col, ascending=False)
        if bb is not None:
            lines.append(f"- Best baseline: **{bb['method_name']}** "
                         f"({headline_label} {_fmt(bb[headline_col])}).")
    lines.append("")

    # Trace-best opportunity-gap interpretation.
    if trace_best_summary is not None and not getattr(trace_best_summary, "empty", True):
        r = trace_best_summary.iloc[0]
        lines.append("## Trace-best opportunity gap\n")
        lines.append(
            "> The opportunity gap is `trace_best_ari - selected_ari`: how much "
            "higher the best ARI in the ClustOpt search trace was than the ARI of "
            "the configuration the internal objective selected. A small gap means "
            "the objective picked nearly the best partition the search found; a "
            "large gap means the search *found* a good partition but the objective "
            "failed to select it (objective failure, not search-space failure).")
        lines.append("")
        lines.append(f"- Mean trace-best ARI: **{_fmt(r.get('mean_trace_best_ari'))}**")
        lines.append(f"- Mean selected ARI: **{_fmt(r.get('mean_selected_ari'))}**")
        lines.append(f"- Mean ARI gap to trace-best: "
                     f"**{_fmt(r.get('mean_ari_gap_to_trace_best'))}**")
        lines.append(f"- Selected-is-trace-best rate: "
                     f"**{_fmt(r.get('selected_is_trace_best_rate'),3)}**")
        lines.append("")

    if warnings:
        lines.append("## Warnings / missing inputs\n")
        for k, v in warnings.items():
            lines.append(f"- {k}: {v}")
        lines.append("")

    lines.append("## Outputs\n")
    lines.append("Structured under this run dir: `raw/`, `standard_summaries/`, "
                 "`winrates/`, `k_analysis/`, `trace_best_gap/`, `cross_analysis/`, "
                 "`metric_selection/`, `statistical_tests/`, `formatted_excel/`, "
                 "`plots/`, `reports/`.")
    lines.append("")
    return "\n".join(lines) + "\n"


def render_global_analysis_report(
    *, overall, family_summary, gap_by_method, winrate, topk,
    overall_best_view=None,
) -> str:
    lines = ["# Global Analysis Report\n"]
    lines.append(f"- Generated: {datetime.now(timezone.utc).isoformat()}\n")
    if overall is not None and not getattr(overall, "empty", True):
        lines.append("## Methods ranked by Mean Best-View ARI\n")
        lines.append("| rank | method | mean_best_view_ari | median_ari | k_accuracy | mean_runtime_s |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        ov = overall.sort_values("mean_ari", ascending=False).reset_index(drop=True)
        for i, r in ov.iterrows():
            lines.append(f"| {i+1} | {r['method_name']} | {_fmt(r['mean_ari'])} | "
                         f"{_fmt(r['median_ari'])} | {_fmt(r.get('k_accuracy'),3)} | "
                         f"{_fmt(r.get('mean_runtime_sec'),2)} |")
        lines.append("")
    if gap_by_method is not None and not getattr(gap_by_method, "empty", True):
        lines.append("## Methods ranked by trace-best gap (lower is better)\n")
        lines.append("| rank | method | mean_gap | selected_is_trace_best_rate |")
        lines.append("| --- | --- | --- | --- |")
        gm = gap_by_method.sort_values("mean_ari_gap_to_trace_best").reset_index(drop=True)
        for i, r in gm.iterrows():
            lines.append(f"| {i+1} | {r['method_name']} | "
                         f"{_fmt(r['mean_ari_gap_to_trace_best'])} | "
                         f"{_fmt(r.get('selected_is_trace_best_rate'),3)} |")
        lines.append("")
    if topk is not None and not getattr(topk, "empty", True) and "top_k" in topk.columns:
        lines.append("## Top-K regressor tradeoff\n")
        lines.append("| top_k | mean_ari | k_accuracy | mean_gap | mean_weight_entropy |")
        lines.append("| --- | --- | --- | --- | --- |")
        for _, r in topk.sort_values("top_k").iterrows():
            lines.append(f"| {int(r['top_k'])} | {_fmt(r.get('mean_ari'))} | "
                         f"{_fmt(r.get('k_accuracy'),3)} | "
                         f"{_fmt(r.get('mean_ari_gap_to_trace_best'))} | "
                         f"{_fmt(r.get('mean_weight_entropy'))} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def render_family_report(
    *, family: str, family_methods, gap_by_method_family, k_by_family,
    top_metrics, winrate_family, family_methods_best_view=None,
) -> str:
    lines = [f"# Family Report — {family}\n"]
    lines.append(f"- Generated: {datetime.now(timezone.utc).isoformat()}\n")
    if family_methods is not None and not getattr(family_methods, "empty", True):
        lines.append("## Methods ranked by Mean Best-View ARI\n")
        lines.append("| rank | method | mean_best_view_ari | median_ari | k_accuracy |")
        lines.append("| --- | --- | --- | --- | --- |")
        fm = family_methods.sort_values("mean_ari", ascending=False).reset_index(drop=True)
        for i, r in fm.iterrows():
            lines.append(f"| {i+1} | {r['method_name']} | {_fmt(r['mean_ari'])} | "
                         f"{_fmt(r['median_ari'])} | {_fmt(r.get('k_accuracy'),3)} |")
        lines.append("")
    if gap_by_method_family is not None and not getattr(gap_by_method_family, "empty", True):
        lines.append("## Methods ranked by trace-best gap (lower=better)\n")
        lines.append("| rank | method | mean_gap | selected_is_trace_best_rate |")
        lines.append("| --- | --- | --- | --- |")
        gm = gap_by_method_family.sort_values("mean_ari_gap_to_trace_best").reset_index(drop=True)
        for i, r in gm.iterrows():
            lines.append(f"| {i+1} | {r['method_name']} | "
                         f"{_fmt(r['mean_ari_gap_to_trace_best'])} | "
                         f"{_fmt(r.get('selected_is_trace_best_rate'),3)} |")
        lines.append("")
    if top_metrics is not None and not getattr(top_metrics, "empty", True):
        lines.append("## Top selected metrics (regressor)\n")
        tm = top_metrics.copy()
        sort_col = "selection_count" if "selection_count" in tm.columns else (
            "top_count" if "top_count" in tm.columns else None)
        if sort_col:
            tm = tm.sort_values(sort_col, ascending=False).head(15)
        metric_col = "metric" if "metric" in tm.columns else (
            "top_metric" if "top_metric" in tm.columns else None)
        if metric_col:
            for _, r in tm.iterrows():
                cnt = r.get(sort_col) if sort_col else ""
                lines.append(f"- {r[metric_col]} ({int(cnt) if pd.notna(cnt) else ''})")
        lines.append("")
    if winrate_family is not None and not getattr(winrate_family, "empty", True):
        lines.append("## Win-rate highlights (regressor vs baseline)\n")
        wf = winrate_family.sort_values("win_rate", ascending=False).head(8)
        lines.append("| method | baseline | win_rate | mean_ari_gain |")
        lines.append("| --- | --- | --- | --- |")
        for _, r in wf.iterrows():
            lines.append(f"| {r['method']} | {r['baseline']} | {_fmt(r['win_rate'],3)} | "
                         f"{_fmt(r['mean_ari_gain'])} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def render_recommendations(overall: pd.DataFrame, winrate: pd.DataFrame,
                           overall_best_view: pd.DataFrame = None) -> str:
    lines = ["# Recommendations\n"]
    if overall is None or getattr(overall, "empty", True):
        lines.append("No successful runs were found; cannot make recommendations.")
        return "\n".join(lines) + "\n"
    best = overall.sort_values("mean_ari", ascending=False).iloc[0]
    lines.append(f"- Highest Mean Best-View ARI: **{best['method_name']}** "
                 f"(mean_best_view_ari={_fmt(best['mean_ari'])}, "
                 f"k_accuracy={_fmt(best.get('k_accuracy'),3)}).")
    if not winrate.empty:
        best_wr = winrate.sort_values("win_rate", ascending=False).iloc[0]
        lines.append(f"- Strongest win rate: **{best_wr['method']}** beats "
                     f"**{best_wr['baseline']}** {_fmt(best_wr['win_rate'],3)} of the time "
                     f"(mean ARI gain {_fmt(best_wr['mean_ari_gain'])}).")
    lines.append("- Cross-check ARI against K accuracy and runtime before drawing "
                 "conclusions: a method can win on ARI while over-/under-clustering.")
    return "\n".join(lines) + "\n"
