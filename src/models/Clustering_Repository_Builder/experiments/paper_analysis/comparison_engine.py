"""Comparison-suite engine: executes one declarative config into artifacts.

Shared machinery for all five suites, so a suite is a *config* rather than a
bespoke script. The engine:

1. resolves the suite's candidate set from the config, replacing every
   ``@representative:<id>`` with the concrete method from the **frozen** selection
   manifest (a suite can never reselect);
2. runs the capability preflight and records which sub-analyses are skipped, for
   which methods and because of which missing field;
3. emits the summary / delta / win-rate / gap / metric-usage / runtime tables the
   config declares;
4. emits the declared plots via :mod:`plots`, each with its data CSV;
5. runs the declared paired hypotheses with per-family Holm correction;
6. writes the five reports and the workbook;
7. registers every artifact in the :class:`~artifact_manifest.ArtifactManifest`.

The five suites are fully implemented here but are **not executed** until the
preliminary study has been reviewed; :func:`run_comparison` is the entry point the
CLI calls for ``--stage comparison-0X``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import capability, frames, gaps, metric_analysis, paths, plots, stats, vbs
from .artifact_manifest import ArtifactManifest
from .config import (
    ConfigError, RepresentativeIndex, RunConfig, SUITE_SUBDIRS, load_config,
    resolve_config,
)
from .excel import comparison_workbook, write_workbook
from .method_registry import METHOD_BY_NAME, short_label
from .portfolio_registry import (
    PORTFOLIO_BY_ID, SUBSET_INVARIANTS as PORTFOLIO_SUBSET_INVARIANTS,
)


@dataclass
class ComparisonContext:
    """Everything a suite needs, assembled once by the CLI."""

    run: RunConfig
    manifest: ArtifactManifest
    best_view: pd.DataFrame
    dataset_level: pd.DataFrame
    capability_matrix: pd.DataFrame
    representatives: RepresentativeIndex
    selection_manifest: pd.DataFrame
    run_frame: Optional[pd.DataFrame] = None
    trace_table: Optional[pd.DataFrame] = None


@dataclass
class ComparisonResult:
    comparison_id: str
    section: str
    tables: Dict[str, Optional[pd.DataFrame]] = field(default_factory=dict)
    # Reproducible bulk data (e.g. every sampled subset) written as Parquet
    # rather than CSV, and gitignorable.
    large_frames: Dict[str, pd.DataFrame] = field(default_factory=dict)
    warnings: pd.DataFrame = field(default_factory=pd.DataFrame)
    plot_results: List[Any] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Candidate resolution
# --------------------------------------------------------------------------- #
def _resolve_candidates(config: Dict[str, Any],
                        index: RepresentativeIndex) -> Tuple[List[str], Dict[str, str]]:
    """Ordered candidate method names + ``method -> display id`` labels.

    Handles the three config shapes in use: ``candidate_pools`` (suite 1),
    ``candidates`` with a ``clustopt``/``external`` split (suite 4), and
    representative grids driven by the frozen manifest (suite 2).
    """
    labels: Dict[str, str] = {}
    ordered: List[str] = []

    def push(name: str, display: str = "") -> None:
        if name not in ordered:
            ordered.append(name)
        if display:
            labels[name] = display

    cands = config.get("candidates")
    if isinstance(cands, dict):
        clustopt = cands.get("clustopt") or {}
        for rep_id in clustopt.get("representative_ids", []) or []:
            push(index.method(rep_id), rep_id)
        for name in cands.get("external", []) or []:
            push(str(name))
        order = cands.get("display_order") or []
        if order:
            resolved_order: List[str] = []
            for item in order:
                item = str(item)
                resolved_order.append(index.method(item) if item in index else item)
            ordered = [m for m in resolved_order if m in ordered] + \
                      [m for m in ordered if m not in resolved_order]
        return ordered, labels

    pools = config.get("candidate_pools")
    if isinstance(pools, dict):
        for key, value in pools.items():
            if key == "display_order" or not isinstance(value, list):
                continue
            for name in value:
                push(str(name))
        order = pools.get("display_order") or []
        if order:
            ordered = [str(m) for m in order if str(m) in ordered] + \
                      [m for m in ordered if m not in [str(x) for x in order]]
        return ordered, labels

    reps = config.get("representatives")
    if isinstance(reps, dict) and reps.get("source") == "frozen_selection_manifest":
        prefix = str(reps.get("representative_id_prefix", ""))
        for rep_id in sorted(index.by_id):
            if prefix and not rep_id.startswith(prefix):
                continue
            push(index.method(rep_id), rep_id)
        return ordered, labels

    # Portfolio-only suites (3 and 5) declare no candidate block. Fall back to
    # the union of their portfolio members so the suite still reports coverage
    # and per-member context rather than an empty candidate set -- the members
    # are real methods and their individual performance is what the portfolio
    # envelope is built from.
    block = config.get("portfolios")
    if isinstance(block, dict):
        for key in ("primary", "optional_matched", "supplementary"):
            for pid in block.get(key, []) or []:
                pf = PORTFOLIO_BY_ID.get(str(pid))
                if pf is None:
                    continue
                for method in pf.methods:
                    push(method)
    return ordered, labels


def _resolve_portfolios(config: Dict[str, Any]) -> List[str]:
    block = config.get("portfolios") or {}
    ids: List[str] = []
    for key in ("primary", "optional_matched", "supplementary"):
        for pid in block.get(key, []) or []:
            if str(pid) not in ids:
                ids.append(str(pid))
    unknown = [p for p in ids if p not in PORTFOLIO_BY_ID]
    if unknown:                                                # pragma: no cover
        raise ConfigError(f"config names unknown portfolios: {unknown}")
    return ids


# --------------------------------------------------------------------------- #
# Hypothesis construction
# --------------------------------------------------------------------------- #
def _build_hypotheses(config: Dict[str, Any], candidates: Sequence[str],
                      index: RepresentativeIndex) -> List[stats.Hypothesis]:
    """Turn the config's ``pairwise_hypotheses`` block into Hypothesis objects."""
    block = config.get("pairwise_hypotheses") or {}
    out: List[stats.Hypothesis] = []
    rep_methods = {index.method(r) for r in index.by_id}

    def is_rep(name: str) -> bool:
        return name in rep_methods

    for item in block.get("primary", []) or []:
        if not isinstance(item, dict) or "a" not in item or "b" not in item:
            continue
        a, b = str(item["a"]), str(item["b"])
        out.append(stats.Hypothesis(
            hypothesis_id=str(item.get("id", f"{a}_vs_{b}")),
            family=str(item.get("family", config.get("comparison_id", "primary"))),
            method_a=a, method_b=b,
            claim_level=stats.CLAIM_CONFIRMATORY,
            direction=str(item.get("direction", "two-sided")),
            rationale=str(item.get("rationale", "")), is_primary=True,
            a_is_representative=is_rep(a), b_is_representative=is_rep(b)))

    grid = block.get("exploratory_grid")
    if isinstance(grid, dict):
        methods = [str(m) for m in grid.get("methods", candidates)]
        references = [str(m) for m in grid.get("references", [])]
        for m in methods:
            for r in references:
                if m == r:
                    continue
                out.append(stats.Hypothesis(
                    hypothesis_id=f"{m}_vs_{r}",
                    family=f"{config.get('comparison_id', 'exploratory')}_exploratory",
                    method_a=m, method_b=r,
                    claim_level=stats.CLAIM_EXPLORATORY, direction="two-sided",
                    rationale="exploratory grid", is_primary=False,
                    a_is_representative=is_rep(m), b_is_representative=is_rep(r)))
    return [h for h in out
            if h.method_a in METHOD_BY_NAME and h.method_b in METHOD_BY_NAME]


# --------------------------------------------------------------------------- #
# Table builders
# --------------------------------------------------------------------------- #
def _candidate_frame(best_view: pd.DataFrame, candidates: Sequence[str]
                     ) -> pd.DataFrame:
    return best_view[best_view["method_name"].isin(list(candidates))].copy()


def _summary_tables(sub: pd.DataFrame, grouping_levels: Sequence[str]
                    ) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    mapping = {
        "global": ("method_name",),
        "family": ("family", "method_name"),
        "subfamily": ("subfamily", "method_name"),
        "difficulty": ("difficulty", "method_name"),
        "cluster_count": ("cluster_count", "method_name"),
    }
    for level in grouping_levels:
        cols = mapping.get(str(level))
        if cols:
            out[str(level)] = frames.summarize_methods(sub, cols)
    return out


def _paired_delta_table(dataset_level: pd.DataFrame, config: Dict[str, Any],
                        candidates: Sequence[str], index: RepresentativeIndex
                        ) -> pd.DataFrame:
    """Per-dataset paired deltas for every declared research-question pair."""
    rows: List[pd.DataFrame] = []
    for rq_id, body in (config.get("research_questions") or {}).items():
        if not isinstance(body, dict):
            continue
        ref = body.get("reference")
        comps = body.get("comparisons") or []
        pairs: List[Tuple[str, str]] = []
        if ref and comps:
            ref_name = index.method(ref[len("@representative:"):]) \
                if isinstance(ref, str) and ref.startswith("@representative:") else str(ref)
            for c in comps:
                c_name = index.method(c[len("@representative:"):]) \
                    if isinstance(c, str) and c.startswith("@representative:") else str(c)
                pairs.append((ref_name, c_name))
        for item in body.get("pairs", []) or []:
            if isinstance(item, dict) and "a" in item and "b" in item:
                pairs.append((str(item["a"]), str(item["b"])))
        for a, b in pairs:
            if a not in METHOD_BY_NAME or b not in METHOD_BY_NAME:
                continue
            merged = gaps._paired(dataset_level, a, b)
            if merged.empty:
                continue
            merged = merged.assign(
                research_question=rq_id,
                reference_method=a, reference_short_label=short_label(a),
                comparison_method=b, comparison_short_label=short_label(b),
                delta=merged["ari_a"] - merged["ari_b"],
                pair_label=f"{short_label(a)} - {short_label(b)}",
                clean_comparison=bool(body.get("clean", False)),
                note=str(body.get("note", "")))
            rows.append(merged)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _research_question_table(config: Dict[str, Any]) -> pd.DataFrame:
    """The suite's research questions as a shipped table (never only in prose)."""
    rows: List[Dict[str, Any]] = []
    for rq_id, body in (config.get("research_questions") or {}).items():
        if not isinstance(body, dict):
            continue
        rows.append({
            "comparison_id": config.get("comparison_id"),
            "research_question_id": rq_id,
            "question": body.get("question", ""),
            "kind": body.get("kind", ""),
            "reference": body.get("reference", ""),
            "comparisons": ";".join(str(c) for c in (body.get("comparisons") or [])),
            "is_clean_comparison": bool(body.get("clean", False)),
            "gap_type": body.get("gap_type", ""),
            "note": body.get("note", ""),
        })
    return pd.DataFrame(rows)


def _runtime_table(sub: pd.DataFrame, matrix: pd.DataFrame) -> pd.DataFrame:
    """Runtime summary with capability flags, never mixing runtime meanings."""
    summ = frames.summarize_methods(sub, ("method_name",))
    if summ.empty:
        return summ
    keep = ["method_name", "method_short_label", "method_family",
            # PRIMARY: obtaining a Best-View result requires all three views.
            "mean_total_runtime_all_views_sec",
            "median_total_runtime_all_views_sec",
            "std_total_runtime_all_views_sec",
            "p90_total_runtime_all_views_sec",
            "sum_total_runtime_all_views_sec",
            # DIAGNOSTIC: the single view whose ARI was reported.
            "mean_selected_best_view_runtime_sec",
            "median_selected_best_view_runtime_sec",
            "best_view_mean_ari",
            "has_total_all_views_runtime", "has_runtime_components"]
    out = summ[[c for c in keep if c in summ.columns]].copy()
    out["primary_cost_column"] = "median_total_runtime_all_views_sec"
    out["selected_view_runtime_is_diagnostic_only"] = True
    out["has_end_to_end_runtime"] = False
    out["runtime_wording"] = "total recorded search/runtime across all three views"
    out["end_to_end_note"] = (
        "NOT complete end-to-end runtime: meta-feature extraction was performed "
        "offline and utility prediction is not timed per run, so those "
        "components are unavailable and are never approximated from runtime_sec")
    if not matrix.empty:
        out = out.merge(
            matrix[["method_name", "runtime_components", "all_view_runtime"]],
            on="method_name", how="left")
    return out


# --------------------------------------------------------------------------- #
# Plot builders
# --------------------------------------------------------------------------- #
def _emit_plots(config: Dict[str, Any], result: ComparisonResult,
                sub: pd.DataFrame, tables: Dict[str, Optional[pd.DataFrame]],
                ctx: ComparisonContext, order: Sequence[str]) -> plots.PlotContext:
    """Emit the declared plots for a suite, skipping any whose inputs are absent."""
    section_dir = ctx.run.output_root / result.section / "plots"
    pctx = plots.PlotContext(output_dir=section_dir,
                             min_importance=ctx.run.min_plot_importance,
                             enabled=ctx.run.make_plots)
    declared = {str(p.get("name")): p for p in (config.get("plots") or [])
                if isinstance(p, dict)}
    cid = str(config.get("comparison_id", ""))
    disclosure = str(config.get("required_disclosure", "")
                     or config.get("causal_caveat", ""))
    label_order = [short_label(m) for m in order]
    global_summary = tables.get("global_summary")

    def spec(name: str, title: str, *, ylabel: str = "", xlabel: str = "",
             caption: str = "", sources: Sequence[str] = ()) -> plots.PlotSpec:
        decl = declared.get(name, {})
        return plots.PlotSpec(
            name=name, title=title, ylabel=ylabel, xlabel=xlabel,
            caption=caption or disclosure,
            importance=str(decl.get("importance", "paper_supplement")),
            source_tables=tuple(sources),
            selection_type="mixed (see the candidate map)")

    if global_summary is not None and not global_summary.empty:
        name = f"{cid}_best_view_ari_bar"
        if name in declared:
            plots.bar_chart(global_summary, spec(
                name, "Mean Best-View ARI by method",
                ylabel="Mean Best-View ARI",
                sources=(f"{cid}_global_summary.csv",)), pctx,
                value_col="best_view_mean_ari",
                error_low_col="best_view_q25_ari", error_high_col="best_view_q75_ari")
        name = f"{cid}_k_accuracy"
        if name in declared:
            plots.bar_chart(global_summary, spec(
                name, "K-selection accuracy by method",
                ylabel="K accuracy (share of datasets)",
                sources=(f"{cid}_global_summary.csv",)), pctx,
                value_col="best_view_k_accuracy")
        name = f"{cid}_success_rate"
        if name in declared:
            plots.bar_chart(global_summary, spec(
                name, "Run success rate by method", ylabel="Success rate",
                sources=(f"{cid}_global_summary.csv",)), pctx,
                value_col="success_rate")
        name = f"{cid}_accuracy_runtime_pareto"
        if name in declared:
            plots.pareto_scatter(global_summary, spec(
                name, "Accuracy vs total recorded runtime across all three views",
                xlabel="Median total runtime, all three views (s)",
                ylabel="Mean Best-View ARI",
                caption=(disclosure + " Operational methods only: a VBS is not "
                                      "one executable method and is excluded. "
                                      "No composite ARI-per-second score is "
                                      "reported; the trade-off is shown as a "
                                      "frontier.").strip(),
                sources=(f"{cid}_runtime_summary.csv",)), pctx,
                x_col="median_total_runtime_all_views_sec",
                y_col="best_view_mean_ari")

    name = f"{cid}_best_view_ari_boxplot"
    if name in declared and not sub.empty:
        plots.boxplot(sub, spec(name, "Best-View ARI distribution by method",
                                ylabel="Best-View ARI",
                                sources=("canonical_best_view_frame",)), pctx,
                      value_col="best_view_ari", order=label_order)
    name = f"{cid}_runtime_boxplot"
    if name in declared and not sub.empty:
        plots.boxplot(sub, spec(
            name, "Total recorded runtime across all three views (log scale)",
            ylabel="Total recorded runtime, all three views (s)",
            caption=("The primary Best-View cost: obtaining a Best-View result "
                     "requires running all three views."),
            sources=("canonical_best_view_frame",)), pctx,
            value_col="total_runtime_all_views_sec", log_y=True)

    family = tables.get("family_summary")
    name = f"{cid}_family_heatmap"
    if name in declared and family is not None and not family.empty:
        pivot = family.pivot_table(index="family", columns="method_short_label",
                                   values="best_view_mean_ari", aggfunc="mean")
        cols = [c for c in label_order if c in pivot.columns]
        plots.heatmap(pivot[cols] if cols else pivot, spec(
            name, "Mean Best-View ARI by structural family",
            xlabel="method", ylabel="structural family",
            sources=(f"{cid}_family_summary.csv",)), pctx)

    deltas = tables.get("pairwise_deltas")
    name = f"{cid}_pairwise_delta_boxplots"
    if name in declared and deltas is not None and not deltas.empty:
        plots.paired_delta_boxplot(deltas, spec(
            name, "Paired per-dataset ARI deltas",
            ylabel="Best-View ARI delta (reference - comparison)",
            sources=(f"{cid}_pairwise_deltas.csv",)), pctx,
            delta_col="delta", group_col="pair_label")

    winrates = tables.get("winrates")
    name = f"{cid}_winrate_heatmap"
    if name in declared and winrates is not None and not winrates.empty:
        pivot = winrates.pivot_table(index="method_short_label",
                                     columns="reference_short_label",
                                     values="win_rate", aggfunc="mean")
        plots.heatmap(pivot, spec(
            name, "Paired win rate (row beats column)",
            xlabel="reference", ylabel="method",
            sources=(f"{cid}_winrates.csv",)), pctx, cmap="RdYlGn", center=0.5)

    cat = tables.get("metric_category_usage")
    for name, value_col, title in (
        (f"{cid}_metric_category_weight_share", "weight_mass_share",
         "Objective weight mass by metric category"),
        (f"{cid}_metric_category_stack", "weight_mass_share",
         "Objective weight mass by metric category"),
    ):
        if name in declared and cat is not None and not cat.empty:
            pivot = cat.pivot_table(index="method_short_label",
                                    columns="metric_category", values=value_col,
                                    aggfunc="sum")
            rows = [r for r in label_order if r in pivot.index]
            plots.stacked_bar(pivot, spec(
                name, title, ylabel="share of weight mass",
                sources=(f"{cid}_metric_category_usage.csv",)), pctx,
                order=rows or None, normalise=True)

    usage = tables.get("metric_usage")
    name = f"{cid}_metric_selection_heatmap"
    if name in declared and usage is not None and not usage.empty:
        top = (usage.groupby("metric_name")["share_of_selected_slots"].sum()
               .sort_values(ascending=False).head(30).index)
        pivot = (usage[usage["metric_name"].isin(top)]
                 .pivot_table(index="metric_name", columns="method_short_label",
                              values="selection_rate_per_run", aggfunc="mean"))
        plots.heatmap(pivot, spec(
            name, "Metric selection rate per run (top 30 metrics)",
            xlabel="method", ylabel="metric",
            caption="Selection rate per run is comparable across Top-K sizes; "
                    "raw counts are not.",
            sources=(f"{cid}_metric_usage.csv",)), pctx, value_fmt="{:.2f}")

    gap = tables.get("gap_summary")
    name = f"{cid}_trace_gap"
    if name in declared and gap is not None and not gap.empty:
        col = ("search_selection_gap_mean" if "search_selection_gap_mean" in gap.columns
               else "view_gap_mean")
        plots.bar_chart(gap, spec(
            name, "Search-selection gap by method (ClustOpt methods only)",
            ylabel="Mean gap (trace best - selected ARI)",
            caption="Unavailable for AutoClust and ML2DAC: their implementations "
                    "expose no compatible per-candidate search trace.",
            sources=(f"{cid}_gap_summary.csv",)), pctx,
            value_col=col, sort="value_asc")

    _emit_meta_learning_plots(cid, declared, spec, tables, pctx, ctx,
                              best_view=sub, order_labels=label_order)
    _emit_portfolio_plots(cid, declared, spec, tables, pctx, ctx)

    ctx.manifest.add_plot_results(pctx.results, comparison_id=cid,
                                  section=result.section)
    return pctx


def _emit_meta_learning_plots(cid, declared, spec, tables, pctx, ctx,
                              best_view=None, order_labels=None) -> None:
    """Suite-2 figures: the 3 x 5 source-by-policy grid and its diagnostics."""
    summary = tables.get("global_summary")
    if summary is None or summary.empty:
        return

    name = f"{cid}_source_by_policy_grouped_bar"
    if name in declared and {"utility_source", "k_policy"} <= set(summary.columns):
        pivot = summary.pivot_table(index="k_policy", columns="utility_source",
                                    values="best_view_mean_ari", aggfunc="mean")
        order = [p for p in ("fixed", "dynamic_predict", "dynamic_real",
                             "not_applicable") if p in pivot.index]
        plots.grouped_bar_chart(
            pivot.reindex(order) if order else pivot,
            spec(name, "Mean Best-View ARI by utility source and metric-count policy",
                 ylabel="Mean Best-View ARI", xlabel="metric-count policy",
                 sources=(f"{cid}_global_summary.csv",)), pctx)

    deltas = tables.get("matched_source_deltas")
    for name, subset, title in (
        (f"{cid}_mlp_vs_knn_paired_delta", "mlp_vs_knn",
         "MLP minus KNN, paired per dataset"),
        (f"{cid}_real_vs_predicted_gap", "utility_prediction_gap",
         "Real utility minus learned utility (prediction loss)"),
        (f"{cid}_dynamic_predict_vs_realK", "metric_count_gap",
         "Dynamic Real-K (oracle-assisted) minus Dynamic Predict-K"),
    ):
        if name not in declared:
            continue
        frame = tables.get(subset) if isinstance(tables.get(subset), pd.DataFrame) \
            else deltas
        if frame is None or frame.empty:
            continue
        value_col = ("gap_mean" if "gap_mean" in frame.columns
                     else ("delta" if "delta" in frame.columns else None))
        label_col = ("comparison_short_label" if "comparison_short_label"
                     in frame.columns else "method_short_label")
        if value_col is None or label_col not in frame.columns:
            continue
        plots.bar_chart(frame, spec(
            name, title, ylabel="Paired ARI delta",
            caption=("A positive value means the reference arm was better. The "
                     "reference in the prediction-loss and metric-count figures "
                     "is an ORACLE, so these are diagnostics, not deployable "
                     "comparisons."),
            sources=(f"{cid}_gap_decomposition.csv",)), pctx,
            value_col=value_col, label_col=label_col, sort="value_desc")

    name = f"{cid}_source_by_policy_boxplots"
    if name in declared and best_view is not None and not best_view.empty:
        # Dataset-level observations, not the aggregated means -- a box plot of
        # means would hide exactly the spread it exists to show.
        plots.boxplot(best_view, spec(
            name, "Best-View ARI distribution by utility source and policy",
            ylabel="Best-View ARI",
            caption="One observation per dataset. Real-utility arms are utility "
                    "oracles and DynRK arms are oracle-assisted; neither is a "
                    "deployable method.",
            sources=("canonical_best_view_frame",)), pctx,
            value_col="best_view_ari", order=order_labels)


def _emit_portfolio_plots(cid, declared, spec, tables, pctx, ctx) -> None:
    """Suite-3 and suite-5 figures: VBS bars, headroom, composition, curves."""
    vbs_summary = tables.get("vbs_summary")
    if vbs_summary is not None and not vbs_summary.empty:
        for name, title in ((f"{cid}_vbs_ari_bar",
                             "Portfolio VBS Mean Best-View ARI (oracle upper bound)"),):
            if name in declared:
                plots.bar_chart(vbs_summary, spec(
                    name, title, ylabel="VBS Mean Best-View ARI (oracle)",
                    caption="Each VBS selects the highest-ARI candidate separately "
                            "for each dataset and is therefore a non-deployable "
                            "upper bound. Portfolio sizes differ; read the "
                            "matched-size curve alongside.",
                    sources=(f"{cid}_vbs_summary.csv",)), pctx,
                    value_col="best_view_mean_ari",
                    label_col="portfolio_short_label")
        name = f"{cid}_full_cost_runtime"
        if name in declared:
            plots.bar_chart(vbs_summary, spec(
                name, "Full portfolio runtime (what realising the VBS would cost)",
                ylabel="Median full portfolio runtime, all views (s)",
                caption="This is the cost of exhaustively running every portfolio "
                        "member. The oracle-selected variant's runtime is a "
                        "diagnostic and is reported separately.",
                sources=(f"{cid}_runtime_diagnostics.csv",)), pctx,
                value_col="portfolio_total_execution_runtime_median_sec",
                label_col="portfolio_short_label", sort="value_asc")

    headroom = tables.get("headroom")
    name = f"{cid}_headroom_bar"
    if name in declared and headroom is not None and not headroom.empty:
        frame = headroom.copy()
        frame["headroom_label"] = (frame["portfolio_id"].astype(str) + " vs "
                                   + frame["reference_short_label"].astype(str))
        plots.bar_chart(frame, spec(
            name, "Policy-selection headroom over the best observed single variant",
            ylabel="Paired VBS headroom (ARI)",
            caption="Headroom is what a better per-dataset selection policy "
                    "could recover. It is an oracle upper bound, not an "
                    "achievable target. The reference is a retrospective "
                    "representative.",
            sources=(f"{cid}_headroom.csv",)), pctx,
            value_col="headroom_paired_mean", label_col="headroom_label",
            horizontal=True)

    # The union headroom is a different table: how much the union of two matched
    # weighting portfolios adds over the BETTER of them.
    union = tables.get("union_headroom")
    name = f"{cid}_union_headroom"
    if name in declared and union is not None and not union.empty:
        frame = union.copy()
        frame["union_label"] = (frame["union_portfolio"].astype(str) + " vs "
                                + frame["best_component"].astype(str))
        plots.bar_chart(frame, spec(
            name, "Union-portfolio headroom over the better component",
            ylabel="Paired VBS headroom (ARI)",
            caption="How much combining the raw and softmax portfolios adds "
                    "beyond the better of the two on its own. Both arms are "
                    "oracle upper bounds.",
            sources=(f"{cid}_union_headroom.csv",)), pctx,
            value_col="headroom_paired_mean", label_col="union_label",
            horizontal=True)

    for name, key, index_col in (
        (f"{cid}_winner_composition", "winner_composition", "winning_method_short_label"),
        (f"{cid}_topk_winner_distribution", "winner_composition_by_policy",
         "winning_k_policy"),
        (f"{cid}_utility_source_winner_distribution", "winner_composition_by_source",
         "winning_utility_source"),
    ):
        frame = tables.get(key)
        if name not in declared or frame is None or frame.empty:
            continue
        if index_col not in frame.columns:
            continue
        pivot = frame.pivot_table(index="portfolio_id", columns=index_col,
                                  values="win_count", aggfunc="sum").fillna(0.0)
        plots.stacked_bar(pivot, spec(
            name, "VBS winner composition",
            ylabel="share of datasets won", xlabel="portfolio",
            caption="Which variants the dataset-level oracle actually picks. "
                    "Winning often is not the same as being irreplaceable -- see "
                    "the unique-coverage table.",
            sources=(f"{cid}_winner_composition.csv",)), pctx, normalise=True)

    curve = tables.get("random_subset_summary")
    name = f"{cid}_portfolio_size_curve"
    if name in declared and curve is not None and not curve.empty:
        usable = curve[~curve["skipped"].astype(bool)]
        if not usable.empty:
            plots.line_curve(usable, spec(
                name, "VBS at matched portfolio size",
                xlabel="portfolio size", ylabel="VBS Mean Best-View ARI",
                caption="Random matched-size subsets with percentile confidence "
                        "bands. Sizes above a portfolio's own membership are "
                        "skipped and logged, never clipped.",
                sources=(f"{cid}_random_subset_summary.csv",)), pctx,
                x_col="size", y_col="vbs_mean", series_col="portfolio_id",
                ci_low_col="ci_low", ci_high_col="ci_high")

    greedy = tables.get("greedy_portfolio")
    name = f"{cid}_marginal_gain_curve"
    if name in declared and greedy is not None and not greedy.empty:
        plots.line_curve(greedy, spec(
            name, "Marginal VBS gain per added variant (greedy forward)",
            xlabel="portfolio size", ylabel="marginal gain in Mean Best-View ARI",
            caption="Greedy forward construction ordered by marginal gain; ties "
                    "resolve lexicographically so the sequence is reproducible.",
            sources=(f"{cid}_greedy_portfolio.csv",)), pctx,
            x_col="step", y_col="marginal_gain", series_col="portfolio_id")

    family_vbs = tables.get("family_vbs_summary")
    for name in (f"{cid}_family_delta_heatmap", f"{cid}_dataset_winner_matrix"):
        if name in declared and family_vbs is not None and not family_vbs.empty:
            pivot = family_vbs.pivot_table(index="family",
                                           columns="portfolio_short_label",
                                           values="best_view_mean_ari",
                                           aggfunc="mean")
            plots.heatmap(pivot, spec(
                name, "VBS Mean Best-View ARI by structural family",
                xlabel="portfolio", ylabel="structural family",
                sources=(f"{cid}_family_summary_vbs.csv",)), pctx)

    deltas = tables.get("raw_vs_softmax_deltas")
    name = f"{cid}_raw_vs_softmax_delta_boxplot"
    if name in declared and deltas is not None and not deltas.empty:
        plots.paired_delta_boxplot(deltas, spec(
            name, "Raw minus softmax VBS, paired per dataset",
            ylabel="Paired VBS ARI delta (raw - softmax)",
            caption="Positive favours raw weighting. Both arms are oracle upper "
                    "bounds over one-to-one matched 17-member portfolios.",
            sources=(f"{cid}_raw_vs_softmax_deltas.csv",)), pctx,
            delta_col="delta", group_col="pair_label")


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #
def _write_reports(config: Dict[str, Any], result: ComparisonResult,
                   ctx: ComparisonContext, candidates: Sequence[str],
                   labels: Dict[str, str], tests: Optional[pd.DataFrame]) -> None:
    from .reports import _table as md_table                    # local: report-only
    cid = str(config.get("comparison_id"))
    out_dir = ctx.run.output_root / result.section / "reports"
    disclosure = str(config.get("required_disclosure", ""))
    caveat = str(config.get("causal_caveat", ""))

    candidate_map = pd.DataFrame([{
        "display_id": labels.get(m, m),
        "selected_concrete_method": m,
        "short_label": short_label(m),
        "paper_label": METHOD_BY_NAME[m].paper_label if m in METHOD_BY_NAME else "",
        "method_family": METHOD_BY_NAME[m].method_family if m in METHOD_BY_NAME else "",
        "oracle_level": METHOD_BY_NAME[m].oracle_level if m in METHOD_BY_NAME else "",
        "selection_type": ("retrospective_global" if labels.get(m) in ctx.representatives
                           else "fixed_variant"),
        "candidate_pool_size": (
            ctx.representatives.record(labels[m]).get("candidate_pool_size")
            if labels.get(m) in ctx.representatives else 1),
        "same_search_space": (METHOD_BY_NAME[m].same_search_space
                              if m in METHOD_BY_NAME else None),
        "search_budget": (METHOD_BY_NAME[m].search_budget
                          if m in METHOD_BY_NAME else None),
    } for m in candidates])
    result.tables["candidate_map"] = candidate_map

    readme = [
        f"# {config.get('title')}",
        "",
        f"Comparison suite `{cid}`. Split "
        f"{ctx.run.split_id} only.",
        "",
    ]
    if disclosure:
        readme += ["> " + disclosure.replace("\n", " "), ""]
    if caveat:
        readme += ["> **Causal caveat.** " + caveat.replace("\n", " "), ""]
    readme += ["## Candidates", ""]
    readme += md_table(candidate_map, [
        "short_label", "selected_concrete_method", "selection_type",
        "candidate_pool_size", "oracle_level", "same_search_space",
        "search_budget"])
    readme += ["## Research questions", ""]
    for rq_id, body in (config.get("research_questions") or {}).items():
        if isinstance(body, dict):
            readme += [f"### {rq_id}", "", str(body.get("question", "")), ""]
            if body.get("note"):
                readme += ["> " + str(body["note"]).replace("\n", " "), ""]
    readme += ["## Files", "",
               "Every table, figure and workbook in this directory is registered "
               "in the run-level `analysis_manifest.csv` with its section, source "
               "tables, paper importance and selection type.", ""]
    ctx.manifest.add_report(
        paths.write_text(out_dir / "README.md", "\n".join(readme)) or
        (out_dir / "README.md"),
        comparison_id=cid, section=result.section,
        description=f"{config.get('title')}: candidates, research questions, "
                    f"required disclosure.",
        paper_importance="paper_main")

    methods_md = [
        f"# Methods and definitions -- {config.get('title')}",
        "",
        "See `../00_protocol_and_definitions/` for the full terminology, oracle "
        "levels, metric definitions and runtime definitions. This file records "
        "only what is specific to this suite.",
        "",
        "## Primary metrics", "",
    ]
    methods_md += [f"- `{m}`" for m in (config.get("primary_metrics") or [])]
    methods_md += ["", "## Secondary / diagnostic metrics", ""]
    methods_md += [f"- `{m}`" for m in (config.get("secondary_metrics") or [])] or ["_none_"]
    methods_md += ["", "## Required capabilities", ""]
    methods_md += [f"- `{c}`: {capability.CAPABILITY_DESCRIPTIONS.get(c, c)}"
                   for c in (config.get("required_capabilities") or [])]
    methods_md += ["", "## Optional capabilities (skipped where unavailable)", ""]
    methods_md += [f"- `{c}`" for c in (config.get("optional_capabilities") or [])] \
        or ["_none_"]
    if config.get("runtime_disclosure_text"):
        methods_md += ["", "## Runtime disclosure", "",
                       str(config["runtime_disclosure_text"]).replace("\n", " "), ""]
    ctx.manifest.add_report(
        paths.write_text(out_dir / "METHODS_AND_DEFINITIONS.md",
                         "\n".join(methods_md)) or
        (out_dir / "METHODS_AND_DEFINITIONS.md"),
        comparison_id=cid, section=result.section,
        description="Suite-specific metrics, capabilities and disclosures.",
        paper_importance="paper_supplement")

    results_md = [f"# Results -- {config.get('title')}", "",
                  "**Facts computed from data.** Interpretation suggestions are "
                  "confined to the final section and marked as suggestions.", ""]
    gsum = result.tables.get("global_summary")
    if gsum is not None and not gsum.empty:
        results_md += ["## Global summary", ""]
        results_md += md_table(gsum, [
            "method_short_label", "best_view_mean_ari", "best_view_median_ari",
            "best_view_std_ari", "best_view_k_accuracy",
            "total_runtime_all_views_median_sec", "n_datasets", "success_rate",
            "oracle_level"])
    if tests is not None and not tests.empty:
        results_md += ["## Paired tests", "",
                       "Tests whose `claim_level` is "
                       "`post_selection_exploratory` involve a retrospectively "
                       "selected representative tested on the same Split-1 data "
                       "and are optimistic by construction.", ""]
        results_md += md_table(tests, [
            "hypothesis_id", "method_a_short_label", "method_b_short_label",
            "n_paired", "mean_delta", "mean_delta_ci_low", "mean_delta_ci_high",
            "wilcoxon_p", "holm_p", "cohens_dz", "rank_biserial", "claim_level"])
    results_md += ["## Interpretation suggestions", "",
                   "_Auto-derived; verify against the CSV tables before quoting._",
                   ""]
    ctx.manifest.add_report(
        paths.write_text(out_dir / "RESULTS.md", "\n".join(results_md)) or
        (out_dir / "RESULTS.md"),
        comparison_id=cid, section=result.section,
        description="Computed results for this suite.",
        paper_importance="paper_main")

    from .reports import limitations_md
    lim = [f"# Limitations -- {config.get('title')}", ""]
    if disclosure:
        lim += ["## Required disclosure", "", disclosure.replace("\n", " "), ""]
    if caveat:
        lim += ["## Causal caveat", "", caveat.replace("\n", " "), ""]
    if config.get("unsupported_note"):
        lim += ["## Unsupported analyses", "",
                str(config["unsupported_note"]).replace("\n", " "), ""]
    lim += [limitations_md(capability_warnings=result.warnings)]
    ctx.manifest.add_report(
        paths.write_text(out_dir / "LIMITATIONS.md", "\n".join(lim)) or
        (out_dir / "LIMITATIONS.md"),
        comparison_id=cid, section=result.section,
        description="Limitations and unsupported analyses for this suite.",
        paper_importance="paper_main")

    cand = [f"# Paper candidates -- {config.get('title')}", "",
            "Artifacts in this suite marked `paper_main` in the artifact "
            "manifest, i.e. the figures and tables proposed for the paper.", ""]
    frame = ctx.manifest.to_frame()
    mine = frame[(frame["comparison_id"] == cid)
                 & (frame["paper_importance"] == "paper_main")]
    cand += md_table(mine, ["artifact_type", "relative_path", "description"],
                     limit=100)
    ctx.manifest.add_report(
        paths.write_text(out_dir / "PAPER_CANDIDATES.md", "\n".join(cand)) or
        (out_dir / "PAPER_CANDIDATES.md"),
        comparison_id=cid, section=result.section,
        description="Paper-candidate artifacts from this suite.",
        paper_importance="paper_supplement")


# --------------------------------------------------------------------------- #
# Suite runner
# --------------------------------------------------------------------------- #
def run_comparison(stage: str, ctx: ComparisonContext) -> ComparisonResult:
    """Execute one comparison suite end to end."""
    config = resolve_config(load_config(stage), ctx.representatives)
    cid = str(config["comparison_id"])
    section = str(config["section"])
    result = ComparisonResult(comparison_id=cid, section=section)
    out_dir = ctx.run.output_root / section
    for sub in SUITE_SUBDIRS:
        paths.mkdirs(out_dir / sub)
    # Snapshot the resolved config: a reader must be able to see exactly which
    # concrete methods the @representative placeholders resolved to for THIS run.
    paths.write_json(out_dir / "config_snapshot" / f"{cid}_resolved_config.json",
                     {k: v for k, v in config.items() if not k.startswith("_")})
    ctx.manifest.add_metadata(
        out_dir / "config_snapshot" / f"{cid}_resolved_config.json",
        comparison_id=cid, section=section,
        description=f"Resolved configuration for {cid}, with every "
                    f"@representative placeholder already substituted.",
        paper_importance="appendix", appendix_only=True)

    candidates, labels = _resolve_candidates(config, ctx.representatives)
    portfolio_ids = _resolve_portfolios(config)
    if not candidates and not portfolio_ids:                   # pragma: no cover
        raise ConfigError(f"{stage}: config resolved to no candidates and no "
                          f"portfolios")

    # ---- 1. capability preflight ---------------------------------------- #
    analyses = list(config.get("required_capabilities") or []) \
        + list(config.get("optional_capabilities") or [])
    supported, warns = capability.preflight(
        ctx.capability_matrix, comparison_id=cid, methods=candidates,
        analyses=analyses, verbose=ctx.run.verbose)
    result.warnings = capability.warnings_frame(warns)

    required = set(config.get("required_capabilities") or [])
    usable = [m for m in candidates
              if all(m in supported.get(a, []) for a in required)]
    dropped = [m for m in candidates if m not in usable]
    if dropped:
        result.notes.append(
            f"{len(dropped)} candidate(s) lack a REQUIRED capability and are "
            f"excluded from this suite: {dropped}")
    sub = _candidate_frame(ctx.best_view, usable)

    # ---- 2. summary tables ---------------------------------------------- #
    grouping = config.get("grouping_levels") or ["global", "family"]
    summaries = _summary_tables(sub, grouping)
    result.tables["global_summary"] = summaries.get("global")
    result.tables["family_summary"] = summaries.get("family")
    result.tables["subfamily_summary"] = summaries.get("subfamily")
    result.tables["difficulty_summary"] = summaries.get("difficulty")
    result.tables["cluster_count_summary"] = summaries.get("cluster_count")
    result.tables["research_questions"] = _research_question_table(config)

    # ---- 3. paired deltas + win rates ----------------------------------- #
    dl = ctx.dataset_level[ctx.dataset_level["method_name"].isin(usable)]
    result.tables["pairwise_deltas"] = _paired_delta_table(
        dl, config, usable, ctx.representatives)
    result.tables["winrates"] = stats.winrates(
        dl, methods=usable, references=usable)
    result.tables["win_tie_loss"] = result.tables["winrates"]

    # ---- 4. gaps -------------------------------------------------------- #
    gap_parts: List[pd.DataFrame] = []
    view_gap = gaps.view_gap_summary(sub, ("method_name",))
    if not view_gap.empty:
        gap_parts.append(view_gap)
    trace_capable = supported.get("trace_best_gap", [])
    if trace_capable and ctx.trace_table is not None and not ctx.trace_table.empty:
        with_trace = gaps.attach_search_selection_gap(
            sub[sub["method_name"].isin(trace_capable)], ctx.trace_table)
        ss = gaps.search_selection_gap_summary(with_trace, ("method_name",))
        if not ss.empty:
            gap_parts.append(ss)
    elif "trace_best_gap" not in analyses:
        result.notes.append(
            "search-selection (trace-best) gap not part of this suite: the "
            "config does not declare the trace_best_gap capability")
    elif ctx.trace_table is None or ctx.trace_table.empty:
        result.notes.append(
            "search-selection (trace-best) gap skipped: the trace table was not "
            "computed (run the preliminary stage with --trace-gaps)")
    else:
        result.notes.append(
            "search-selection (trace-best) gap skipped: no candidate in this "
            "suite exposes a compatible per-candidate search trace")
    result.tables["gap_summary"] = (pd.concat(gap_parts, ignore_index=True)
                                   if gap_parts else pd.DataFrame())
    result.tables["gap_definitions"] = gaps.gap_definitions_frame()

    # Matched-arm gap tables. Only computed when the suite actually declares a
    # research question of that kind, so a suite does not carry tables it never
    # discusses.
    rq_kinds = {str(b.get("kind", "")) for b in
                (config.get("research_questions") or {}).values()
                if isinstance(b, dict)}
    if "matched_source_delta" in rq_kinds:
        upg = gaps.utility_prediction_gaps(ctx.best_view)
        result.tables["utility_prediction_gap"] = upg
        result.tables["matched_source_deltas"] = _matched_source_deltas(
            ctx.best_view, config)
        result.tables["gap_decomposition"] = pd.concat(
            [f for f in (upg, gaps.weighting_gaps(ctx.best_view),
                         gaps.metric_count_gaps(ctx.best_view)) if not f.empty],
            ignore_index=True) if not upg.empty else pd.DataFrame()
    if "matched_pair" in rq_kinds and cid == "comparison_02":
        result.tables["metric_count_gap"] = gaps.metric_count_gaps(ctx.best_view)
        result.tables["dynamic_predict_vs_realK"] = result.tables["metric_count_gap"]
    if "family_reselection_headroom" in rq_kinds:
        result.notes.append(
            "family-reselection headroom is read from the frozen preliminary "
            "family selections (01_preliminary_selection/family_divergence.csv); "
            "this suite does not reselect")

    # ---- 5. metric selection -------------------------------------------- #
    sel_capable = supported.get("metric_selection", [])
    if sel_capable:
        long_frame = metric_analysis.explode_selections(sub, methods=sel_capable)
        result.tables["metric_usage"] = metric_analysis.metric_usage(
            long_frame, ("method_name",))
        result.tables["metric_category_usage"] = metric_analysis.category_usage(
            long_frame, ("method_name",))
        result.tables["new_metric_usage"] = metric_analysis.new_metric_usage(
            long_frame, ("method_name",))
        result.tables["metric_entropy"] = metric_analysis.entropy_summary(
            sub[sub["method_name"].isin(sel_capable)], ("method_name",))
    elif "metric_selection" not in analyses:
        result.notes.append(
            "metric-selection analyses not part of this suite: the config does "
            "not declare the metric_selection capability")
    else:
        result.notes.append(
            "metric-selection analyses skipped: no candidate in this suite "
            "records a metric selection")

    # ---- 6. runtime + Pareto -------------------------------------------- #
    result.tables["runtime_summary"] = _runtime_table(sub, ctx.capability_matrix)
    gsum = result.tables.get("global_summary")
    if gsum is not None and not gsum.empty:
        # Operational Pareto only: gsum contains methods, never portfolios, so
        # no VBS point can enter the frontier.
        pareto = plots.pareto_frontier_table(
            gsum, x_col="median_total_runtime_all_views_sec",
            y_col="best_view_mean_ari")
        if not pareto.empty:
            pareto["contains_vbs_points"] = False
            pareto["frontier_scope"] = "operational methods only"
        result.tables["pareto_frontier"] = pareto

    # ---- 7. portfolios (suites 3 and 5) --------------------------------- #
    if portfolio_ids:
        vbs_frames = vbs.build_all_vbs_frames(ctx.best_view, portfolio_ids,
                                              verbose=ctx.run.verbose)
        result.tables["vbs_summary"] = vbs.vbs_summary(vbs_frames)
        result.tables["family_vbs_summary"] = vbs.vbs_summary(vbs_frames, ("family",))
        result.tables["dataset_winners"] = pd.concat(
            [f for f in vbs_frames.values() if not f.empty], ignore_index=True) \
            if vbs_frames else pd.DataFrame()
        result.tables["winner_composition"] = vbs.winner_composition(
            vbs_frames, ("winning_method_name",))
        result.tables["winner_composition_by_policy"] = vbs.winner_composition(
            vbs_frames, ("winning_k_policy",))
        result.tables["winner_composition_by_source"] = vbs.winner_composition(
            vbs_frames, ("winning_utility_source",))
        result.tables["winner_composition_by_weighting"] = vbs.winner_composition(
            vbs_frames, ("winning_weighting_mode",))
        result.tables["unique_coverage"] = vbs.unique_coverage(
            vbs_frames, ctx.best_view)
        refs = _reference_methods(config, ctx.representatives)
        if refs:
            result.tables["headroom"] = vbs.headroom(
                vbs_frames, ctx.best_view, reference_methods=refs)
        size_cfg = _size_sensitivity_config(config)
        if size_cfg:
            greedy_parts, curve_parts, sample_parts = [], [], []
            for pid in portfolio_ids:
                g = vbs.greedy_portfolio(ctx.best_view, pid)
                if not g.empty:
                    greedy_parts.append(g)
                c, samples = vbs.random_subset_curve(
                    ctx.best_view, pid, sizes=size_cfg["sizes"],
                    n_samples=size_cfg["n_samples"],
                    confidence=size_cfg["confidence"], seed=size_cfg["seed"],
                    return_samples=True)
                if not c.empty:
                    curve_parts.append(c)
                if not samples.empty:
                    sample_parts.append(samples)
            greedy = pd.concat(greedy_parts, ignore_index=True) if greedy_parts \
                else pd.DataFrame()
            curve = (pd.concat(curve_parts, ignore_index=True) if curve_parts
                     else pd.DataFrame())
            result.tables["greedy_portfolio"] = greedy
            result.tables["greedy_portfolio_membership"] = vbs.greedy_membership(greedy)
            result.tables["random_subset_summary"] = curve
            result.tables["portfolio_size_curve"] = curve
            result.tables["matched_size_vbs"] = vbs.matched_size_table(
                curve, sizes=size_cfg["sizes"])
            # Minimum useful portfolio at both reporting thresholds.
            min_parts = []
            for share in size_cfg["gain_shares"]:
                part = vbs.min_size_for_gain_share(greedy, share)
                if not part.empty:
                    min_parts.append(part)
            result.tables["minimum_portfolio_size"] = (
                pd.concat(min_parts, ignore_index=True) if min_parts
                else pd.DataFrame())
            result.tables["min_size_for_90pct_gain"] = result.tables[
                "minimum_portfolio_size"]
            if sample_parts:
                result.large_frames["random_subset_vbs_samples"] = pd.concat(
                    sample_parts, ignore_index=True)
            if not curve.empty and bool(curve["skipped"].any()):
                dropped = curve[curve["skipped"].astype(bool)]
                result.notes.append(
                    f"{len(dropped)} portfolio-size point(s) skipped because the "
                    f"requested size exceeds the portfolio's own membership: "
                    + "; ".join(
                        f"{r['portfolio_id']}@{int(r['size'])}"
                        for _, r in dropped.iterrows())
                    + ". Recorded in the random-subset summary rather than clipped.")
        result.tables["runtime_diagnostics"] = _vbs_runtime_diagnostics(
            result.tables.get("vbs_summary"))
        result.tables["raw_vs_softmax_deltas"] = _portfolio_paired_deltas(
            vbs_frames, config)
        result.tables["union_headroom"] = _union_headroom(vbs_frames, config)

    # ---- 8. statistics --------------------------------------------------- #
    tests: Optional[pd.DataFrame] = None
    if ctx.run.make_stats:
        hyps = _build_hypotheses(config, usable, ctx.representatives)
        if hyps:
            tests = stats.run_hypotheses(
                dl, hyps, n_bootstrap=ctx.run.bootstrap_samples,
                seed=ctx.run.seed, verbose=ctx.run.verbose)
            result.tables["paired_tests"] = tests

    # ---- 8b. frozen-manifest excerpts + transparency ---------------------- #
    result.tables["selection_manifest_excerpt"] = _selection_excerpt(
        ctx.selection_manifest, labels)
    if portfolio_ids:
        result.tables["portfolio_manifest_excerpt"] = pd.DataFrame(
            [PORTFOLIO_BY_ID[p].to_record() for p in portfolio_ids
             if p in PORTFOLIO_BY_ID])
    if labels:
        result.tables["candidate_selection_transparency"] = _transparency_table(
            usable, labels, ctx)

    # ---- 9. write tables ------------------------------------------------- #
    _write_tables(config, result, ctx)
    _write_large_frames(config, result, ctx)

    # ---- 10. plots ------------------------------------------------------- #
    pctx = _emit_plots(config, result, sub, result.tables, ctx, usable)
    result.plot_results = pctx.results

    # ---- 11. reports + workbook ------------------------------------------ #
    _write_reports(config, result, ctx, usable, labels, tests)
    if ctx.run.make_excel and config.get("workbook"):
        _write_workbook(config, result, ctx)
    _write_validation(config, result, ctx, usable, portfolio_ids)
    _write_suite_specific_reports(config, result, ctx, usable, labels)
    return result


def _suite_number(cid: str) -> str:
    return cid.split("_")[-1]


def _write_suite_specific_reports(config: Dict[str, Any],
                                  result: ComparisonResult,
                                  ctx: ComparisonContext,
                                  candidates: Sequence[str],
                                  labels: Dict[str, str]) -> None:
    """The named reports each suite owes beyond the five common ones."""
    from .reports import _table as md_table
    cid = str(config["comparison_id"])
    num = _suite_number(cid)
    reports_dir = ctx.run.output_root / result.section / "reports"

    def emit(filename: str, lines: List[str], description: str,
             importance: str = "paper_main") -> None:
        path = reports_dir / filename
        ctx.manifest.add_report(
            paths.write_text(path, "\n".join(lines)) or path,
            comparison_id=cid, section=result.section, description=description,
            paper_importance=importance)

    # ---- COMPARISON_0X_REPORT.md: the suite's headline numbers ------------ #
    lines = [f"# Comparison {num} -- {config.get('title')}", "",
             f"Split {ctx.run.split_id}. Facts computed from data; every "
             f"retrospective or oracle row is labelled as such.", ""]
    disclosure = str(config.get("required_disclosure", ""))
    caveat = str(config.get("causal_caveat", ""))
    if disclosure:
        lines += ["> " + disclosure.replace("\n", " "), ""]
    if caveat:
        lines += ["> **Causal caveat.** " + caveat.replace("\n", " "), ""]

    gsum = result.tables.get("global_summary")
    if gsum is not None and not gsum.empty:
        lines += ["## Primary results", "",
                  "Primary metric: Mean Best-View ARI. Primary cost: total "
                  "recorded runtime across all three views (a Best-View result "
                  "requires running all three).", ""]
        lines += md_table(gsum, [
            "method_short_label", "best_view_mean_ari", "best_view_median_ari",
            "best_view_std_ari", "best_view_k_accuracy",
            "median_total_runtime_all_views_sec",
            "mean_total_runtime_all_views_sec",
            "p90_total_runtime_all_views_sec", "success_rate", "oracle_level"],
            headers=["method", "mean BV-ARI", "median", "std", "K acc",
                     "median all-view runtime (s)", "mean (s)", "p90 (s)",
                     "success", "oracle"], limit=40)
    vsum = result.tables.get("vbs_summary")
    if vsum is not None and not vsum.empty:
        lines += ["## Portfolio VBS (oracle upper bounds)", "",
                  "Every row is a non-deployable per-dataset oracle. "
                  "`fully_operational` says whether all members could actually "
                  "be run without test-time ground truth.", ""]
        lines += md_table(vsum, [
            "portfolio_short_label", "portfolio_size", "best_view_mean_ari",
            "best_view_median_ari", "best_view_k_accuracy",
            "fully_operational", "oracle_contamination_reason",
            "n_distinct_winners",
            "portfolio_total_execution_runtime_median_sec"],
            headers=["portfolio", "n", "mean BV-ARI", "median", "K acc",
                     "operational", "contamination", "distinct winners",
                     "full portfolio cost (s)"], limit=30)
    hr = result.tables.get("headroom")
    if hr is not None and not hr.empty:
        lines += ["## Policy-selection headroom", "",
                  "`headroom_paired_mean` is the defensible figure. It is what a "
                  "better per-dataset selection policy could recover, and it is "
                  "an oracle upper bound, not an achievable target.", ""]
        lines += md_table(hr, [
            "portfolio_id", "reference_short_label", "vbs_best_view_mean_ari",
            "reference_best_view_mean_ari", "headroom_paired_mean",
            "headroom_paired_median", "n_paired"], limit=40)
    tests = result.tables.get("paired_tests")
    if tests is not None and not tests.empty:
        lines += ["## Paired tests", "",
                  "`post_selection_exploratory` marks any test with a "
                  "retrospectively selected representative on either arm: "
                  "optimistic by construction, not confirmatory evidence.", ""]
        lines += md_table(tests, [
            "hypothesis_id", "method_a_short_label", "method_b_short_label",
            "n_paired", "mean_delta", "mean_delta_ci_low", "mean_delta_ci_high",
            "wilcoxon_p", "holm_p", "cohens_dz", "rank_biserial",
            "claim_level"], limit=40)
    if result.notes:
        lines += ["## Notes", ""] + [f"- {n}" for n in result.notes] + [""]
    emit(f"COMPARISON_{num}_REPORT.md", lines,
         f"Headline results for comparison {num}.")

    # ---- suite-specific -------------------------------------------------- #
    if cid == "comparison_02":
        mapping = result.tables.get("candidate_selection_transparency")
        lines = [f"# Comparison {num} -- representative mapping", "",
                 "Every representative in this suite, with the concrete method "
                 "behind it. No table or figure in this suite shows an opaque "
                 "representative id alone.", ""]
        lines += md_table(mapping, [
            "representative_id", "short_label", "selected_concrete_method",
            "candidate_pool_id", "candidate_pool_size", "selection_type",
            "runner_up_short_label", "delta_to_runner_up",
            "selection_stability", "oracle_level"], limit=60)
        lines += [
            "The 15 primary cells use `dynamic_predictK_*` for the MLP/KNN "
            "Dynamic policy. The `DynRK` (Dynamic Real-K) cells take the metric "
            "COUNT from the real utility vector and are reported separately as "
            "oracle-assisted diagnostics.", ""]
        emit(f"COMPARISON_{num}_REPRESENTATIVE_MAPPING.md", lines,
             "Representative id -> concrete method mapping for comparison 2.")

    if cid == "comparison_03":
        pm = result.tables.get("portfolio_manifest_excerpt")
        lines = [f"# Comparison {num} -- portfolio definitions", "",
                 "Exact membership and oracle status of every portfolio in this "
                 "suite. The matched raw/softmax portfolios are one-to-one: each "
                 "raw member has exactly one softmax counterpart with the same "
                 "utility source, metric-count policy and dynamic-K source.", ""]
        lines += md_table(pm, [
            "portfolio_id", "portfolio_size", "fully_operational",
            "oracle_contamination_reason", "n_contaminated_members",
            "candidate_short_labels"], limit=20)
        lines += [
            "", "## Why two groups", "",
            "`raw_only_vbs` / `softmax_only_vbs` / `raw_softmax_union_vbs` "
            "contain real-utility and Dynamic Real-K members, so they answer "
            "\"what is the achievable envelope?\" but are not deployable. "
            "`deployable_raw_vbs` / `deployable_softmax_vbs` / "
            "`deployable_raw_softmax_union_vbs` repeat the analysis with those "
            "members removed and are fully operational.", ""]
        emit(f"COMPARISON_{num}_PORTFOLIO_DEFINITIONS.md", lines,
             "Exact portfolio membership and oracle status for comparison 3.")

    if cid == "comparison_04":
        t = result.tables.get("candidate_selection_transparency")
        lines = [f"# Comparison {num} -- selection transparency", "", ]
        if disclosure:
            lines += ["> " + disclosure.replace("\n", " "), ""]
        lines += [
            "Each row states whether the candidate is one predefined method or "
            "the winner of a retrospective search over a pool, how large that "
            "pool was, how close the decision was, and how stable it is under "
            "resampling.", ""]
        lines += md_table(t, [
            "short_label", "selected_concrete_method", "selection_type",
            "candidate_pool_size", "runner_up_short_label",
            "delta_to_runner_up", "bootstrap_selection_frequency",
            "selection_stability", "disclosure"], limit=30)
        lines += [
            "", "## How to read `selection_stability`", "",
            "It is the share of bootstrap resamples in which the same candidate "
            "was re-selected (the selection step is repeated inside every "
            "resample). A value well below 1 means the point representative is a "
            "fragile choice and the comparison should be read as a benchmark "
            "summary rather than a statement about one configuration.", ""]
        emit(f"COMPARISON_{num}_SELECTION_TRANSPARENCY.md", lines,
             "Candidate-pool asymmetry and selection stability for comparison 4.")

        rt = result.tables.get("runtime_summary")
        lines = [f"# Comparison {num} -- runtime limitations", "",
                 "## What the runtime numbers are", "",
                 "The headline runtime is the **total recorded search/runtime "
                 "across all three views**, because obtaining a Best-View result "
                 "requires running all three. It is reported as mean, median, "
                 "standard deviation and p90.", "",
                 "The runtime of the single view whose ARI was reported "
                 "(`*_selected_best_view_runtime_sec`) is retained only as a "
                 "diagnostic and is never the headline.", "",
                 "## What they are NOT", "",
                 "They are **not complete end-to-end runtime**. Meta-features "
                 "were extracted offline into `features/features_records.csv` "
                 "and reused, and the utility-prediction step is not timed per "
                 "run, so `meta_feature_extraction_sec`, "
                 "`utility_prediction_sec` and `knn_query_sec` are unavailable "
                 "for every method. They are reported as unavailable and are "
                 "never approximated from `runtime_sec`.", "",
                 "For the external methods the adapters record a single "
                 "`runtime_sec` per view with no fit/predict vs objective split, "
                 "so `runtime_components` is unavailable for them too.", "",
                 "## Recorded runtime", ""]
        lines += md_table(rt, [
            "method_short_label", "mean_total_runtime_all_views_sec",
            "median_total_runtime_all_views_sec",
            "std_total_runtime_all_views_sec",
            "p90_total_runtime_all_views_sec",
            "median_selected_best_view_runtime_sec",
            "has_runtime_components", "has_end_to_end_runtime"], limit=30)
        emit(f"COMPARISON_{num}_RUNTIME_LIMITATIONS.md", lines,
             "What the runtime numbers in comparison 4 do and do not measure.")

    if cid == "comparison_05":
        matched = result.tables.get("matched_size_vbs")
        minimum = result.tables.get("minimum_portfolio_size")
        greedy = result.tables.get("greedy_portfolio_membership")
        lines = [f"# Comparison {num} -- matched-size analysis", "",
                 "## The question", "",
                 "> Is the ClustOpt VBS advantage caused only by a larger "
                 "portfolio, or does a small ClustOpt portfolio remain "
                 "competitive at matched size?", "",
                 "ClustOpt portfolios hold up to 34 variants while the external "
                 "ones hold 2, so a raw full-VBS comparison conflates \"better "
                 "candidates\" with \"more candidates\". The table below "
                 "controls for that by comparing at equal size.", ""]
        lines += md_table(matched, [
            "size", "portfolio_short_label", "portfolio_full_size",
            "fully_operational", "vbs_mean", "ci_low", "ci_high", "vbs_max",
            "n_samples", "is_exhaustive"], limit=80)
        lines += ["", "## Minimum useful portfolio", "",
                  "Smallest greedy portfolio reaching a given share of the "
                  "achievable gain (single-best -> full VBS).", ""]
        lines += md_table(minimum, [
            "portfolio_id", "target_share_of_achievable_gain", "min_size",
            "portfolio_size", "single_best_vbs", "full_vbs",
            "achievable_gain"], limit=60)
        lines += ["", "## Greedy portfolio membership", "",
                  "Which variants do the saturating, in the order the greedy "
                  "construction adds them.", ""]
        lines += md_table(greedy, [
            "portfolio_id", "step", "added_short_label", "utility_source",
            "k_policy", "weighting_mode", "vbs_best_view_mean_ari",
            "marginal_gain", "share_of_achievable_gain"], limit=60)
        emit(f"COMPARISON_{num}_MATCHED_SIZE_ANALYSIS.md", lines,
             "Size-controlled VBS comparison and minimum useful portfolio size.")

        pm = result.tables.get("portfolio_manifest_excerpt")
        lines = [f"# Comparison {num} -- portfolio membership", "",
                 "Exact membership of every portfolio in this suite, with its "
                 "operational status. A portfolio name never implies "
                 "deployability its membership does not have.", ""]
        lines += md_table(pm, [
            "portfolio_id", "portfolio_size", "fully_operational",
            "oracle_contamination_reason", "n_contaminated_members",
            "frameworks"], limit=30)
        if pm is not None and not pm.empty:
            lines += ["", "## Full membership", ""]
            for _, r in pm.iterrows():
                lines += [f"### `{r['portfolio_id']}` "
                          f"({int(r['portfolio_size'])} members, "
                          f"{'fully operational' if r['fully_operational'] else 'oracle-contaminated: ' + str(r['oracle_contamination_reason'])})",
                          "", str(r.get("notes", "")), "",
                          ", ".join(f"`{m}`" for m in
                                    str(r["candidate_short_labels"]).split(";")),
                          ""]
        emit(f"COMPARISON_{num}_PORTFOLIO_MEMBERSHIP.md", lines,
             "Exact portfolio membership and operational status for comparison 5.")


def _write_validation(config: Dict[str, Any], result: ComparisonResult,
                      ctx: ComparisonContext, candidates: Sequence[str],
                      portfolio_ids: Sequence[str]) -> None:
    """Per-suite integrity record: coverage, invariants, artifacts, skips."""
    from .reports import _table as md_table
    cid = str(config["comparison_id"])
    out_dir = ctx.run.output_root / result.section / "validation"
    sub = ctx.best_view[ctx.best_view["method_name"].isin(list(candidates))]

    checks: List[Dict[str, Any]] = [
        {"check": "split_id", "value": ctx.run.split_id, "status": "ok"},
        {"check": "datasets covered",
         "value": int(sub["dataset_id"].nunique()) if len(sub) else 0,
         "status": "ok"},
        {"check": "families covered",
         "value": int(sub["family"].nunique()) if len(sub) else 0, "status": "ok"},
        {"check": "subfamilies covered",
         "value": int(sub["subfamily"].nunique()) if len(sub) else 0,
         "status": "ok"},
        {"check": "candidates in comparison", "value": len(candidates),
         "status": "ok"},
        {"check": "portfolios in comparison", "value": len(portfolio_ids),
         "status": "ok"},
        {"check": "rows with >=1 successful view",
         "value": int(sub["has_success"].astype(bool).sum()) if len(sub) else 0,
         "status": "ok"},
        {"check": "rows with NO successful view",
         "value": int((~sub["has_success"].astype(bool)).sum()) if len(sub) else 0,
         "status": "ok"},
    ]
    # VBS invariants, re-checked here so a suite carries its own proof.
    if portfolio_ids:
        vbs_frames = {pid: build for pid, build in
                      ((p, vbs.build_vbs_frame(ctx.best_view, p, verbose=False))
                       for p in portfolio_ids)}
        problems = vbs.validate_vbs(
            vbs_frames, ctx.best_view,
            subset_pairs=[(a, b) for a, b in PORTFOLIO_SUBSET_INVARIANTS
                          if a in vbs_frames and b in vbs_frames])
        checks.append({
            "check": "VBS invariants",
            "value": "passed" if not problems else "; ".join(problems),
            "status": "ok" if not problems else "FAIL"})
        if problems:
            result.notes.append(f"VBS invariant failures: {problems}")

    checks.append({
        "check": "representative references resolved",
        "value": len([m for m in candidates if m in METHOD_BY_NAME]),
        "status": "ok"})
    checks.append({
        "check": "primary runtime column",
        "value": "median_total_runtime_all_views_sec (all three views)",
        "status": "ok"})
    checks.append({
        "check": "capability warnings",
        "value": int(len(result.warnings)), "status": "ok"})

    frame = ctx.manifest.to_frame()
    mine = frame[frame["comparison_id"] == cid]
    checks += [
        {"check": "artifacts declared", "value": int(len(mine)), "status": "ok"},
        {"check": "artifacts created",
         "value": int(mine["created"].sum()) if len(mine) else 0, "status": "ok"},
        {"check": "artifacts skipped (expected capability skips)",
         "value": int((~mine["created"]).sum()) if len(mine) else 0,
         "status": "ok"},
    ]
    checks_frame = pd.DataFrame(checks)
    ctx.manifest.add_table(
        paths.write_csv(out_dir / f"{cid}_validation_checks.csv", checks_frame,
                        index=False),
        comparison_id=cid, section=result.section,
        description=f"Integrity checks for {cid}: coverage, VBS invariants, "
                    f"runtime semantics, artifact accounting.",
        paper_importance="paper_supplement")

    lines = [f"# Validation -- {config.get('title')}", "",
             f"Suite `{cid}`, Split {ctx.run.split_id}.", ""]
    lines += md_table(checks_frame, ["check", "value", "status"], limit=60)
    if not result.warnings.empty:
        lines += ["## Expected capability skips", "",
                  "Each row names the exact method and the exact missing field. "
                  "No value was substituted or null-filled.", ""]
        grouped = (result.warnings.groupby(["analysis", "missing_capabilities"])
                   ["method_short_label"]
                   .apply(lambda s: ", ".join(sorted(set(s)))).reset_index())
        lines += md_table(grouped, ["analysis", "missing_capabilities",
                                    "method_short_label"], limit=40)
    if result.notes:
        lines += ["## Suite notes", ""] + [f"- {n}" for n in result.notes] + [""]
    path = out_dir / f"COMPARISON_{cid.split('_')[-1]}_VALIDATION.md"
    ctx.manifest.add_report(
        paths.write_text(path, "\n".join(lines)) or path,
        comparison_id=cid, section=result.section,
        description=f"Integrity review for {cid}.",
        paper_importance="paper_supplement")


def _selection_excerpt(selection_manifest: pd.DataFrame,
                       labels: Dict[str, str]) -> pd.DataFrame:
    """The rows of the frozen selection manifest this suite actually relies on."""
    if selection_manifest is None or selection_manifest.empty or not labels:
        return pd.DataFrame()
    wanted = set(labels.values())
    sub = selection_manifest[
        selection_manifest["representative_id"].astype(str).isin(wanted)]
    return sub.reset_index(drop=True)


def _transparency_table(candidates: Sequence[str], labels: Dict[str, str],
                        ctx: ComparisonContext) -> pd.DataFrame:
    """Per-candidate selection transparency (§11.2 of the amendment).

    Every row states whether the candidate is one predefined method or the winner
    of a retrospective search, how large the pool was, and how stable that choice
    is under resampling. A reader must never have to guess which is which.
    """
    rows: List[Dict[str, Any]] = []
    manifest = ctx.selection_manifest
    for method in candidates:
        rep_id = labels.get(method, "")
        rec: Dict[str, Any] = {}
        if rep_id and not manifest.empty:
            match = manifest[manifest["representative_id"].astype(str) == rep_id]
            if not match.empty:
                rec = match.iloc[0].to_dict()
        spec = METHOD_BY_NAME.get(method)
        is_rep = bool(rec)
        rows.append({
            "display_id": rep_id or method,
            "representative_id": rep_id,
            "selected_concrete_method": method,
            "short_label": short_label(method),
            "display_name": spec.display_name if spec else method,
            "selection_type": ("retrospective_global" if is_rep
                               else "fixed_method"),
            "candidate_pool_id": rec.get("candidate_pool_id", ""),
            "candidate_pool_size": rec.get("candidate_pool_size", 1),
            "candidate_methods": rec.get("candidate_methods", method),
            "selection_split": rec.get("selection_split", 1),
            "selection_metric": rec.get("selection_metric",
                                        "n/a (predefined method)"),
            "runner_up": rec.get("runner_up_method", ""),
            "runner_up_short_label": rec.get("runner_up_short_label", ""),
            "delta_to_runner_up": rec.get("delta_to_runner_up"),
            "bootstrap_selection_frequency": rec.get(
                "bootstrap_selection_frequency"),
            "selection_stability": rec.get("selection_stability"),
            "n_tied_on_primary": rec.get("n_tied_on_primary", 1),
            "tie_break_reason": rec.get("tie_break_reason", ""),
            "oracle_level": spec.oracle_level if spec else "",
            "same_search_space": spec.same_search_space if spec else None,
            "search_budget": spec.search_budget if spec else None,
            "is_independently_preselected": False,
            "paper_claim_level": ("retrospective_summary" if is_rep
                                  else "fixed_observed"),
            "disclosure": (
                "retrospective global representative selected on Split 1 from "
                f"{rec.get('candidate_pool_size', '?')} evaluated variants"
                if is_rep else
                "one predefined method; no selection freedom"),
        })
    return pd.DataFrame(rows)


def _matched_source_deltas(best_view: pd.DataFrame, config: Dict[str, Any]
                           ) -> pd.DataFrame:
    """MLP-vs-KNN (and real-vs-learned) deltas at every matched policy.

    Matched means: same metric-count policy and same weighting mode, so the only
    thing that differs between the arms is the utility source.
    """
    from .method_registry import VARIANT_FAMILIES

    rows: List[Dict[str, Any]] = []
    by_key: Dict[Tuple[str, str], Any] = {
        (f.utility_source, f.policy_id): f for f in VARIANT_FAMILIES}
    policies = sorted({f.policy_id for f in VARIANT_FAMILIES})
    for policy in policies:
        mlp = by_key.get(("mlp", policy))
        knn = by_key.get(("knn", policy))
        real = by_key.get(("real", policy)) or by_key.get(("real", "dynamic"))
        for weighting, attr in (("raw", "raw_method"),
                                ("softmax", "softmax_method")):
            if mlp is not None and knn is not None:
                rows.append(gaps.paired_gap_row(
                    best_view, gap_type="mlp_vs_knn",
                    reference=getattr(mlp, attr), comparison=getattr(knn, attr),
                    group={"policy_id": policy, "weighting": weighting,
                           "delta_kind": "mlp_vs_knn"},
                    notes="both arms are deployable-source: the one fully "
                          "operational comparison in this suite"))
            if real is not None and mlp is not None:
                rows.append(gaps.paired_gap_row(
                    best_view, gap_type="utility_prediction_gap",
                    reference=getattr(real, attr), comparison=getattr(mlp, attr),
                    group={"policy_id": policy, "weighting": weighting,
                           "delta_kind": "real_vs_mlp"},
                    notes="reference is a utility ORACLE"))
            if real is not None and knn is not None:
                rows.append(gaps.paired_gap_row(
                    best_view, gap_type="utility_prediction_gap",
                    reference=getattr(real, attr), comparison=getattr(knn, attr),
                    group={"policy_id": policy, "weighting": weighting,
                           "delta_kind": "real_vs_knn"},
                    notes="reference is a utility ORACLE"))
    return pd.DataFrame(rows)


def _portfolio_paired_deltas(vbs_frames: Dict[str, pd.DataFrame],
                             config: Dict[str, Any]) -> pd.DataFrame:
    """Per-dataset paired deltas between the declared portfolio pairs."""
    rows: List[pd.DataFrame] = []
    for rq_id, body in (config.get("research_questions") or {}).items():
        if not isinstance(body, dict):
            continue
        if body.get("kind") not in ("paired_portfolio_delta", "win_tie_loss"):
            continue
        a, b = str(body.get("a", "")), str(body.get("b", ""))
        fa, fb = vbs_frames.get(a), vbs_frames.get(b)
        if fa is None or fb is None or fa.empty or fb.empty:
            continue
        merged = fa[["dataset_id", "family", "subfamily",
                     "winning_best_view_ari", "winning_method_short_label"]].merge(
            fb[["dataset_id", "winning_best_view_ari",
                "winning_method_short_label"]],
            on="dataset_id", suffixes=("_a", "_b"))
        merged["delta"] = (pd.to_numeric(merged["winning_best_view_ari_a"],
                                         errors="coerce")
                           - pd.to_numeric(merged["winning_best_view_ari_b"],
                                           errors="coerce"))
        merged["research_question"] = rq_id
        merged["portfolio_a"] = a
        merged["portfolio_b"] = b
        merged["pair_label"] = f"{a} - {b}"
        merged["a_wins"] = merged["delta"] > 1e-12
        merged["b_wins"] = merged["delta"] < -1e-12
        merged["tied"] = merged["delta"].abs() <= 1e-12
        merged["selection_type"] = "dataset_vbs"
        merged["paper_claim_level"] = "upper_bound"
        rows.append(merged)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _union_headroom(vbs_frames: Dict[str, pd.DataFrame], config: Dict[str, Any]
                    ) -> pd.DataFrame:
    """Union-portfolio headroom over the BETTER of its two components."""
    rows: List[Dict[str, Any]] = []
    for rq_id, body in (config.get("research_questions") or {}).items():
        if not isinstance(body, dict) or body.get("kind") != "union_headroom":
            continue
        union_id = str(body.get("union", ""))
        components = [str(c) for c in (body.get("components") or [])]
        union = vbs_frames.get(union_id)
        if union is None or union.empty:
            continue
        union_mean = float(pd.to_numeric(union["winning_best_view_ari"],
                                         errors="coerce").mean())
        comp_means = {}
        for cid in components:
            frame = vbs_frames.get(cid)
            if frame is not None and not frame.empty:
                comp_means[cid] = float(pd.to_numeric(
                    frame["winning_best_view_ari"], errors="coerce").mean())
        if not comp_means:
            continue
        best_component = max(comp_means, key=comp_means.get)
        paired = union[["dataset_id", "winning_best_view_ari"]].merge(
            vbs_frames[best_component][["dataset_id", "winning_best_view_ari"]],
            on="dataset_id", suffixes=("_union", "_comp"))
        delta = (pd.to_numeric(paired["winning_best_view_ari_union"], errors="coerce")
                 - pd.to_numeric(paired["winning_best_view_ari_comp"],
                                 errors="coerce")).dropna()
        rows.append({
            "research_question": rq_id,
            "union_portfolio": union_id,
            "component_portfolios": ";".join(components),
            "baseline_rule": str(body.get("baseline", "better_of_components")),
            "best_component": best_component,
            "union_best_view_mean_ari": union_mean,
            "best_component_best_view_mean_ari": comp_means[best_component],
            "headroom_unpaired": union_mean - comp_means[best_component],
            "n_paired": int(len(delta)),
            "headroom_paired_mean": float(delta.mean()) if len(delta) else float("nan"),
            "headroom_paired_median": (float(delta.median()) if len(delta)
                                       else float("nan")),
            "share_datasets_union_strictly_better": (
                float((delta > 1e-12).mean()) if len(delta) else float("nan")),
            "selection_type": "dataset_vbs",
            "paper_claim_level": "upper_bound",
            "notes": "headroom is measured over the BETTER of the two component "
                     "portfolios, not over their mean",
        })
    return pd.DataFrame(rows)


def _reference_methods(config: Dict[str, Any], index: RepresentativeIndex
                       ) -> Dict[str, str]:
    block = (config.get("references") or {}).get("best_observed_single_variants") or {}
    out: Dict[str, str] = {}
    for label, rep_id in block.items():
        rep = str(rep_id)
        if rep in index:
            out[str(label)] = index.method(rep)
        elif rep in METHOD_BY_NAME:
            out[str(label)] = rep
    for name in (config.get("references") or {}).get("external_leaders", []) or []:
        if str(name) in METHOD_BY_NAME:
            out[f"external_{short_label(str(name))}"] = str(name)
    return out


def _size_sensitivity_config(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for body in (config.get("research_questions") or {}).values():
        if isinstance(body, dict) and body.get("kind") == "portfolio_size_sensitivity":
            rs = body.get("random_subset") or {}
            return {
                "sizes": tuple(int(s) for s in
                               (body.get("sizes") or (1, 2, 3, 5, 10))),
                "n_samples": int(rs.get("n_samples", 200)),
                "confidence": float(rs.get("confidence", 0.95)),
                "seed": int(rs.get("seed", 20260725)),
                "gain_shares": tuple(float(x) for x in
                                     (body.get("gain_shares") or (0.90, 0.95))),
            }
    return None


def _vbs_runtime_diagnostics(vbs_summary: Optional[pd.DataFrame]) -> pd.DataFrame:
    if vbs_summary is None or vbs_summary.empty:
        return pd.DataFrame()
    keep = ["portfolio_id", "portfolio_short_label", "portfolio_size",
            "best_view_mean_ari", "fully_operational",
            "oracle_contamination_reason",
            "selected_member_total_all_views_runtime_mean_sec",
            "selected_member_total_all_views_runtime_median_sec",
            "portfolio_total_execution_runtime_mean_sec",
            "portfolio_total_execution_runtime_median_sec",
            "portfolio_total_execution_runtime_total_sec"]
    out = vbs_summary[[c for c in keep if c in vbs_summary.columns]].copy()
    out["operational_runtime_claim_allowed"] = False
    out["excluded_from_operational_pareto"] = True
    out["primary_vbs_cost_column"] = "portfolio_total_execution_runtime_median_sec"
    out["runtime_note"] = (
        "A VBS is not a single executable method: it selects the best member "
        "only after every member has been evaluated. "
        "portfolio_total_execution_runtime is the cost of evaluating the FULL "
        "portfolio and is the only quantity that may be described as the cost of "
        "the VBS; selected_member_* runtimes describe the method the oracle "
        "picked and are diagnostics. No VBS point appears in an operational "
        "accuracy-runtime Pareto plot.")
    return out


_TABLE_KEY_TO_FILE: Dict[str, str] = {
    "global_summary": "{cid}_global_summary.csv",
    "family_summary": "{cid}_family_summary.csv",
    "subfamily_summary": "{cid}_subfamily_summary.csv",
    "difficulty_summary": "{cid}_difficulty_summary.csv",
    "cluster_count_summary": "{cid}_cluster_count_summary.csv",
    "pairwise_deltas": "{cid}_pairwise_deltas.csv",
    "winrates": "{cid}_winrates.csv",
    "gap_summary": "{cid}_gap_summary.csv",
    "gap_definitions": "{cid}_gap_definitions.csv",
    "metric_usage": "{cid}_metric_usage.csv",
    "metric_category_usage": "{cid}_metric_category_usage.csv",
    "new_metric_usage": "{cid}_new_metric_usage.csv",
    "metric_entropy": "{cid}_metric_entropy.csv",
    "runtime_summary": "{cid}_runtime_summary.csv",
    "pareto_frontier": "{cid}_pareto_frontier.csv",
    "candidate_map": "{cid}_candidate_map.csv",
    "research_questions": "{cid}_research_questions.csv",
    "paired_tests": "{cid}_paired_tests.csv",
    "vbs_summary": "{cid}_vbs_summary.csv",
    "family_vbs_summary": "{cid}_family_summary_vbs.csv",
    "dataset_winners": "{cid}_dataset_winners.csv",
    "winner_composition": "{cid}_winner_composition.csv",
    "winner_composition_by_policy": "{cid}_winner_composition_by_policy.csv",
    "winner_composition_by_source": "{cid}_winner_composition_by_source.csv",
    "winner_composition_by_weighting": "{cid}_winner_composition_by_weighting.csv",
    "unique_coverage": "{cid}_winner_coverage.csv",
    "headroom": "{cid}_headroom.csv",
    "greedy_portfolio": "{cid}_greedy_portfolio.csv",
    "random_subset_summary": "{cid}_random_subset_summary.csv",
    "portfolio_size_curve": "{cid}_portfolio_size_curve.csv",
    "min_size_for_90pct_gain": "{cid}_min_portfolio_size.csv",
    "runtime_diagnostics": "{cid}_runtime_diagnostics.csv",
    "matched_source_deltas": "{cid}_matched_source_deltas.csv",
    "utility_prediction_gap": "{cid}_utility_prediction_gap.csv",
    "metric_count_gap": "{cid}_metric_count_gap.csv",
    "dynamic_predict_vs_realK": "{cid}_dynamic_predict_vs_realK.csv",
    "gap_decomposition": "{cid}_gap_decomposition.csv",
    "raw_vs_softmax_deltas": "{cid}_raw_vs_softmax_deltas.csv",
    "union_headroom": "{cid}_union_headroom.csv",
    "new_metric_usage": "{cid}_new_metric_usage.csv",
    "metric_entropy": "{cid}_metric_entropy.csv",
    "matched_size_vbs": "{cid}_matched_size_vbs.csv",
    "greedy_portfolio_membership": "{cid}_greedy_portfolio_membership.csv",
    "minimum_portfolio_size": "{cid}_minimum_portfolio_size.csv",
    "portfolio_manifest_excerpt": "{cid}_portfolio_manifest_excerpt.csv",
    "selection_manifest_excerpt": "{cid}_selection_manifest_excerpt.csv",
    "representative_stability": "{cid}_representative_stability.csv",
    "candidate_selection_transparency": "{cid}_candidate_selection_transparency.csv",
    "selection_aware_bootstrap": "{cid}_selection_aware_bootstrap.csv",
    "win_tie_loss": "{cid}_win_tie_loss.csv",
}

_IMPORTANCE_BY_KEY: Dict[str, str] = {
    "global_summary": "paper_main", "family_summary": "paper_main",
    "pairwise_deltas": "paper_main", "vbs_summary": "paper_main",
    "headroom": "paper_main", "candidate_map": "paper_main",
    "research_questions": "paper_main", "paired_tests": "paper_main",
    "pareto_frontier": "paper_main", "winner_composition": "paper_main",
    "unique_coverage": "paper_main", "greedy_portfolio": "paper_main",
    "random_subset_summary": "paper_main", "portfolio_size_curve": "paper_main",
    "metric_category_usage": "paper_main",
    "matched_source_deltas": "paper_main", "gap_decomposition": "paper_main",
    "dynamic_predict_vs_realK": "paper_main",
    "raw_vs_softmax_deltas": "paper_main", "union_headroom": "paper_main",
    "matched_size_vbs": "paper_main",
    "greedy_portfolio_membership": "paper_main",
    "minimum_portfolio_size": "paper_main",
    "candidate_selection_transparency": "paper_main",
    "selection_aware_bootstrap": "paper_main",
    "representative_stability": "paper_main", "win_tie_loss": "paper_main",
    "subfamily_summary": "appendix", "dataset_winners": "appendix",
    "gap_definitions": "appendix", "selection_manifest_excerpt": "appendix",
    "portfolio_manifest_excerpt": "appendix",
}


def _write_large_frames(config: Dict[str, Any], result: ComparisonResult,
                        ctx: ComparisonContext) -> None:
    """Write bulk reproducible data as Parquet, beside the CSV tables."""
    cid = str(config["comparison_id"])
    out_dir = ctx.run.output_root / result.section / "tables"
    for key, frame in result.large_frames.items():
        if frame is None or frame.empty:
            continue
        path = out_dir / f"{cid}_{key}.parquet"
        written = paths.write_parquet(path, frame)
        ctx.manifest.add_frame(
            written, comparison_id=cid, section=result.section,
            description=(f"{key.replace('_', ' ')} for {cid} "
                         f"({len(frame):,} rows; reproducible bulk data)"),
            paper_importance="appendix", appendix_only=True,
            notes="regenerated deterministically from the fixed seed in the "
                  "comparison config; safe to gitignore")


def _write_tables(config: Dict[str, Any], result: ComparisonResult,
                  ctx: ComparisonContext) -> None:
    cid = str(config["comparison_id"])
    tables_dir = ctx.run.output_root / result.section / "tables"
    for key, frame in result.tables.items():
        template = _TABLE_KEY_TO_FILE.get(key)
        if template is None:
            template = f"{{cid}}_{key}.csv"
        path = tables_dir / template.format(cid=cid)
        if frame is None or frame.empty:
            ctx.manifest.add_table(
                None, comparison_id=cid, section=result.section,
                description=f"{key.replace('_', ' ')} for {cid}",
                paper_importance=_IMPORTANCE_BY_KEY.get(key, "paper_supplement"),
                created=False,
                notes="not created: the analysis produced no rows (see the "
                      "capability warnings and the suite notes)")
            continue
        written = paths.write_csv(path, frame, index=False)
        ctx.manifest.add_table(
            written, comparison_id=cid, section=result.section,
            description=f"{key.replace('_', ' ')} for {cid}",
            paper_importance=_IMPORTANCE_BY_KEY.get(key, "paper_supplement"),
            paper_candidate=_IMPORTANCE_BY_KEY.get(key) == "paper_main",
            appendix_only=_IMPORTANCE_BY_KEY.get(key) == "appendix",
            supports_external_methods=key not in (
                "metric_usage", "metric_category_usage", "new_metric_usage",
                "metric_entropy"))
    if not result.warnings.empty:
        path = tables_dir / f"{cid}_capability_warnings.csv"
        ctx.manifest.add_table(
            paths.write_csv(path, result.warnings, index=False),
            comparison_id=cid, section=result.section,
            description=f"Sub-analyses skipped in {cid}, with the exact missing "
                        f"capability per method.",
            paper_importance="paper_supplement")
    if result.notes:
        path = ctx.run.output_root / result.section / "SUITE_NOTES.md"
        ctx.manifest.add_report(
            paths.write_text(path, "\n".join(
                [f"# Suite notes -- {cid}", ""] + [f"- {n}" for n in result.notes]
            )) or path,
            comparison_id=cid, section=result.section,
            description="Runtime notes: skipped analyses, excluded candidates, "
                        "dropped portfolio-size points.",
            paper_importance="paper_supplement")


def _write_workbook(config: Dict[str, Any], result: ComparisonResult,
                    ctx: ComparisonContext) -> None:
    cid = str(config["comparison_id"])
    plan = [(key.replace("_", " ").title()[:31], key,
             f"{key.replace('_', ' ')} for {cid}")
            for key, frame in result.tables.items()
            if frame is not None and not frame.empty]
    spec = comparison_workbook(
        cid, str(config["workbook"]), str(config.get("title", cid)),
        str(config.get("required_disclosure", "")
            or config.get("causal_caveat", "")), result.tables, plan)
    portfolio_manifest = result.tables.get("portfolio_manifest_excerpt")
    path = write_workbook(
        spec, ctx.run.output_root / result.section / "formatted_excel",
        selection_manifest=ctx.selection_manifest,
        portfolio_manifest=(portfolio_manifest
                            if isinstance(portfolio_manifest, pd.DataFrame)
                            and not portfolio_manifest.empty else None),
        capability_warnings=result.warnings if not result.warnings.empty else None,
        dry_run=ctx.run.dry_run)
    ctx.manifest.add_workbook(
        path, comparison_id=cid, section=result.section,
        description=f"Formatted workbook for {config.get('title')}.",
        paper_importance="paper_supplement")
