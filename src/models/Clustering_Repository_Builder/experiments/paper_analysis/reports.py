"""Markdown report writers.

Every report keeps five things visually separate, because collapsing them is how
a benchmark summary turns into an overclaim:

1. **facts computed from data** -- numbers straight out of the frames;
2. **retrospective selections** -- which representative was chosen and from what;
3. **oracle / VBS results** -- upper bounds, explicitly labelled;
4. **interpretation suggestions** -- clearly marked as suggestions;
5. **limitations and unsupported analyses**.

The wording rules of §34 are applied here (and only here, so they cannot drift):
"best observed single variant", "retrospective global representative", "Virtual
Best Solver", "oracle upper bound", "policy-selection headroom" -- never "best
deployable model", "independently selected winner" or "X outperformed Y" when the
claim rests on retrospective selection.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from .capability import CAPABILITIES, CAPABILITY_DESCRIPTIONS
from .method_registry import (
    ALL_METRICS_COUNT, CANDIDATE_POOLS, CLASSIC_CVI_BASELINE_METRICS,
    EXPECTED_COUNTS, METHODS, PRIMARY_VARIANT_FAMILIES, VARIANT_FAMILIES,
)
from .metric_registry import METRIC_AUDIT, METRIC_CATEGORIES
from .portfolio_registry import PORTFOLIOS

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _table(frame: Optional[pd.DataFrame], columns: Sequence[str],
           headers: Optional[Sequence[str]] = None, *, limit: int = 60,
           digits: int = 4) -> List[str]:
    """Render a frame as a markdown table, skipping absent columns."""
    if frame is None or frame.empty:
        return ["_(no rows)_", ""]
    cols = [c for c in columns if c in frame.columns]
    if not cols:
        return ["_(none of the expected columns are present)_", ""]
    heads = list(headers) if headers else [c.replace("_", " ") for c in cols]
    heads = heads[: len(cols)]
    lines = ["| " + " | ".join(heads) + " |",
             "|" + "---|" * len(cols)]
    for _, r in frame.head(limit).iterrows():
        lines.append("| " + " | ".join(_fmt(r[c], digits) for c in cols) + " |")
    if len(frame) > limit:
        lines.append(f"| _... {len(frame) - limit} more rows in the CSV_ |"
                     + " |" * (len(cols) - 1))
    lines.append("")
    return lines


def _section(title: str, level: int = 2) -> List[str]:
    return ["", "#" * level + " " + title, ""]


# --------------------------------------------------------------------------- #
# 00_protocol_and_definitions
# --------------------------------------------------------------------------- #
def terminology_md() -> str:
    return "\n".join([
        "# Terminology",
        "",
        "This file is the terminology contract for the whole analysis. Every "
        "table, figure, workbook and report below uses these terms with exactly "
        "these meanings.",
        "",
        "## The constraint that shapes everything",
        "",
        "All downstream clustering experiments were executed on **`split_id = 1` "
        "only**. There are no downstream ClustOpt clustering experiments on "
        "splits 2-16, so it is impossible to select a policy on independent data "
        "and then evaluate it on Split 1.",
        "",
        "Consequently, a ClustOpt representative chosen from Split-1 results must "
        "**never** be described as any of:",
        "",
        "- ~~preselected deployable winner~~",
        "- ~~independently selected deployment policy~~",
        "- ~~held-out selected configuration~~",
        "",
        "## The four things that must never be confused",
        "",
        "### 1. Fixed evaluated variant",
        "",
        "A concrete method run exactly as defined -- e.g. `regressor_top5_raw`, "
        "`knn_top3_softmax_t05`, `real_top_dynamic_raw`, "
        "`AutoClust_Extended_InDomain_same_search_space`. The method itself is "
        "operational as a fixed algorithmic configuration, even when it was not "
        "independently selected before Split 1 was examined.",
        "",
        "`selection_type = fixed_variant`, `paper_claim_level = fixed_observed`.",
        "",
        "### 2. Retrospective global representative",
        "",
        "One fixed evaluated variant, selected from a predefined candidate pool "
        "because it achieved the highest global Mean Best-View ARI on Split 1.",
        "",
        "Preferred wording: **best observed single variant on Split 1**, "
        "**retrospective global representative**, **best observed variant within "
        "the evaluated candidate pool**.",
        "",
        "> Among the evaluated MLP-based ClustOpt variants, method X achieved the ",
        "> highest Mean Best-View ARI on Split 1 and is therefore used as the ",
        "> retrospective global representative in subsequent summaries.",
        "",
        "`selection_type = retrospective_global`, "
        "`paper_claim_level = retrospective_summary`, "
        "`is_independently_preselected = False`.",
        "",
        "### 3. Retrospective family representative",
        "",
        "One fixed evaluated variant selected separately inside each of the "
        "structural families, using that family's Mean Best-View ARI on Split 1. "
        "This is **not** a global operational policy -- it is a family-level "
        "retrospective **policy oracle** and is reported as headroom only.",
        "",
        "`selection_type = retrospective_family`, "
        "`paper_claim_level = diagnostic_oracle`.",
        "",
        "### 4. Dataset-level VBS",
        "",
        "A Virtual Best Solver that selects a different candidate method for each "
        "dataset using that dataset's true external ARI. A non-deployable oracle "
        "upper bound.",
        "",
        "> The VBS selects the highest-ARI candidate separately for each dataset ",
        "> and therefore represents a non-deployable upper bound on portfolio ",
        "> performance.",
        "",
        "`selection_type = dataset_vbs`, `paper_claim_level = upper_bound`, "
        "`uses_variant_oracle = True`.",
        "",
        "## Selection metadata carried by every row",
        "",
        "| field | values |",
        "|---|---|",
        "| `selection_type` | `fixed_variant`, `retrospective_global`, "
        "`retrospective_family`, `dataset_vbs` |",
        "| `selection_split` | `1` |",
        "| `selection_metric` | `best_view_mean_ari` |",
        "| `selection_scope` | `global`, `family`, `dataset` |",
        "| `uses_view_oracle` | always `True` (Best-View is a view oracle) |",
        "| `uses_variant_oracle` | `True` for retrospective representatives and VBS |",
        "| `uses_real_utility` | `True` for the `real_*` methods |",
        "| `uses_real_dynamic_k` | `True` for `*_dynamic_realK_*` and `real_top_dynamic_*` |",
        "| `is_operational_fixed_method` | `True` for every concrete variant |",
        "| `is_independently_preselected` | always `False` |",
        "| `paper_claim_level` | `fixed_observed`, `retrospective_summary`, "
        "`diagnostic_oracle`, `upper_bound` |",
        "",
        "## Wording rules for generated prose",
        "",
        "**Use:** best observed single variant; retrospective global "
        "representative; retrospective family representative; Virtual Best "
        "Solver; oracle upper bound; policy-selection headroom.",
        "",
        "**Avoid** (whenever the claim relies on retrospective selection or a "
        "VBS): best deployable model; independently selected winner; held-out "
        "selected configuration; \"ClustOpt outperformed AutoClust\".",
        "",
        "Standard disclosure for any ClustOpt-vs-external table:",
        "",
        "> The ClustOpt representative was retrospectively selected from N ",
        "> evaluated variants, while each external row represents one predefined ",
        "> method. Results should therefore be interpreted as a benchmark summary ",
        "> rather than an independently selected deployment comparison.",
        "",
    ])


def oracle_levels_md() -> str:
    return "\n".join([
        "# Oracle levels",
        "",
        "Four *different* oracles appear in this analysis. They are kept "
        "separately named because collapsing them into one \"uses ground truth\" "
        "label would make several results uninterpretable.",
        "",
        "| # | oracle | what it uses ground truth for | applies to |",
        "|---|---|---|---|",
        "| 1 | **view oracle** (Best-View) | picks which of the three views to "
        "report, by external ARI | *every* method, uniformly |",
        "| 2 | **utility oracle** | the objective's metric weights come from the "
        "REAL utility vector | the 10 `real_*` methods |",
        "| 3 | **dynamic-K oracle assistance** | metric ranking is learned, but "
        "the metric COUNT comes from the real utility vector | the 4 "
        "`*_dynamic_realK_*` MLP/KNN methods |",
        "| 4 | **policy-selection oracle** (VBS) | picks which *method* to use, "
        "per dataset, by external ARI | portfolios, not methods |",
        "",
        "## Why they must stay separate",
        "",
        "- Oracle 1 is a *constant* across the comparison: it neither favours nor "
        "penalises any method, but it does mean the primary metric is an "
        "upper-ish bound on single-view deployment, and that the honest cost of a "
        "Best-View result is the runtime of **all three** views.",
        "- Oracle 2 marks the `real_*` family as an *upper bound on what utility "
        "guidance could achieve*, not as a deployable method. Real-vs-learned "
        "comparisons measure prediction loss.",
        "- Oracle 3 is narrower than oracle 2: only the metric count leaks. It "
        "exists to answer \"how much of the Dynamic loss is metric-count "
        "prediction?\" and is excluded from the primary 15-cell meta-learning "
        "grid for exactly that reason.",
        "- Oracle 4 is a property of a *portfolio*. A VBS number is never a "
        "method's performance and never has an operational runtime.",
        "",
        "## Combination table",
        "",
        "| row type | view oracle | utility oracle | dyn-K assist | policy oracle |",
        "|---|---|---|---|---|",
        "| fixed MLP/KNN variant | yes | no | no | no |",
        "| fixed `real_*` variant | yes | yes | (implied) | no |",
        "| `*_dynamic_realK_*` variant | yes | no | yes | no |",
        "| external same-search-space method | yes | no | no | no |",
        "| retrospective global representative | yes | inherited | inherited | "
        "no (selected on aggregate, not per dataset) |",
        "| retrospective family representative | yes | inherited | inherited | "
        "partial (per family) |",
        "| dataset-level VBS | yes | inherited | inherited | yes |",
        "",
    ])


def runtime_definitions_md(*, end_to_end_available: bool = False) -> str:
    lines = [
        "# Runtime definitions",
        "",
        "Runtime fields are never mixed. Each has one meaning.",
        "",
        "## `best_view_runtime_sec` -- diagnostic only",
        "",
        "Runtime of the single view whose ARI was selected. It is **not** the cost "
        "of obtaining a Best-View result, because you cannot know which view wins "
        "without running all three.",
        "",
        "## `total_runtime_all_views_sec` -- the primary cost",
        "",
        "`x_only + y_only + xy_2d` for one `dataset x method`. This is the cost "
        "associated with the primary metric and is what the runtime box plots and "
        "the accuracy-runtime Pareto frontier use.",
        "",
        "Every operational summary therefore reports "
        "`mean_total_runtime_all_views_sec`, "
        "`median_total_runtime_all_views_sec`, "
        "`std_total_runtime_all_views_sec` and "
        "`p90_total_runtime_all_views_sec` as the headline runtime block, with "
        "`mean_selected_best_view_runtime_sec` / "
        "`median_selected_best_view_runtime_sec` retained only as diagnostics.",
        "",
        "Wording: this is the **total recorded search/runtime across all three "
        "views**. It must not be called complete end-to-end runtime -- see below.",
        "",
        "## `portfolio_total_execution_runtime_sec` vs "
        "`selected_member_*_runtime_sec` -- VBS",
        "",
        "A VBS is **not a single executable method**: it selects the best member "
        "only after every member has been evaluated. Two numbers are therefore "
        "always reported together:",
        "",
        "- `selected_member_best_view_runtime_sec` and "
        "`selected_member_total_all_views_runtime_sec`: describe the *method the "
        "VBS selected* on that dataset. **Diagnostic only.**",
        "- `portfolio_total_execution_runtime_sec`: the cost of running **every** "
        "member across all required views,",
        "",
        "      T_portfolio(d) = sum over m in P of T_all_views(d, m)",
        "",
        "  This is the only quantity that may be described as the cost of the "
        "VBS, and it is an oracle-cost diagnostic rather than an operational "
        "runtime.",
        "",
        "By construction `portfolio_total_execution_runtime_sec >= "
        "selected_member_total_all_views_runtime_sec` on every dataset with "
        "finite runtimes; the VBS validator asserts it. A VBS point never appears "
        "in an operational accuracy-runtime Pareto plot as though it were one "
        "executable method.",
        "",
        "## Runtime components",
        "",
        "| field | meaning | availability |",
        "|---|---|---|",
        "| `fit_predict_sec` | clustering fit/predict time summed over trials | "
        "ClustOpt methods |",
        "| `cvi_eval_sec` | objective (CVI) evaluation time summed over trials | "
        "ClustOpt methods |",
        "| `cvi_normalize_sec`, `cvi_aggregate_sec` | objective normalisation / "
        "aggregation | ClustOpt methods |",
        "| `n_cvi_evaluations` | number of objective evaluations | ClustOpt methods |",
        "",
        "## End-to-end runtime -- NOT AVAILABLE",
        "",
    ]
    if end_to_end_available:
        lines += ["End-to-end components are available; see the capability matrix.", ""]
    else:
        lines += [
            "`meta_feature_extraction_sec`, `utility_prediction_sec`, "
            "`knn_query_sec` and `search_runtime_sec` are **not recorded per run** "
            "by any adapter in this experiment set. Meta-features were extracted "
            "offline into `features/features_records.csv` and reused, and the "
            "utility-prediction step is not separately timed.",
            "",
            "These fields are therefore reported as unavailable "
            "(`has_end_to_end_runtime = False` for every method) and are **never** "
            "approximated from `runtime_sec`. Any analysis requiring them is "
            "skipped with a recorded capability warning.",
            "",
        ]
    return "\n".join(lines)


def metric_definitions_md() -> str:
    return "\n".join([
        "# Metric definitions",
        "",
        "## Primary result block",
        "",
        "| column | definition |",
        "|---|---|",
        "| `best_view_mean_ari` | **PRIMARY METRIC.** Mean over datasets of each "
        "dataset's Best-View ARI. |",
        "| `best_view_median_ari` | Median of the same per-dataset values. |",
        "| `best_view_std_ari` | Sample standard deviation of the same values. |",
        "| `best_view_q25_ari`, `best_view_q75_ari` | Quartiles. |",
        "| `best_view_min_ari`, `best_view_max_ari` | Extremes. |",
        "| `best_view_k_accuracy` | Share of datasets where the selected cluster "
        "count equals the true count. |",
        "| `best_view_mean_k_abs_error` | Mean absolute cluster-count error. |",
        "| `best_view_overcluster_rate` / `best_view_undercluster_rate` | Share "
        "with signed K error > 0 / < 0. |",
        "| `total_runtime_all_views_*_sec` | **PRIMARY COST.** See "
        "`runtime_definitions.md`. |",
        "",
        "Column names are deliberately explicit: there is no bare `mean_ari` on a "
        "frame that has already been collapsed to best views, so a reader can "
        "never mistake a mean-over-best-views for a mean-over-all-views.",
        "",
        "## Supplementary all-view diagnostics",
        "",
        "| column | definition |",
        "|---|---|",
        "| `all_views_mean_ari` | Mean over datasets of the per-dataset mean "
        "across the three views. |",
        "| `all_views_median_ari` | Median counterpart. |",
        "| `view_gap` | `best_view_ari - view_ari_mean`: dependence on picking "
        "the right representation. |",
        "",
        "## Audit fields",
        "",
        "`n_datasets`, `n_success`, `success_rate`, `failure_rate` accompany every "
        "summary row. Rows for datasets where a method had **no** successful view "
        "are retained in the best-view frame (with null ARI), so success-rate "
        "accounting stays honest.",
        "",
        "## Metric inventory",
        "",
        f"The utility inventory contains **{ALL_METRICS_COUNT}** metrics in "
        f"{len(METRIC_CATEGORIES)} categories "
        f"({', '.join('`' + c + '`' for c in METRIC_CATEGORIES)}). The "
        "`uniform_classic_cvi` baseline objective uses "
        f"{len(CLASSIC_CVI_BASELINE_METRICS)}: "
        + ", ".join(f"`{m}`" for m in CLASSIC_CVI_BASELINE_METRICS) + ".",
        "",
        "See `metric_registry_audit.md` for the full taxonomy and the two "
        "separately-named notions of \"classic\".",
        "",
    ])


# --------------------------------------------------------------------------- #
# Method registry audit
# --------------------------------------------------------------------------- #
def method_registry_audit_md(coverage: Optional[pd.DataFrame] = None,
                             registry_problems: Sequence[str] = ()) -> str:
    lines = [
        "# Method Registry Audit",
        "",
        f"**{len(METHODS)} methods registered** "
        f"({sum(1 for m in METHODS if m.is_paper_method)} paper methods, "
        f"{sum(1 for m in METHODS if not m.is_paper_method)} supplementary).",
        "",
        "## Expected vs registered counts",
        "",
        "| method family | expected | registered |",
        "|---|---|---|",
    ]
    counts: Dict[str, int] = {}
    for m in METHODS:
        counts[m.method_family] = counts.get(m.method_family, 0) + 1
    for fam, want in EXPECTED_COUNTS.items():
        got = counts.get(fam, 0)
        flag = "" if got == want else "  **MISMATCH**"
        lines.append(f"| `{fam}` | {want} | {got}{flag} |")
    lines += [
        "",
        "ClustOpt variants (MLP + KNN + real): "
        f"**{counts.get('clustopt_mlp', 0) + counts.get('clustopt_knn', 0) + counts.get('clustopt_real', 0)}** "
        "(expected 34).",
        "",
        "## Validation",
        "",
    ]
    if registry_problems:
        lines += [f"- **PROBLEM:** {p}" for p in registry_problems] + [""]
    else:
        lines += ["No registry problems: canonical names, on-disk names and short "
                  "plot labels are all unique; every taxonomy field is within its "
                  "controlled vocabulary; every candidate pool resolves; the 17 "
                  "variant families exactly partition the 34 ClustOpt variants.",
                  ""]

    lines += _section("Structural variant families (utility source x metric-count policy)")
    lines += [
        f"{len(PRIMARY_VARIANT_FAMILIES)} primary cells + "
        f"{len(VARIANT_FAMILIES) - len(PRIMARY_VARIANT_FAMILIES)} oracle-assisted "
        "diagnostic cells. Each cell holds the matched raw/softmax pair from which "
        "Comparison 2 picks a representative.",
        "",
        "| family id | source | policy | raw variant | softmax variant | primary |",
        "|---|---|---|---|---|---|",
    ]
    for f in VARIANT_FAMILIES:
        lines.append(f"| `{f.family_id}` | {f.utility_source} | {f.policy_label} "
                     f"| `{f.raw_method}` | `{f.softmax_method}` "
                     f"| {'yes' if f.is_primary else 'no (oracle-assisted)'} |")

    lines += _section("Candidate pools")
    lines += ["| pool id | size | purpose |", "|---|---|---|"]
    for p in CANDIDATE_POOLS:
        lines.append(f"| `{p.pool_id}` | {p.size} | {p.description} |")

    lines += _section("Registered methods")
    lines += [
        "| short label | method | family | utility | K policy | weighting | "
        "oracle | same SS | budget | paper |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for m in METHODS:
        lines.append(
            f"| `{m.short_label}` | `{m.method_name}` | {m.method_family} "
            f"| {m.utility_source} | {m.k_policy} | {m.weighting_mode} "
            f"| {m.oracle_level} | {'yes' if m.same_search_space else 'no'} "
            f"| {m.search_budget or 'n/a'} "
            f"| {'yes' if m.is_paper_method else 'no'} |")

    if coverage is not None and not coverage.empty:
        lines += _section("On-disk coverage (Split 1)")
        lines += _table(coverage, [
            "method_short_label", "n_datasets", "n_runs", "n_success",
            "n_failed", "n_missing", "n_datasets_with_all_views",
            "n_datasets_with_no_success", "n_provenance_mismatch"], limit=60)
        mismatch = coverage[coverage["n_provenance_mismatch"] > 0]
        lines += [
            f"Provenance cross-check: the registry taxonomy was compared against "
            f"the `phaseD_metadata.json` block each run wrote for itself. "
            + ("**No mismatches.**" if mismatch.empty else
               f"**{len(mismatch)} method(s) disagree** -- see the coverage table."),
            "",
        ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Preliminary selection report
# --------------------------------------------------------------------------- #
def preliminary_selection_report_md(
    *,
    split_id: int,
    coverage_summary: Dict[str, Any],
    global_reps: pd.DataFrame,
    candidate_rankings: pd.DataFrame,
    comparison_01: pd.DataFrame,
    comparison_02_global: pd.DataFrame,
    comparison_02_family: pd.DataFrame,
    family_divergence: pd.DataFrame,
    comparison_04: pd.DataFrame,
    portfolio_manifest: pd.DataFrame,
    method_summary: pd.DataFrame,
    capability_warnings: pd.DataFrame,
    selection_bootstrap: Optional[pd.DataFrame] = None,
    real_dynamic_row: Optional[Dict[str, Any]] = None,
) -> str:
    lines = [
        "# Preliminary Retrospective Representative-Selection Study",
        "",
        f"**Split {split_id} only.** This report freezes every representative and "
        "portfolio that comparison suites 1-5 will consume. It performs no "
        "comparison of its own.",
        "",
        "> **Read this first.** Because downstream clustering experiments were",
        "> performed only on Split 1, ClustOpt single-variant representatives are",
        "> selected retrospectively from the evaluated candidate pools. These",
        "> representatives summarize the strongest observed fixed configurations",
        "> on the benchmark and are reported separately from dataset-level VBS",
        "> upper bounds. None of them is an independently preselected deployment",
        "> policy.",
        "",
    ]

    # ---- 1. facts from data --------------------------------------------- #
    lines += _section("1. Data coverage (facts computed from data)")
    lines += ["| quantity | value |", "|---|---|"]
    for key, label in (
        ("split_id", "split id"),
        ("n_datasets", "datasets"),
        ("n_families", "structural families"),
        ("n_subfamilies", "subfamilies"),
        ("n_methods", "methods collected"),
        ("n_methods_expected", "methods expected"),
        ("n_views", "views per dataset x method"),
        ("n_runs", "run rows (dataset x method x view)"),
        ("n_success", "successful runs"),
        ("n_failed", "failed runs"),
        ("n_missing", "missing runs"),
        ("n_best_view_rows", "best-view rows (dataset x method)"),
        ("n_best_view_with_success", "best-view rows with >=1 successful view"),
        ("n_provenance_mismatch", "provenance mismatches"),
    ):
        if key in coverage_summary:
            lines.append(f"| {label} | {_fmt(coverage_summary[key])} |")
    lines.append("")

    # ---- 2. retrospective selections ------------------------------------ #
    lines += _section("2. Retrospective global representatives")
    lines += [
        "Each row is the fixed variant with the highest global Mean Best-View ARI "
        "on Split 1 inside its candidate pool -- the **best observed single "
        "variant** in that pool, not an independently selected policy.",
        "",
    ]
    lines += _table(global_reps, [
        "representative_id", "candidate_pool_id", "candidate_pool_size",
        "selected_short_label", "selected_method",
        "selected_best_view_mean_ari", "selected_best_view_median_ari",
        "selected_best_view_std_ari", "selected_k_accuracy",
        "selected_median_total_all_views_runtime_sec", "n_tied_on_primary",
        "tie_breaks_invoked", "selection_type"], limit=60)

    lines += _section("2.1 Comparison 1 helper selection", 3)
    lines += [
        "Candidate pool: the four fixed Real-utility Top-K **raw** variants. "
        "`real_top_dynamic_raw` is recorded separately and is deliberately **not** "
        "a pool member, so the \"best fixed count\" and the \"dynamic count\" arms "
        "of RQ1-D stay distinct.",
        "",
    ]
    lines += _table(comparison_01, [
        "representative_id", "selection_scope", "family",
        "selected_short_label", "selected_method", "selected_best_view_mean_ari",
        "selected_best_view_median_ari", "n_tied_on_primary",
        "tie_breaks_invoked"], limit=40)
    if real_dynamic_row:
        lines += [
            "Recorded separately (not merged into the fixed winner):",
            "",
            "| method | Mean Best-View ARI | Median | K accuracy |",
            "|---|---|---|---|",
            f"| `{real_dynamic_row.get('method_name')}` "
            f"| {_fmt(real_dynamic_row.get('best_view_mean_ari'))} "
            f"| {_fmt(real_dynamic_row.get('best_view_median_ari'))} "
            f"| {_fmt(real_dynamic_row.get('best_view_k_accuracy'))} |",
            "",
        ]

    lines += _section("2.2 Comparison 2 raw-vs-softmax representatives", 3)
    lines += [
        f"One representative per (utility source x metric-count policy) cell: "
        f"{len(PRIMARY_VARIANT_FAMILIES)} primary cells plus "
        f"{len(VARIANT_FAMILIES) - len(PRIMARY_VARIANT_FAMILIES)} "
        "oracle-assisted Dynamic Real-K diagnostic cells. For the MLP/KNN primary "
        "Dynamic cells only `dynamic_predictK_*` is eligible; `dynamic_realK_*` is "
        "kept as a separately-labelled diagnostic.",
        "",
    ]
    lines += _table(comparison_02_global, [
        "representative_id", "candidate_pool_id", "selected_short_label",
        "selected_method", "selected_weighting_mode",
        "selected_best_view_mean_ari", "selected_best_view_median_ari",
        "n_tied_on_primary", "tie_breaks_invoked", "selected_oracle_level"],
        limit=40)

    lines += _section("2.3 Comparison 4 fixed representatives", 3)
    lines += [
        "The best observed **fixed** ClustOpt-MLP and ClustOpt-KNN variants (8 "
        "candidates each: 4 Top-K sizes x raw/softmax). Supplemental pools that "
        "also admit Dynamic Predict-K are recorded but do not replace these.",
        "",
    ]
    lines += _table(comparison_04, [
        "representative_id", "candidate_pool_id", "candidate_pool_size",
        "selected_short_label", "selected_method",
        "selected_best_view_mean_ari", "selected_best_view_median_ari",
        "selected_best_view_std_ari", "selected_k_accuracy",
        "selected_median_total_all_views_runtime_sec", "n_tied_on_primary",
        "tie_breaks_invoked"], limit=40)

    # ---- 3. oracle results --------------------------------------------- #
    lines += _section("3. Family-level reselection (oracle / headroom only)")
    lines += [
        "Re-picking raw vs softmax **inside each family** is a family-level "
        "retrospective policy oracle. It is reported as headroom and is never "
        "mixed with the primary fixed-global representative results.",
        "",
    ]
    if family_divergence is not None and not family_divergence.empty:
        rates = (family_divergence[["representative_id", "n_families",
                                    "n_differing", "divergence_rate"]]
                 .drop_duplicates()
                 .sort_values("divergence_rate", ascending=False))
        lines += _table(rates, ["representative_id", "n_families", "n_differing",
                                "divergence_rate"], limit=40)
        lines += [
            "A high divergence rate means a per-family weighting policy would "
            "diverge often from the single global choice -- i.e. more family-level "
            "headroom, **not** evidence that any single deployable method is "
            "better.",
            "",
        ]
    else:
        lines += ["_(no family-level selections available)_", ""]

    lines += _section("4. Frozen VBS portfolios (oracle upper bounds)")
    lines += [
        "Every portfolio's VBS picks a different member per dataset using that "
        "dataset's true external ARI, and is therefore a non-deployable upper "
        "bound. Contamination flags make oracle membership explicit.",
        "",
    ]
    lines += _table(portfolio_manifest, [
        "portfolio_id", "portfolio_short_label", "portfolio_size",
        "contains_real_utility", "contains_oracle_assisted_dynamic_k",
        "frameworks", "notes"], limit=40)

    # ---- statistical uncertainty --------------------------------------- #
    if selection_bootstrap is not None and not selection_bootstrap.empty:
        lines += _section("5. Selection-aware uncertainty")
        lines += [
            "The interval below re-runs the selection rule inside every bootstrap "
            "resample, so it prices in **both** dataset sampling and "
            "representative selection. `selection_stability` is the share of "
            "resamples in which the same candidate won -- a low value means the "
            "point choice is fragile.",
            "",
        ]
        lines += _table(selection_bootstrap, [
            "label", "candidate_pool_size", "point_selected_short_label",
            "competitor_short_label", "point_delta", "bootstrap_mean_delta",
            "ci_low", "ci_high", "selection_stability", "n_distinct_winners",
            "claim_level"], limit=30)

    # ---- observed performance table ------------------------------------ #
    lines += _section("6. All collected methods (facts, no selection applied)")
    lines += _table(method_summary, [
        "method_short_label", "method_family", "best_view_mean_ari",
        "best_view_median_ari", "best_view_std_ari", "best_view_k_accuracy",
        "median_total_runtime_all_views_sec", "n_datasets", "success_rate",
        "oracle_level", "is_paper_method"], limit=60)

    # ---- unsupported analyses ------------------------------------------ #
    lines += _section("7. Unsupported analyses")
    if capability_warnings is None or capability_warnings.empty:
        lines += ["Every collected method supports every requested analysis.", ""]
    else:
        grouped = (capability_warnings.groupby(
            ["analysis", "missing_capabilities"])["method_short_label"]
            .apply(lambda s: ", ".join(sorted(set(s)))).reset_index())
        lines += ["| analysis | missing capability | affected methods |",
                  "|---|---|---|"]
        for _, r in grouped.iterrows():
            lines.append(f"| `{r['analysis']}` | `{r['missing_capabilities']}` | "
                         f"{r['method_short_label']} |")
        lines += [
            "",
            "Trace-best (search-selection) gap and metric-selection analyses are "
            "unavailable for AutoClust and ML2DAC because the original "
            "implementations do not expose a compatible per-candidate search "
            "trace or a metric-selection record. Those sections will therefore "
            "include ClustOpt methods only, and the omission is stated rather "
            "than null-filled.",
            "",
            "End-to-end runtime components are unavailable for **all** methods: "
            "meta-feature extraction and utility prediction are not timed per "
            "run. They are never approximated from `runtime_sec`.",
            "",
        ]

    # ---- limitations ---------------------------------------------------- #
    lines += _section("8. Limitations")
    lines += [
        "1. **Split-1 only.** No downstream clustering experiments exist on "
        "splits 2-16, so no representative here was selected on independent data. "
        "Every representative is retrospective and every statistical test "
        "involving one is `post_selection_exploratory`.",
        "2. **Best-View is a view oracle.** The primary metric uses external ARI "
        "to choose the representation. The matching honest cost is "
        "`total_runtime_all_views_sec`.",
        "3. **Candidate-pool asymmetry.** A ClustOpt representative is the best of "
        "8-10 evaluated variants; each external row is one predefined method. "
        "Comparison 4 discloses this in every table.",
        "4. **Real-utility and Dynamic Real-K arms are oracles**, not deployable "
        "methods. They bound what utility guidance could achieve and diagnose "
        "where learned prediction loses ground.",
        "5. **VBS numbers are upper bounds** with no operational runtime; the full "
        "portfolio cost is always reported beside them.",
        "",
    ]
    lines += _section("9. Interpretation suggestions (not conclusions)")
    lines += [
        "_The following are auto-derived starting points that must be verified "
        "against the CSV tables before being quoted._",
        "",
    ]
    if method_summary is not None and not method_summary.empty:
        top = method_summary.sort_values("best_view_mean_ari", ascending=False)
        # Suggestions are drawn from the PAPER methods only. The supplementary
        # rows (unmatched search space, original authors' code) are collected for
        # auditability and must not become headline claims.
        paper = (top[top["is_paper_method"].astype(bool)]
                 if "is_paper_method" in top.columns else top)
        if not paper.empty:
            best = paper.iloc[0]
            lines += [
                f"- The highest observed Mean Best-View ARI among the paper "
                f"methods on Split 1 is **{_fmt(best['best_view_mean_ari'])}** "
                f"(`{best['method_short_label']}`, oracle level "
                f"`{best.get('oracle_level', 'n/a')}`). If that oracle level is "
                f"not `none`, this is an upper bound rather than a deployable "
                f"result.",
            ]
        deployable = (paper[paper["oracle_level"] == "none"]
                      if "oracle_level" in paper.columns else pd.DataFrame())
        if not deployable.empty:
            d = deployable.iloc[0]
            lines.append(
                f"- The best observed paper method with **no** method-level "
                f"oracle is `{d['method_short_label']}` at "
                f"{_fmt(d['best_view_mean_ari'])}.")
        clustopt = (paper[paper["method_family"].astype(str).str.startswith("clustopt")
                          & (paper["oracle_level"] == "none")]
                    if {"method_family", "oracle_level"} <= set(paper.columns)
                    else pd.DataFrame())
        if not clustopt.empty:
            c = clustopt.iloc[0]
            lines.append(
                f"- The best observed ClustOpt variant with no method-level "
                f"oracle is `{c['method_short_label']}` at "
                f"{_fmt(c['best_view_mean_ari'])}. Comparing it to an external "
                f"row is a retrospective benchmark summary, not an "
                f"independently selected deployment comparison.")
    lines += ["", "_End of preliminary report._", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Architecture / readiness
# --------------------------------------------------------------------------- #
def readiness_report_md(*, sections: Dict[str, Sequence[str]],
                        ready: bool, blockers: Sequence[str] = ()) -> str:
    lines = [
        "# Architecture and Readiness Report",
        "",
        f"**Ready to run comparison suites 1-5 sequentially: "
        f"{'YES' if ready else 'NO'}**",
        "",
    ]
    if not ready and blockers:
        lines += ["## Blockers", ""] + [f"- {b}" for b in blockers] + [""]
    for title, body in sections.items():
        lines += _section(title)
        lines += list(body) + [""]
    return "\n".join(lines)


def start_here_md(*, split_id: int, output_root: str,
                  stages_done: Sequence[str],
                  stages_pending: Sequence[str]) -> str:
    return "\n".join([
        "# START HERE",
        "",
        f"Paper-oriented analysis of the ClustOpt experiments, Split {split_id}.",
        "",
        "## Reading order",
        "",
        "1. `00_protocol_and_definitions/terminology.md` -- **required.** What "
        "each kind of \"winner\" in this analysis does and does not mean.",
        "2. `00_protocol_and_definitions/oracle_levels.md` -- the four different "
        "oracles and why they stay separate.",
        "3. `00_protocol_and_definitions/runtime_definitions.md` -- why "
        "`total_runtime_all_views_sec` is the primary cost.",
        "4. `01_preliminary_selection/PRELIMINARY_SELECTION_REPORT.md` -- the "
        "frozen representatives and portfolios every comparison consumes.",
        "5. `ARCHITECTURE_AND_READINESS_REPORT.md` -- what is implemented, "
        "validated and ready to run.",
        "",
        "## The one-sentence caveat",
        "",
        "> All downstream clustering experiments were run on Split 1 only, so ",
        "> every ClustOpt single-variant representative is selected ",
        "> retrospectively; representatives summarise the strongest observed ",
        "> fixed configurations and are reported separately from dataset-level VBS ",
        "> upper bounds.",
        "",
        "## Directory map",
        "",
        "| directory | contents |",
        "|---|---|",
        "| `00_protocol_and_definitions/` | terminology, oracle levels, metric "
        "and runtime definitions, method label map |",
        "| `01_preliminary_selection/` | frozen selection + portfolio manifests, "
        "candidate rankings, preliminary workbook and report |",
        "| `02_comparison_01_metric_utility/` | metric inventory and "
        "utility-selection ablation |",
        "| `03_comparison_02_meta_learning/` | real vs MLP vs KNN |",
        "| `04_comparison_03_raw_softmax_vbs/` | raw vs softmax weighting VBS |",
        "| `05_comparison_04_operational_benchmarks/` | best observed fixed "
        "ClustOpt vs external competitors |",
        "| `06_comparison_05_vbs_headroom/` | portfolio VBS and policy-selection "
        "headroom |",
        "| `07_metric_selection/` | metric taxonomy and new-metric usage |",
        "| `08_gap_decomposition/` | the seven separately-named gaps |",
        "| `09_runtime_and_pareto/` | runtime distributions and Pareto frontier |",
        "| `10_statistical_tests/` | paired tests, Holm correction, bootstraps |",
        "| `11_family_reports/` | per-family rollups |",
        "| `12_subfamily_appendix/` | per-subfamily appendix |",
        "| `13_raw_and_reproducibility/` | canonical frames, registries, "
        "capability matrix, run config |",
        "| `14_packaged_for_paper/` | advisor-facing package |",
        "",
        "## Stage status",
        "",
        "| stage | status |",
        "|---|---|",
        *[f"| `{s}` | complete |" for s in stages_done],
        *[f"| `{s}` | not yet run (awaiting review of the preliminary study) |"
          for s in stages_pending],
        "",
        f"Output root: `{output_root}`",
        "",
        "## Manifest",
        "",
        "`analysis_manifest.csv` / `.json` list every generated artifact with its "
        "section, type, source tables, paper importance and selection type. The "
        "packager reads that manifest -- it does not guess from filenames.",
        "",
    ])


def limitations_md(*, capability_warnings: Optional[pd.DataFrame] = None) -> str:
    lines = [
        "# Limitations",
        "",
        "## 1. Single evaluation split",
        "",
        "Downstream clustering experiments exist for `split_id = 1` only. Every "
        "ClustOpt representative in this analysis is therefore selected "
        "**retrospectively on the same data it is reported on**. Consequences:",
        "",
        "- no representative is an independently preselected deployment policy;",
        "- statistical tests involving a representative are optimistic and are "
        "labelled `post_selection_exploratory`;",
        "- the selection-aware bootstrap is the appropriate uncertainty measure "
        "for those comparisons, and it is wider than a naive paired bootstrap.",
        "",
        "## 2. Best-View is a view oracle",
        "",
        "The primary metric selects the best of three representations using "
        "external ARI. It is a uniform advantage across methods, but it is still "
        "an oracle: a deployed system would have to pick a view without ground "
        "truth. The matching honest cost is the runtime of all three views.",
        "",
        "## 3. Candidate-pool asymmetry against the external methods",
        "",
        "A ClustOpt row in Comparison 4 is the best of 8-10 evaluated variants; "
        "each external row is one predefined method. The search space and the "
        "50-evaluation budget are matched, but the *selection freedom* is not.",
        "",
        "## 4. Oracle contamination inside portfolios",
        "",
        "`real_utility_vbs` and any portfolio containing `real_*` or "
        "`*_dynamic_realK_*` members inherits an oracle. Those flags "
        "(`contains_real_utility`, `contains_oracle_assisted_dynamic_k`) are on "
        "every portfolio row; `strict_deployable_clustopt_vbs` is the only "
        "portfolio whose members are all fully operational.",
        "",
        "## 5. The isolated effect of the new metrics is not identified",
        "",
        "Comparisons between Real Top-K and the classic baselines change the "
        "metric inventory, the metric selection, the utility weighting and "
        "possibly the metric count **at the same time**. Without a classic-only "
        "real-utility Top-K run, the causal contribution of the new metrics alone "
        "cannot be identified from the existing experiments. New-metric weight "
        "mass vs ARI gain is reported as an *association* only.",
        "",
        "## 6. End-to-end runtime is not measured",
        "",
        "Meta-feature extraction, utility prediction and KNN query time are not "
        "recorded per run by any adapter. Those fields are reported as "
        "unavailable and never approximated.",
        "",
    ]
    if capability_warnings is not None and not capability_warnings.empty:
        lines += ["## 7. Analyses skipped for specific methods", ""]
        grouped = (capability_warnings.groupby(
            ["analysis", "missing_capabilities"])["method_short_label"]
            .apply(lambda s: ", ".join(sorted(set(s)))).reset_index())
        lines += ["| analysis | missing capability | affected methods |",
                  "|---|---|---|"]
        for _, r in grouped.iterrows():
            lines.append(f"| `{r['analysis']}` | `{r['missing_capabilities']}` "
                         f"| {r['method_short_label']} |")
        lines.append("")
    return "\n".join(lines)


def capability_audit_md(matrix_md: str, warnings: Optional[pd.DataFrame]) -> str:
    lines = [matrix_md, "", "## Warnings raised during preflight", ""]
    if warnings is None or warnings.empty:
        lines += ["_None._", ""]
    else:
        lines += _table(warnings, ["comparison_id", "analysis",
                                   "method_short_label", "missing_capabilities",
                                   "action"], limit=200)
    return "\n".join(lines)
