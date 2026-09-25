"""Metric-selection and new-metric-usage analysis.

Works for **every** method that records a metric selection -- MLP, KNN,
real-utility and the uniform ensembles -- not just the regressor group the
previous implementation handled.

Normalisation (§17.2) -- this is the part that is easy to get wrong
-------------------------------------------------------------------
A Top-10 method fills ten selection slots per run; a Top-1 method fills one. Raw
selection counts are therefore **not** comparable across metric-count policies.
Every rate here is normalised explicitly and the denominator is named in the
column:

``selection_rate_per_run``
    runs in which the metric was selected / runs of that method. Comparable
    across policies -- "how often does this method reach for this metric?"
``share_of_selected_slots``
    times selected / total slots the method filled. Sums to 1 across metrics
    within a method, so it is the composition measure.
``mean_weight_when_selected``
    mean objective weight *conditional* on being selected.
``mean_unconditional_weight_mass``
    total weight assigned / number of runs. Comparable across policies because a
    run's weights always sum to ~1 regardless of K.
``dataset_coverage``
    distinct datasets in which the metric was selected / distinct datasets.

Raw ``selection_count`` is still emitted, but always beside its denominator.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .collector import parse_metric_list, parse_weights
from .method_registry import METHOD_BY_NAME, short_label
from .metric_registry import (
    METRIC_CATEGORIES, canonical_metric_name, category_of, is_new_metric,
    subcategory_of,
)

# --------------------------------------------------------------------------- #
# Long-form explosion
# --------------------------------------------------------------------------- #
SELECTION_LONG_COLUMNS: Tuple[str, ...] = (
    "dataset_id", "family", "family_id", "subfamily", "difficulty",
    "cluster_count", "method_name", "method_short_label", "method_family",
    "utility_source", "k_policy", "weighting_mode", "metric_name",
    "metric_name_as_recorded", "metric_category", "metric_subcategory",
    "is_new_metric", "weight", "utility", "is_top_metric", "n_selected_in_run",
    "is_known_metric",
)


def explode_selections(best_view: pd.DataFrame,
                       *, methods: Optional[Sequence[str]] = None
                       ) -> pd.DataFrame:
    """Long frame: one row per ``(dataset, method, selected metric)``.

    Built from the best-view frame, so each run contributes the selection of the
    view that was actually reported. Metric names are canonicalised through the
    alias table but the recorded spelling is kept in
    ``metric_name_as_recorded`` for auditability.
    """
    sub = best_view
    if methods is not None:
        sub = sub[sub["method_name"].isin(list(methods))]
    sub = sub[sub.get("has_metric_selection", pd.Series(True, index=sub.index))
              .astype("boolean").fillna(False)]
    if sub.empty:
        return pd.DataFrame(columns=list(SELECTION_LONG_COLUMNS))

    rows: List[Dict[str, Any]] = []
    for _, r in sub.iterrows():
        weights = parse_weights(r.get("selected_metric_weights"))
        selected = parse_metric_list(r.get("selected_metrics")) or list(weights)
        if not selected:
            continue
        utilities = parse_weights(r.get("selected_metric_utilities"))
        top = r.get("top_metric")
        n_sel = len(selected)
        base = {
            "dataset_id": r.get("dataset_id"),
            "family": r.get("family"), "family_id": r.get("family_id"),
            "subfamily": r.get("subfamily"), "difficulty": r.get("difficulty"),
            "cluster_count": r.get("cluster_count"),
            "method_name": r.get("method_name"),
            "method_short_label": r.get("method_short_label"),
            "method_family": r.get("method_family"),
            "utility_source": r.get("utility_source"),
            "k_policy": r.get("k_policy"),
            "weighting_mode": r.get("weighting_mode"),
            "n_selected_in_run": n_sel,
        }
        for name in selected:
            canon = canonical_metric_name(name)
            rows.append({
                **base,
                "metric_name": canon,
                "metric_name_as_recorded": str(name),
                "metric_category": category_of(name),
                "metric_subcategory": subcategory_of(name),
                "is_new_metric": is_new_metric(name),
                "is_known_metric": category_of(name) != "unknown",
                "weight": float(weights.get(name, weights.get(canon, np.nan))),
                "utility": float(utilities.get(name, utilities.get(canon, np.nan))),
                "is_top_metric": bool(top is not None
                                      and canonical_metric_name(str(top)) == canon),
            })
    return pd.DataFrame(rows, columns=list(SELECTION_LONG_COLUMNS))


def _run_denominators(long_frame: pd.DataFrame,
                      group_cols: Sequence[str]) -> pd.DataFrame:
    """Per-group run/slot/dataset denominators for correct normalisation."""
    cols = list(group_cols)
    grouped = long_frame.groupby(cols, dropna=False)
    return grouped.agg(
        n_runs=("dataset_id", "nunique"),
        n_slots=("metric_name", "size"),
        n_datasets=("dataset_id", "nunique"),
    ).reset_index()


# --------------------------------------------------------------------------- #
# 17.2 Per-metric measures
# --------------------------------------------------------------------------- #
def metric_usage(long_frame: pd.DataFrame,
                 group_cols: Sequence[str] = ("method_name",)) -> pd.DataFrame:
    """Normalised per-metric usage, grouped by ``group_cols``."""
    if long_frame.empty:
        return pd.DataFrame()
    cols = [c for c in group_cols if c in long_frame.columns]
    denom = _run_denominators(long_frame, cols).set_index(cols) if cols else None
    total_runs = int(long_frame["dataset_id"].nunique())
    total_slots = int(len(long_frame))

    keys = cols + ["metric_name"]
    agg = long_frame.groupby(keys, dropna=False).agg(
        selection_count=("metric_name", "size"),
        n_datasets_selected=("dataset_id", "nunique"),
        top_metric_count=("is_top_metric", "sum"),
        mean_weight_when_selected=("weight", "mean"),
        median_weight_when_selected=("weight", "median"),
        total_weight_mass=("weight", "sum"),
        mean_utility_when_selected=("utility", "mean"),
    ).reset_index()

    if cols:
        agg = agg.merge(denom.reset_index(), on=cols, how="left")
    else:
        agg["n_runs"] = total_runs
        agg["n_slots"] = total_slots
        agg["n_datasets"] = total_runs

    agg["selection_rate_per_run"] = agg["selection_count"] / agg["n_runs"]
    agg["share_of_selected_slots"] = agg["selection_count"] / agg["n_slots"]
    agg["top_metric_rate"] = agg["top_metric_count"] / agg["n_runs"]
    agg["mean_unconditional_weight_mass"] = agg["total_weight_mass"] / agg["n_runs"]
    agg["dataset_coverage"] = agg["n_datasets_selected"] / agg["n_datasets"]
    agg["metric_category"] = agg["metric_name"].map(category_of)
    agg["metric_subcategory"] = agg["metric_name"].map(subcategory_of)
    agg["is_new_metric"] = agg["metric_name"].map(is_new_metric)
    if "method_name" in agg.columns:
        agg["method_short_label"] = agg["method_name"].map(short_label)
    agg["normalisation_note"] = (
        "selection_rate_per_run and mean_unconditional_weight_mass are the "
        "policy-comparable measures; raw selection_count is NOT comparable "
        "across different Top-K sizes")
    sort_cols = cols + ["share_of_selected_slots"]
    return agg.sort_values(sort_cols, ascending=[True] * len(cols) + [False],
                          kind="mergesort").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 17.3 Per-category measures
# --------------------------------------------------------------------------- #
def category_usage(long_frame: pd.DataFrame,
                   group_cols: Sequence[str] = ("method_name",),
                   *, level: str = "metric_category") -> pd.DataFrame:
    """Category (or subcategory) level usage shares."""
    if long_frame.empty:
        return pd.DataFrame()
    cols = [c for c in group_cols if c in long_frame.columns]
    keys = cols + [level]
    denom = _run_denominators(long_frame, cols) if cols else None

    agg = long_frame.groupby(keys, dropna=False).agg(
        selected_slot_count=(level, "size"),
        weight_mass=("weight", "sum"),
        top_metric_count=("is_top_metric", "sum"),
        n_datasets_selected=("dataset_id", "nunique"),
        n_distinct_metrics=("metric_name", "nunique"),
    ).reset_index()
    if cols:
        agg = agg.merge(denom, on=cols, how="left")
    else:
        agg["n_runs"] = int(long_frame["dataset_id"].nunique())
        agg["n_slots"] = int(len(long_frame))
        agg["n_datasets"] = agg["n_runs"]

    agg["selected_slot_share"] = agg["selected_slot_count"] / agg["n_slots"]
    total_mass = agg.groupby(cols, dropna=False)["weight_mass"].transform("sum") \
        if cols else agg["weight_mass"].sum()
    agg["weight_mass_share"] = agg["weight_mass"] / total_mass
    total_top = agg.groupby(cols, dropna=False)["top_metric_count"].transform("sum") \
        if cols else agg["top_metric_count"].sum()
    agg["top_metric_share"] = agg["top_metric_count"] / total_top.replace(0, np.nan)
    agg["dataset_coverage"] = agg["n_datasets_selected"] / agg["n_datasets"]
    agg["mean_selected_count_per_run"] = agg["selected_slot_count"] / agg["n_runs"]

    # Median per-run count needs a per-run tally, not a ratio of totals.
    per_run = (long_frame.groupby(cols + ["dataset_id", level], dropna=False)
               .size().rename("cnt").reset_index())
    med = (per_run.groupby(keys, dropna=False)["cnt"].median()
           .rename("median_selected_count_per_run").reset_index())
    agg = agg.merge(med, on=keys, how="left")

    if "method_name" in agg.columns:
        agg["method_short_label"] = agg["method_name"].map(short_label)
    sort_cols = cols + ["weight_mass_share"]
    return agg.sort_values(sort_cols, ascending=[True] * len(cols) + [False],
                          kind="mergesort").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 17.4 New-metric usage
# --------------------------------------------------------------------------- #
def per_run_new_metric_usage(long_frame: pd.DataFrame) -> pd.DataFrame:
    """One row per ``(dataset, method)``: how much new-metric mass it used."""
    if long_frame.empty:
        return pd.DataFrame()
    keys = ["dataset_id", "method_name"]
    grouped = long_frame.groupby(keys, dropna=False)
    out = grouped.agg(
        family=("family", "first"),
        family_id=("family_id", "first"),
        subfamily=("subfamily", "first"),
        difficulty=("difficulty", "first"),
        cluster_count=("cluster_count", "first"),
        method_short_label=("method_short_label", "first"),
        utility_source=("utility_source", "first"),
        k_policy=("k_policy", "first"),
        weighting_mode=("weighting_mode", "first"),
        n_selected=("metric_name", "size"),
        new_metric_count=("is_new_metric", "sum"),
        total_weight=("weight", "sum"),
    ).reset_index()
    new_mass = (long_frame[long_frame["is_new_metric"]]
                .groupby(keys, dropna=False)["weight"].sum()
                .rename("new_metric_weight_mass").reset_index())
    out = out.merge(new_mass, on=keys, how="left")
    out["new_metric_weight_mass"] = out["new_metric_weight_mass"].fillna(0.0)
    out["new_metric_fraction"] = out["new_metric_count"] / out["n_selected"]
    out["new_metric_weight_share"] = (out["new_metric_weight_mass"]
                                      / out["total_weight"].replace(0, np.nan))
    out["has_new_metric"] = out["new_metric_count"] > 0
    top_new = (long_frame[long_frame["is_top_metric"]]
               .groupby(keys, dropna=False)["is_new_metric"].max()
               .rename("top_metric_is_new").reset_index())
    out = out.merge(top_new, on=keys, how="left")
    out["top_metric_is_new"] = out["top_metric_is_new"].fillna(False).astype(bool)
    return out


def new_metric_usage(long_frame: pd.DataFrame,
                     group_cols: Sequence[str] = ("method_name",)) -> pd.DataFrame:
    """Aggregate new-metric usage per group."""
    per_run = per_run_new_metric_usage(long_frame)
    if per_run.empty:
        return pd.DataFrame()
    cols = [c for c in group_cols if c in per_run.columns]
    if not cols:
        per_run = per_run.assign(_all="all")
        cols = ["_all"]
    agg = per_run.groupby(cols, dropna=False).agg(
        n_runs=("dataset_id", "nunique"),
        mean_new_metric_count=("new_metric_count", "mean"),
        mean_new_metric_fraction=("new_metric_fraction", "mean"),
        median_new_metric_fraction=("new_metric_fraction", "median"),
        mean_new_metric_weight_mass=("new_metric_weight_mass", "mean"),
        median_new_metric_weight_mass=("new_metric_weight_mass", "median"),
        mean_new_metric_weight_share=("new_metric_weight_share", "mean"),
        share_runs_with_new_metric=("has_new_metric", "mean"),
        share_runs_top_metric_is_new=("top_metric_is_new", "mean"),
        mean_n_selected=("n_selected", "mean"),
    ).reset_index()
    if "method_name" in agg.columns:
        agg["method_short_label"] = agg["method_name"].map(short_label)
    return agg.sort_values(cols, kind="mergesort").reset_index(drop=True)


def new_metric_vs_ari_gain(
    best_view: pd.DataFrame, long_frame: pd.DataFrame, *,
    reference_method: str = "uniform_classic_cvi",
) -> pd.DataFrame:
    """Associate per-run new-metric weight mass with ARI gain over a reference.

    This is an **association**, not causal evidence: a run that leans on new
    metrics may also differ in metric count and weighting. The returned frame
    says so in ``interpretation_note`` and the report repeats it.
    """
    per_run = per_run_new_metric_usage(long_frame)
    if per_run.empty:
        return pd.DataFrame()
    ref = best_view.loc[best_view["method_name"] == reference_method,
                        ["dataset_id", "best_view_ari"]].rename(
        columns={"best_view_ari": "reference_ari"})
    if ref.empty:
        return pd.DataFrame()
    own = best_view[["dataset_id", "method_name", "best_view_ari"]]
    merged = (per_run.merge(own, on=["dataset_id", "method_name"], how="left")
              .merge(ref, on="dataset_id", how="left"))
    merged["ari_gain_over_reference"] = (
        pd.to_numeric(merged["best_view_ari"], errors="coerce")
        - pd.to_numeric(merged["reference_ari"], errors="coerce"))
    merged["reference_method"] = reference_method

    rows: List[Dict[str, Any]] = []
    for method, grp in merged.groupby("method_name", dropna=False):
        mass = pd.to_numeric(grp["new_metric_weight_mass"], errors="coerce")
        gain = pd.to_numeric(grp["ari_gain_over_reference"], errors="coerce")
        mask = mass.notna() & gain.notna()
        n = int(mask.sum())
        pearson = spearman = float("nan")
        if n >= 3 and mass[mask].nunique() > 1 and gain[mask].nunique() > 1:
            pearson = float(np.corrcoef(mass[mask], gain[mask])[0, 1])
            spearman = float(pd.Series(mass[mask]).corr(
                pd.Series(gain[mask]), method="spearman"))
        rows.append({
            "method_name": method,
            "method_short_label": short_label(str(method)),
            "reference_method": reference_method,
            "n_paired": n,
            "mean_new_metric_weight_mass": float(mass[mask].mean()) if n else float("nan"),
            "mean_ari_gain_over_reference": float(gain[mask].mean()) if n else float("nan"),
            "pearson_r": pearson,
            "spearman_rho": spearman,
            "interpretation_note":
                "ASSOCIATION ONLY. Runs differ simultaneously in metric "
                "inventory, selection, weighting and count, so this correlation "
                "cannot isolate a causal contribution of the new metrics.",
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Entropy / diversity
# --------------------------------------------------------------------------- #
def entropy_summary(best_view: pd.DataFrame,
                    group_cols: Sequence[str] = ("method_name",)) -> pd.DataFrame:
    """Weight-distribution entropy + selection diversity per group."""
    sub = best_view[best_view.get("has_metric_selection",
                                  pd.Series(True, index=best_view.index))
                    .astype("boolean").fillna(False)]
    if sub.empty:
        return pd.DataFrame()
    cols = [c for c in group_cols if c in sub.columns]
    work = sub.assign(_ent=pd.to_numeric(sub["metric_weight_entropy"],
                                         errors="coerce"),
                      _n=pd.to_numeric(sub["selected_metric_count"],
                                       errors="coerce"))
    agg = work.groupby(cols, dropna=False).agg(
        n=("_ent", "size"),
        mean_weight_entropy=("_ent", "mean"),
        median_weight_entropy=("_ent", "median"),
        min_weight_entropy=("_ent", "min"),
        mean_selected_metric_count=("_n", "mean"),
        median_selected_metric_count=("_n", "median"),
        n_distinct_top_metrics=("top_metric", "nunique"),
    ).reset_index()
    if "method_name" in agg.columns:
        agg["method_short_label"] = agg["method_name"].map(short_label)
    return agg.sort_values(cols, kind="mergesort").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Audit
# --------------------------------------------------------------------------- #
def usage_audit(long_frame: pd.DataFrame) -> Dict[str, Any]:
    """Coverage audit of the metric-usage analysis (all metrics mapped?)."""
    if long_frame.empty:
        return {"n_selection_rows": 0, "n_unknown_metric_rows": 0,
                "unknown_metrics": [], "n_distinct_metrics": 0,
                "n_distinct_recorded_spellings": 0,
                "counts_by_category": {}, "aliased_spellings": []}
    unknown = long_frame[~long_frame["is_known_metric"]]
    aliased = long_frame[long_frame["metric_name"]
                         != long_frame["metric_name_as_recorded"]]
    return {
        "n_selection_rows": int(len(long_frame)),
        "n_unknown_metric_rows": int(len(unknown)),
        "unknown_metrics": sorted(unknown["metric_name_as_recorded"].unique().tolist()),
        "n_distinct_metrics": int(long_frame["metric_name"].nunique()),
        "n_distinct_recorded_spellings": int(
            long_frame["metric_name_as_recorded"].nunique()),
        "counts_by_category": long_frame["metric_category"].value_counts().to_dict(),
        "aliased_spellings": sorted(
            aliased[["metric_name_as_recorded", "metric_name"]]
            .drop_duplicates().itertuples(index=False, name=None)),
        "n_methods_with_selection": int(long_frame["method_name"].nunique()),
    }


def validate_usage(long_frame: pd.DataFrame) -> List[str]:
    """Fail-fast checks: 100% mapped, no unknown category, shares sum to 1."""
    problems: List[str] = []
    if long_frame.empty:
        problems.append("no metric-selection rows to analyse")
        return problems
    audit = usage_audit(long_frame)
    if audit["n_unknown_metric_rows"]:
        problems.append(f"{audit['n_unknown_metric_rows']} selection rows use "
                        f"metrics missing from the taxonomy: "
                        f"{audit['unknown_metrics']}")
    bad_cat = set(long_frame["metric_category"].unique()) - set(METRIC_CATEGORIES)
    if bad_cat:
        problems.append(f"unknown metric categories present: {sorted(bad_cat)}")

    # share_of_selected_slots must sum to 1 within every method.
    usage = metric_usage(long_frame, ("method_name",))
    if not usage.empty:
        sums = usage.groupby("method_name")["share_of_selected_slots"].sum()
        bad = sums[~np.isclose(sums, 1.0, atol=1e-9)]
        if len(bad):
            problems.append(f"share_of_selected_slots does not sum to 1 for "
                            f"{len(bad)} method(s): {bad.head(3).to_dict()}")
    cats = category_usage(long_frame, ("method_name",))
    if not cats.empty:
        sums = cats.groupby("method_name")["selected_slot_share"].sum()
        bad = sums[~np.isclose(sums, 1.0, atol=1e-9)]
        if len(bad):
            problems.append(f"selected_slot_share does not sum to 1 for "
                            f"{len(bad)} method(s)")
    return problems
