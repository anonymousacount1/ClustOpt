"""Canonical frame derivations: best-view, dataset-level, representative.

Frame contract
--------------
``run_frame``
    one row per ``dataset x method x view`` (see :mod:`collector`).

``best_view_frame``
    one row per ``dataset x method``. The view with the highest external ARI
    among the *successful* views is selected -- a **view oracle**, so every row
    carries ``uses_view_oracle = True``. Datasets where a method had no
    successful view are **kept** (all Best-View measures null, ``n_success = 0``)
    so success-rate accounting stays honest.

``dataset_level_frame``
    the best-view frame plus per-dataset cross-method context (rank of the
    method on that dataset, delta to the dataset's best method). Built once and
    reused by the win-rate/paired analyses.

``representative_frame``
    one row per ``dataset x representative_id`` -- the best-view rows of the one
    fixed method the selection engine chose, re-labelled with the selection
    provenance. A *global* representative is the same method on every dataset;
    a *family* representative varies only across families.

Runtime semantics (never mixed -- see ``runtime_definitions.md``)
----------------------------------------------------------------
``best_view_runtime_sec``
    runtime of the single view whose ARI was selected. Diagnostic only.
``total_runtime_all_views_sec``
    ``x_only + y_only + xy_2d`` runtime. This is the **primary** cost of a
    Best-View result, because all three views must be run to obtain one.
``has_*`` capability flags
    per-row availability, so a comparison can skip an unsupported runtime column
    rather than substituting a different field.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .collector import STATUS_MISSING, STATUS_SUCCESS
from .method_registry import METHOD_BY_NAME, VIEW_IDS

# Columns copied verbatim from the winning view's run row.
_CARRY_IDENTITY: Tuple[str, ...] = (
    "dataset_id", "dataset_dir", "split_id", "family", "family_id", "subfamily",
    "difficulty", "cluster_count", "n_points",
)
_CARRY_PROVENANCE: Tuple[str, ...] = (
    "method_name", "method_full_name", "method_short_label", "paper_label",
    "on_disk_name", "schema", "framework", "method_family", "utility_source",
    "k_policy", "fixed_metric_count", "dynamic_k_source", "weighting_mode",
    "softmax_temperature", "metric_inventory", "metric_inventory_size",
    "selection_capability", "oracle_level", "uses_real_utility",
    "uses_real_dynamic_k", "is_oracle_assisted", "is_utility_oracle",
    "is_meta_learning", "is_external", "is_paper_method", "same_search_space",
    "search_budget",
)
_CARRY_RESULT: Tuple[str, ...] = (
    "selected_algorithm", "selected_params", "selected_k", "true_k", "k_correct",
    "k_abs_error", "k_signed_error", "best_objective_score", "n_trials_completed",
    "used_fallback", "nmi", "ami", "v_measure", "homogeneity", "completeness",
    "fowlkes_mallows", "purity", "predicted_noise_ratio", "true_noise_ratio",
    "noise_precision", "noise_recall", "noise_f1",
    "selected_metrics", "selected_metric_weights", "selected_metric_utilities",
    "selected_metric_count", "metric_weight_entropy", "top_metric", "cvi_type",
    "dynamic_topk_policy", "run_dir", "trace_path", "provenance_mismatch",
)

BEST_VIEW_EXTRA_COLUMNS: Tuple[str, ...] = (
    "best_view_id", "best_view_ari", "best_view_runtime_sec",
    "total_runtime_all_views_sec", "view_ari_mean", "view_ari_median",
    "view_ari_std", "view_ari_min", "view_ari_max", "view_gap",
    "n_views_attempted", "n_views_success", "n_success", "has_success",
    "view_tie_count", "view_tie_broken_by",
    "best_view_fit_predict_sec", "best_view_cvi_eval_sec",
    "total_fit_predict_all_views_sec", "total_cvi_eval_all_views_sec",
    "uses_view_oracle", "selection_type", "paper_claim_level",
    "has_best_view_runtime", "has_total_all_views_runtime",
    "has_end_to_end_runtime", "has_runtime_components",
    "has_metric_selection", "has_trace",
)

# Deterministic view order used for tie-breaking and for the all-views sum.
_VIEW_RANK = {v: i for i, v in enumerate(VIEW_IDS)}

# End-to-end runtime components (meta-feature extraction, utility prediction,
# KNN query, ...) are NOT recorded per run by any method in this experiment set.
# The flag is therefore constant False; the value must never be synthesised.
HAS_END_TO_END_RUNTIME = False


# --------------------------------------------------------------------------- #
# Best-view frame
# --------------------------------------------------------------------------- #
def build_best_view_frame(run_frame: pd.DataFrame, *, verbose: bool = True
                          ) -> pd.DataFrame:
    """Collapse the run frame to one row per ``dataset x method``."""
    if run_frame.empty:
        return pd.DataFrame(columns=list(
            _CARRY_IDENTITY + _CARRY_PROVENANCE + _CARRY_RESULT
            + BEST_VIEW_EXTRA_COLUMNS))

    work = run_frame.copy()
    work["_ari"] = pd.to_numeric(work["ari"], errors="coerce")
    work["_runtime"] = pd.to_numeric(work["runtime_sec"], errors="coerce")
    work["_fit"] = pd.to_numeric(work.get("fit_predict_sec"), errors="coerce")
    work["_cvi"] = pd.to_numeric(work.get("cvi_eval_sec"), errors="coerce")
    work["_view_rank"] = work["view_id"].map(_VIEW_RANK).fillna(99).astype(int)
    work["_ok"] = work["status"].eq(STATUS_SUCCESS) & work["_ari"].notna()

    rows: List[Dict[str, Any]] = []
    for (_dataset_id, _method), grp in work.groupby(["dataset_id", "method_name"],
                                                    sort=False, dropna=False):
        rows.append(_best_view_row(grp))

    frame = pd.DataFrame(rows)
    order = [c for c in (_CARRY_IDENTITY + _CARRY_PROVENANCE + _CARRY_RESULT
                         + BEST_VIEW_EXTRA_COLUMNS) if c in frame.columns]
    frame = frame[order]
    if verbose:
        n_ok = int(frame["has_success"].sum())
        print(f"[best-view] {len(frame)} dataset x method rows "
              f"({n_ok} with >=1 successful view, "
              f"{len(frame) - n_ok} with none)", flush=True)
    return frame.reset_index(drop=True)


def _best_view_row(grp: pd.DataFrame) -> Dict[str, Any]:
    """One best-view row from a ``dataset x method`` group of view rows.

    Tie-break for equal ARI (documented and deterministic):
    1. higher ARI;
    2. lower runtime of that view;
    3. canonical view order ``x_only < y_only < xy_2d``.
    """
    ok = grp[grp["_ok"]]
    ari = ok["_ari"]
    n_attempt = int(len(grp))
    n_ok = int(len(ok))
    total_runtime = float(grp["_runtime"].sum(skipna=True)) if n_attempt else float("nan")
    total_fit = float(grp["_fit"].sum(skipna=True)) if grp["_fit"].notna().any() else float("nan")
    total_cvi = float(grp["_cvi"].sum(skipna=True)) if grp["_cvi"].notna().any() else float("nan")

    method_name = str(grp["method_name"].iloc[0])
    spec = METHOD_BY_NAME.get(method_name)
    base: Dict[str, Any] = {}
    src = ok.iloc[0] if n_ok else grp.iloc[0]
    for col in _CARRY_IDENTITY + _CARRY_PROVENANCE:
        if col in grp.columns:
            base[col] = src[col]

    common = {
        "total_runtime_all_views_sec": total_runtime,
        "total_fit_predict_all_views_sec": total_fit,
        "total_cvi_eval_all_views_sec": total_cvi,
        "n_views_attempted": n_attempt,
        "n_views_success": n_ok,
        "n_success": n_ok,
        "has_success": n_ok > 0,
        "uses_view_oracle": True,
        "selection_type": "fixed_variant",
        "paper_claim_level": "fixed_observed",
        "has_best_view_runtime": bool(n_ok and ok["_runtime"].notna().any()),
        "has_total_all_views_runtime": bool(
            grp["_runtime"].notna().sum() == n_attempt and n_attempt > 0),
        "has_end_to_end_runtime": HAS_END_TO_END_RUNTIME,
        "has_runtime_components": bool(grp["_fit"].notna().any()
                                       and grp["_cvi"].notna().any()),
        "has_metric_selection": bool(spec.metric_selection_capable) if spec else False,
        "has_trace": bool(spec.trace_best_capable) if spec else False,
    }

    if not n_ok:
        # No successful view: keep the row so coverage accounting is complete.
        nulls = {c: None for c in _CARRY_RESULT if c in grp.columns}
        nulls.update({
            "best_view_id": None, "best_view_ari": np.nan,
            "best_view_runtime_sec": np.nan,
            "best_view_fit_predict_sec": np.nan, "best_view_cvi_eval_sec": np.nan,
            "view_ari_mean": np.nan, "view_ari_median": np.nan,
            "view_ari_std": np.nan, "view_ari_min": np.nan, "view_ari_max": np.nan,
            "view_gap": np.nan, "view_tie_count": 0, "view_tie_broken_by": "",
        })
        # Preserve the failure reason from the first attempted view.
        nulls["provenance_mismatch"] = grp["provenance_mismatch"].iloc[0] \
            if "provenance_mismatch" in grp.columns else ""
        return {**base, **nulls, **common}

    top = float(ari.max())
    tied = ok[np.isclose(ok["_ari"], top, rtol=0.0, atol=0.0)]
    tie_count = int(len(tied))
    if tie_count > 1:
        ranked = tied.sort_values(
            ["_runtime", "_view_rank"], ascending=[True, True], kind="mergesort")
        tie_reason = ("runtime" if ranked["_runtime"].notna().any()
                      and ranked["_runtime"].nunique() > 1 else "view_order")
        best = ranked.iloc[0]
    else:
        tie_reason = ""
        best = tied.iloc[0]

    result = {c: best[c] for c in _CARRY_RESULT if c in grp.columns}
    view_mean = float(ari.mean())
    return {
        **base, **result,
        "best_view_id": best["view_id"],
        "best_view_ari": float(best["_ari"]),
        "best_view_runtime_sec": (float(best["_runtime"])
                                  if pd.notna(best["_runtime"]) else np.nan),
        "best_view_fit_predict_sec": (float(best["_fit"])
                                      if pd.notna(best["_fit"]) else np.nan),
        "best_view_cvi_eval_sec": (float(best["_cvi"])
                                   if pd.notna(best["_cvi"]) else np.nan),
        "view_ari_mean": view_mean,
        "view_ari_median": float(ari.median()),
        "view_ari_std": float(ari.std(ddof=1)) if n_ok > 1 else 0.0,
        "view_ari_min": float(ari.min()),
        "view_ari_max": float(ari.max()),
        "view_gap": float(best["_ari"]) - view_mean,
        "view_tie_count": tie_count,
        "view_tie_broken_by": tie_reason,
        **common,
    }


# --------------------------------------------------------------------------- #
# Dataset-level frame
# --------------------------------------------------------------------------- #
def build_dataset_level_frame(best_view: pd.DataFrame, *,
                              methods: Optional[Sequence[str]] = None
                              ) -> pd.DataFrame:
    """Best-view frame + per-dataset cross-method context.

    ``methods`` restricts the *comparison set* used for the rank/delta columns
    (they are only meaningful relative to an explicit candidate set). The
    returned frame still contains one row per input ``dataset x method``.
    """
    if best_view.empty:
        return best_view.copy()
    out = best_view.copy()
    pool = out if methods is None else out[out["method_name"].isin(list(methods))]
    pool_size = int(pool["method_name"].nunique())

    ari = pd.to_numeric(pool["best_view_ari"], errors="coerce")
    agg = pool.assign(_ari=ari).groupby("dataset_id")["_ari"].agg(
        dataset_best_ari="max", dataset_mean_ari="mean",
        dataset_median_ari="median", dataset_worst_ari="min",
        dataset_n_candidates="count")
    out = out.merge(agg, left_on="dataset_id", right_index=True, how="left")
    out["comparison_pool_size"] = pool_size
    out["delta_to_dataset_best"] = (
        pd.to_numeric(out["best_view_ari"], errors="coerce") - out["dataset_best_ari"])
    out["is_dataset_best"] = np.isclose(
        out["delta_to_dataset_best"].fillna(-np.inf), 0.0, atol=1e-12)
    # Rank within the comparison pool (1 = best); non-pool rows stay null.
    in_pool = out["method_name"].isin(pool["method_name"].unique())
    ranks = (out.assign(_ari=pd.to_numeric(out["best_view_ari"], errors="coerce"))
             .where(in_pool)
             .groupby("dataset_id")["_ari"]
             .rank(ascending=False, method="min"))
    out["rank_within_pool"] = ranks.where(in_pool)
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Representative frame
# --------------------------------------------------------------------------- #
REPRESENTATIVE_META_COLUMNS: Tuple[str, ...] = (
    "representative_id", "selected_method_name", "selected_short_label",
    "selection_type", "selection_scope", "selection_metric", "selection_split",
    "candidate_pool_id", "candidate_pool_size", "is_independently_preselected",
    "paper_claim_level", "uses_variant_oracle",
)


def build_representative_frame(
    best_view: pd.DataFrame,
    *,
    representative_id: str,
    selection: Dict[str, Any] | Sequence[Dict[str, Any]],
    selection_scope: str,
    scope_column: str = "family",
) -> pd.DataFrame:
    """One row per ``dataset x representative_id``.

    ``selection`` is a single selection record (``selection_scope='global'``) or a
    sequence of per-scope records (``selection_scope='family'``), each with at
    least ``selected_method``, ``candidate_pool_id``, ``candidate_pool_size`` and
    -- for family scope -- the scope key under ``scope_column``.

    A *global* representative yields the same ``selected_method_name`` on every
    dataset; a *family* representative varies only across ``scope_column`` values.
    Both are labelled ``retrospective_global`` / ``retrospective_family`` -- never
    "preselected" or "deployable" (see ``terminology.md``).
    """
    if best_view.empty:
        return best_view.copy()
    records = [selection] if isinstance(selection, dict) else list(selection)
    if not records:
        return best_view.iloc[0:0].copy()

    sel_type = ("retrospective_global" if selection_scope == "global"
                else "retrospective_family")
    parts: List[pd.DataFrame] = []
    for rec in records:
        method = str(rec["selected_method"])
        rows = best_view[best_view["method_name"] == method].copy()
        if selection_scope != "global":
            key = rec.get(scope_column)
            rows = rows[rows[scope_column] == key]
        if rows.empty:
            continue
        rows["representative_id"] = representative_id
        rows["selected_method_name"] = method
        rows["selected_short_label"] = rows["method_short_label"]
        rows["selection_type"] = sel_type
        rows["selection_scope"] = selection_scope
        rows["selection_metric"] = rec.get("selection_metric", "best_view_mean_ari")
        rows["selection_split"] = rec.get("selection_split")
        rows["candidate_pool_id"] = rec.get("candidate_pool_id")
        rows["candidate_pool_size"] = rec.get("candidate_pool_size")
        rows["is_independently_preselected"] = False
        rows["paper_claim_level"] = "retrospective_summary"
        rows["uses_variant_oracle"] = True
        parts.append(rows)
    if not parts:
        return best_view.iloc[0:0].copy()
    return pd.concat(parts, ignore_index=True)


# --------------------------------------------------------------------------- #
# Frame validation
# --------------------------------------------------------------------------- #
def validate_frames(run_frame: pd.DataFrame, best_view: pd.DataFrame,
                    *, expected_views: int = len(VIEW_IDS)) -> List[str]:
    """Structural checks on the canonical frames (empty list == valid)."""
    problems: List[str] = []
    if run_frame.empty:
        problems.append("run_frame is empty")
        return problems

    dupes = run_frame.duplicated(subset=["dataset_id", "method_name", "view_id"])
    if bool(dupes.any()):
        problems.append(f"run_frame has {int(dupes.sum())} duplicate "
                        f"(dataset, method, view) rows")

    n_expected = (run_frame["dataset_id"].nunique()
                  * run_frame["method_name"].nunique() * expected_views)
    if len(run_frame) != n_expected:
        problems.append(f"run_frame has {len(run_frame)} rows, expected "
                        f"{n_expected} (datasets x methods x {expected_views} views)")

    if best_view.empty:
        problems.append("best_view_frame is empty")
        return problems

    dupes = best_view.duplicated(subset=["dataset_id", "method_name"])
    if bool(dupes.any()):
        problems.append(f"best_view_frame has {int(dupes.sum())} duplicate "
                        f"(dataset, method) rows")

    n_pairs = run_frame.groupby(["dataset_id", "method_name"]).ngroups
    if len(best_view) != n_pairs:
        problems.append(f"best_view_frame has {len(best_view)} rows but the run "
                        f"frame has {n_pairs} (dataset, method) pairs")

    # Best-View ARI must equal the max successful view ARI, per pair.
    ok = run_frame[run_frame["status"] == STATUS_SUCCESS].copy()
    ok["_ari"] = pd.to_numeric(ok["ari"], errors="coerce")
    truth = ok.groupby(["dataset_id", "method_name"])["_ari"].max()
    got = best_view.set_index(["dataset_id", "method_name"])["best_view_ari"]
    joined = pd.concat([truth.rename("truth"), got.rename("got")], axis=1,
                       join="inner")
    bad = joined[~np.isclose(joined["truth"], joined["got"], equal_nan=True)]
    if len(bad):
        problems.append(f"{len(bad)} best_view_ari values differ from the max "
                        f"successful view ARI (e.g. {bad.head(3).to_dict()})")

    # Best-View ARI must never be below any individual view's ARI.
    per_view_max = truth.reindex(joined.index)
    if bool((joined["got"] < per_view_max - 1e-12).any()):
        problems.append("best_view_ari below the per-view maximum for some rows")

    # Rows without a successful view must have a null Best-View ARI.
    no_succ = best_view[~best_view["has_success"].astype(bool)]
    if len(no_succ) and bool(pd.to_numeric(no_succ["best_view_ari"],
                                          errors="coerce").notna().any()):
        problems.append("rows with no successful view carry a non-null "
                        "best_view_ari")

    if bool((~best_view["uses_view_oracle"].astype(bool)).any()):
        problems.append("best_view_frame has rows with uses_view_oracle != True")

    return problems


# --------------------------------------------------------------------------- #
# Summaries over the best-view frame
# --------------------------------------------------------------------------- #
PRIMARY_SUMMARY_COLUMNS: Tuple[str, ...] = (
    "n_datasets", "n_success", "success_rate", "failure_rate",
    "best_view_mean_ari", "best_view_median_ari", "best_view_std_ari",
    "best_view_q25_ari", "best_view_q75_ari", "best_view_min_ari",
    "best_view_max_ari", "best_view_k_accuracy", "best_view_mean_k_abs_error",
    "best_view_median_k_abs_error", "best_view_overcluster_rate",
    "best_view_undercluster_rate", "best_view_exact_k_rate",
    # PRIMARY runtime: obtaining a Best-View result requires running all three
    # views, so the all-view total is the operational cost.
    "mean_total_runtime_all_views_sec", "median_total_runtime_all_views_sec",
    "std_total_runtime_all_views_sec", "p90_total_runtime_all_views_sec",
    "sum_total_runtime_all_views_sec",
    # DIAGNOSTIC runtime: the single view whose ARI was reported. Never the
    # headline runtime in an operational comparison.
    "mean_selected_best_view_runtime_sec", "median_selected_best_view_runtime_sec",
    # Retained aliases so the selection tie-break chain and earlier manifests
    # keep resolving.
    "total_runtime_all_views_mean_sec", "total_runtime_all_views_median_sec",
    "total_runtime_all_views_sum_sec", "best_view_runtime_mean_sec",
    "best_view_runtime_median_sec", "all_views_mean_ari", "all_views_median_ari",
    "all_views_std_ari", "mean_view_gap", "mean_best_view_ari_x_only_share",
    "mean_nmi", "mean_ami", "mean_v_measure", "mean_purity",
    "mean_selected_metric_count", "mean_metric_weight_entropy",
    "has_total_all_views_runtime", "has_runtime_components",
    "has_metric_selection", "has_trace",
)


def _num(series: Optional[pd.Series]) -> pd.Series:
    if series is None:
        return pd.Series(dtype="float64")
    return pd.to_numeric(series, errors="coerce")


def summary_block(grp: pd.DataFrame) -> Dict[str, Any]:
    """The canonical primary-metric block for one group of best-view rows."""
    n = int(len(grp))
    ok = grp[grp["has_success"].astype("boolean").fillna(False)]
    n_ok = int(len(ok))
    ari = _num(ok.get("best_view_ari")).dropna()
    total_rt = _num(ok.get("total_runtime_all_views_sec")).dropna()
    bv_rt = _num(ok.get("best_view_runtime_sec")).dropna()
    k_signed = _num(ok.get("k_signed_error"))
    n_k = int(k_signed.notna().sum())
    k_correct = ok.get("k_correct")

    def q(series: pd.Series, fn, *args) -> float:
        return float(fn(series, *args)) if len(series) else float("nan")

    block: Dict[str, Any] = {
        "n_datasets": n,
        "n_success": n_ok,
        "success_rate": float(n_ok / n) if n else float("nan"),
        "failure_rate": float((n - n_ok) / n) if n else float("nan"),
        "best_view_mean_ari": q(ari, np.mean),
        "best_view_median_ari": q(ari, np.median),
        "best_view_std_ari": float(ari.std(ddof=1)) if len(ari) > 1 else float("nan"),
        "best_view_q25_ari": q(ari, np.quantile, 0.25),
        "best_view_q75_ari": q(ari, np.quantile, 0.75),
        "best_view_min_ari": q(ari, np.min),
        "best_view_max_ari": q(ari, np.max),
        "best_view_k_accuracy": (
            float(k_correct.dropna().astype(bool).mean())
            if k_correct is not None and int(k_correct.notna().sum()) else float("nan")),
        "best_view_mean_k_abs_error": q(_num(ok.get("k_abs_error")).dropna(), np.mean),
        "best_view_median_k_abs_error": q(_num(ok.get("k_abs_error")).dropna(), np.median),
        "best_view_overcluster_rate": (float((k_signed > 0).sum() / n_k)
                                       if n_k else float("nan")),
        "best_view_undercluster_rate": (float((k_signed < 0).sum() / n_k)
                                        if n_k else float("nan")),
        "best_view_exact_k_rate": (float((k_signed == 0).sum() / n_k)
                                   if n_k else float("nan")),
        # ---- PRIMARY runtime (all three views) --------------------------- #
        "mean_total_runtime_all_views_sec": q(total_rt, np.mean),
        "median_total_runtime_all_views_sec": q(total_rt, np.median),
        "std_total_runtime_all_views_sec": (float(total_rt.std(ddof=1))
                                            if len(total_rt) > 1 else float("nan")),
        "p90_total_runtime_all_views_sec": q(total_rt, np.quantile, 0.90),
        "sum_total_runtime_all_views_sec": (float(total_rt.sum()) if len(total_rt)
                                            else float("nan")),
        # ---- DIAGNOSTIC runtime (selected view only) --------------------- #
        "mean_selected_best_view_runtime_sec": q(bv_rt, np.mean),
        "median_selected_best_view_runtime_sec": q(bv_rt, np.median),
        # ---- retained aliases -------------------------------------------- #
        "total_runtime_all_views_mean_sec": q(total_rt, np.mean),
        "total_runtime_all_views_median_sec": q(total_rt, np.median),
        "total_runtime_all_views_sum_sec": float(total_rt.sum()) if len(total_rt) else float("nan"),
        "best_view_runtime_mean_sec": q(bv_rt, np.mean),
        "best_view_runtime_median_sec": q(bv_rt, np.median),
        "primary_runtime_column": "median_total_runtime_all_views_sec",
        "selected_view_runtime_is_diagnostic_only": True,
        "all_views_mean_ari": q(_num(ok.get("view_ari_mean")).dropna(), np.mean),
        "all_views_median_ari": q(_num(ok.get("view_ari_median")).dropna(), np.median),
        "all_views_std_ari": q(_num(ok.get("view_ari_std")).dropna(), np.mean),
        "mean_view_gap": q(_num(ok.get("view_gap")).dropna(), np.mean),
        "mean_nmi": q(_num(ok.get("nmi")).dropna(), np.mean),
        "mean_ami": q(_num(ok.get("ami")).dropna(), np.mean),
        "mean_v_measure": q(_num(ok.get("v_measure")).dropna(), np.mean),
        "mean_purity": q(_num(ok.get("purity")).dropna(), np.mean),
        "mean_selected_metric_count": q(
            _num(ok.get("selected_metric_count")).dropna(), np.mean),
        "mean_metric_weight_entropy": q(
            _num(ok.get("metric_weight_entropy")).dropna(), np.mean),
    }
    if "best_view_id" in ok.columns and n_ok:
        share = (ok["best_view_id"] == "x_only").mean()
        block["mean_best_view_ari_x_only_share"] = float(share)
    else:
        block["mean_best_view_ari_x_only_share"] = float("nan")
    for flag in ("has_total_all_views_runtime", "has_runtime_components",
                 "has_metric_selection", "has_trace"):
        block[flag] = (bool(grp[flag].astype("boolean").fillna(False).all())
                       if flag in grp.columns and n else False)
    return block


def summarize(best_view: pd.DataFrame, group_cols: Sequence[str] = ("method_name",),
              *, extra_cols: Sequence[str] = ()) -> pd.DataFrame:
    """One primary-metric summary row per unique ``group_cols`` combination.

    ``extra_cols`` are carried through from the first row of each group (used for
    short labels and provenance so plotting/reporting never re-joins them).
    """
    cols = [c for c in group_cols if c in best_view.columns]
    if best_view.empty or not cols:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for key, grp in best_view.groupby(cols, dropna=False, sort=True):
        key_tuple = key if isinstance(key, tuple) else (key,)
        row: Dict[str, Any] = dict(zip(cols, key_tuple))
        for col in extra_cols:
            if col in grp.columns:
                row[col] = grp[col].iloc[0]
        row.update(summary_block(grp))
        rows.append(row)
    out = pd.DataFrame(rows)
    sort_cols = [c for c in cols if c != "method_name"]
    if "best_view_mean_ari" in out.columns:
        out = out.sort_values(sort_cols + ["best_view_mean_ari"],
                             ascending=[True] * len(sort_cols) + [False],
                             kind="mergesort")
    return out.reset_index(drop=True)


DEFAULT_EXTRA_COLS: Tuple[str, ...] = (
    "method_short_label", "paper_label", "method_family", "framework",
    "utility_source", "k_policy", "weighting_mode", "metric_inventory",
    "oracle_level", "is_paper_method", "same_search_space", "search_budget",
    "selection_type", "paper_claim_level",
)


def summarize_methods(best_view: pd.DataFrame,
                      group_cols: Sequence[str] = ("method_name",)) -> pd.DataFrame:
    """:func:`summarize` with the standard provenance columns carried through."""
    return summarize(best_view, group_cols, extra_cols=DEFAULT_EXTRA_COLS)
