"""Gap decomposition: where the ARI actually goes missing.

Seven distinct, separately-named gaps (§18). Signs are consistent throughout:
**a positive gap always means "there was something better available"**.

======================  ==================================================  ============
gap                     definition                                          availability
======================  ==================================================  ============
search-selection        ``trace_best_ari - selected_ari`` per run           ClustOpt only
view                    ``best_view_ari - view_ari_mean``                   all methods
utility-prediction      ``real ARI - {MLP,KNN} ARI`` at matched policy      ClustOpt
weighting               ``raw ARI - softmax ARI`` at matched policy         ClustOpt
metric-count            ``dynamic_realK ARI - dynamic_predictK ARI``        MLP, KNN
policy / VBS            ``portfolio VBS ARI - representative ARI``          all methods
external-leader         ``external ARI - ClustOpt representative ARI``      all methods
======================  ==================================================  ============

The search-selection gap is preserved from the previous implementation (it is the
most expensive analysis and the logic was already correct); it needs the
per-candidate ``clustopt_results.csv`` trace and is therefore unavailable for the
external methods. The capability preflight skips it for them and the report says
so, rather than emitting nulls without explanation.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import paths
from .collector import STATUS_SUCCESS
from .method_registry import METHOD_BY_NAME, VARIANT_FAMILIES, short_label

EPS = 1e-9

GAP_TYPES: Tuple[str, ...] = (
    "search_selection_gap", "view_gap", "utility_prediction_gap",
    "weighting_gap", "metric_count_gap", "policy_vbs_gap",
    "external_leader_gap",
)

GAP_DEFINITIONS: Dict[str, str] = {
    "search_selection_gap":
        "trace_best_ari - selected_ari: the search found a better partition but "
        "the internal objective did not select it (objective failure, not "
        "search-space failure). Requires the per-candidate ClustOpt trace.",
    "view_gap":
        "best_view_ari - view_ari_mean: how much of the Best-View result depends "
        "on picking the right representation. Best-View is a view oracle.",
    "utility_prediction_gap":
        "real-utility ARI - learned-utility ARI at a matched metric-count policy "
        "and matched weighting: the cost of replacing real utility with a "
        "predictor.",
    "weighting_gap":
        "raw ARI - softmax ARI at a matched utility source and metric-count "
        "policy: the effect of the weighting transform alone.",
    "metric_count_gap":
        "dynamic_realK ARI - dynamic_predictK ARI: how much of the Dynamic loss "
        "is caused by predicting the metric count. The Real-K arm is "
        "oracle-assisted and is a diagnostic, not a deployable method.",
    "policy_vbs_gap":
        "portfolio VBS ARI - selected single representative ARI: the "
        "policy-selection headroom. The VBS is a non-deployable oracle.",
    "external_leader_gap":
        "external method ARI - ClustOpt representative ARI: signed lead of an "
        "external competitor over the retrospectively selected ClustOpt "
        "representative.",
}


# --------------------------------------------------------------------------- #
# 18.1 Search-selection gap (per-candidate trace)
# --------------------------------------------------------------------------- #
_ARI_COLS = ("ARI", "ari", "external_ari", "adjusted_rand_index")
_OBJ_COLS = ("aggregate_score", "score", "objective", "best_objective_score",
             "aggregate_objective", "cvi_score")
_VALID_COLS = ("valid",)

TRACE_COLUMNS: Tuple[str, ...] = (
    "trace_best_ari", "selected_ari", "search_selection_gap",
    "relative_search_selection_gap", "selected_is_trace_best",
    "trace_best_rank_by_internal_objective", "selected_rank_by_ari",
    "n_trace_candidates", "trace_best_selected_algorithm",
    "trace_best_internal_objective",
)


def _find_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def trace_gap_for_run(trace_path: Optional[str], *, selected_ari: Optional[float]
                      ) -> Tuple[Dict[str, Any], Optional[str]]:
    """Search-selection gap for one run. Returns ``(fields, warning-or-None)``."""
    nan_fields: Dict[str, Any] = {c: float("nan") for c in TRACE_COLUMNS}
    nan_fields["trace_best_selected_algorithm"] = None
    if not trace_path or not paths.exists(trace_path):
        return nan_fields, "missing_trace"
    try:
        trace = pd.read_csv(paths.ext(trace_path))
    except Exception as exc:
        return nan_fields, f"unreadable_trace:{type(exc).__name__}"
    if trace.empty:
        return nan_fields, "empty_trace"

    ari_col = _find_col(trace, _ARI_COLS)
    if ari_col is None:
        return nan_fields, "trace_has_no_ari_column"
    obj_col = _find_col(trace, _OBJ_COLS)

    work = trace.copy()
    work["_ari"] = pd.to_numeric(work[ari_col], errors="coerce")
    work["_obj"] = (pd.to_numeric(work[obj_col], errors="coerce")
                    if obj_col is not None else np.nan)
    valid_col = _find_col(work, _VALID_COLS)
    if valid_col is not None:
        mask = work[valid_col].astype("object").apply(
            lambda v: str(v).strip().lower() in ("true", "1", "1.0", "yes"))
        if int(mask.sum()) > 0:
            work = work[mask]

    cand = work[np.isfinite(work["_ari"])].reset_index(drop=True)
    if cand.empty:
        return nan_fields, "trace_has_no_finite_ari"

    best_idx = int(cand["_ari"].idxmax())
    trace_best = float(cand.loc[best_idx, "_ari"])
    sel_idx = int(cand["_obj"].idxmax()) if cand["_obj"].notna().any() else None
    sel_ari = (float(selected_ari)
               if selected_ari is not None and np.isfinite(selected_ari)
               else (float(cand.loc[sel_idx, "_ari"]) if sel_idx is not None
                     else float("nan")))

    gap = trace_best - sel_ari if np.isfinite(sel_ari) else float("nan")
    algo_col = _find_col(cand, ("algorithm", "selected_algorithm"))
    return {
        "trace_best_ari": trace_best,
        "selected_ari": sel_ari,
        "search_selection_gap": gap,
        "relative_search_selection_gap": (gap / max(abs(trace_best), EPS)
                                          if np.isfinite(gap) else float("nan")),
        "selected_is_trace_best": (float(abs(gap) <= EPS) if np.isfinite(gap)
                                   else float("nan")),
        "trace_best_rank_by_internal_objective": (
            float(cand["_obj"].rank(ascending=False, method="min").iloc[best_idx])
            if cand["_obj"].notna().any() else float("nan")),
        "selected_rank_by_ari": (
            float(cand["_ari"].rank(ascending=False, method="min").iloc[sel_idx])
            if sel_idx is not None else float("nan")),
        "n_trace_candidates": float(len(cand)),
        "trace_best_selected_algorithm": (str(cand.loc[best_idx, algo_col])
                                          if algo_col is not None else None),
        "trace_best_internal_objective": (float(cand.loc[best_idx, "_obj"])
                                          if cand["_obj"].notna().any()
                                          else float("nan")),
    }, None


def compute_search_selection_gaps(
    run_frame: pd.DataFrame, *, method_names: Optional[Sequence[str]] = None,
    max_workers: Optional[int] = None, verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Per-run search-selection gap for every trace-capable successful run."""
    key_cols = ["dataset_id", "split_id", "view_id", "method_name"]
    capable = {name for name, spec in METHOD_BY_NAME.items()
               if spec.trace_best_capable}
    if method_names is not None:
        capable &= set(method_names)
    ok = run_frame[(run_frame["status"] == STATUS_SUCCESS)
                   & run_frame["method_name"].isin(capable)]
    if ok.empty:
        return pd.DataFrame(columns=key_cols + list(TRACE_COLUMNS)), {}

    rows: List[Dict[str, Any]] = []
    warnings: Dict[str, int] = {}
    records = [r for _, r in ok.iterrows()]
    n = len(records)

    def _one(r: pd.Series) -> Tuple[Dict[str, Any], Optional[str]]:
        sel = pd.to_numeric(pd.Series([r.get("ari")]), errors="coerce").iloc[0]
        fields, warn = trace_gap_for_run(r.get("trace_path"), selected_ari=sel)
        return ({k: r.get(k) for k in key_cols} | fields), warn

    if max_workers and int(max_workers) > 1:
        done = 0
        with ThreadPoolExecutor(max_workers=int(max_workers)) as pool:
            for row, warn in pool.map(_one, records):
                rows.append(row)
                if warn:
                    warnings[warn] = warnings.get(warn, 0) + 1
                done += 1
                if verbose and done % 5000 == 0:
                    print(f"[gap:search] {done}/{n} runs...", flush=True)
    else:
        for i, r in enumerate(records, start=1):
            row, warn = _one(r)
            rows.append(row)
            if warn:
                warnings[warn] = warnings.get(warn, 0) + 1
            if verbose and i % 5000 == 0:
                print(f"[gap:search] {i}/{n} runs...", flush=True)

    table = pd.DataFrame(rows, columns=key_cols + list(TRACE_COLUMNS))
    if verbose:
        print(f"[gap:search] {len(table)} runs; warnings={warnings or 'none'}",
              flush=True)
    return table, warnings


def attach_search_selection_gap(best_view: pd.DataFrame, trace_table: pd.DataFrame
                                ) -> pd.DataFrame:
    """Merge each best-view row's own run's search-selection gap onto it."""
    if best_view.empty or trace_table.empty:
        out = best_view.copy()
        for c in TRACE_COLUMNS:
            out.setdefault(c, np.nan)
        return out
    tt = trace_table.rename(columns={"view_id": "best_view_id"})
    keys = ["dataset_id", "best_view_id", "method_name"]
    keep = keys + [c for c in TRACE_COLUMNS if c in tt.columns]
    return best_view.merge(tt[keep].drop_duplicates(subset=keys), on=keys, how="left")


# --------------------------------------------------------------------------- #
# Matched paired gaps (18.3 - 18.5)
# --------------------------------------------------------------------------- #
def _paired(best_view: pd.DataFrame, method_a: str, method_b: str
            ) -> pd.DataFrame:
    """Datasets where both methods have a Best-View ARI, aligned by dataset_id."""
    cols = ["dataset_id", "family", "subfamily", "difficulty", "cluster_count",
            "best_view_ari"]
    a = best_view.loc[best_view["method_name"] == method_a,
                      [c for c in cols if c in best_view.columns]]
    b = best_view.loc[best_view["method_name"] == method_b,
                      ["dataset_id", "best_view_ari"]]
    merged = a.merge(b, on="dataset_id", suffixes=("_a", "_b"))
    merged["ari_a"] = pd.to_numeric(merged["best_view_ari_a"], errors="coerce")
    merged["ari_b"] = pd.to_numeric(merged["best_view_ari_b"], errors="coerce")
    return merged.dropna(subset=["ari_a", "ari_b"])


def paired_gap_row(best_view: pd.DataFrame, *, gap_type: str, reference: str,
                   comparison: str, group: Optional[Dict[str, Any]] = None,
                   notes: str = "") -> Dict[str, Any]:
    """``reference - comparison`` paired gap summary (one row).

    ``reference`` is the arm expected to be *better* under the gap's definition,
    so a positive gap means the comparison arm lost something.
    """
    merged = _paired(best_view, reference, comparison)
    delta = merged["ari_a"] - merged["ari_b"] if len(merged) else pd.Series(dtype=float)
    ref_spec, cmp_spec = METHOD_BY_NAME.get(reference), METHOD_BY_NAME.get(comparison)
    row: Dict[str, Any] = dict(group or {})
    row.update({
        "gap_type": gap_type,
        "gap_definition": GAP_DEFINITIONS.get(gap_type, ""),
        "reference_method": reference,
        "reference_short_label": short_label(reference),
        "comparison_method": comparison,
        "comparison_short_label": short_label(comparison),
        "n_paired": int(len(delta)),
        "reference_mean_ari": float(merged["ari_a"].mean()) if len(merged) else float("nan"),
        "comparison_mean_ari": float(merged["ari_b"].mean()) if len(merged) else float("nan"),
        "gap_mean": float(delta.mean()) if len(delta) else float("nan"),
        "gap_median": float(delta.median()) if len(delta) else float("nan"),
        "gap_std": float(delta.std(ddof=1)) if len(delta) > 1 else float("nan"),
        "gap_q25": float(delta.quantile(0.25)) if len(delta) else float("nan"),
        "gap_q75": float(delta.quantile(0.75)) if len(delta) else float("nan"),
        "share_reference_better": (float((delta > 1e-12).mean()) if len(delta)
                                   else float("nan")),
        "share_equal": (float(np.isclose(delta, 0.0, atol=1e-12).mean())
                        if len(delta) else float("nan")),
        "share_comparison_better": (float((delta < -1e-12).mean()) if len(delta)
                                    else float("nan")),
        "reference_oracle_level": ref_spec.oracle_level if ref_spec else "",
        "comparison_oracle_level": cmp_spec.oracle_level if cmp_spec else "",
        "reference_uses_real_utility": bool(ref_spec.uses_real_utility) if ref_spec else False,
        "reference_is_oracle_assisted": bool(ref_spec.is_oracle_assisted) if ref_spec else False,
        "paper_claim_level": ("diagnostic_oracle"
                              if (ref_spec and (ref_spec.uses_real_utility
                                                or ref_spec.is_oracle_assisted))
                              else "fixed_observed"),
        "notes": notes,
    })
    return row


def utility_prediction_gaps(best_view: pd.DataFrame,
                            group_cols: Sequence[str] = ()) -> pd.DataFrame:
    """Real-utility vs each learned predictor at every matched policy+weighting."""
    rows: List[Dict[str, Any]] = []
    real_by_key = {(f.policy_id, w): getattr(f, f"{w}_method")
                   for f in VARIANT_FAMILIES if f.utility_source == "real"
                   for w in ("raw", "softmax")}
    for fam in VARIANT_FAMILIES:
        if fam.utility_source == "real":
            continue
        # Match Dynamic Predict-K and Dynamic Real-K both against Real Dynamic.
        policy_key = "dynamic" if fam.policy_id.startswith("dynamic") else fam.policy_id
        for weighting, attr in (("raw", "raw_method"), ("softmax", "softmax_method")):
            reference = real_by_key.get((policy_key, weighting))
            comparison = getattr(fam, attr)
            if reference is None:
                continue
            for grp_vals, sub in _iter_groups(best_view, group_cols):
                rows.append(paired_gap_row(
                    sub, gap_type="utility_prediction_gap", reference=reference,
                    comparison=comparison,
                    group={**grp_vals, "utility_source": fam.utility_source,
                           "policy_id": fam.policy_id,
                           "policy_label": fam.policy_label,
                           "weighting": weighting,
                           "oracle_assisted_comparison": fam.oracle_assisted},
                    notes=("reference uses real (oracle) utility; the gap is the "
                           "cost of substituting a learned predictor")))
    return pd.DataFrame(rows)


def weighting_gaps(best_view: pd.DataFrame,
                   group_cols: Sequence[str] = ()) -> pd.DataFrame:
    """Raw vs softmax at every matched (utility source x metric-count policy)."""
    rows: List[Dict[str, Any]] = []
    for fam in VARIANT_FAMILIES:
        for grp_vals, sub in _iter_groups(best_view, group_cols):
            rows.append(paired_gap_row(
                sub, gap_type="weighting_gap", reference=fam.raw_method,
                comparison=fam.softmax_method,
                group={**grp_vals, "variant_family": fam.family_id,
                       "utility_source": fam.utility_source,
                       "policy_id": fam.policy_id,
                       "policy_label": fam.policy_label,
                       "oracle_assisted": fam.oracle_assisted},
                notes="positive gap favours raw weights; the sign convention is "
                      "raw - softmax, so a negative value means softmax wins"))
    return pd.DataFrame(rows)


def metric_count_gaps(best_view: pd.DataFrame,
                      group_cols: Sequence[str] = ()) -> pd.DataFrame:
    """Dynamic Real-K (oracle-assisted) vs Dynamic Predict-K, per source+weighting."""
    rows: List[Dict[str, Any]] = []
    for source, prefix in (("mlp", "regressor"), ("knn", "knn")):
        for weighting, suffix in (("raw", "raw"), ("softmax", "softmax_t05")):
            reference = f"{prefix}_top_dynamic_realK_{suffix}"
            comparison = f"{prefix}_top_dynamic_predictK_{suffix}"
            for grp_vals, sub in _iter_groups(best_view, group_cols):
                rows.append(paired_gap_row(
                    sub, gap_type="metric_count_gap", reference=reference,
                    comparison=comparison,
                    group={**grp_vals, "utility_source": source,
                           "weighting": weighting},
                    notes="reference is ORACLE-ASSISTED (metric count taken from "
                          "the real utility vector); diagnostic only"))
    return pd.DataFrame(rows)


def view_gap_summary(best_view: pd.DataFrame,
                     group_cols: Sequence[str] = ("method_name",)) -> pd.DataFrame:
    """View-oracle dependence per group: ``best_view_ari - view_ari_mean``."""
    if best_view.empty:
        return pd.DataFrame()
    cols = [c for c in group_cols if c in best_view.columns]
    rows: List[Dict[str, Any]] = []
    for key, grp in best_view.groupby(cols, dropna=False, sort=True):
        key_tuple = key if isinstance(key, tuple) else (key,)
        gap = pd.to_numeric(grp["view_gap"], errors="coerce").dropna()
        bv = pd.to_numeric(grp["best_view_ari"], errors="coerce").dropna()
        mean_v = pd.to_numeric(grp["view_ari_mean"], errors="coerce").dropna()
        row: Dict[str, Any] = dict(zip(cols, key_tuple))
        if "method_name" in cols:
            row["method_short_label"] = short_label(str(row["method_name"]))
        row.update({
            "gap_type": "view_gap",
            "gap_definition": GAP_DEFINITIONS["view_gap"],
            "n": int(len(gap)),
            "best_view_mean_ari": float(bv.mean()) if len(bv) else float("nan"),
            "all_views_mean_ari": float(mean_v.mean()) if len(mean_v) else float("nan"),
            "view_gap_mean": float(gap.mean()) if len(gap) else float("nan"),
            "view_gap_median": float(gap.median()) if len(gap) else float("nan"),
            "view_gap_q75": float(gap.quantile(0.75)) if len(gap) else float("nan"),
            "view_gap_max": float(gap.max()) if len(gap) else float("nan"),
            "best_view_x_only_share": (float((grp["best_view_id"] == "x_only").mean())
                                       if "best_view_id" in grp else float("nan")),
            "best_view_y_only_share": (float((grp["best_view_id"] == "y_only").mean())
                                       if "best_view_id" in grp else float("nan")),
            "best_view_xy_2d_share": (float((grp["best_view_id"] == "xy_2d").mean())
                                      if "best_view_id" in grp else float("nan")),
            "paper_claim_level": "diagnostic_oracle",
        })
        rows.append(row)
    return pd.DataFrame(rows)


def search_selection_gap_summary(best_view_with_trace: pd.DataFrame,
                                 group_cols: Sequence[str] = ("method_name",)
                                 ) -> pd.DataFrame:
    """Search-selection gap per group (ClustOpt methods only)."""
    if best_view_with_trace.empty or "search_selection_gap" not in best_view_with_trace:
        return pd.DataFrame()
    cols = [c for c in group_cols if c in best_view_with_trace.columns]
    rows: List[Dict[str, Any]] = []
    for key, grp in best_view_with_trace.groupby(cols, dropna=False, sort=True):
        gap = pd.to_numeric(grp["search_selection_gap"], errors="coerce").dropna()
        if gap.empty:
            continue
        key_tuple = key if isinstance(key, tuple) else (key,)
        row: Dict[str, Any] = dict(zip(cols, key_tuple))
        if "method_name" in cols:
            row["method_short_label"] = short_label(str(row["method_name"]))
        tb = pd.to_numeric(grp["trace_best_ari"], errors="coerce").dropna()
        sel = pd.to_numeric(grp["selected_ari"], errors="coerce").dropna()
        is_best = pd.to_numeric(grp["selected_is_trace_best"], errors="coerce").dropna()
        row.update({
            "gap_type": "search_selection_gap",
            "gap_definition": GAP_DEFINITIONS["search_selection_gap"],
            "n_with_trace": int(len(gap)),
            "mean_trace_best_ari": float(tb.mean()) if len(tb) else float("nan"),
            "mean_selected_ari": float(sel.mean()) if len(sel) else float("nan"),
            "search_selection_gap_mean": float(gap.mean()),
            "search_selection_gap_median": float(gap.median()),
            "search_selection_gap_q75": float(gap.quantile(0.75)),
            "search_selection_gap_max": float(gap.max()),
            "selected_is_trace_best_rate": (float(is_best.mean()) if len(is_best)
                                            else float("nan")),
            "mean_n_trace_candidates": float(
                pd.to_numeric(grp["n_trace_candidates"], errors="coerce").mean()),
            "paper_claim_level": "fixed_observed",
        })
        rows.append(row)
    return pd.DataFrame(rows)


def external_leader_gaps(best_view: pd.DataFrame, *,
                         external_methods: Sequence[str],
                         clustopt_representatives: Dict[str, str],
                         group_cols: Sequence[str] = ()) -> pd.DataFrame:
    """External method vs each ClustOpt retrospective representative."""
    rows: List[Dict[str, Any]] = []
    for ext_method in external_methods:
        for rep_label, rep_method in clustopt_representatives.items():
            for grp_vals, sub in _iter_groups(best_view, group_cols):
                row = paired_gap_row(
                    sub, gap_type="external_leader_gap", reference=ext_method,
                    comparison=rep_method,
                    group={**grp_vals, "representative_label": rep_label},
                    notes="the ClustOpt arm is a RETROSPECTIVE representative "
                          "selected from an evaluated candidate pool on split 1, "
                          "while the external arm is one predefined method")
                row["comparison_selection_type"] = "retrospective_global"
                row["paper_claim_level"] = "retrospective_summary"
                rows.append(row)
    return pd.DataFrame(rows)


def policy_vbs_gap_table(headroom_frame: pd.DataFrame) -> pd.DataFrame:
    """Re-label the VBS headroom table as the canonical policy/VBS gap."""
    if headroom_frame.empty:
        return headroom_frame
    out = headroom_frame.copy()
    out["gap_type"] = "policy_vbs_gap"
    out["gap_definition"] = GAP_DEFINITIONS["policy_vbs_gap"]
    out["gap_mean"] = out["headroom_paired_mean"]
    out["gap_median"] = out["headroom_paired_median"]
    return out


def _iter_groups(best_view: pd.DataFrame, group_cols: Sequence[str]):
    """Yield ``(group values dict, sub-frame)``; no grouping -> one whole-frame pass."""
    cols = [c for c in group_cols if c in best_view.columns]
    if not cols:
        yield {}, best_view
        return
    for key, sub in best_view.groupby(cols, dropna=False, sort=True):
        key_tuple = key if isinstance(key, tuple) else (key,)
        yield dict(zip(cols, key_tuple)), sub


def gap_definitions_frame() -> pd.DataFrame:
    """The gap glossary, shipped alongside every gap table."""
    return pd.DataFrame([
        {"gap_type": g, "definition": GAP_DEFINITIONS[g],
         "sign_convention": "positive = something better was available",
         "availability": _AVAILABILITY[g]}
        for g in GAP_TYPES
    ])


_AVAILABILITY: Dict[str, str] = {
    "search_selection_gap": "ClustOpt methods only (needs the per-candidate trace)",
    "view_gap": "all methods",
    "utility_prediction_gap": "ClustOpt utility-driven variants only",
    "weighting_gap": "ClustOpt utility-driven variants only",
    "metric_count_gap": "MLP and KNN dynamic variants only",
    "policy_vbs_gap": "all methods (per portfolio)",
    "external_leader_gap": "external same-search-space methods vs ClustOpt "
                           "representatives",
}
