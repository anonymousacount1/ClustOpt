"""Grouped summary statistics over the raw_results table.

``summarize_by`` computes the full statistic block (counts, ARI distribution,
mean external metrics, runtime, trials, K accuracy / error / over-under rates)
for an arbitrary set of grouping columns, always including ``method_name``.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _summary_for_group(g: pd.DataFrame) -> dict:
    n = len(g)
    success = g[g["status"] == "success"]
    n_success = len(success)
    ari = _num(success["ari"])
    runtime = _num(success["runtime_sec"])
    trials = _num(success["n_trials_completed"])

    k_correct = success["k_correct"]
    k_abs = _num(success["k_abs_error"])
    k_signed = _num(success["k_signed_error"])
    n_k = int(k_signed.notna().sum())

    def _mean(col: str) -> float:
        return float(_num(success[col]).mean()) if n_success else float("nan")

    block = {
        "count": int(n),
        "n_success": int(n_success),
        "success_rate": float(n_success / n) if n else 0.0,
        "failure_rate": float((g["status"] == "failed").sum() / n) if n else 0.0,
        "mean_ari": float(ari.mean()) if n_success else float("nan"),
        "median_ari": float(ari.median()) if n_success else float("nan"),
        "std_ari": float(ari.std()) if n_success else float("nan"),
        "q25_ari": float(ari.quantile(0.25)) if n_success else float("nan"),
        "q75_ari": float(ari.quantile(0.75)) if n_success else float("nan"),
        "min_ari": float(ari.min()) if n_success else float("nan"),
        "max_ari": float(ari.max()) if n_success else float("nan"),
        "mean_nmi": _mean("nmi"),
        "mean_ami": _mean("ami"),
        "mean_v_measure": _mean("v_measure"),
        "mean_homogeneity": _mean("homogeneity"),
        "mean_completeness": _mean("completeness"),
        "mean_fowlkes_mallows": _mean("fowlkes_mallows"),
        "mean_purity": _mean("purity"),
        "mean_runtime_sec": float(runtime.mean()) if n_success else float("nan"),
        "median_runtime_sec": float(runtime.median()) if n_success else float("nan"),
        "mean_trials_completed": float(trials.mean()) if n_success else float("nan"),
        "k_accuracy": float(k_correct.dropna().astype(bool).mean()) if n_k else float("nan"),
        "mean_k_abs_error": float(k_abs.mean()) if n_k else float("nan"),
        "median_k_abs_error": float(k_abs.median()) if n_k else float("nan"),
        "mean_k_signed_error": float(k_signed.mean()) if n_k else float("nan"),
        "overcluster_rate": float((k_signed > 0).sum() / n_k) if n_k else float("nan"),
        "undercluster_rate": float((k_signed < 0).sum() / n_k) if n_k else float("nan"),
        "exact_k_rate": float((k_signed == 0).sum() / n_k) if n_k else float("nan"),
    }
    block.update(_gap_extras(success, n_success))
    return block


def _gap_extras(success: pd.DataFrame, n_success: int) -> dict:
    """Trace-best gap metrics appended to a summary block when available."""
    if "ari_gap_to_trace_best" not in success.columns or not n_success:
        return {}
    gap = _num(success["ari_gap_to_trace_best"])
    if int(gap.notna().sum()) == 0:
        return {}
    tba = _num(success.get("trace_best_ari"))
    is_best = _num(success.get("selected_is_trace_best"))
    return {
        "mean_trace_best_ari": float(tba.mean()),
        "mean_ari_gap_to_trace_best": float(gap.mean()),
        "median_ari_gap_to_trace_best": float(gap.median()),
        "selected_is_trace_best_rate": float(is_best.mean()),
    }


def summarize_by(raw: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """Return one summary row per unique combination of ``group_cols``."""
    cols = [c for c in group_cols if c in raw.columns]
    if raw.empty or not cols:
        return pd.DataFrame()
    out_rows = []
    for key, g in raw.groupby(cols, dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        row = {c: v for c, v in zip(cols, key_tuple)}
        row.update(_summary_for_group(g))
        out_rows.append(row)
    df = pd.DataFrame(out_rows)
    sort_cols = [c for c in cols if c != "method_name"] + (["method_name"] if "method_name" in cols else [])
    return df.sort_values(sort_cols).reset_index(drop=True) if sort_cols else df


VIEW_ORDER = ["x_only", "y_only", "xy_2d"]


def collapse_best_view(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per (dataset, method): the best (highest-ARI) **successful** view.

    This is the canonical analysis frame for the whole pipeline: every reported
    *mean* is therefore a **mean over best views** (one best view per
    dataset x method), not a mean over all three views. The full per-view picture
    is retained separately in ``raw_results.csv`` and the per-view summary.

    Unlike :func:`best_view_rows` (success-only), this keeps a row even when a
    method has *no* successful view for a dataset -- it carries the original
    failed/missing row -- so ``count`` / ``success_rate`` / ``failure_rate`` in
    the downstream summaries still reflect every attempted dataset x method.
    """
    if raw.empty:
        return raw.copy()
    out_rows: List[pd.Series] = []
    for _, grp in raw.groupby(["dataset_id", "method_name"], dropna=False):
        ari = pd.to_numeric(grp["ari"], errors="coerce")
        succ_mask = grp["status"].eq("success") & ari.notna()
        if bool(succ_mask.any()):
            out_rows.append(grp.loc[ari[succ_mask].idxmax()])
        elif bool(grp["status"].eq("success").any()):
            out_rows.append(grp[grp["status"].eq("success")].iloc[0])
        else:
            out_rows.append(grp.iloc[0])
    return pd.DataFrame(out_rows, columns=raw.columns).reset_index(drop=True)


def best_view_rows(raw: pd.DataFrame, group_cols: List[str] | None = None
                   ) -> pd.DataFrame:
    """Collapse to one row per (group..., dataset, method): the highest-ARI view.

    Returns the full original row of the best (oracle) view per dataset/method,
    so downstream stats (NMI, runtime, K) reflect the selected view. This is the
    same *oracle best-view* selection used by ``dataset_level_selector``; kept
    here so method-level summaries can be computed with :func:`summarize_by`.
    """
    if raw.empty:
        return raw.copy()
    base = [c for c in (group_cols or []) if c in raw.columns]
    keys = base + ["dataset_id", "method_name"]
    succ = raw[raw["status"] == "success"].copy()
    if succ.empty:
        return succ
    succ["_ari"] = pd.to_numeric(succ["ari"], errors="coerce")
    succ = succ.dropna(subset=["_ari"])
    if succ.empty:
        return succ
    best_idx = succ.groupby(keys)["_ari"].idxmax()
    return succ.loc[best_idx].drop(columns=["_ari"])


def best_view_summarize_by(raw: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """Like :func:`summarize_by` but on best-view-collapsed rows.

    ``mean_ari`` in the result is therefore the **Best-View Mean ARI** (primary
    metric). An ``all_views_mean_ari`` column is added for reference, plus
    best-view win counts per view (``best_view_<view>``).
    """
    cols = [c for c in group_cols if c in raw.columns]
    bv = best_view_rows(raw, [c for c in cols if c != "method_name"])
    out = summarize_by(bv, group_cols)
    if out.empty:
        return out
    # mean_ari here is best-view; expose it explicitly and attach all-views mean.
    out = out.rename(columns={"mean_ari": "best_view_mean_ari"})
    out["mean_ari"] = out["best_view_mean_ari"]  # keep 'mean_ari' as primary
    all_views = summarize_by(raw, group_cols)
    if not all_views.empty and "mean_ari" in all_views.columns:
        merge_keys = cols
        out = out.merge(
            all_views[merge_keys + ["mean_ari"]].rename(
                columns={"mean_ari": "all_views_mean_ari"}),
            on=merge_keys, how="left")
    # Best-view win counts per view.
    if not bv.empty and "view_id" in bv.columns:
        vc = (bv.groupby(cols + ["view_id"]).size()
              .unstack(fill_value=0))
        for v in VIEW_ORDER:
            if v not in vc.columns:
                vc[v] = 0
        vc = vc[VIEW_ORDER].rename(
            columns={v: f"best_view_{v}" for v in VIEW_ORDER}).reset_index()
        out = out.merge(vc, on=cols, how="left")
    return out


def standard_summaries(best_view_raw: pd.DataFrame,
                       all_views_raw: pd.DataFrame | None = None) -> dict:
    """Return the standard grouping-level summaries keyed by output filename.

    Every ``*_method_summary`` is computed on the **best-view frame**
    (``best_view_raw`` = one best view per dataset x method), so ``mean_ari`` and
    all other means are **mean-best-view** values -- the single headline mean.
    The per-view ``view_method_summary`` is the only summary that intentionally
    spans all three views (it is *about* the views) and so reads
    ``all_views_raw``.
    """
    av = all_views_raw if all_views_raw is not None else best_view_raw
    return {
        "overall_method_summary": summarize_by(best_view_raw, ["method_name"]),
        "family_method_summary": summarize_by(best_view_raw, ["family", "method_name"]),
        "subfamily_method_summary": summarize_by(best_view_raw, ["subfamily", "method_name"]),
        "difficulty_method_summary": summarize_by(best_view_raw, ["difficulty", "method_name"]),
        "cluster_count_method_summary": summarize_by(best_view_raw, ["cluster_count", "method_name"]),
        "view_method_summary": summarize_by(av, ["view_id", "method_name"]),
        "algorithm_method_summary": summarize_by(best_view_raw, ["selected_algorithm", "method_name"]),
    }
