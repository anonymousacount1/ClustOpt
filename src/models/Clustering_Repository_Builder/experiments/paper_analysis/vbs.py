"""Virtual-Best-Solver (VBS) engine and portfolio-size sensitivity.

For each dataset a portfolio's VBS selects whichever member achieved the highest
**Best-View ARI** on that dataset -- using the dataset's true external ARI. That
makes it a *policy-selection oracle*, stacked on top of the Best-View *view
oracle*. It is a non-deployable upper bound and every emitted row says so
(``selection_type = dataset_vbs``, ``uses_variant_oracle = True``,
``paper_claim_level = upper_bound``).

A VBS is **not a single executable method**: it selects the best member only
after every member has been evaluated. Runtime is therefore reported as two
clearly separate, explicitly named quantities (amendment §3.2):

``selected_member_best_view_runtime_sec``
``selected_member_total_all_views_runtime_sec``
    describe the *method the VBS selected* on that dataset. **Diagnostic only.**
    Neither is the cost of obtaining the VBS result.

``portfolio_total_execution_runtime_sec``
    the cost of running every member of the portfolio across all required views,

        T_portfolio(d) = sum over m in P of T_all_views(d, m)

    This is the only quantity that may be described as the cost of the VBS, and
    it is an oracle-cost diagnostic rather than an operational runtime.

By construction ``portfolio_total_execution_runtime_sec >=
selected_member_total_all_views_runtime_sec`` on every dataset with finite
runtimes; :func:`validate_vbs` asserts it. A VBS point must never appear in an
operational accuracy-runtime Pareto plot as though it were one executable
method.

Tie-break (§5.4, deterministic):
1. higher ``best_view_ari``
2. higher ``view_ari_median`` then ``view_ari_mean`` (more robust across views)
3. lower ``total_runtime_all_views_sec``
4. lexicographically smaller canonical method name
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .frames import summarize_methods
from .method_registry import METHOD_BY_NAME, short_label
from .portfolio_registry import Portfolio, PORTFOLIO_BY_ID, portfolio

VBS_FRAME_COLUMNS: Tuple[str, ...] = (
    "portfolio_id", "portfolio_label", "portfolio_size", "dataset_id", "family",
    "family_id", "subfamily", "difficulty", "cluster_count",
    "winning_method_name", "winning_method_short_label", "winning_view_id",
    "winning_best_view_ari", "winning_k_correct", "winning_selected_k",
    "winning_true_k", "winning_utility_source", "winning_k_policy",
    "winning_weighting_mode", "winning_method_family", "winning_oracle_level",
    "selected_member_best_view_runtime_sec",
    "selected_member_total_all_views_runtime_sec",
    "portfolio_total_execution_runtime_sec",
    "n_candidates_available", "n_candidates_successful", "tie_count",
    "tie_broken_by", "portfolio_mean_ari", "portfolio_median_ari",
    "portfolio_min_ari", "uses_view_oracle", "uses_variant_oracle",
    "selection_type", "paper_claim_level", "contains_real_utility",
    "contains_oracle_dynamic_k", "contains_oracle_assisted_dynamic_k",
    "fully_operational", "oracle_contamination_reason",
)


# --------------------------------------------------------------------------- #
# Dataset-level VBS
# --------------------------------------------------------------------------- #
def build_vbs_frame(best_view: pd.DataFrame, pf: Portfolio | str,
                    *, verbose: bool = False) -> pd.DataFrame:
    """One row per dataset: the portfolio's oracle winner on that dataset."""
    pf = portfolio(pf) if isinstance(pf, str) else pf
    sub = best_view[best_view["method_name"].isin(pf.methods)].copy()
    if sub.empty:
        return pd.DataFrame(columns=list(VBS_FRAME_COLUMNS))

    sub["_ari"] = pd.to_numeric(sub["best_view_ari"], errors="coerce")
    sub["_median"] = pd.to_numeric(sub.get("view_ari_median"), errors="coerce")
    sub["_mean"] = pd.to_numeric(sub.get("view_ari_mean"), errors="coerce")
    sub["_total_rt"] = pd.to_numeric(sub["total_runtime_all_views_sec"], errors="coerce")
    sub["_ok"] = sub["_ari"].notna()

    rows: List[Dict[str, Any]] = []
    for dataset_id, grp in sub.groupby("dataset_id", sort=True):
        rows.append(_vbs_row(dataset_id, grp, pf))
    frame = pd.DataFrame(rows, columns=list(VBS_FRAME_COLUMNS))
    if verbose:
        n_tie = int((frame["tie_count"] > 1).sum())
        print(f"[vbs] {pf.portfolio_id}: {len(frame)} datasets, "
              f"mean best-view ARI={frame['winning_best_view_ari'].mean():.4f}, "
              f"{n_tie} datasets needed a tie-break", flush=True)
    return frame


def _vbs_row(dataset_id: Any, grp: pd.DataFrame, pf: Portfolio) -> Dict[str, Any]:
    ok = grp[grp["_ok"]]
    # Full portfolio cost counts every attempted member, successful or not.
    full_cost = float(grp["_total_rt"].sum(skipna=True))
    first = grp.iloc[0]
    base: Dict[str, Any] = {
        "portfolio_id": pf.portfolio_id,
        "portfolio_label": pf.label,
        "portfolio_size": pf.size,
        "dataset_id": dataset_id,
        "family": first.get("family"),
        "family_id": first.get("family_id"),
        "subfamily": first.get("subfamily"),
        "difficulty": first.get("difficulty"),
        "cluster_count": first.get("cluster_count"),
        "n_candidates_available": int(len(grp)),
        "n_candidates_successful": int(len(ok)),
        "portfolio_total_execution_runtime_sec": full_cost,
        "uses_view_oracle": True,
        "uses_variant_oracle": True,
        "selection_type": "dataset_vbs",
        "paper_claim_level": "upper_bound",
        "contains_real_utility": pf.contains_real_utility,
        "contains_oracle_dynamic_k": pf.contains_oracle_dynamic_k,
        "contains_oracle_assisted_dynamic_k": pf.contains_oracle_dynamic_k,
        "fully_operational": pf.fully_operational,
        "oracle_contamination_reason": pf.oracle_contamination_reason,
    }
    if ok.empty:
        return {**base, **{c: None for c in VBS_FRAME_COLUMNS if c not in base}}

    top = float(ok["_ari"].max())
    tied = ok[ok["_ari"] == top]
    tie_count = int(len(tied))
    tie_reason = ""
    if tie_count > 1:
        ranked = tied.sort_values(
            ["_median", "_mean", "_total_rt", "method_name"],
            ascending=[False, False, True, True], kind="mergesort")
        if ranked["_median"].nunique(dropna=False) > 1:
            tie_reason = "view_ari_median"
        elif ranked["_mean"].nunique(dropna=False) > 1:
            tie_reason = "view_ari_mean"
        elif ranked["_total_rt"].nunique(dropna=False) > 1:
            tie_reason = "total_runtime_all_views_sec"
        else:
            tie_reason = "method_name"
        win = ranked.iloc[0]
    else:
        win = tied.iloc[0]

    spec = METHOD_BY_NAME.get(str(win["method_name"]))
    return {
        **base,
        "winning_method_name": win["method_name"],
        "winning_method_short_label": win.get("method_short_label"),
        "winning_view_id": win.get("best_view_id"),
        "winning_best_view_ari": float(win["_ari"]),
        "winning_k_correct": win.get("k_correct"),
        "winning_selected_k": win.get("selected_k"),
        "winning_true_k": win.get("true_k"),
        "winning_utility_source": spec.utility_source if spec else None,
        "winning_k_policy": spec.k_policy if spec else None,
        "winning_weighting_mode": spec.weighting_mode if spec else None,
        "winning_method_family": spec.method_family if spec else None,
        "winning_oracle_level": spec.oracle_level if spec else None,
        "selected_member_best_view_runtime_sec": win.get("best_view_runtime_sec"),
        "selected_member_total_all_views_runtime_sec": (
            float(win["_total_rt"]) if pd.notna(win["_total_rt"]) else np.nan),
        "tie_count": tie_count,
        "tie_broken_by": tie_reason,
        "portfolio_mean_ari": float(ok["_ari"].mean()),
        "portfolio_median_ari": float(ok["_ari"].median()),
        "portfolio_min_ari": float(ok["_ari"].min()),
    }


def build_all_vbs_frames(best_view: pd.DataFrame,
                         portfolio_ids: Sequence[str],
                         *, verbose: bool = True) -> Dict[str, pd.DataFrame]:
    return {pid: build_vbs_frame(best_view, pid, verbose=verbose)
            for pid in portfolio_ids}


# --------------------------------------------------------------------------- #
# Portfolio summaries
# --------------------------------------------------------------------------- #
def vbs_summary(vbs_frames: Dict[str, pd.DataFrame],
                group_cols: Sequence[str] = ()) -> pd.DataFrame:
    """Aggregate VBS performance per portfolio (optionally grouped)."""
    rows: List[Dict[str, Any]] = []
    for pid, frame in vbs_frames.items():
        if frame.empty:
            continue
        pf = PORTFOLIO_BY_ID.get(pid)
        groups = ([((), frame)] if not group_cols
                  else list(frame.groupby(list(group_cols), dropna=False, sort=True)))
        for key, grp in groups:
            key_tuple = key if isinstance(key, tuple) else (key,)
            ari = pd.to_numeric(grp["winning_best_view_ari"], errors="coerce").dropna()
            oracle_rt = pd.to_numeric(grp["selected_member_total_all_views_runtime_sec"],
                                      errors="coerce").dropna()
            full_rt = pd.to_numeric(grp["portfolio_total_execution_runtime_sec"],
                                    errors="coerce").dropna()
            kc = grp["winning_k_correct"]
            row: Dict[str, Any] = dict(zip(group_cols, key_tuple))
            row.update({
                "portfolio_id": pid,
                "portfolio_label": pf.label if pf else pid,
                "portfolio_short_label": pf.short_label if pf else pid,
                "portfolio_size": pf.size if pf else grp["portfolio_size"].iloc[0],
                "n_datasets": int(len(grp)),
                "n_success": int(len(ari)),
                "success_rate": float(len(ari) / len(grp)) if len(grp) else float("nan"),
                "best_view_mean_ari": float(ari.mean()) if len(ari) else float("nan"),
                "best_view_median_ari": float(ari.median()) if len(ari) else float("nan"),
                "best_view_std_ari": (float(ari.std(ddof=1)) if len(ari) > 1
                                     else float("nan")),
                "best_view_q25_ari": (float(ari.quantile(0.25)) if len(ari)
                                      else float("nan")),
                "best_view_q75_ari": (float(ari.quantile(0.75)) if len(ari)
                                      else float("nan")),
                "best_view_min_ari": float(ari.min()) if len(ari) else float("nan"),
                "best_view_max_ari": float(ari.max()) if len(ari) else float("nan"),
                "best_view_k_accuracy": (
                    float(kc.dropna().astype(bool).mean())
                    if int(kc.notna().sum()) else float("nan")),
                "selected_member_total_all_views_runtime_mean_sec": (
                    float(oracle_rt.mean()) if len(oracle_rt) else float("nan")),
                "selected_member_total_all_views_runtime_median_sec": (
                    float(oracle_rt.median()) if len(oracle_rt) else float("nan")),
                "portfolio_total_execution_runtime_mean_sec": (
                    float(full_rt.mean()) if len(full_rt) else float("nan")),
                "portfolio_total_execution_runtime_median_sec": (
                    float(full_rt.median()) if len(full_rt) else float("nan")),
                "portfolio_total_execution_runtime_total_sec": (
                    float(full_rt.sum()) if len(full_rt) else float("nan")),
                "n_distinct_winners": int(grp["winning_method_name"].nunique()),
                "n_tie_datasets": int((pd.to_numeric(grp["tie_count"],
                                                     errors="coerce") > 1).sum()),
                "uses_view_oracle": True,
                "uses_variant_oracle": True,
                "selection_type": "dataset_vbs",
                "paper_claim_level": "upper_bound",
                "contains_real_utility": (pf.contains_real_utility if pf
                                          else grp["contains_real_utility"].iloc[0]),
                "contains_oracle_dynamic_k": (
                    pf.contains_oracle_dynamic_k if pf
                    else grp["contains_oracle_dynamic_k"].iloc[0]),
                "contains_oracle_assisted_dynamic_k": (
                    pf.contains_oracle_dynamic_k if pf
                    else grp["contains_oracle_dynamic_k"].iloc[0]),
                "fully_operational": (pf.fully_operational if pf
                                      else grp["fully_operational"].iloc[0]),
                "oracle_contamination_reason": (
                    pf.oracle_contamination_reason if pf
                    else grp["oracle_contamination_reason"].iloc[0]),
                "runtime_note": (
                    "portfolio_total_execution_runtime is the cost of evaluating "
                    "the FULL portfolio; selected_member_* runtimes are "
                    "diagnostics for the method the oracle picked and are never "
                    "the cost of obtaining the VBS result"),
            })
            rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    sort_cols = list(group_cols) + ["best_view_mean_ari"]
    return out.sort_values(sort_cols,
                           ascending=[True] * len(group_cols) + [False],
                           kind="mergesort").reset_index(drop=True)


def winner_composition(vbs_frames: Dict[str, pd.DataFrame],
                       by: Sequence[str] = ("winning_method_name",)) -> pd.DataFrame:
    """Winner-frequency composition of each portfolio's VBS."""
    rows: List[Dict[str, Any]] = []
    for pid, frame in vbs_frames.items():
        if frame.empty:
            continue
        total = int(frame["winning_method_name"].notna().sum())
        cols = [c for c in by if c in frame.columns]
        if not cols:
            continue
        counts = frame.dropna(subset=["winning_method_name"]).groupby(
            cols, dropna=False).size().rename("win_count").reset_index()
        for _, r in counts.iterrows():
            row: Dict[str, Any] = {"portfolio_id": pid}
            row.update({c: r[c] for c in cols})
            row["win_count"] = int(r["win_count"])
            row["win_share"] = float(r["win_count"] / total) if total else float("nan")
            row["n_datasets"] = total
            if "winning_method_name" in cols:
                row["winning_method_short_label"] = short_label(
                    str(r["winning_method_name"]))
            rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["portfolio_id", "win_count"],
                          ascending=[True, False], kind="mergesort"
                          ).reset_index(drop=True)


def unique_coverage(vbs_frames: Dict[str, pd.DataFrame],
                    best_view: pd.DataFrame) -> pd.DataFrame:
    """Per-method *unique* coverage inside each portfolio.

    A method uniquely covers a dataset when it is the sole portfolio member
    reaching that dataset's portfolio-best ARI -- i.e. removing it would strictly
    lower the VBS. This separates "wins often" from "is irreplaceable".
    """
    rows: List[Dict[str, Any]] = []
    for pid, frame in vbs_frames.items():
        if frame.empty:
            continue
        pf = PORTFOLIO_BY_ID.get(pid)
        members = list(pf.methods) if pf else sorted(
            frame["winning_method_name"].dropna().unique())
        sub = best_view[best_view["method_name"].isin(members)].copy()
        sub["_ari"] = pd.to_numeric(sub["best_view_ari"], errors="coerce")
        best_per_ds = sub.groupby("dataset_id")["_ari"].transform("max")
        sub["_at_best"] = np.isclose(sub["_ari"], best_per_ds, equal_nan=False)
        n_at_best = sub.groupby("dataset_id")["_at_best"].transform("sum")
        sub["_unique"] = sub["_at_best"] & (n_at_best == 1)
        for method, grp in sub.groupby("method_name"):
            wins = int((frame["winning_method_name"] == method).sum())
            rows.append({
                "portfolio_id": pid,
                "method_name": method,
                "method_short_label": short_label(str(method)),
                "vbs_win_count": wins,
                "n_at_portfolio_best": int(grp["_at_best"].sum()),
                "n_unique_coverage": int(grp["_unique"].sum()),
                "unique_coverage_share": (
                    float(grp["_unique"].sum() / len(frame)) if len(frame)
                    else float("nan")),
                "n_datasets": int(len(frame)),
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["portfolio_id", "n_unique_coverage"],
                          ascending=[True, False], kind="mergesort"
                          ).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Headroom
# --------------------------------------------------------------------------- #
def headroom(vbs_frames: Dict[str, pd.DataFrame], best_view: pd.DataFrame,
             *, reference_methods: Dict[str, str]) -> pd.DataFrame:
    """VBS headroom over a named reference single variant.

    ``reference_methods`` maps a label (e.g. ``"best_observed_mlp"``) to a concrete
    method name. Headroom is reported both unpaired (mean minus mean) and paired
    (mean of the per-dataset differences over the shared datasets), because the
    two answer different questions and the paired form is the defensible one.
    """
    rows: List[Dict[str, Any]] = []
    ref_summ = summarize_methods(
        best_view[best_view["method_name"].isin(reference_methods.values())],
        ("method_name",))
    ref_by_method = ref_summ.set_index("method_name") if not ref_summ.empty else None

    for pid, frame in vbs_frames.items():
        if frame.empty:
            continue
        pf = PORTFOLIO_BY_ID.get(pid)
        vbs_ari = pd.to_numeric(frame["winning_best_view_ari"], errors="coerce")
        vbs_mean = float(vbs_ari.mean())
        for ref_label, ref_method in reference_methods.items():
            ref_rows = best_view[best_view["method_name"] == ref_method]
            if ref_rows.empty or ref_by_method is None or ref_method not in ref_by_method.index:
                continue
            ref_mean = float(ref_by_method.loc[ref_method, "best_view_mean_ari"])
            paired = frame[["dataset_id", "winning_best_view_ari"]].merge(
                ref_rows[["dataset_id", "best_view_ari"]], on="dataset_id",
                how="inner")
            delta = (pd.to_numeric(paired["winning_best_view_ari"], errors="coerce")
                     - pd.to_numeric(paired["best_view_ari"], errors="coerce")).dropna()
            rows.append({
                "portfolio_id": pid,
                "portfolio_label": pf.label if pf else pid,
                "portfolio_size": pf.size if pf else np.nan,
                "reference_label": ref_label,
                "reference_method": ref_method,
                "reference_short_label": short_label(ref_method),
                "vbs_best_view_mean_ari": vbs_mean,
                "reference_best_view_mean_ari": ref_mean,
                "headroom_unpaired": vbs_mean - ref_mean,
                "n_paired": int(len(delta)),
                "headroom_paired_mean": float(delta.mean()) if len(delta) else float("nan"),
                "headroom_paired_median": (float(delta.median()) if len(delta)
                                           else float("nan")),
                "headroom_paired_q25": (float(delta.quantile(0.25)) if len(delta)
                                        else float("nan")),
                "headroom_paired_q75": (float(delta.quantile(0.75)) if len(delta)
                                        else float("nan")),
                "share_datasets_vbs_strictly_better": (
                    float((delta > 1e-12).mean()) if len(delta) else float("nan")),
                "share_datasets_equal": (
                    float(np.isclose(delta, 0.0, atol=1e-12).mean()) if len(delta)
                    else float("nan")),
                "uses_variant_oracle": True,
                "selection_type": "dataset_vbs",
                "paper_claim_level": "upper_bound",
                "reference_selection_type": "retrospective_global",
                "notes": "headroom_paired_mean is the defensible figure; the "
                         "unpaired difference is provided for continuity only. "
                         "The VBS is a non-deployable oracle upper bound.",
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["reference_label", "headroom_paired_mean"],
                          ascending=[True, False], kind="mergesort"
                          ).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Portfolio-size sensitivity
# --------------------------------------------------------------------------- #
def _pivot_ari(best_view: pd.DataFrame, methods: Sequence[str]) -> pd.DataFrame:
    """``dataset_id x method`` matrix of Best-View ARI (NaN where unavailable)."""
    sub = best_view[best_view["method_name"].isin(list(methods))]
    if sub.empty:
        return pd.DataFrame()
    return sub.pivot_table(index="dataset_id", columns="method_name",
                           values="best_view_ari", aggfunc="max")


def _vbs_mean(matrix: pd.DataFrame, methods: Sequence[str]) -> float:
    cols = [m for m in methods if m in matrix.columns]
    if not cols:
        return float("nan")
    return float(matrix[cols].max(axis=1).mean())


def greedy_portfolio(best_view: pd.DataFrame, pf: Portfolio | str,
                     *, max_size: Optional[int] = None) -> pd.DataFrame:
    """Greedy forward construction by marginal VBS gain.

    At each step add the member that raises the portfolio's VBS mean Best-View ARI
    the most (ties -> lexicographically smaller method name, so the sequence is
    reproducible). Reports the VBS curve and the marginal gain per addition.
    """
    pf = portfolio(pf) if isinstance(pf, str) else pf
    matrix = _pivot_ari(best_view, pf.methods)
    if matrix.empty:
        return pd.DataFrame()
    remaining = [m for m in pf.methods if m in matrix.columns]
    limit = int(max_size) if max_size else len(remaining)
    chosen: List[str] = []
    rows: List[Dict[str, Any]] = []
    prev = float("nan")
    full = _vbs_mean(matrix, remaining)
    while remaining and len(chosen) < limit:
        scored = sorted(
            ((_vbs_mean(matrix, chosen + [m]), m) for m in remaining),
            key=lambda t: (-t[0], t[1]))
        gain_value, pick = scored[0]
        marginal = gain_value - prev if chosen else gain_value
        chosen.append(pick)
        remaining.remove(pick)
        rows.append({
            "portfolio_id": pf.portfolio_id,
            "step": len(chosen),
            "added_method": pick,
            "added_short_label": short_label(pick),
            "portfolio_members": ";".join(chosen),
            "vbs_best_view_mean_ari": gain_value,
            "marginal_gain": marginal,
            "cumulative_gain_from_step1": gain_value - rows[0]["vbs_best_view_mean_ari"]
            if rows else 0.0,
            "share_of_full_vbs": (gain_value / full) if full and np.isfinite(full)
            else float("nan"),
            "full_vbs_best_view_mean_ari": full,
            "selection_type": "dataset_vbs",
            "paper_claim_level": "upper_bound",
        })
        prev = gain_value
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # Fraction of the *achievable gain* (single-best -> full VBS) reached so far.
    single = out.iloc[0]["vbs_best_view_mean_ari"]
    span = full - single
    out["share_of_achievable_gain"] = (
        (out["vbs_best_view_mean_ari"] - single) / span if span > 1e-12 else 1.0)
    return out


def greedy_membership(greedy: pd.DataFrame) -> pd.DataFrame:
    """Flat membership table: which method entered a greedy portfolio at step k.

    The curve alone says how fast the VBS saturates; this says *which* variants
    do the saturating, which is the part that generalises to a real portfolio.
    """
    if greedy.empty:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for _, r in greedy.iterrows():
        members = str(r["portfolio_members"]).split(";")
        added = str(r["added_method"])
        spec = METHOD_BY_NAME.get(added)
        rows.append({
            "portfolio_id": r["portfolio_id"],
            "step": int(r["step"]),
            "added_method": added,
            "added_short_label": short_label(added),
            "utility_source": spec.utility_source if spec else "",
            "k_policy": spec.k_policy if spec else "",
            "weighting_mode": spec.weighting_mode if spec else "",
            "oracle_level": spec.oracle_level if spec else "",
            "portfolio_size_after_step": len(members),
            "portfolio_members_after_step": r["portfolio_members"],
            "vbs_best_view_mean_ari": r["vbs_best_view_mean_ari"],
            "marginal_gain": r["marginal_gain"],
            "share_of_achievable_gain": r.get("share_of_achievable_gain"),
            "selection_type": "dataset_vbs",
            "paper_claim_level": "upper_bound",
        })
    return pd.DataFrame(rows)


def matched_size_table(curves: pd.DataFrame, *, sizes: Sequence[int] = (1, 2, 3, 5, 10)
                       ) -> pd.DataFrame:
    """Side-by-side VBS at matched portfolio sizes across frameworks.

    This is the size-confound control: ClustOpt portfolios have up to 34 members
    while the external ones have 2, so comparing full VBS across frameworks
    conflates "better candidates" with "more candidates". At a matched size the
    comparison is fair.
    """
    if curves.empty:
        return pd.DataFrame()
    wanted = curves[curves["size"].isin(list(sizes)) & ~curves["skipped"].astype(bool)]
    if wanted.empty:
        return pd.DataFrame()
    keep = ["portfolio_id", "size", "n_samples", "is_exhaustive", "vbs_mean",
            "vbs_std", "ci_low", "ci_high", "vbs_min", "vbs_max",
            "full_vbs_mean", "share_of_full_vbs"]
    out = wanted[[c for c in keep if c in wanted.columns]].copy()
    out["portfolio_short_label"] = out["portfolio_id"].map(
        lambda p: PORTFOLIO_BY_ID[p].short_label if p in PORTFOLIO_BY_ID else p)
    out["fully_operational"] = out["portfolio_id"].map(
        lambda p: PORTFOLIO_BY_ID[p].fully_operational if p in PORTFOLIO_BY_ID
        else None)
    out["portfolio_full_size"] = out["portfolio_id"].map(
        lambda p: PORTFOLIO_BY_ID[p].size if p in PORTFOLIO_BY_ID else None)
    out["selection_type"] = "dataset_vbs"
    out["paper_claim_level"] = "upper_bound"
    out["matched_size_note"] = (
        "VBS at a matched portfolio size, so a framework with more candidates "
        "does not win on count alone")
    return out.sort_values(["size", "vbs_mean"], ascending=[True, False],
                          kind="mergesort").reset_index(drop=True)


def random_subset_curve(best_view: pd.DataFrame, pf: Portfolio | str,
                        *, sizes: Sequence[int] = (1, 2, 3, 5, 10),
                        n_samples: int = 200, seed: int = 20260725,
                        confidence: float = 0.95,
                        return_samples: bool = False
                        ) -> pd.DataFrame | Tuple[pd.DataFrame, pd.DataFrame]:
    """Matched-size random-subset VBS with percentile confidence intervals.

    Answers "how much of the VBS gain needs a *large* portfolio?" fairly: the
    ClustOpt portfolios have far more members than the 2-member external ones, so
    a raw full-VBS comparison across frameworks is size-confounded. Sizes above
    the portfolio's own size are skipped (and reported), never clipped silently.
    """
    pf = portfolio(pf) if isinstance(pf, str) else pf
    matrix = _pivot_ari(best_view, pf.methods)
    if matrix.empty:
        return (pd.DataFrame(), pd.DataFrame()) if return_samples else pd.DataFrame()
    members = [m for m in pf.methods if m in matrix.columns]
    rng = np.random.default_rng(int(seed))
    full = _vbs_mean(matrix, members)
    lo_q, hi_q = (1 - confidence) / 2, 1 - (1 - confidence) / 2

    rows: List[Dict[str, Any]] = []
    samples: List[Dict[str, Any]] = []
    wanted = sorted({int(sz) for sz in sizes} | {len(members)})
    for size in wanted:
        if size < 1 or size > len(members):
            rows.append({
                "portfolio_id": pf.portfolio_id, "size": size,
                "n_samples": 0, "skipped": True,
                "skip_reason": f"portfolio has only {len(members)} members",
                "vbs_mean": float("nan"), "vbs_std": float("nan"),
                "ci_low": float("nan"), "ci_high": float("nan"),
                "vbs_min": float("nan"), "vbs_max": float("nan"),
                "full_vbs_mean": full, "share_of_full_vbs": float("nan"),
                "is_exhaustive": False,
                "selection_type": "dataset_vbs", "paper_claim_level": "upper_bound",
            })
            continue
        if size == len(members):
            values = np.array([full])
            exhaustive = True
            n_draws = 1
            drawn = [tuple(members)]
        else:
            n_draws = int(n_samples)
            values = np.empty(n_draws, dtype=float)
            drawn = []
            for i in range(n_draws):
                pick = rng.choice(len(members), size=size, replace=False)
                subset = tuple(sorted(members[j] for j in pick))
                drawn.append(subset)
                values[i] = _vbs_mean(matrix, list(subset))
            exhaustive = False
        if return_samples:
            for subset, value in zip(drawn, values):
                samples.append({
                    "portfolio_id": pf.portfolio_id, "size": size,
                    "members": ";".join(subset),
                    "vbs_best_view_mean_ari": float(value),
                })
        n_unique = len({tuple(d) for d in drawn})
        rows.append({
            "portfolio_id": pf.portfolio_id, "size": size,
            "n_samples": n_draws, "skipped": False, "skip_reason": "",
            "vbs_mean": float(values.mean()),
            "vbs_std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "ci_low": float(np.quantile(values, lo_q)),
            "ci_high": float(np.quantile(values, hi_q)),
            "vbs_min": float(values.min()), "vbs_max": float(values.max()),
            "full_vbs_mean": full,
            "share_of_full_vbs": (float(values.mean() / full)
                                  if full and np.isfinite(full) else float("nan")),
            "vbs_best_sampled_subset": (
                ";".join(drawn[int(np.argmax(values))]) if len(drawn) else ""),
            "n_unique_subsets_evaluated": int(n_unique),
            "n_possible_subsets": int(_n_choose_k(len(members), size)),
            "is_exhaustive": exhaustive,
            "confidence": confidence,
            "seed": int(seed),
            "selection_type": "dataset_vbs", "paper_claim_level": "upper_bound",
        })
    curve = pd.DataFrame(rows)
    if return_samples:
        return curve, pd.DataFrame(samples)
    return curve


def _n_choose_k(n: int, k: int) -> int:
    from math import comb
    try:
        return comb(int(n), int(k))
    except ValueError:                                         # pragma: no cover
        return 0


def min_size_for_gain_share(greedy: pd.DataFrame, share: float = 0.90
                            ) -> pd.DataFrame:
    """Smallest greedy portfolio size reaching ``share`` of the achievable gain."""
    if greedy.empty or "share_of_achievable_gain" not in greedy.columns:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for pid, grp in greedy.groupby("portfolio_id"):
        hit = grp[grp["share_of_achievable_gain"] >= float(share)]
        rows.append({
            "portfolio_id": pid,
            "target_share_of_achievable_gain": float(share),
            "min_size": int(hit.iloc[0]["step"]) if len(hit) else None,
            "portfolio_size": int(grp["step"].max()),
            "single_best_vbs": float(grp.iloc[0]["vbs_best_view_mean_ari"]),
            "full_vbs": float(grp.iloc[-1]["vbs_best_view_mean_ari"]),
            "achievable_gain": float(grp.iloc[-1]["vbs_best_view_mean_ari"]
                                     - grp.iloc[0]["vbs_best_view_mean_ari"]),
            "selection_type": "dataset_vbs", "paper_claim_level": "upper_bound",
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Invariants
# --------------------------------------------------------------------------- #
def validate_vbs(vbs_frames: Dict[str, pd.DataFrame], best_view: pd.DataFrame,
                 *, subset_pairs: Sequence[Tuple[str, str]] = ()) -> List[str]:
    """Check the VBS invariants (empty list == valid)."""
    problems: List[str] = []
    for pid, frame in vbs_frames.items():
        if frame.empty:
            problems.append(f"{pid}: empty VBS frame")
            continue
        dupes = frame.duplicated(subset=["dataset_id"])
        if bool(dupes.any()):
            problems.append(f"{pid}: {int(dupes.sum())} duplicate dataset rows "
                            f"(VBS must emit exactly one winner per dataset)")
        pf = PORTFOLIO_BY_ID.get(pid)
        if pf is not None:
            outside = set(frame["winning_method_name"].dropna()) - set(pf.methods)
            if outside:
                problems.append(f"{pid}: winners outside the portfolio: "
                                f"{sorted(outside)}")
            if int(frame["portfolio_size"].iloc[0]) != pf.size:
                problems.append(f"{pid}: portfolio_size column disagrees with the "
                                f"registry ({frame['portfolio_size'].iloc[0]} "
                                f"vs {pf.size})")
            # Full portfolio runtime must be the sum of member costs.
            sub = best_view[best_view["method_name"].isin(pf.methods)]
            want = (sub.assign(_rt=pd.to_numeric(sub["total_runtime_all_views_sec"],
                                                 errors="coerce"))
                    .groupby("dataset_id")["_rt"].sum())
            got = frame.set_index("dataset_id")["portfolio_total_execution_runtime_sec"]
            joined = pd.concat([want.rename("want"), got.rename("got")], axis=1,
                               join="inner")
            bad = joined[~np.isclose(joined["want"], joined["got"], rtol=1e-9,
                                     atol=1e-6, equal_nan=True)]
            # A VBS is not one executable method: realising it costs the WHOLE
            # portfolio, so the full-execution runtime can never be below the
            # runtime of the single member the oracle happened to pick.
            rt = frame[["portfolio_total_execution_runtime_sec",
                        "selected_member_total_all_views_runtime_sec"]].apply(
                pd.to_numeric, errors="coerce").dropna()
            violated = rt[rt["portfolio_total_execution_runtime_sec"]
                          < rt["selected_member_total_all_views_runtime_sec"] - 1e-6]
            if len(violated):
                problems.append(
                    f"{pid}: portfolio_total_execution_runtime_sec below "
                    f"selected_member_total_all_views_runtime_sec on "
                    f"{len(violated)} dataset(s)")
            # A multi-member portfolio must genuinely cost more than one member.
            if pf.size > 1 and len(rt):
                equal = rt[np.isclose(
                    rt["portfolio_total_execution_runtime_sec"],
                    rt["selected_member_total_all_views_runtime_sec"],
                    rtol=0, atol=1e-9)]
                if len(equal) == len(rt):
                    problems.append(
                        f"{pid}: full portfolio cost equals the selected "
                        f"member's cost on every dataset, which cannot be true "
                        f"for a {pf.size}-member portfolio")
            # Declared oracle status must match the registry.
            for col, want in (("fully_operational", pf.fully_operational),
                              ("contains_real_utility", pf.contains_real_utility),
                              ("contains_oracle_dynamic_k",
                               pf.contains_oracle_dynamic_k)):
                if col in frame.columns and bool(frame[col].iloc[0]) != bool(want):
                    problems.append(f"{pid}: {col} column says "
                                    f"{frame[col].iloc[0]} but the registry says "
                                    f"{want}")
            if len(bad):
                problems.append(f"{pid}: portfolio_total_execution_runtime_sec != sum of "
                                f"member runtimes on {len(bad)} datasets")
            # VBS must equal the per-dataset max over members.
            want_ari = (sub.assign(_a=pd.to_numeric(sub["best_view_ari"],
                                                    errors="coerce"))
                        .groupby("dataset_id")["_a"].max())
            got_ari = frame.set_index("dataset_id")["winning_best_view_ari"]
            j2 = pd.concat([want_ari.rename("want"), got_ari.rename("got")],
                           axis=1, join="inner")
            bad2 = j2[~np.isclose(j2["want"], j2["got"], equal_nan=True)]
            if len(bad2):
                problems.append(f"{pid}: winning_best_view_ari != max member ARI "
                                f"on {len(bad2)} datasets")
            # VBS can never be below the best single member on the same dataset.
            if bool((j2["got"] < j2["want"] - 1e-12).any()):
                problems.append(f"{pid}: VBS below the best single candidate")

    # Monotonicity: a superset portfolio's VBS dominates its subset's, per dataset.
    for sub_id, sup_id in subset_pairs:
        a, b = vbs_frames.get(sub_id), vbs_frames.get(sup_id)
        if a is None or b is None or a.empty or b.empty:
            continue
        merged = a[["dataset_id", "winning_best_view_ari"]].merge(
            b[["dataset_id", "winning_best_view_ari"]], on="dataset_id",
            suffixes=("_sub", "_sup"))
        worse = merged[merged["winning_best_view_ari_sup"]
                       < merged["winning_best_view_ari_sub"] - 1e-12]
        if len(worse):
            problems.append(f"{sup_id} VBS below its subset {sub_id} on "
                            f"{len(worse)} datasets (monotonicity violated)")
    return problems
