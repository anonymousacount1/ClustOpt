"""Stage 3B step 5: how ClustOpt predicts and uses individual metrics.

No model is trained, loaded, or run here. The predicted utilities are the frozen
cross-fitted out-of-fold values already stored in the Stage-2A1 package; the
selection is reconstructed from them with the *existing* deployed semantics --
Top-5 by predicted utility with normalised-positive (RAW) weighting, the rule
named in the frozen Stage-1 arm ``stage1_full60_mlp_top5_raw_fixed50``.

The point of this file is to separate two very different statements:

    "metric m is useful"                     -> true utility
    "ClustOpt knows when m is useful"        -> predicted vs true agreement

A metric can be excellent and still contribute nothing, if the utility model
cannot tell when to reach for it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from . import common as C
from .atlas import UTIL_PREFIX_TRUE, UTIL_PREFIX_PRED


def usage_and_quality(oof: pd.DataFrame, A: pd.DataFrame, log
                      ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tcols = C.metric_cols(oof, UTIL_PREFIX_TRUE)
    metrics = C.strip(tcols, UTIL_PREFIX_TRUE)
    pcols = [UTIL_PREFIX_PRED + m for m in metrics]
    missing = [c for c in pcols if c not in oof.columns]
    if missing:
        raise AssertionError("missing predicted columns: %s" % missing[:5])

    keys = ["dataset_id", "family_name", "subfamily_id", "view_mode"]
    df = oof[keys + tcols + pcols].copy()

    T = df[tcols].to_numpy(float)
    Pm = df[pcols].to_numpy(float)

    # --- the frozen deployed selection: Top-5 by PREDICTED utility, per view ---
    order = np.argsort(-np.nan_to_num(Pm, nan=-np.inf), axis=1)
    sel = np.zeros_like(Pm, dtype=bool)
    rows_idx = np.arange(Pm.shape[0])[:, None]
    sel[rows_idx, order[:, :C.CLUSTOPT_TOP_K]] = True
    # normalised-positive weighting: a selected metric with non-positive
    # predicted utility receives zero weight, i.e. it is selected but inert.
    pos = np.nan_to_num(Pm, nan=0.0) > 0
    eff = sel & pos
    wsum = np.where(eff, np.nan_to_num(Pm, nan=0.0), 0.0).sum(axis=1, keepdims=True)
    W = np.divide(np.where(eff, np.nan_to_num(Pm, nan=0.0), 0.0),
                  np.where(wsum == 0, np.nan, wsum))
    W = np.nan_to_num(W, nan=0.0)

    Rt = pd.DataFrame(T).rank(axis=1, ascending=False, method="average").to_numpy()
    Rp = pd.DataFrame(Pm).rank(axis=1, ascending=False, method="average").to_numpy()

    rows = []
    for i, m in enumerate(metrics):
        t, p = T[:, i], Pm[:, i]
        ok = np.isfinite(t) & np.isfinite(p)
        rows.append({
            "metric_id": m,
            "n_view_rows": int(ok.sum()),
            "mean_true_utility": float(np.nanmean(t)),
            "mean_predicted_utility": float(np.nanmean(p)),
            "mae_pred_vs_true": float(np.nanmean(np.abs(p[ok] - t[ok]))) if ok.any() else np.nan,
            "bias_pred_minus_true": float(np.nanmean(p[ok] - t[ok])) if ok.any() else np.nan,
            "spearman_pred_true": (float(pd.Series(p[ok]).corr(pd.Series(t[ok]),
                                                               method="spearman"))
                                   if ok.sum() > 10 else np.nan),
            "true_rank": float(np.nanmean(Rt[:, i])),
            "pred_rank": float(np.nanmean(Rp[:, i])),
            "rank_error": float(np.nanmean(Rp[:, i] - Rt[:, i])),
            "pred_top1_freq": float(np.nanmean(Rp[:, i] <= 1)),
            "pred_top3_freq": float(np.nanmean(Rp[:, i] <= 3)),
            "pred_top5_freq": float(np.nanmean(Rp[:, i] <= 5)),
            "pred_top10_freq": float(np.nanmean(Rp[:, i] <= 10)),
            "true_top5_freq": float(np.nanmean(Rt[:, i] <= 5)),
            "selected_freq": float(np.mean(sel[:, i])),
            "effective_selected_freq": float(np.mean(eff[:, i])),
            "mean_weight_when_selected": (float(W[sel[:, i], i].mean())
                                          if sel[:, i].any() else 0.0),
            "total_weight_share": float(W[:, i].mean()),
            "top5_recall": (float(np.mean(Rp[Rt[:, i] <= 5, i] <= 5))
                            if (Rt[:, i] <= 5).any() else np.nan),
        })
    U = pd.DataFrame(rows).merge(
        A[["metric_id", "group", "subgroup", "is_live"]], on="metric_id", how="left")
    U = U.sort_values("selected_freq", ascending=False).reset_index(drop=True)

    # ---- prediction-quality summary by group ----
    qrows = []
    for key, col, val in (("Original", "group", "Original"),
                          ("Established", "group", "Established"),
                          ("New46", "group", "New46"),
                          ("New46-Pattern", "subgroup", "New46-Pattern"),
                          ("New46-Image", "subgroup", "New46-Image")):
        s = U[U[col] == val]
        if not len(s):
            continue
        qrows.append({
            "group": key, "n_metrics": int(len(s)),
            "mean_mae": float(s["mae_pred_vs_true"].mean()),
            "mean_bias": float(s["bias_pred_minus_true"].mean()),
            "mean_spearman_pred_true": float(s["spearman_pred_true"].mean()),
            "mean_rank_error": float(s["rank_error"].mean()),
            "mean_top5_recall": float(s["top5_recall"].mean()),
            "sum_selected_freq": float(s["selected_freq"].sum()),
            "share_of_selection_slots": float(s["selected_freq"].sum() / C.CLUSTOPT_TOP_K),
            "sum_weight_share": float(s["total_weight_share"].sum()),
        })
    Q = pd.DataFrame(qrows)

    # ---- family-level selection usage ----
    fam = df["family_name"].to_numpy()
    frows = []
    for g in pd.unique(fam):
        s = fam == g
        for i, m in enumerate(metrics):
            if sel[s, i].mean() == 0:
                continue
            frows.append({"family_name": g, "metric_id": m,
                          "n_view_rows": int(s.sum()),
                          "selected_freq": float(sel[s, i].mean()),
                          "mean_true_utility": float(np.nanmean(T[s, i])),
                          "mean_predicted_utility": float(np.nanmean(Pm[s, i]))})
    F = pd.DataFrame(frows).merge(A[["metric_id", "group", "subgroup"]],
                                  on="metric_id", how="left")
    g_sel = U.set_index("metric_id")["selected_freq"]
    F["global_selected_freq"] = F["metric_id"].map(g_sel)
    F["selection_enrichment"] = F["selected_freq"] / F["global_selected_freq"].replace(0, np.nan)

    new_share = U[U.group == "New46"]["selected_freq"].sum() / C.CLUSTOPT_TOP_K
    log("ClustOpt Top-%d selection: New46 holds %.1f%% of selection slots"
        % (C.CLUSTOPT_TOP_K, 100 * new_share))
    log("prediction quality (mean Spearman pred-vs-true): " + ", ".join(
        "%s=%.3f" % (r["group"], r["mean_spearman_pred_true"]) for _, r in Q.iterrows()))
    return U, Q, F


def stage1_context(repo: Path) -> Dict[str, object]:
    """The authoritative Stage-1 2x3 arm results, read, never recomputed."""
    out = {}
    for tag, rel in (("offline_fixed32",
                      "offline_fixed32/offline_2x3_overall_summary.csv"),
                     ("online_fixed50",
                      "online_fixed50/online_2x3_overall_summary.csv")):
        f = repo / C.STAGE1_REL / rel
        if not f.exists():
            continue
        d = pd.read_csv(C.ext(f))
        out[tag] = {
            "source": str(f),
            "arms": {r["arm"]: {
                "method_id": r["method_id"],
                "metric_inventory": r["metric_inventory"],
                "utility_strategy": r["utility_strategy"],
                "deployable": bool(r["deployable"]),
                "mean": float(r.get("mean_best_view_ari", r.get("mean_ari"))),
                "ci_low": float(r["mean_ci_lo"]), "ci_high": float(r["mean_ci_hi"]),
                "n_datasets": int(r["n_datasets"]),
            } for _, r in d.iterrows()},
        }
    return out
