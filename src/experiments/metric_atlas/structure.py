"""Stage 3B steps 3-4: family/subfamily structure and New46 complementarity.

All of this is *association*, never causation: it says where a metric ranks and
how often it beats a comparator, not what removing it would do to any method's
ARI. That distinction is enforced in the report, and every table produced here
carries an ``evidence_level`` of 2.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from . import common as C
from .atlas import UTIL_PREFIX_TRUE, TOPK_LEVELS


def _matrix(D: pd.DataFrame, metrics: List[str]) -> np.ndarray:
    return D[[UTIL_PREFIX_TRUE + m for m in metrics]].to_numpy(float)


def family_maps(oof: pd.DataFrame) -> pd.DataFrame:
    """dataset_id -> family / subfamily, one row per dataset."""
    return (oof[["dataset_id", "family_name", "subfamily_id"]]
            .drop_duplicates("dataset_id").reset_index(drop=True))


def group_utility(D: pd.DataFrame, metrics: List[str], keys: pd.DataFrame,
                  by: str, log) -> pd.DataFrame:
    """Per-(group, metric) utility/rank summary. Dataset is the unit."""
    M = _matrix(D, metrics)
    R = pd.DataFrame(M).rank(axis=1, ascending=False, method="average").to_numpy()
    lab = keys.set_index("dataset_id").loc[D["dataset_id"], by].to_numpy()
    rows = []
    for g in pd.unique(lab):
        sel = lab == g
        Mg, Rg = M[sel], R[sel]
        n = int(sel.sum())
        with np.errstate(invalid="ignore"):
            for i, m in enumerate(metrics):
                col = Mg[:, i]
                rows.append({
                    by: g, "metric_id": m, "n_datasets": n,
                    "n_defined": int(np.isfinite(col).sum()),
                    "mean_utility": float(np.nanmean(col)),
                    "median_utility": float(np.nanmedian(col)),
                    "mean_rank": float(np.nanmean(Rg[:, i])),
                    "top1_freq": float(np.nanmean(Rg[:, i] <= 1)),
                    "top3_freq": float(np.nanmean(Rg[:, i] <= 3)),
                    "top5_freq": float(np.nanmean(Rg[:, i] <= 5)),
                    "top10_freq": float(np.nanmean(Rg[:, i] <= 10)),
                })
    T = pd.DataFrame(rows)
    T["rank_within_group"] = T.groupby(by)["mean_utility"].rank(ascending=False,
                                                                method="min")
    log("%s matrix: %d rows (%d %ss x %d metrics)"
        % (by, len(T), T[by].nunique(), by, len(metrics)))
    return T


def enrichment(T: pd.DataFrame, G: pd.DataFrame, by: str) -> pd.DataFrame:
    """Top-5 frequency of a metric in a family relative to its global rate."""
    g = G.set_index("metric_id")["top5_freq"]
    T = T.copy()
    T["global_top5_freq"] = T["metric_id"].map(g)
    T["top5_enrichment"] = T["top5_freq"] / T["global_top5_freq"].replace(0, np.nan)
    return T


def complementarity(D: pd.DataFrame, A: pd.DataFrame, keys: pd.DataFrame,
                    log) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """New46 complementarity against the Original+Established comparator pool.

    Three predeclared quantities per New46 metric (§13 A-C), each aggregated with
    the dataset as the unit and BH-FDR applied across the family of New46 tests.
    """
    have = set(c[len(UTIL_PREFIX_TRUE):] for c in D.columns
               if c.startswith(UTIL_PREFIX_TRUE))
    comparator = [m for m in A[A.group.isin(["Original", "Established"])]["metric_id"]
                  if m in have]
    new46 = [m for m in A[(A.group == "New46") & A.is_live]["metric_id"] if m in have]
    allm = comparator + new46
    log("complementarity: %d comparator (Original+Established in Full-60), "
        "%d live New46" % (len(comparator), len(new46)))

    Mc = _matrix(D, comparator)
    Mn = _matrix(D, new46)
    Ma = _matrix(D, allm)
    with np.errstate(invalid="ignore"):
        best_c = np.nanmax(Mc, axis=1)          # best comparator per dataset
        best_a = np.nanmax(Ma, axis=1)          # best overall per dataset
    argbest = np.array(allm)[np.nanargmax(np.nan_to_num(Ma, nan=-np.inf), axis=1)]
    is_new_best = np.isin(argbest, new46)

    rows = []
    for i, m in enumerate(new46):
        d = Mn[:, i] - best_c                    # A. best-comparator difference
        lo, hi = C.boot_ci_vector(d)
        wins = (argbest == m)                    # B. oracle-winner frequency
        rows.append({
            "metric_id": m, "n_datasets": int(np.isfinite(d).sum()),
            "mean_vs_best_comparator": float(np.nanmean(d)),
            "median_vs_best_comparator": float(np.nanmedian(d)),
            "ci_low": lo, "ci_high": hi,
            "frac_beats_best_comparator": float(np.nanmean(d > 0)),
            "wilcoxon_p": C.wilcoxon_p(d),
            "oracle_win_freq": float(np.mean(wins)),
            "oracle_wins": int(wins.sum()),
        })
    Cp = pd.DataFrame(rows)
    Cp["wilcoxon_q_bh"] = C.bh_fdr(Cp["wilcoxon_p"].to_numpy())
    Cp["ci_excludes_zero"] = (Cp["ci_low"] > 0) | (Cp["ci_high"] < 0)
    Cp = Cp.merge(A[["metric_id", "subgroup"]], on="metric_id", how="left")
    Cp = Cp.sort_values("mean_vs_best_comparator", ascending=False).reset_index(drop=True)

    # C. New46 unique-win coverage, globally and by family/subfamily
    km = keys.set_index("dataset_id")
    fam = km.loc[D["dataset_id"], "family_name"].to_numpy()
    sub = km.loc[D["dataset_id"], "subfamily_id"].to_numpy()
    cov_rows = [{"level": "global", "key": "ALL", "n_datasets": len(D),
                 "new46_best_frac": float(np.mean(is_new_best)),
                 "new46_best_n": int(is_new_best.sum())}]
    for lvl, lab in (("family", fam), ("subfamily", sub)):
        for g in pd.unique(lab):
            s = lab == g
            cov_rows.append({"level": lvl, "key": g, "n_datasets": int(s.sum()),
                             "new46_best_frac": float(np.mean(is_new_best[s])),
                             "new46_best_n": int(is_new_best[s].sum())})
    Cov = pd.DataFrame(cov_rows)

    # D. group oracle coverage: best achievable utility per inventory
    est = [m for m in A[A.group == "Established"]["metric_id"] if m in have]
    orig = [m for m in A[A.group == "Original"]["metric_id"] if m in have]
    invs = {
        "A0_Original": orig,
        "A1_Original_plus_Established": orig + est,
        "A2_Original_plus_New46": orig + new46,
        "A3_Extended": orig + est + new46,
    }
    grows = []
    base = None
    for name, ms in invs.items():
        if not ms:
            continue
        with np.errstate(invalid="ignore"):
            b = np.nanmax(_matrix(D, ms), axis=1)
        if base is None:
            base = b
        lo, hi = C.boot_ci_vector(b)
        d = b - base
        dlo, dhi = C.boot_ci_vector(d)
        grows.append({
            "inventory": name, "n_metrics_in_utility_space": len(ms),
            "n_datasets": int(np.isfinite(b).sum()),
            "mean_best_utility": float(np.nanmean(b)),
            "median_best_utility": float(np.nanmedian(b)),
            "ci_low": lo, "ci_high": hi,
            "mean_gain_vs_A0": float(np.nanmean(d)),
            "gain_ci_low": dlo, "gain_ci_high": dhi,
            "frac_datasets_improved_vs_A0": float(np.nanmean(d > 0)),
        })
    Go = pd.DataFrame(grows)
    log("group oracle coverage: " + ", ".join(
        "%s=%.4f" % (r["inventory"], r["mean_best_utility"]) for _, r in Go.iterrows()))
    log("New46 is the best available metric on %.1f%% of datasets"
        % (100 * np.mean(is_new_best)))
    return Cp, Cov, Go
