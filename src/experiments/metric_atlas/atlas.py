"""Stage 3B step 1-2: canonical metric atlas and the per-index utility analysis.

Everything here is descriptive or inferential over artefacts that already exist.
The utility values are the ones the project already computed and froze; this
module reads them, it does not define them.

Population note. The canonical per-dataset utilities live in the leakage-free
out-of-fold table for **Splits 2-16** (16,013 datasets x 3 views). That is the
ClustOpt development population and the only place where a *true* utility and a
*cross-fitted predicted* utility exist side by side for the same record. Split 1
is deliberately absent from it, so no Split-1 outcome informs any number in this
file.

Vocabulary note. The ClustOpt utility vocabulary is the Full-60 inventory
(Head-14 classic + New-46). Three of the seven ORIGINAL baseline CVIs
(``dunn_index``, ``cop``, ``coggins_jain_index``) are ML2DAC-side indices that
ClustOpt never computes a utility for; they appear in the atlas with an explicit
NA reason rather than being silently dropped or imputed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from . import common as C

UTIL_PREFIX_TRUE = "true_"
UTIL_PREFIX_PRED = "pred_"
TOPK_LEVELS = (1, 3, 5, 10)


# --------------------------------------------------------------- 1. the atlas
def build_atlas(repo: Path, log) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    P = C.partitions(repo)
    impl = C.implementation_subgroups(repo)
    hg = C.head_groups()
    reg = pd.read_parquet(repo / C.MKR_REL / "master" / "metric_registry.parquet")
    reg_by = {r["metric_id"]: r for _, r in reg.iterrows()}

    oof = C.load_oof(repo)
    util_metrics = set(C.strip(C.metric_cols(oof, UTIL_PREFIX_TRUE), UTIL_PREFIX_TRUE))

    rows: List[Dict[str, Any]] = []
    for m in P["original"] + P["established"] + P["new46"]:
        grp = C.group_for(m, P)
        r = reg_by.get(m, {})
        rows.append({
            "metric_id": m,
            "canonical_name": r.get("metric_name", m),
            "group": grp,
            "subgroup": C.subgroup_for(m, impl, grp),
            "head_group": hg.get(m, "NA"),
            "implementation_subpackage": impl.get(m, "NA"),
            "implementation_source": r.get("implementation_source", "NA"),
            "direction": r.get("direction", "NA"),
            "is_live": m not in P["dead"],
            "used_by_ml2dac": bool(r.get("used_by_ml2dac", False)),
            "used_by_autoclust": bool(r.get("used_by_autoclust", False)),
            "used_by_clustopt": bool(r.get("used_by_clustopt", False)),
            "has_clustopt_utility": m in util_metrics,
            "clustopt_utility_na_reason": (
                "" if m in util_metrics else
                ("dead: all-NaN on the training repository" if m in P["dead"]
                 else "ML2DAC-side CVI; not part of the ClustOpt Full-60 utility "
                      "vocabulary, so no canonical utility exists")),
            "requires_ground_truth": bool(r.get("requires_ground_truth", False)),
        })
    A = pd.DataFrame(rows).sort_values(["group", "subgroup", "metric_id"])
    dead = A[~A["is_live"]].copy()
    dead["dead_reason"] = "100% NaN on the training repository (deterministic)"
    dead["retained_in_contribution_count"] = True
    live = A[A["is_live"]]
    log("atlas: %d requested, %d live, %d dead | groups %s"
        % (len(A), len(live), len(dead),
           live.groupby("group").size().to_dict()))
    log("New46 subgroups: %s"
        % live[live.group == "New46"].groupby("subgroup").size().to_dict())

    defn = utility_definition(repo, sorted(util_metrics))
    return A, dead, defn


def utility_definition(repo: Path, util_metrics: List[str]) -> Dict[str, Any]:
    """The canonical ClustOpt utility, recorded verbatim from the frozen source."""
    src = (repo / "models/ClustOpt/external_evaluation/compute_metric_utility.py")
    return {
        "name": "ClustOpt metric utility_score",
        "authoritative_source": str(src),
        "function": "compute_metric_utilities",
        "invented_for_stage3b": False,
        "definition": (
            "Per (dataset, view), over that record's evaluated clustering "
            "configurations, each metric's vector of values is compared with the "
            "vector of ARI values. utility_score is a fixed weighted sum of eight "
            "bounded ranking-agreement terms."),
        "formula": ("0.18*spearman_pos + 0.12*kendall_pos + 0.20*pairwise_accuracy"
                    " + 0.20*ndcg_all + 0.05*topk_overlap + 0.05*bottomk_overlap"
                    " + 0.10*topk_ndcg + 0.10*bottomk_ndcg"),
        "weights": {"w_spearman": 0.18, "w_kendall": 0.12, "w_pairwise": 0.20,
                    "w_ndcg_all": 0.20, "w_top_overlap": 0.05,
                    "w_bottom_overlap": 0.05, "w_top_ndcg": 0.10,
                    "w_bottom_ndcg": 0.10},
        "candidate_unit": "one evaluated clustering configuration of that record",
        "orientation": (
            "POSITIVE-ONLY. spearman_pos = max(0, spearman_signed) and likewise "
            "for Kendall: a metric that anti-correlates with ARI is misleading the "
            "search and is NOT rewarded. abs() is deliberately not used. Metric "
            "columns are expected normalised to higher = better before this step."),
        "range": "[0, 1], clipped; NaN/inf mapped to 0",
        "higher_is_better": True,
        "nan_handling": (
            "only finite (metric, ARI) pairs are used; a metric is skipped for a "
            "record unless at least max(10, 0.2*n_valid_configs) finite pairs "
            "remain; records with fewer than 10 valid configurations produce no "
            "utilities at all"),
        "min_valid_candidates": "max(10, 0.2 * n_valid_configs)",
        "view_semantics": (
            "computed independently per view (x_only, y_only, xy_2d); a metric may "
            "be useful on one projection and useless on another"),
        "distinct_notions_kept_separate": {
            "true_utility": "the canonical quantity defined above (true_<metric>)",
            "predicted_utility": ("cross-fitted out-of-fold prediction of the same "
                                  "quantity by the deployed utility model "
                                  "(pred_<metric>) -- NEVER mixed with true"),
            "raw_correlation": ("spearman_signed / kendall_signed are retained by "
                                "the source for diagnosis and are NOT the utility"),
            "downstream_ari": ("Stage-1/Stage-3A arm ARIs are a different quantity "
                               "entirely and are never compared numerically with a "
                               "utility"),
        },
        "population_used_in_stage3b": {
            "table": C.OOF_REL,
            "splits": list(range(2, 17)),
            "datasets": 16013, "records": 48039,
            "split_1_present": False,
            "n_metrics_with_utility": len(util_metrics),
        },
        "inferential_unit": "dataset (views averaged); view level reported descriptively",
        "created_at": C.now(),
    }


# ------------------------------------------------- 2. global per-index results
def global_utility(repo: Path, oof: pd.DataFrame, A: pd.DataFrame, log
                   ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tcols = C.metric_cols(oof, UTIL_PREFIX_TRUE)
    metrics = C.strip(tcols, UTIL_PREFIX_TRUE)
    keep = ["dataset_id", "family_name", "subfamily_id", "view_mode", "heldout_split"]
    df = oof[keep + tcols].copy()

    # ---- view-level descriptive (never used for inference) ----
    V = pd.DataFrame({
        "metric_id": metrics,
        "view_mean_utility": [float(np.nanmean(df[c])) for c in tcols],
        "view_n_defined": [int(np.isfinite(df[c]).sum()) for c in tcols],
        "n_view_rows": len(df),
    })

    # ---- dataset-level (the inferential unit) ----
    D = C.dataset_level(df, tcols)
    M = D[tcols].to_numpy(float)
    log("dataset-level utility matrix: %d datasets x %d metrics" % M.shape)
    lo, hi = C.boot_ci_matrix(M)

    # ranks among the metrics defined on that dataset; higher utility = rank 1
    R = pd.DataFrame(M).rank(axis=1, ascending=False, method="average").to_numpy()
    topk = {}
    for k in TOPK_LEVELS:
        topk[k] = np.nanmean(R <= k, axis=0)

    with np.errstate(invalid="ignore"):
        rows = []
        for i, m in enumerate(metrics):
            col = M[:, i]
            fin = np.isfinite(col)
            rows.append({
                "metric_id": m,
                "n_datasets_defined": int(fin.sum()),
                "frac_defined": float(fin.mean()),
                "mean_utility": float(np.nanmean(col)),
                "median_utility": float(np.nanmedian(col)),
                "std_utility": float(np.nanstd(col, ddof=1)),
                "q25_utility": float(np.nanpercentile(col, 25)),
                "q75_utility": float(np.nanpercentile(col, 75)),
                "ci_low": float(lo[i]), "ci_high": float(hi[i]),
                "mean_rank": float(np.nanmean(R[:, i])),
                "median_rank": float(np.nanmedian(R[:, i])),
                "top1_freq": float(topk[1][i]), "top3_freq": float(topk[3][i]),
                "top5_freq": float(topk[5][i]), "top10_freq": float(topk[10][i]),
                "frac_positive_utility": float(np.nanmean(col > 0)),
                "frac_utility_ge_0p5": float(np.nanmean(col >= 0.5)),
                "frac_invalid": float(1.0 - fin.mean()),
            })
    G = pd.DataFrame(rows)
    G = G.merge(A[["metric_id", "group", "subgroup", "head_group", "is_live"]],
                on="metric_id", how="left")
    G["utility_rank_global"] = G["mean_utility"].rank(ascending=False, method="min")
    G = G.sort_values("mean_utility", ascending=False).reset_index(drop=True)
    V = V.merge(A[["metric_id", "group", "subgroup"]], on="metric_id", how="left")
    return G, V, D


#: the five reporting groups, and how each selects its metrics from the atlas
GROUP_SELECTORS = (
    ("Original", "group", "Original"),
    ("Established", "group", "Established"),
    ("New46", "group", "New46"),
    ("New46-Pattern", "subgroup", "New46-Pattern"),
    ("New46-Image", "subgroup", "New46-Image"),
)
#: pairwise group contrasts, fixed BEFORE any value was read
GROUP_CONTRASTS = (
    ("New46", "Original"), ("New46", "Established"),
    ("Established", "Original"), ("New46-Pattern", "New46-Image"),
)


def group_summary(G: pd.DataFrame, D: pd.DataFrame, log
                  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Group-level utility summary plus the predeclared between-group contrasts.

    The contrasts are paired *per dataset*: within a dataset, each group is
    represented by the mean utility of its own metrics, and the groups are then
    compared on the same datasets. The dataset remains the unit of inference.
    """
    rows = []
    members: Dict[str, List[str]] = {}
    for key, col, val in GROUP_SELECTORS:
        sub = G[(G[col] == val) & (G["n_datasets_defined"] > 0)]
        members[key] = list(sub["metric_id"])
        if not len(sub):
            rows.append({"group": key, "n_metrics": 0})
            continue
        rows.append({
            "group": key, "n_metrics": int(len(sub)),
            "mean_of_metric_means": float(sub["mean_utility"].mean()),
            "median_of_metric_means": float(sub["mean_utility"].median()),
            "min_metric_mean": float(sub["mean_utility"].min()),
            "max_metric_mean": float(sub["mean_utility"].max()),
            "mean_top5_freq": float(sub["top5_freq"].mean()),
            "sum_top5_freq": float(sub["top5_freq"].sum()),
            "mean_rank_among_metrics": float(sub["mean_rank"].mean()),
            "n_in_global_top10": int((sub["utility_rank_global"] <= 10).sum()),
            "best_metric": str(sub.sort_values("mean_utility",
                                               ascending=False)["metric_id"].iloc[0]),
            "best_metric_mean_utility": float(sub["mean_utility"].max()),
        })
    S = pd.DataFrame(rows)

    per_group = {}
    for key, ms in members.items():
        cols = [UTIL_PREFIX_TRUE + m for m in ms if UTIL_PREFIX_TRUE + m in D.columns]
        if cols:
            per_group[key] = D[cols].mean(axis=1, skipna=True).to_numpy(float)
    crows = []
    for a, b in GROUP_CONTRASTS:
        if a not in per_group or b not in per_group:
            continue
        d = per_group[a] - per_group[b]
        lo, hi = C.boot_ci_vector(d)
        crows.append({
            "contrast": "%s_minus_%s" % (a, b), "n_datasets": int(np.isfinite(d).sum()),
            "mean_diff": float(np.nanmean(d)), "median_diff": float(np.nanmedian(d)),
            "ci_low": lo, "ci_high": hi, "wilcoxon_p": C.wilcoxon_p(d),
            "frac_positive": float(np.nanmean(d > 0)),
            "note": "within-dataset mean utility of each group's own metrics",
        })
    Ct = pd.DataFrame(crows)
    if len(Ct):
        Ct["wilcoxon_q_bh"] = C.bh_fdr(Ct["wilcoxon_p"].to_numpy())
    log("group contrasts: " + ", ".join(
        "%s %+.4f (q=%.3g)" % (r["contrast"], r["mean_diff"], r["wilcoxon_q_bh"])
        for _, r in Ct.iterrows()))
    return S, Ct
