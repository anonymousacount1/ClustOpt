"""Stage 3B step 8: cross-method summary, publication tables, forward plan.

The single most important rule in this file: evidence types are kept apart. A
permutation reliance score, a selection frequency, a utility rank and an ARI
ablation are four different quantities on four different scales, and none of
them is ever numerically compared with another. Each column carries the evidence
level it belongs to.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from . import common as C

#: modern structural CVIs the eventual metrics paper would be expected to
#: compare against. Presence is decided against the frozen registry, never
#: assumed; nothing here is implemented or evaluated in Stage 3B.
MODERN_COMPARATORS = {
    "DBCV": ["dbcv"],
    "S_Dbw": ["s_dbw"],
    "Silhouette": ["silhouette"],
    "Noise-aware Silhouette": ["noise_aware_silhouette"],
    "Calinski-Harabasz": ["calinski_harabasz"],
    "Davies-Bouldin": ["davies_bouldin"],
    "Dunn": ["dunn_index"],
    "COP": ["cop"],
    "Coggins-Jain": ["coggins_jain_index"],
    "CDbw": ["cdbw", "c_dbw"],
    "CVNN": ["cvnn"],
    "CVDD": ["cvdd"],
    "DCSI": ["dcsi"],
}


def comparator_coverage(A: pd.DataFrame, repo: Path) -> Dict[str, Any]:
    live = set(A[A.is_live]["metric_id"])
    allm = set(A["metric_id"])
    ev_cols = set()
    try:
        import pyarrow.parquet as pq
        ev_cols = set(pq.ParquetFile(
            repo / C.MKR_REL / "master" / "evaluations.parquet").schema_arrow.names)
    except Exception:
        pass
    avail, unavail = {}, {}
    for name, aliases in MODERN_COMPARATORS.items():
        hit = [a for a in aliases if a in allm or a in ev_cols]
        if hit:
            avail[name] = {"metric_id": hit[0],
                           "live": hit[0] in live,
                           "in_evaluations_table": hit[0] in ev_cols,
                           "group": str(A.set_index("metric_id")["group"].get(hit[0], "NA"))}
        else:
            unavail[name] = {"searched_aliases": aliases,
                             "present_anywhere_in_evaluations": False}
    return {
        "stage": "3B", "created_at": C.now(),
        "policy": ("Coverage is decided by what already exists. Stage 3B does NOT "
                   "implement or evaluate any missing comparator; listing one as "
                   "unavailable is a statement about this repository, not about "
                   "the metric."),
        "already_available": avail,
        "unavailable": unavail,
        "n_available": len(avail), "n_unavailable": len(unavail),
        "alias_equivalent_notes": {
            "convexity_solidity_image": "alias of convexity_ratio_image_based (dead)",
            "hough_circle_arc_strength": "alias of hough_arc_circle_strength (dead)",
            "ML2DAC short names": "SIL/CH/DBI/DBCV/DI/CJI/COP map to canonical ids",
        },
        "future_external_domain_requirement": sorted(unavail),
        "claims_permitted_now": ("comparisons against the modern CVIs listed as "
                                 "available only; no claim may be made against "
                                 "CDbw/CVNN/CVDD/DCSI unless they appear above"),
    }


def metric_publication_table(A: pd.DataFrame, G: pd.DataFrame, Cp: pd.DataFrame,
                             U: pd.DataFrame, MS: pd.DataFrame, FAM: pd.DataFrame,
                             AR: pd.DataFrame) -> pd.DataFrame:
    """One compact row per live metric. NA is used with an explicit reason."""
    T = A[A.is_live][["metric_id", "group", "subgroup", "head_group",
                      "has_clustopt_utility", "clustopt_utility_na_reason"]].copy()
    g = G.set_index("metric_id")
    T["global_utility"] = T["metric_id"].map(g["mean_utility"])
    T["global_utility_rank"] = T["metric_id"].map(g["utility_rank_global"])
    T["top5_freq"] = T["metric_id"].map(g["top5_freq"])

    cp = Cp.set_index("metric_id")
    T["oracle_win_freq"] = T["metric_id"].map(cp["oracle_win_freq"])
    T["beats_best_comparator_rate"] = T["metric_id"].map(cp["frac_beats_best_comparator"])

    best_fam = (FAM.sort_values("mean_utility", ascending=False)
                .drop_duplicates("metric_id").set_index("metric_id"))
    T["strongest_family"] = T["metric_id"].map(best_fam["family_name"])
    T["strongest_family_top5_enrichment"] = T["metric_id"].map(best_fam["top5_enrichment"])

    u = U.set_index("metric_id")
    T["clustopt_selected_freq"] = T["metric_id"].map(u["selected_freq"])
    T["clustopt_pred_top5_freq"] = T["metric_id"].map(u["pred_top5_freq"])
    T["clustopt_pred_true_spearman"] = T["metric_id"].map(u["spearman_pred_true"])

    a3 = MS[MS.arm == "A3"].set_index("metric_id")
    T["ml2dac_a3_selection_freq"] = T["metric_id"].map(a3["selection_freq"]).fillna(0.0)

    ar = AR.set_index("metric_id")
    T["autoclust_reliance_rmse_increase"] = T["metric_id"].map(ar["rmse_increase"])
    T["autoclust_reliance_rank"] = T["metric_id"].map(ar["reliance_rank"])

    T["na_reason_utility"] = np.where(
        T["has_clustopt_utility"], "", T["clustopt_utility_na_reason"])
    T["na_reason_clustopt_usage"] = np.where(
        T["has_clustopt_utility"], "",
        "not in the ClustOpt Full-60 utility vocabulary")
    T["evidence_levels"] = ("utility/top5/oracle/selection/reliance = LEVEL 2 "
                            "association; no LEVEL 3 single-metric causal value "
                            "exists for any metric")
    return T.sort_values(["group", "subgroup", "global_utility"],
                         ascending=[True, True, False]).reset_index(drop=True)


def family_publication_table(A: pd.DataFrame, FAM: pd.DataFrame, Cov: pd.DataFrame,
                             CF: pd.DataFrame, MF: pd.DataFrame, AF: pd.DataFrame,
                             fam_map: Dict[str, str]) -> pd.DataFrame:
    """``fam_map`` translates the raw family id used by the baseline tables into
    the display family name used by the utility tables; without it the ML2DAC and
    AutoClust columns silently join to nothing."""
    MF = MF.copy()
    MF["family_name"] = MF["family"].map(fam_map).fillna(MF["family"])
    AF = AF.copy()
    AF["family_name"] = AF["family"].map(fam_map).fillna(AF["family"])
    grp = A.set_index("metric_id")["group"].to_dict()
    sub = A.set_index("metric_id")["subgroup"].to_dict()
    F = FAM.copy()
    F["group"] = F["metric_id"].map(grp)
    F["subgroup"] = F["metric_id"].map(sub)

    def best(fdf, mask, col="mean_utility"):
        s = fdf[mask]
        if not len(s):
            return "", np.nan
        r = s.sort_values(col, ascending=False).iloc[0]
        return str(r["metric_id"]), float(r[col])

    cov = Cov[Cov.level == "family"].set_index("key")
    rows = []
    for fam, fdf in F.groupby("family_name"):
        bo, bov = best(fdf, fdf.group == "Original")
        be, bev = best(fdf, fdf.group == "Established")
        bn, bnv = best(fdf, fdf.group == "New46")
        bp, bpv = best(fdf, fdf.subgroup == "New46-Pattern")
        bi, biv = best(fdf, fdf.subgroup == "New46-Image")
        cf = CF[CF.family_name == fam]
        new_sel = cf[cf.group == "New46"]["selected_freq"].sum() / C.CLUSTOPT_TOP_K
        mf = MF[(MF.family_name == fam) & (MF.arm == "A3")]
        af = AF[AF.family_name == fam]
        rows.append({
            "family": fam,
            "best_original_metric": bo, "best_original_utility": bov,
            "best_established_metric": be, "best_established_utility": bev,
            "best_new46_metric": bn, "best_new46_utility": bnv,
            "best_new46_pattern_metric": bp, "best_new46_pattern_utility": bpv,
            "best_new46_image_metric": bi, "best_new46_image_utility": biv,
            "new46_oracle_win_rate": float(cov["new46_best_frac"].get(fam, np.nan)),
            "new46_complementarity_gain": bnv - max(bov if np.isfinite(bov) else -np.inf,
                                                    bev if np.isfinite(bev) else -np.inf),
            "clustopt_new46_selection_rate": float(new_sel),
            "ml2dac_a3_new46_selection_rate": (float(mf["share_New46"].iloc[0])
                                               if len(mf) else np.nan),
            "autoclust_new46_pattern_reliance_pct": (
                float(af[af.group == "New46-Pattern"]["rmse_increase_pct"].iloc[0])
                if len(af[af.group == "New46-Pattern"]) else np.nan),
            "autoclust_new46_image_reliance_pct": (
                float(af[af.group == "New46-Image"]["rmse_increase_pct"].iloc[0])
                if len(af[af.group == "New46-Image"]) else np.nan),
        })
    return pd.DataFrame(rows).sort_values("new46_oracle_win_rate", ascending=False)


def metric_group_summary(S: pd.DataFrame, Q: pd.DataFrame, Arm: pd.DataFrame,
                         Gp: pd.DataFrame) -> pd.DataFrame:
    """Cross-method group summary. Every column names its evidence type."""
    rows = []
    a3 = Arm[Arm.arm == "A3"].iloc[0] if len(Arm[Arm.arm == "A3"]) else None
    ml_share = {"Original": "selected_share_Original",
                "Established": "selected_share_Established",
                "New46": "selected_share_New46"}
    for _, r in S.iterrows():
        g = r["group"]
        q = Q[Q.group == g]
        gp = Gp[Gp.group == g]
        rows.append({
            "group": g, "n_metrics": r.get("n_metrics"),
            "A_intrinsic_mean_utility": r.get("mean_of_metric_means"),
            "A_intrinsic_best_metric": r.get("best_metric"),
            "A_n_in_global_top10": r.get("n_in_global_top10"),
            "B_clustopt_selection_slot_share": (float(q["share_of_selection_slots"].iloc[0])
                                                if len(q) else np.nan),
            "B_clustopt_pred_true_spearman": (float(q["mean_spearman_pred_true"].iloc[0])
                                              if len(q) else np.nan),
            "C_ml2dac_a3_selected_share": (float(a3[ml_share[g]])
                                           if a3 is not None and g in ml_share else np.nan),
            "D_autoclust_reliance_rmse_pct": (float(gp["rmse_increase_pct"].iloc[0])
                                              if len(gp) else np.nan),
            "D_autoclust_reliance_per_feature": (
                float(gp["rmse_increase"].iloc[0] / gp["n_features_permuted"].iloc[0])
                if len(gp) else np.nan),
            "E_causal_bundle_note": (
                "AutoClust MainEffect_New46 +0.017540 (p=3.3e-23); ML2DAC "
                "MainEffect_New46 -0.005414 (ns). Bundle-level ARI ablation, "
                "Stage 3A. NOT comparable in scale with any column above."),
            "evidence_levels": "A/B/C/D = LEVEL 2 association; E = LEVEL 1 causal",
        })
    return pd.DataFrame(rows)
