"""Unified-vs-dedicated contrasts and the three GO-U indicators."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from models.Clustering_Repository_Builder.experiments.paper_analysis import stats as S

GO_U1_RECOVERY = 0.45
GO_U2_RECOVERY = 0.475
HISTORICAL = {"autoclust_extended": 0.6145, "fp_mlp_top5_selected": 0.5779,
              "fp_online_full_slate_oracle": 0.6549,
              "required_recovery": (0.6145 - 0.5779) / (0.6549 - 0.5779)}
VIEWS = ("x_only", "y_only", "xy_2d")


def paired_row(name: str, a, b, la: str, lb: str, level: str = "dataset"
               ) -> Dict[str, Any]:
    d = np.asarray(a, float) - np.asarray(b, float)
    pt, lo, hi = S.bootstrap_ci(d, statistic="mean")
    row = {"comparison": name, "level": level, "arm_a": la, "arm_b": lb,
           "n": int(len(d)), "mean_a": float(np.mean(a)),
           "mean_b": float(np.mean(b)), "mean_delta": float(pt),
           "median_delta": float(np.median(d)), "ci_lo": float(lo),
           "ci_hi": float(hi), "wins_a": int((d > 1e-12).sum()),
           "wins_b": int((d < -1e-12).sum()),
           "ties": int((np.abs(d) <= 1e-12).sum()),
           "cohens_dz": float(S.cohens_dz(d))}
    try:
        from scipy import stats as sst
        nz = d[np.abs(d) > 1e-12]
        row["wilcoxon_p"] = float(sst.wilcoxon(nz).pvalue) if len(nz) else 1.0
    except Exception:                                              # noqa: BLE001
        row["wilcoxon_p"] = float("nan")
    return row


def regime_summary(g: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "condition_id": g["condition_id"].iloc[0],
        "utility_source": g["utility_source"].iloc[0],
        "k_mode": g["k_mode"].iloc[0], "n_folds": int(len(g)),
        "macro_bestview": float(g["mean_bestview_selected_ari"].mean()),
        "median_bestview": float(g["mean_bestview_selected_ari"].median()),
        "sd_bestview": float(g["mean_bestview_selected_ari"].std(ddof=1)),
        "min_bestview": float(g["mean_bestview_selected_ari"].min()),
        "max_bestview": float(g["mean_bestview_selected_ari"].max()),
        "macro_bestview_oracle": float(g["mean_bestview_candidate_oracle"].mean()),
        "macro_view_selected_ari": float(g["mean_selected_ari"].mean()),
        "macro_oracle_regret": float(g["mean_oracle_regret"].mean()),
        "macro_frac_exact_oracle": float(g["frac_exact_oracle"].mean()),
        "macro_frac_within_001": float(g["frac_within_001"].mean()),
        "macro_frac_within_005": float(g["frac_within_005"].mean()),
        "macro_frac_within_01": float(g["frac_within_01"].mean()),
        "macro_dedicated_local_bestview":
            float(g["dedicated_local_bestview"].mean()),
    }
    pt, lo, hi = S.bootstrap_ci(g["mean_bestview_selected_ari"].to_numpy(float),
                                statistic="mean")
    out["bestview_ci_lo"], out["bestview_ci_hi"] = float(lo), float(hi)
    for lab in ("raw", "softmax"):
        v = g[f"recovery_{lab}"].to_numpy(float)
        p2, l2, h2 = S.bootstrap_ci(v, statistic="mean")
        out[f"macro_b_{lab}_bestview"] = float(g[f"b_{lab}_bestview"].mean())
        out[f"macro_recovery_{lab}"] = float(v.mean())
        out[f"median_recovery_{lab}"] = float(np.median(v))
        out[f"sd_recovery_{lab}"] = float(v.std(ddof=1))
        out[f"min_recovery_{lab}"] = float(v.min())
        out[f"max_recovery_{lab}"] = float(v.max())
        out[f"recovery_{lab}_ci_lo"], out[f"recovery_{lab}_ci_hi"] = (
            float(l2), float(h2))
        out[f"folds_beating_{lab}"] = int(g[f"beats_{lab}"].sum())
    out["pipeline_delta_vs_dedicated_local"] = (
        out["macro_bestview"] - out["macro_dedicated_local_bestview"])
    out["folds_beating_dedicated_local"] = int(
        (g["mean_bestview_selected_ari"].to_numpy()
         > g["dedicated_local_bestview"].to_numpy()).sum())
    for v in VIEWS:
        out[f"{v}__macro_selected_ari"] = float(g[f"{v}__mean_selected_ari"].mean())
        out[f"{v}__macro_regret"] = float(g[f"{v}__mean_regret"].mean())
        out[f"{v}__macro_frac_exact"] = float(g[f"{v}__frac_exact"].mean())
    return out


def go_u1(summ: pd.DataFrame) -> Dict[str, Any]:
    rows = []
    for _, r in summ.iterrows():
        ok = bool(r["macro_recovery_raw"] >= GO_U1_RECOVERY
                  and r["folds_beating_raw"] >= 12)
        rows.append({"condition_id": r["condition_id"],
                     "macro_recovery_raw": float(r["macro_recovery_raw"]),
                     "folds_beating_raw": int(r["folds_beating_raw"]),
                     "PASS": ok})
    best = max(rows, key=lambda x: x["macro_recovery_raw"])
    return {"PASS": any(x["PASS"] for x in rows), "best": best,
            "threshold": GO_U1_RECOVERY, "evaluated": rows}


def go_u2(summ: pd.DataFrame) -> Dict[str, Any]:
    r = summ[summ["condition_id"] == "MLP_TOP5"].iloc[0]
    a = bool(r["macro_recovery_raw"] >= GO_U2_RECOVERY)
    b = bool(r["folds_beating_raw"] >= 12)
    return {"condition_id": "MLP_TOP5",
            "macro_recovery_raw": float(r["macro_recovery_raw"]),
            "recovery_ci": [float(r["recovery_raw_ci_lo"]),
                            float(r["recovery_raw_ci_hi"])],
            "min_recovery_raw": float(r["min_recovery_raw"]),
            "max_recovery_raw": float(r["max_recovery_raw"]),
            "folds_beating_raw": int(r["folds_beating_raw"]),
            "A_recovery_ge_0.475": a, "B_folds_ge_12": b,
            "PASS": bool(a and b), "historical": HISTORICAL,
            "caveat": "fixed32 development magnitude only; not an AutoClust "
                      "comparison and not an online result"}


def go_u3(uni_folds: pd.DataFrame, ded_folds: pd.DataFrame,
          dataset_delta: Dict[str, Any]) -> Dict[str, Any]:
    u = uni_folds.sort_values("outer_split")["mean_bestview_selected_ari"].to_numpy()
    d = ded_folds.sort_values("outer_split")["mean_bestview_selected_ari"].to_numpy()
    wins = int((u > d).sum())
    macro_delta = float(u.mean() - d.mean())
    a = bool(macro_delta > 0)
    b = bool(dataset_delta["mean_delta"] > 0)
    c = bool(wins >= 10)
    return {"comparison": "Unified C2 MLP_TOP10 vs Dedicated C2 MLP_TOP10",
            "unified_macro_bestview": float(u.mean()),
            "dedicated_macro_bestview": float(d.mean()),
            "A_positive_macro_delta": a, "macro_delta": macro_delta,
            "B_positive_dataset_paired_delta": b,
            "dataset_paired_delta": dataset_delta["mean_delta"],
            "dataset_paired_ci": [dataset_delta["ci_lo"], dataset_delta["ci_hi"]],
            "C_wins_ge_10_of_15": c, "folds_won": wins,
            "PASS": bool(a and b and c),
            "note": "this is the only mechanism-clean contrast in Stage 2B-3: "
                    "same regime, context, target, model and folds; the only "
                    "difference is dedicated vs mixed-regime training"}
