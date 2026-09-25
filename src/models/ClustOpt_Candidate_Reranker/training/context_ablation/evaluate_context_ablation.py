"""Paired context contrasts and the GO-CONTEXT gate."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from models.Clustering_Repository_Builder.experiments.paper_analysis import stats as S

from . import context_feature_builder as CFB

# The six pre-registered contrasts (brief section 24).
CONTRASTS: Tuple[Tuple[str, str, str], ...] = (
    ("A_local_vs_meta", "C1_LOCAL_META", "C0_LOCAL"),
    ("B_local_vs_util", "C2_LOCAL_UTIL", "C0_LOCAL"),
    ("C_local_vs_meta_util", "C3_LOCAL_META_UTIL", "C0_LOCAL"),
    ("D_meta_increment_after_util", "C3_LOCAL_META_UTIL", "C2_LOCAL_UTIL"),
    ("E_util_increment_after_meta", "C3_LOCAL_META_UTIL", "C1_LOCAL_META"),
    ("F_util_vs_meta", "C2_LOCAL_UTIL", "C1_LOCAL_META"),
)
GO_RECOVERY_THRESHOLD = 0.45
VERY_STRONG_THRESHOLD = 0.475
HISTORICAL = {"autoclust_extended": 0.6145, "fp_mlp_top5_selected": 0.5779,
              "fp_online_full_slate_oracle": 0.6549,
              "required_recovery": (0.6145 - 0.5779) / (0.6549 - 0.5779)}


def paired_row(name: str, a, b, la: str, lb: str, level: str = "dataset"
               ) -> Dict[str, Any]:
    d = np.asarray(a, float) - np.asarray(b, float)
    pt, lo, hi = S.bootstrap_ci(d, statistic="mean")
    row = {"contrast": name, "level": level, "arm_a": la, "arm_b": lb,
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


def condition_summary(fold_rows: pd.DataFrame) -> Dict[str, Any]:
    d = fold_rows
    cid = d["condition_id"].iloc[0]
    out: Dict[str, Any] = {
        "condition_id": cid, "input_dim": int(d["n_features"].iloc[0]),
        "n_folds": int(len(d)),
        "macro_bestview": float(d["mean_bestview_selected_ari"].mean()),
        "median_bestview": float(d["mean_bestview_selected_ari"].median()),
        "sd_bestview": float(d["mean_bestview_selected_ari"].std()),
        "min_bestview": float(d["mean_bestview_selected_ari"].min()),
        "max_bestview": float(d["mean_bestview_selected_ari"].max()),
        "macro_bestview_oracle": float(d["mean_bestview_candidate_oracle"].mean()),
        "macro_view_selected_ari": float(d["mean_selected_ari"].mean()),
        "macro_oracle_regret": float(d["mean_oracle_regret"].mean()),
        "macro_median_regret": float(d["median_oracle_regret"].mean()),
        "macro_frac_exact_oracle": float(d["frac_exact_oracle"].mean()),
        "macro_frac_within_001": float(d["frac_within_001"].mean()),
        "macro_frac_within_005": float(d["frac_within_005"].mean()),
        "macro_frac_within_01": float(d["frac_within_01"].mean()),
        "macro_b_raw_bestview": float(d["b_raw_bestview"].mean()),
        "macro_b_softmax_bestview": float(d["b_softmax_bestview"].mean()),
        "folds_beating_raw": int(d["beats_raw"].sum()),
        "folds_beating_softmax": int(d["beats_softmax"].sum()),
    }
    pt, lo, hi = S.bootstrap_ci(d["mean_bestview_selected_ari"].to_numpy(float),
                                statistic="mean")
    out["bestview_ci_lo"], out["bestview_ci_hi"] = float(lo), float(hi)
    for lab in ("raw", "softmax"):
        v = d[f"recovery_{lab}"].to_numpy(float)
        p2, l2, h2 = S.bootstrap_ci(v, statistic="mean")
        out[f"macro_recovery_{lab}"] = float(v.mean())
        out[f"median_recovery_{lab}"] = float(np.median(v))
        out[f"sd_recovery_{lab}"] = float(v.std(ddof=1))
        out[f"min_recovery_{lab}"] = float(v.min())
        out[f"max_recovery_{lab}"] = float(v.max())
        out[f"recovery_{lab}_ci_lo"], out[f"recovery_{lab}_ci_hi"] = (
            float(l2), float(h2))
        out[f"folds_recovery_{lab}_positive"] = int((v > 0).sum())
    for v in ("x_only", "y_only", "xy_2d"):
        out[f"{v}__macro_selected_ari"] = float(d[f"{v}__mean_selected_ari"].mean())
        out[f"{v}__macro_regret"] = float(d[f"{v}__mean_regret"].mean())
        out[f"{v}__macro_frac_exact"] = float(d[f"{v}__frac_exact"].mean())
    out["total_fit_sec"] = float(d["fit_sec"].sum()) if "fit_sec" in d else 0.0
    return out


def go_context(sel: Dict[str, Any], folds_beating_c0: int,
               paired_delta_vs_c0: float, c0_bestview: float) -> Dict[str, Any]:
    a = bool(sel["macro_recovery_raw"] >= GO_RECOVERY_THRESHOLD)
    b = bool(folds_beating_c0 >= 12)
    c = bool(paired_delta_vs_c0 > 0)
    dd = bool(sel["macro_bestview"] > c0_bestview)
    return {
        "condition_id": sel["condition_id"],
        "A_macro_recovery_raw_ge_0.45": a, "A_value": sel["macro_recovery_raw"],
        "B_beats_C0_ge_12_of_15": b, "B_value": folds_beating_c0,
        "C_positive_paired_delta_vs_C0": c, "C_value": paired_delta_vs_c0,
        "D_macro_bestview_gt_C0": dd,
        "D_values": [sel["macro_bestview"], c0_bestview],
        "PASS": bool(a and b and c and dd),
        "very_strong_indicator_ge_0.475":
            bool(sel["macro_recovery_raw"] >= VERY_STRONG_THRESHOLD),
        "historical_anchor": HISTORICAL,
        "interpretation_note": (
            "0.475 is a development recovery MAGNITUDE anchor computed on the "
            "historical online FP MLP_TOP5 slate. Reaching it on fixed32 "
            "MLP_TOP10 does not mean AutoClust is beaten."),
    }


def select_context(summ: pd.DataFrame, beats: Dict[str, int]) -> Dict[str, Any]:
    """Deterministic ladder from brief section 27."""
    r = summ.copy()
    r["folds_beating_c0"] = r["condition_id"].map(beats).fillna(0)
    r["_cx"] = r["condition_id"].map(CFB.COMPLEXITY_ORDER)
    r = r.sort_values(
        by=["macro_bestview", "macro_recovery_raw", "macro_recovery_softmax",
            "folds_beating_c0", "min_bestview", "_cx"],
        ascending=[False, False, False, False, False, True])
    w = r.iloc[0]
    return {"condition_id": w["condition_id"],
            "input_dim": int(w["input_dim"]),
            "macro_bestview": float(w["macro_bestview"]),
            "macro_recovery_raw": float(w["macro_recovery_raw"]),
            "macro_recovery_softmax": float(w["macro_recovery_softmax"]),
            "folds_beating_c0": int(w["folds_beating_c0"]),
            "min_bestview": float(w["min_bestview"]),
            "ladder": r["condition_id"].tolist(),
            "purpose": "freezes ONE context representation for later reranker "
                       "development; does not freeze a final reranker or a "
                       "final ClustOpt policy"}
