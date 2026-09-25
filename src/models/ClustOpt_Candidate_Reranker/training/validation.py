"""Stage-2B1 integrity assertions."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

N_CONDITIONS = 11
N_FOLDS = 15
N_UNITS = 165
N_DATASETS = 16013
N_SLATES = 48039
N_CANDIDATES = 1537248
CANDIDATES_PER_SLATE = 32
FORBIDDEN_SPLIT = 1
OUTER_SPLITS = tuple(range(2, 17))

# Pre-registered feasibility interpretation bands (research bands, not tests).
BANDS = (
    (-np.inf, 0.0, "no useful reranking signal"),
    (0.0, 0.20, "weak"),
    (0.20, 0.35, "learnable but likely insufficient alone"),
    (0.35, 0.475, "promising / close to historical requirement"),
    (0.475, 0.60, "strong historical-transfer candidate"),
    (0.60, np.inf, "very strong evidence reranking is the dominant Stage-2 path"),
)
HISTORICAL_REQUIREMENT = 0.475


def band_of(recovery: float) -> str:
    if not np.isfinite(recovery):
        return "undefined"
    for lo, hi, label in BANDS:
        if lo < recovery <= hi or (lo == -np.inf and recovery <= hi):
            return label
    return "undefined"


class Check:
    def __init__(self) -> None:
        self.rows: List[Dict[str, Any]] = []

    def add(self, cid: str, desc: str, ok: bool, detail: Any = "") -> bool:
        self.rows.append({"check_id": cid, "description": desc,
                          "status": "PASS" if ok else "FAIL",
                          "detail": str(detail)})
        print(f"  [{'PASS' if ok else 'FAIL'}] {cid:<5} {desc}"
              + (f"  ({detail})" if detail != "" else ""), flush=True)
        return ok

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    @property
    def n_failed(self) -> int:
        return sum(1 for r in self.rows if r["status"] == "FAIL")


def go_a(anchor: Dict[str, Any]) -> Dict[str, Any]:
    """Historical anchor: MLP_TOP5 vs its RAW native baseline."""
    a1 = bool(anchor["macro_recovery_raw"] >= HISTORICAL_REQUIREMENT)
    a2 = bool(anchor["folds_beating_raw"] >= 12)
    a3 = bool(anchor["macro_bestview"] > anchor["macro_b_raw_bestview"])
    return {"A1_macro_recovery_raw_ge_0.475": a1,
            "A1_value": anchor["macro_recovery_raw"],
            "A2_beats_raw_ge_12_of_15": a2,
            "A2_value": anchor["folds_beating_raw"],
            "A3_macro_bestview_gt_raw": a3,
            "A3_values": [anchor["macro_bestview"],
                          anchor["macro_b_raw_bestview"]],
            "PASS": bool(a1 and a2 and a3)}


def go_b(rows: pd.DataFrame) -> Dict[str, Any]:
    """Any deployable regime reaching macro recovery >= 0.45 against RAW or SOFTMAX."""
    best, detail = None, []
    for _, r in rows.iterrows():
        for lab in ("raw", "softmax"):
            rec = r[f"macro_recovery_{lab}"]
            b1 = bool(np.isfinite(rec) and rec >= 0.45)
            b2 = bool(r[f"folds_beating_{lab}"] >= 12)
            b3 = bool(r[f"mean_paired_bestview_delta_{lab}"] > 0)
            ok = b1 and b2 and b3
            detail.append({"condition_id": r["condition_id"], "baseline": lab,
                           "B1_recovery_ge_0.45": b1, "recovery": float(rec),
                           "B2_folds_ge_12": b2,
                           "folds": int(r[f"folds_beating_{lab}"]),
                           "B3_positive_paired_delta": b3,
                           "paired_delta": float(
                               r[f"mean_paired_bestview_delta_{lab}"]),
                           "PASS": ok})
            if ok and (best is None or rec > best["recovery"]):
                best = detail[-1]
    return {"PASS": best is not None, "winner": best, "evaluated": detail}


def pick_best_deployable(rows: pd.DataFrame) -> Dict[str, Any]:
    """Deterministic descriptive ladder (brief section 36)."""
    k_order = {"top1": 1, "top3": 3, "top5": 5, "dynamic": 5.5, "top10": 10}
    r = rows.copy()
    r["_k"] = r["k_mode"].map(k_order)
    r = r.sort_values(
        by=["macro_bestview", "macro_recovery_raw", "macro_recovery_softmax",
            "folds_beating_both", "macro_oracle_regret", "_k"],
        ascending=[False, False, False, False, True, True])
    win = r.iloc[0]
    return {"condition_id": win["condition_id"],
            "utility_source": win["utility_source"], "k_mode": win["k_mode"],
            "macro_bestview": float(win["macro_bestview"]),
            "macro_recovery_raw": float(win["macro_recovery_raw"]),
            "macro_recovery_softmax": float(win["macro_recovery_softmax"]),
            "macro_oracle_regret": float(win["macro_oracle_regret"]),
            "ladder": r["condition_id"].tolist(),
            "purpose": "designs Stage 2B-2 only; does not freeze a final reranker"}
