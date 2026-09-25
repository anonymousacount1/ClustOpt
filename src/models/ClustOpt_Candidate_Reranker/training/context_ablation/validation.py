"""Stage-2B2 integrity assertions."""
from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

N_NEW_CONDITIONS = 3
N_ALL_CONDITIONS = 4
N_FOLDS = 15
N_NEW_UNITS = 45
N_DATASETS = 16013
N_SLATES = 48039
FORBIDDEN_SPLIT = 1
OUTER_SPLITS = tuple(range(2, 17))

FORBIDDEN_FEATURE_TOKENS = {
    "dataset_id", "split_id", "family", "subfamily", "record_id",
    "candidate_ari", "slate_regret", "slate_oracle_ari", "rank_group",
    "is_slate_best", "policy_id", "objective_j", "weighting", "w_raw", "w_soft",
    "trial", "search_history",
}


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


def forbidden_feature_names(names: List[str]) -> List[str]:
    """Feature names that would smuggle identity, target or policy context in."""
    bad = []
    for n in names:
        low = n.lower()
        for tok in FORBIDDEN_FEATURE_TOKENS:
            # Context blocks are prefixed (meta__/ctxutil__); the guard applies
            # to the UNPREFIXED semantic name so a legitimate meta-feature is
            # never flagged by an accidental substring.
            base = low.split("__", 1)[-1] if "__" in low else low
            if base == tok:
                bad.append(n)
                break
    return bad
