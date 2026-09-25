"""Stage-2B0 integrity assertions, kept separate from the builders that use them."""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

EXPECTED_DATASETS = 16013
EXPECTED_SLATES = 48039
EXPECTED_CANDIDATES_PER_SLATE = 32
EXPECTED_CANDIDATE_ROWS = 1537248
EXPECTED_REGIMES = 10
EXPECTED_POLICY_CONTEXTS = 20
EXPECTED_MASK_ROWS = 480390
LOGICAL_CANDIDATE_REGIME_OBSERVATIONS = 15372480     # 1,537,248 x 10
LOGICAL_CANDIDATE_POLICY_OBSERVATIONS = 30744960     # 1,537,248 x 20
FORBIDDEN_SPLIT = 1
SPLITS = tuple(range(2, 17))


class Check:
    """Accumulates named PASS/FAIL rows for the gate table."""

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


def slate_sizes_ok(ident: pd.DataFrame) -> bool:
    g = ident.groupby(["dataset_id", "view_id"]).size()
    return bool((g == EXPECTED_CANDIDATES_PER_SLATE).all())


def candidate_ids_unique_within_slate(ident: pd.DataFrame) -> bool:
    d = ident.duplicated(subset=["dataset_id", "view_id", "candidate_key"])
    return not bool(d.any())


def regret_valid(targets: pd.DataFrame, partition: pd.DataFrame) -> Dict[str, Any]:
    v = partition["part__valid"].to_numpy(bool)
    r = targets["slate_regret"].to_numpy(float)
    return {"min_regret_valid": float(r[v].min()) if v.any() else float("nan"),
            "n_negative_valid": int((r[v] < -1e-9).sum()),
            "n_nan": int(np.isnan(r).sum())}


def zero_regret_per_slate(ident: pd.DataFrame, targets: pd.DataFrame,
                          partition: pd.DataFrame) -> int:
    """Slates with no zero-regret VALID candidate (must be 0)."""
    f = pd.DataFrame({
        "dataset_id": ident["dataset_id"].to_numpy(),
        "view_id": ident["view_id"].to_numpy(),
        "regret": targets["slate_regret"].to_numpy(float),
        "valid": partition["part__valid"].to_numpy(bool),
    })
    f = f[f["valid"]]
    mins = f.groupby(["dataset_id", "view_id"])["regret"].min()
    return int((mins > 1e-9).sum())


def rank_groups_valid(ident: pd.DataFrame, targets: pd.DataFrame) -> int:
    """Slates whose dense rank groups do not start at 0 (must be 0)."""
    f = pd.DataFrame({
        "dataset_id": ident["dataset_id"].to_numpy(),
        "view_id": ident["view_id"].to_numpy(),
        "rank_group": targets["rank_group"].to_numpy(int),
    })
    mins = f.groupby(["dataset_id", "view_id"])["rank_group"].min()
    return int((mins != 0).sum())


def no_split_one(*frames: pd.DataFrame) -> bool:
    for f in frames:
        if "split_id" in f.columns and (f["split_id"] == FORBIDDEN_SPLIT).any():
            return False
    return True


def join_is_exact(left_keys: Sequence, right_keys: Sequence) -> bool:
    return set(left_keys) == set(right_keys)
