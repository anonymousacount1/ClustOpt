"""Candidate targets T1/T2/T3 and the future loss formulations they support.

Targets attach to the UNIQUE candidate (dataset x view x candidate). No policy
context, masking regime or weighting changes a candidate's ARI, so every target
is stored exactly once and joined -- never duplicated across the 10 regimes or
the 20 policy contexts.

Eligibility note: production selects only among ``valid`` candidates
(``offline_candidate_execution.run_arm_offline``), so the slate optimum used for
regret is the max over VALID candidates. The all-candidate max is stored beside
it for transparency; the two differ only when an invalid candidate outscores
every valid one, which the integrity gate counts.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

TARGETS: Tuple[str, ...] = ("T1_ARI", "T2_SLATE_REGRET", "T3_TIE_AWARE_RANK")
FUTURE_LOSSES: Tuple[str, ...] = (
    "T1 pointwise ARI regression",
    "T2 pointwise slate-regret regression",
    "T3 within-slate centered regression",
    "T4 pairwise ranking (within slate)",
    "T5 listwise ranking / NDCG (within slate)",
)
TIE_TOL = 1e-12


def slate_optimum(ari: np.ndarray, valid: np.ndarray) -> Tuple[float, float]:
    """(optimum over VALID candidates, optimum over all candidates)."""
    a = np.asarray(ari, dtype=np.float64)
    v = np.asarray(valid, dtype=bool)
    all_max = float(np.nanmax(a)) if a.size else float("nan")
    val_max = float(np.nanmax(a[v])) if v.any() else float("nan")
    return val_max, all_max


def tie_aware_rank(ari: np.ndarray, *, tol: float = TIE_TOL) -> Dict[str, np.ndarray]:
    """Dense rank groups over a slate, ties preserved rather than broken.

    ``rank_group`` is 0 for the best ARI value, 1 for the next DISTINCT value and
    so on: candidates with equal ARI share a group. ``n_at_rank`` records the
    group size, which is what makes the tie structure recoverable later for
    pairwise/listwise losses without re-deriving it.
    """
    a = np.asarray(ari, dtype=np.float64)
    order = np.argsort(-a, kind="stable")
    group = np.empty(len(a), dtype=np.int32)
    g, prev = 0, None
    for pos in order:
        if prev is not None and abs(a[pos] - prev) > tol:
            g += 1
        group[pos] = g
        prev = a[pos]
    sizes = pd.Series(group).map(pd.Series(group).value_counts()).to_numpy(np.int32)
    return {"rank_group": group, "n_at_rank": sizes,
            "is_slate_best": (group == 0).astype(np.int8)}


def candidate_targets(ari: np.ndarray, valid: np.ndarray) -> Dict[str, Any]:
    """All stored targets for one slate."""
    a = np.asarray(ari, dtype=np.float64)
    val_max, all_max = slate_optimum(a, valid)
    r = tie_aware_rank(a)
    regret = val_max - a
    return {
        "candidate_ari": a,
        "slate_regret": regret,
        "slate_oracle_ari_valid": val_max,
        "slate_oracle_ari_all": all_max,
        **r,
    }


def validate_slate_targets(t: Dict[str, Any], valid: np.ndarray) -> None:
    """Contract for one slate's targets."""
    v = np.asarray(valid, dtype=bool)
    reg = np.asarray(t["slate_regret"], dtype=np.float64)
    if v.any():
        if reg[v].min() < -1e-9:
            raise ValueError("negative regret on a valid candidate")
        if not np.isclose(reg[v].min(), 0.0, atol=1e-9):
            raise ValueError("no zero-regret valid candidate in slate")
    if int(np.asarray(t["rank_group"]).min()) != 0:
        raise ValueError("rank groups do not start at 0")
