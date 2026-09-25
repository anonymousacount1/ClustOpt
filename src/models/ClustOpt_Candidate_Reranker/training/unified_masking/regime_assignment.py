"""Deterministic, target-independent, balanced regime assignment.

One regime per outer-TRAINING slate. The assignment is a pure function of
identity strings -- it never touches ARI, regret, oracle or any model output, so
it cannot leak the target. It is recomputed identically on every resume rather
than stored as the source of truth.

Balance is exact by construction: within each view the slates are sorted by their
identity hash and dealt round-robin over the frozen 10-regime order, so any two
regimes differ by at most one slate per view.
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from ...protocol import masking_regimes as MR

SEED = 42
NAMESPACE = "stage2b3"
VIEWS: Tuple[str, ...] = ("x_only", "y_only", "xy_2d")

# Frozen deal order. RICH is absent by design: it is not a deployable regime.
REGIME_ORDER: Tuple[str, ...] = tuple(
    f"{s}__{k}" for s in MR.SOURCES for k in MR.K_MODES)
CONDITION_ORDER: Tuple[str, ...] = tuple(
    f"{s.upper()}_{k.upper()}" for s in MR.SOURCES for k in MR.K_MODES)
N_REGIMES = len(REGIME_ORDER)


def regime_condition_id(regime_id: str) -> str:
    src, km = regime_id.split("__")
    return f"{src.upper()}_{km.upper()}"


def slate_hash(outer_split: int, dataset_id: str, view_id: str) -> str:
    payload = "|".join([NAMESPACE, f"seed={SEED}", f"outer={outer_split}",
                        dataset_id, view_id])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def assign(outer_split: int, dataset_ids: Sequence[str],
           view_ids: Sequence[str]) -> pd.DataFrame:
    """One regime per slate, dealt round-robin within each view by hash order."""
    df = pd.DataFrame({"dataset_id": np.asarray(dataset_ids),
                       "view_id": np.asarray(view_ids)})
    df["assignment_hash"] = [slate_hash(outer_split, d, v)
                             for d, v in zip(df["dataset_id"], df["view_id"])]
    df["assigned_regime"] = ""
    for v in VIEWS:
        m = df["view_id"] == v
        if not m.any():
            continue
        idx = df.index[m]
        order = np.argsort(df.loc[idx, "assignment_hash"].to_numpy(), kind="stable")
        ranked = idx[order]
        df.loc[ranked, "assigned_regime"] = [
            REGIME_ORDER[i % N_REGIMES] for i in range(len(ranked))]
    df["outer_split"] = int(outer_split)
    df["regime_index"] = df["assigned_regime"].map(
        {r: i for i, r in enumerate(REGIME_ORDER)})
    return df


def manifest_hash(df: pd.DataFrame) -> str:
    """Stable hash of the whole assignment, for the unit identity."""
    key = (df["dataset_id"].astype(str) + "|" + df["view_id"].astype(str)
           + "|" + df["assigned_regime"].astype(str))
    return hashlib.sha256("\n".join(sorted(key.tolist())).encode()).hexdigest()[:16]


def balance_report(df: pd.DataFrame) -> Dict[str, Any]:
    overall = df["assigned_regime"].value_counts().reindex(
        list(REGIME_ORDER)).fillna(0).astype(int)
    per_view = (df.groupby(["view_id", "assigned_regime"]).size()
                .unstack(fill_value=0).reindex(columns=list(REGIME_ORDER),
                                               fill_value=0))
    within = int((per_view.max(axis=1) - per_view.min(axis=1)).max())
    return {
        "n_slates": int(len(df)),
        "counts": overall.to_dict(),
        "overall_imbalance": int(overall.max() - overall.min()),
        "per_view_counts": {v: per_view.loc[v].to_dict()
                            for v in per_view.index},
        "max_within_view_imbalance": within,
        "balanced_within_view": bool(within <= 1),
    }


def validate(df: pd.DataFrame, expected_slates: int) -> None:
    """Hard contract for one fold's assignment."""
    if len(df) != expected_slates:
        raise AssertionError(f"assignment covers {len(df)} slates, "
                             f"expected {expected_slates}")
    if df.duplicated(subset=["dataset_id", "view_id"]).any():
        raise AssertionError("a slate received more than one regime")
    if set(df["assigned_regime"]) - set(REGIME_ORDER):
        raise AssertionError("unknown regime in assignment")
    rep = balance_report(df)
    if rep["max_within_view_imbalance"] > 1:
        raise AssertionError(
            f"within-view imbalance {rep['max_within_view_imbalance']} > 1")


def target_independence_statement() -> Dict[str, Any]:
    return {
        "inputs": ["namespace 'stage2b3'", "seed=42", "outer_split",
                   "dataset_id", "view_id"],
        "excluded": ["ARI", "slate regret", "candidate oracle", "any model "
                     "output", "any fold performance"],
        "argument": "the assignment is a pure function of identity strings; no "
                    "target value is read anywhere in this module, so the "
                    "assignment cannot encode the target",
        "determinism": "SHA-256 over a fixed payload, stable sort, fixed deal "
                       "order -- recomputed identically on every resume",
    }
