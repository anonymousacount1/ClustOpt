"""Target formulations T1/T2/T3 and their selection rules.

All three keep the full 20-dimensional continuous vector. No hard multiclass
target is ever constructed: Stage 2A-2 measured ~99% exact ties and ~13% of rows
where all 20 policies tie, so a single "best policy" label would be ill-posed.

Targets are used in their native scale -- the 20 outputs are NOT independently
standardised, because argmax/argmin selection requires them to stay comparable
across policies.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

TARGETS: Tuple[str, ...] = ("ARI", "REGRET", "CENTERED")


def transform(ari: np.ndarray, target: str) -> np.ndarray:
    """ARI matrix (n x 20) -> training target in the requested formulation."""
    if target == "ARI":                       # T1: absolute
        return ari.astype(np.float32)
    if target == "REGRET":                    # T2: r = rowmax - ARI  (>= 0)
        return (ari.max(axis=1, keepdims=True) - ari).astype(np.float32)
    if target == "CENTERED":                  # T3: c = ARI - rowmean
        return (ari - ari.mean(axis=1, keepdims=True)).astype(np.float32)
    raise ValueError(f"unknown target {target!r}")


def select_policy(pred: np.ndarray, target: str) -> np.ndarray:
    """Predicted 20-vector -> selected policy index, per row.

    ARI and CENTERED select the argmax; REGRET selects the argmin. Ties in the
    PREDICTION are broken by lowest index, which is deterministic and does not
    consult any target value.
    """
    if target in ("ARI", "CENTERED"):
        return np.asarray(pred).argmax(axis=1)
    if target == "REGRET":
        return np.asarray(pred).argmin(axis=1)
    raise ValueError(f"unknown target {target!r}")


def all_tied_rows(ari: np.ndarray) -> np.ndarray:
    """Rows where every policy ties -- regret and centered are then all-zero."""
    return np.isclose(ari.max(axis=1), ari.min(axis=1), atol=1e-12)
