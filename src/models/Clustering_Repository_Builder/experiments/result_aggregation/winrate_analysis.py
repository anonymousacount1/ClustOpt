"""Win/tie/loss rate analysis between methods and baselines.

Compares paired (dataset-level) ARI: a method "wins" over a baseline on a
dataset when ``ARI(method) > ARI(baseline) + epsilon``. Pairing is done on
``dataset_id`` using the dataset-level (best-view) ARI.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .method_registry import (
    BASELINE_METHODS as _REGISTRY_BASELINES,
    REGRESSOR_METHODS as _REGISTRY_REGRESSORS,
    is_regressor,
)

EPSILON = 1e-6

# Registry defaults (phase-tagged). The summary helpers default to *dynamic*
# detection from the data so each run compares the regressors against whichever
# non-regressor methods are present (phase-A baselines OR phase-B externals).
REGRESSOR_METHODS = _REGISTRY_REGRESSORS
BASELINE_METHODS = _REGISTRY_BASELINES


def regressors_present(dataset_level: pd.DataFrame) -> List[str]:
    present = set(dataset_level["method_name"].unique())
    return [m for m in present if is_regressor(m)]


def baselines_present(dataset_level: pd.DataFrame) -> List[str]:
    """All non-regressor methods present (phase-A baselines or phase-B externals)."""
    present = set(dataset_level["method_name"].unique())
    return sorted(m for m in present if not is_regressor(m))


def _paired(
    dataset_level: pd.DataFrame, method: str, baseline: str
) -> Tuple[np.ndarray, np.ndarray]:
    m = dataset_level[dataset_level["method_name"] == method][["dataset_id", "best_ari"]]
    b = dataset_level[dataset_level["method_name"] == baseline][["dataset_id", "best_ari"]]
    merged = m.merge(b, on="dataset_id", suffixes=("_m", "_b")).dropna()
    return (
        pd.to_numeric(merged["best_ari_m"], errors="coerce").to_numpy(),
        pd.to_numeric(merged["best_ari_b"], errors="coerce").to_numpy(),
    )


def winrate_row(dataset_level: pd.DataFrame, method: str, baseline: str,
                epsilon: float = EPSILON) -> dict:
    a, b = _paired(dataset_level, method, baseline)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    n = int(a.size)
    if n == 0:
        return {
            "method": method, "baseline": baseline, "n_paired": 0,
            "win_rate": float("nan"), "tie_rate": float("nan"),
            "loss_rate": float("nan"), "mean_ari_gain": float("nan"),
            "median_ari_gain": float("nan"), "q25_ari_gain": float("nan"),
            "q75_ari_gain": float("nan"),
        }
    diff = a - b
    wins = int(np.sum(diff > epsilon))
    losses = int(np.sum(diff < -epsilon))
    ties = n - wins - losses
    return {
        "method": method, "baseline": baseline, "n_paired": n,
        "win_rate": wins / n, "tie_rate": ties / n, "loss_rate": losses / n,
        "mean_ari_gain": float(np.mean(diff)),
        "median_ari_gain": float(np.median(diff)),
        "q25_ari_gain": float(np.quantile(diff, 0.25)),
        "q75_ari_gain": float(np.quantile(diff, 0.75)),
    }


def winrate_summary(
    dataset_level: pd.DataFrame,
    *,
    methods: Optional[List[str]] = None,
    baselines: Optional[List[str]] = None,
    epsilon: float = EPSILON,
) -> pd.DataFrame:
    if dataset_level.empty:
        return pd.DataFrame()
    if methods is None:
        methods = regressors_present(dataset_level)
    if baselines is None:
        baselines = baselines_present(dataset_level)
    present = set(dataset_level["method_name"].unique())
    rows = []
    for method in methods:
        if method not in present:
            continue
        for baseline in baselines:
            if baseline not in present or baseline == method:
                continue
            rows.append(winrate_row(dataset_level, method, baseline, epsilon))
    return pd.DataFrame(rows)
