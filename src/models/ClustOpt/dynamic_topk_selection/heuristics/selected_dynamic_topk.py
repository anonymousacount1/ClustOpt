"""The selected Dynamic Top-K heuristic: ``dynamic_topk_relsoft_a092_k5_ceil``.

Chosen in Phase 2 as the best candidate by composite score: a relative-threshold
selector at ``alpha = 0.92`` with a *soft* cap (ceil) at base 5 and a hard final
cap at 10. It picks how many metrics ``K`` to aggregate from the shape of a single
utility vector.

Definition (matches the Phase-2 column ``k_relsoft_a092_k5_ceil`` exactly):

    u      = clip(nan/inf -> 0, then [0, 1])
    u_max  = max(u)
    k_rel  = 1                       if u_max <= eps
             count(u_i >= 0.92*u_max) otherwise
    K      = k_rel                   if k_rel <= 5
             ceil((k_rel + 5) / 2)   if k_rel  > 5      (soft cap)
    K      = clip(K, 1, 10)                              (hard cap)

Selected metrics are the top-K by utility, descending.

The module exposes a clear single-record API (:func:`select`) and a vectorised
batch API (:func:`select_batch`) for the full repository.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# --- fixed hyperparameters of the selected heuristic -------------------------
HEURISTIC_NAME = "dynamic_topk_relsoft_a092_k5_ceil"
ALPHA = 0.92
SOFT_CAP_BASE = 5
MAX_K = 10
MIN_K = 1
EPS = 1e-12


def clean_utilities(u: np.ndarray) -> np.ndarray:
    """NaN/inf -> 0, then clip to [0, 1] (same cleaning as Phase 1/2)."""
    c = np.nan_to_num(np.asarray(u, dtype=float), nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(c, 0.0, 1.0)


def k_rel_count(u_clean: np.ndarray, u_max: float) -> int:
    """Number of metrics within ``alpha`` of the best (relative threshold)."""
    if u_max <= EPS:
        return MIN_K
    return int((u_clean >= ALPHA * u_max).sum())


def soft_cap(k_rel: int) -> int:
    """Soft cap (ceil) at base 5, then hard cap to [1, 10]."""
    K = k_rel if k_rel <= SOFT_CAP_BASE else math.ceil((k_rel + SOFT_CAP_BASE) / 2)
    return int(min(MAX_K, max(MIN_K, K)))


@dataclass
class Selection:
    selected_k: int
    k_rel_raw: int
    u_max: float
    selected_metric_names: list[str] = field(default_factory=list)
    selected_metric_utilities: list[float] = field(default_factory=list)
    selected_metric_ranks: list[int] = field(default_factory=list)


def select(metric_names: list[str], utilities: np.ndarray) -> Selection:
    """Run the heuristic on one record's utility vector.

    Selection ordering uses the cleaned utilities (descending, stable). Returned
    utilities are the *raw* values for the chosen metrics so callers see the true
    scores.
    """
    raw = np.asarray(utilities, dtype=float)
    u = clean_utilities(raw)
    u_max = float(u.max()) if u.size else 0.0
    k_rel = k_rel_count(u, u_max)
    K = soft_cap(k_rel)

    # top-K by cleaned utility, descending; stable for ties (mergesort).
    order = np.argsort(-u, kind="stable")[:K]
    names = [metric_names[i] for i in order]
    utils = [float(raw[i]) for i in order]
    ranks = list(range(1, len(order) + 1))
    return Selection(selected_k=K, k_rel_raw=k_rel, u_max=u_max,
                     selected_metric_names=names,
                     selected_metric_utilities=utils,
                     selected_metric_ranks=ranks)


def select_batch(values: np.ndarray, u_max_eps: float = EPS):
    """Vectorised K selection over an (N x M) utility matrix.

    Returns ``(selected_k, k_rel_raw, u_max)`` as length-N int/float arrays.
    Metric-name extraction is left to the caller (see :func:`select`), since it
    needs the column-name list and is cheap to do per row.
    """
    raw = np.asarray(values, dtype=float)
    u = np.clip(np.nan_to_num(raw, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    u_max = u.max(axis=1)
    safe = np.where(u_max > u_max_eps, u_max, 1.0)
    k_rel = (u >= ALPHA * safe[:, None]).sum(axis=1).astype(int)
    k_rel = np.where(u_max > u_max_eps, k_rel, MIN_K)

    # soft cap (ceil) at base 5, vectorised, then hard cap [1, 10]
    K = np.where(k_rel <= SOFT_CAP_BASE, k_rel,
                 np.ceil((k_rel + SOFT_CAP_BASE) / 2.0)).astype(int)
    K = np.clip(K, MIN_K, MAX_K)
    return K, k_rel, u_max
