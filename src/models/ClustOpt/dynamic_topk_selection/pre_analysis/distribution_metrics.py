"""Objective quality metrics for a heuristic's K distribution (Phase 2, section 6).

A "K distribution" is the empirical distribution of selected K (in 1..MAX_K) that
a heuristic produces across all records. We score *how desirable* that shape is,
replacing the visual inspection used in Phase 1.

Desired shape (qualitative spec):
  * few K=1
  * peak around K=3-5
  * monotonic decay afterwards
  * little pile-up at the clip boundary (K=10)

Bell score
----------
We do NOT assume Gaussian. We define an explicit discrete "ideal" bell vector
over K=1..10 and score a distribution by its closeness to it via total-variation
distance:

    bell_fit = 1 - 0.5 * sum_K | P(K) - IDEAL(K) |        in [0, 1], higher better

``IDEAL`` peaks at K=3-4, allots only ~4% to K=1, and decays monotonically. This
is fully documented (the vector below) and reproducible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MAX_K = 10

# Discrete ideal bell over K = 1..10 (sums to 1.0). Peak at 3-4, few K=1,
# monotone decay. This is the reference shape for ``bell_fit``.
IDEAL_BELL = np.array(
    [0.04, 0.13, 0.24, 0.24, 0.16, 0.09, 0.05, 0.025, 0.015, 0.01], dtype=float)
assert abs(IDEAL_BELL.sum() - 1.0) < 1e-9


def k_probabilities(k_array: np.ndarray) -> np.ndarray:
    """Return P(K=1..MAX_K) as a length-MAX_K vector (normalised)."""
    k = np.asarray(k_array, dtype=float)
    k = k[np.isfinite(k)].astype(int)
    p = np.zeros(MAX_K, dtype=float)
    if k.size == 0:
        return p
    for i in range(1, MAX_K + 1):
        p[i - 1] = np.mean(k == i)
    return p


def bell_fit(p: np.ndarray) -> float:
    """1 - TVD between the K distribution and IDEAL_BELL (in [0, 1])."""
    return float(1.0 - 0.5 * np.abs(p - IDEAL_BELL).sum())


def _entropy(p: np.ndarray) -> float:
    nz = p[p > 0]
    return float(-(nz * np.log(nz)).sum())


def _decay_violation(p: np.ndarray) -> float:
    """Sum of increases in P after the peak (0 = perfectly monotone decay)."""
    peak = int(np.argmax(p))
    return float(np.clip(np.diff(p[peak:]), 0, None).sum())


def k_distribution_metrics(k_array: np.ndarray) -> dict:
    """All section-6 quality metrics for one heuristic's K vector."""
    k = np.asarray(k_array, dtype=float)
    k = k[np.isfinite(k)]
    out: dict = {"n": int(k.size)}
    if k.size == 0:
        return out
    p = k_probabilities(k)
    out["mean_K"] = float(k.mean())
    out["median_K"] = float(np.median(k))
    out["std_K"] = float(k.std(ddof=0))
    for i in range(1, MAX_K + 1):
        out[f"P_K{i}"] = float(p[i - 1])
    out["P_2_5"] = float(p[1:5].sum())          # preferred region P(2<=K<=5)
    out["P_gt5"] = float(p[5:].sum())           # P(K>5)
    out["P_gt7"] = float(p[7:].sum())           # P(K>7)
    out["P_eq10"] = float(p[9])                 # P(K==10) clip pile-up
    out["entropy_K"] = _entropy(p)
    out["smoothness"] = float(np.abs(np.diff(p)).sum())  # lower = smoother
    out["peak_K"] = int(np.argmax(p) + 1)
    out["decay_violation"] = _decay_violation(p)
    out["bell_fit"] = bell_fit(p)
    return out


# --------------------------------------------------------------------------- #
# Composite selection score                                                   #
# --------------------------------------------------------------------------- #
def composite_score(row: pd.Series) -> float:
    """Headline ranking score combining the desiderata, transparently.

        composite = bell_fit                       (closeness to ideal shape)
                    - 0.25 * P(K==10)              (clip pile-up is undesirable)
                    - 0.15 * min(std_K/3, 1)       (cross-record dispersion)
                    - 0.20 * max(0, P(K=1) - 0.12) (excess K=1 over ~12%)

    bell_fit dominates; the penalties break ties toward stable, non-saturating,
    not-too-many-K=1 distributions. Higher is better.
    """
    bell = row.get("bell_fit", 0.0)
    p10 = row.get("P_eq10", 0.0)
    std = row.get("std_K", 0.0)
    p1 = row.get("P_K1", 0.0)
    return float(bell - 0.25 * p10 - 0.15 * min(std / 3.0, 1.0)
                 - 0.20 * max(0.0, p1 - 0.12))


# --------------------------------------------------------------------------- #
# Stability across repository structure (section 7)                           #
# --------------------------------------------------------------------------- #
def stability(df: pd.DataFrame, kcol: str) -> dict:
    """Std of per-group mean-K across families / subfamilies / views.

    Lower = the heuristic behaves more consistently across structure.
    Also reports the std of per-group P(2<=K<=5) for the family grouping.
    """
    out = {}
    for by, tag in (("family_id", "family"), ("subfamily_id", "subfamily"),
                    ("view_type", "view")):
        gm = df.groupby(by)[kcol].mean()
        out[f"{tag}_meanK_std"] = float(gm.std(ddof=0))
        out[f"{tag}_meanK_range"] = float(gm.max() - gm.min())
    # cross-family variability of the preferred-region mass
    reg = df.groupby("family_id")[kcol].apply(
        lambda s: float(((s >= 2) & (s <= 5)).mean()))
    out["family_P2_5_std"] = float(reg.std(ddof=0))
    return out


# --------------------------------------------------------------------------- #
# Agreement between heuristics (section 8)                                     #
# --------------------------------------------------------------------------- #
def agreement_matrices(df: pd.DataFrame, cols: list[str]):
    """Return (mean_abs_diff, exact_agreement_pct) DataFrames over ``cols``."""
    n = len(cols)
    mad = np.zeros((n, n))
    exact = np.zeros((n, n))
    arrs = {c: df[c].to_numpy(dtype=float) for c in cols}
    for i, ci in enumerate(cols):
        for j, cj in enumerate(cols):
            a, b = arrs[ci], arrs[cj]
            mad[i, j] = float(np.nanmean(np.abs(a - b)))
            exact[i, j] = float(np.nanmean(a == b) * 100)
    return (pd.DataFrame(mad, index=cols, columns=cols),
            pd.DataFrame(exact, index=cols, columns=cols))
