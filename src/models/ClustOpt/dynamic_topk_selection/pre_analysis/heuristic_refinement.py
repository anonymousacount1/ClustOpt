"""New Phase-2 candidate heuristics (section 5), built by *reusing* the Phase-1
record tables -- no rescanning of utility files.

All heuristics live in the Relative-Threshold family, which Phase 1 identified as
the only viable one (utility vectors are too flat for mass/effective-K signals).
We explore it thoroughly: a fine alpha sweep, hard caps, soft caps, elbow guards
and gap-confidence guards. Every heuristic returns K clipped to [1, 10].

Inputs (vectorised over all records):
  * wide matrix V of the 60 per-metric utilities (``metric_*`` columns)
  * Phase-1 sorted-curve gap stats (``gap_1..gap_10``, ``largest_gap_value_1_to_10``)
  * Phase-1 elbow K (``k_elbow_10``)

Everything is pure numpy / pandas. Column-name conventions:
  alpha 0.95 -> 'a095' | cap 7 -> 'c7' | knee 7 -> 'k7' |
  gap thr 0.05 -> 'g005' | conf thr 1.5 -> 'cf15'
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_K, MAX_K = 1, 10
EPS = 1e-12

# Section 5.1 fine relative-threshold sweep.
FINE_ALPHAS = [0.95, 0.94, 0.93, 0.92, 0.91, 0.90, 0.89, 0.88, 0.87, 0.86, 0.85]
# 5.2 hard caps.
CAPS = [5, 6, 7, 8, 9, 10]
# 5.3 soft-cap knees.
SOFT_KNEES = [5, 6, 7]
# 5.4 elbow-guard absolute-gap thresholds.
GAP_THRESHOLDS = [0.05, 0.06, 0.07, 0.08]
# 5.5 gap-confidence thresholds (largest_gap / mean(first-10 gaps)).
CONF_THRESHOLDS = [1.5, 2.0, 2.5, 3.0]


def _a(alpha: float) -> str:
    return f"a{int(round(alpha * 100)):03d}"


def _g(thr: float) -> str:
    return f"g{int(round(thr * 100)):03d}"


def _cf(thr: float) -> str:
    return f"cf{int(round(thr * 10)):02d}"


def _clip(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, MIN_K, MAX_K).astype(int)


def build_new_heuristics(merged: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame (aligned to ``merged``) of all new heuristic K columns."""
    metric_cols = [c for c in merged.columns if c.startswith("metric_")]
    V = merged[metric_cols].to_numpy(dtype=float)
    Vc = np.clip(np.nan_to_num(V, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    M = Vc.max(axis=1)
    M_safe = np.where(M > EPS, M, 1.0)

    k_elbow = merged["k_elbow_10"].to_numpy(dtype=float)
    largest_gap = merged["largest_gap_value_1_to_10"].to_numpy(dtype=float)
    gap_cols = [f"gap_{i}" for i in range(1, 11)]
    mean_gap10 = merged[gap_cols].to_numpy(dtype=float).mean(axis=1)
    confidence = largest_gap / np.where(mean_gap10 > EPS, mean_gap10, np.nan)

    out: dict[str, np.ndarray] = {}

    # raw count of metrics >= alpha * max, per alpha
    counts: dict[float, np.ndarray] = {}
    for alpha in FINE_ALPHAS:
        cnt = (Vc >= alpha * M_safe[:, None]).sum(axis=1).astype(float)
        cnt = np.where(M > EPS, cnt, 1.0)              # degenerate -> K=1
        counts[alpha] = cnt
        # 5.1 fine sweep (clipped)
        out[f"k_rel_{_a(alpha)}"] = _clip(cnt)

    # 5.2 hard cap: min(count, cap)
    for alpha in FINE_ALPHAS:
        cnt = counts[alpha]
        for cap in CAPS:
            out[f"k_relcap_{_a(alpha)}_c{cap}"] = _clip(np.minimum(cnt, cap))

    # 5.3 soft cap: if K>knee, K=(round|ceil)((K+knee)/2)
    for alpha in FINE_ALPHAS:
        cnt = counts[alpha]
        for knee in SOFT_KNEES:
            soft_r = np.where(cnt > knee, np.round((cnt + knee) / 2.0), cnt)
            soft_c = np.where(cnt > knee, np.ceil((cnt + knee) / 2.0), cnt)
            out[f"k_relsoft_{_a(alpha)}_k{knee}_round"] = _clip(soft_r)
            out[f"k_relsoft_{_a(alpha)}_k{knee}_ceil"] = _clip(soft_c)

    # 5.4 elbow guard: if largest_gap >= thr, K=min(K_rel, K_elbow) else K_rel
    for alpha in FINE_ALPHAS:
        krel = _clip(counts[alpha])
        guarded = np.minimum(krel, k_elbow)
        for thr in GAP_THRESHOLDS:
            use = largest_gap >= thr
            out[f"k_relelbow_{_a(alpha)}_{_g(thr)}"] = _clip(
                np.where(use, guarded, krel))

    # 5.5 gap-confidence guard: if confidence >= thr, K=min(K_rel, K_elbow)
    for alpha in FINE_ALPHAS:
        krel = _clip(counts[alpha])
        guarded = np.minimum(krel, k_elbow)
        for thr in CONF_THRESHOLDS:
            use = np.nan_to_num(confidence, nan=0.0) >= thr
            out[f"k_relconf_{_a(alpha)}_{_cf(thr)}"] = _clip(
                np.where(use, guarded, krel))

    res = pd.DataFrame(out, index=merged.index)
    res["_confidence"] = confidence            # exposed for reporting / plots
    return res


def new_heuristic_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if c.startswith(("k_rel_a", "k_relcap_", "k_relsoft_",
                             "k_relelbow_", "k_relconf_"))]


def shortlist_columns() -> list[str]:
    """Representative heuristics for the agreement matrix / focused plots:
    spans the new families plus the key Phase-1 baselines."""
    return [
        # Phase-1 baselines kept for reference
        "k_rel_090", "k_elbow_10", "k_elbow_norm_10",
        # fine sweep extremes / centre
        "k_rel_a095", "k_rel_a092", "k_rel_a090", "k_rel_a088",
        # caps on a centre alpha
        "k_relcap_a092_c5", "k_relcap_a090_c5", "k_relcap_a090_c6",
        "k_relcap_a090_c7",
        # soft cap
        "k_relsoft_a092_k5_round", "k_relsoft_a090_k6_round",
        # guards
        "k_relelbow_a090_g005", "k_relelbow_a092_g006",
        "k_relconf_a090_cf20", "k_relconf_a092_cf20",
    ]
