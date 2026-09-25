"""Candidate dynamic-K heuristics (task section 8).

Each heuristic maps a utility vector -> an integer K (number of metrics to
aggregate). We record, for every record, the K each candidate *would* choose --
ClustOpt itself is untouched. This is pure simulation / what-if analysis.

For every heuristic we keep two columns:
* ``<name>``       -- the K clipped to [MIN_K, MAX_K] (what ClustOpt would use).
* ``<name>_raw``   -- the K *before* clipping (so we can report how often a
  heuristic naturally wanted K > MAX_K, i.e. pct_K_gt_10_before_clipping).

All heuristics operate on the cleaned, descending-sorted curve ``s`` plus a few
scalar shape features already computed in :mod:`utility_vector_stats`.
"""

from __future__ import annotations

import math

import numpy as np

MIN_K = 1
MAX_K = 10
EPS = 1e-12

# Hyperparameter ladders (section 8).
REL_ALPHAS = (0.90, 0.85, 0.80, 0.75, 0.70)
MASS_TAUS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75)
HYBRID_ALPHAS = (0.85, 0.80, 0.75)
HYBRID_TAUS = (0.60, 0.65, 0.70)


def _suf(x: float) -> str:
    return f"{int(round(x * 100)):03d}"


def _clip(k: int) -> int:
    return int(min(MAX_K, max(MIN_K, k)))


def _k_relative(s: np.ndarray, alpha: float) -> int:
    """Count metrics with utility >= alpha * max."""
    mx = s[0] if s.size else 0.0
    if mx <= EPS:
        return MIN_K
    return int((s >= alpha * mx).sum())


def _k_mass(csum: np.ndarray, total: float, tau: float) -> int:
    """Smallest K with cumulative top-K mass >= tau."""
    if total <= EPS:
        return MIN_K
    k = int(np.searchsorted(csum, tau * total) + 1)
    return min(k, csum.size)


def _k_elbow(gaps: np.ndarray, upto: int) -> int:
    """Position of the largest absolute drop in the sorted curve."""
    g = gaps[:upto]
    if g.size == 0:
        return MIN_K
    return int(np.argmax(g)) + 1


def _k_elbow_norm(s: np.ndarray, gaps: np.ndarray, upto: int) -> int:
    """Position of the largest *relative* drop gap_i / max(s_i, eps)."""
    g = gaps[:upto]
    if g.size == 0:
        return MIN_K
    denom = np.maximum(s[:g.size], EPS)
    return int(np.argmax(g / denom)) + 1


def compute_heuristics(s: np.ndarray, participation_ratio: float,
                       perplexity: float) -> dict:
    """Return all candidate-K columns (raw + clipped) for one record.

    Parameters
    ----------
    s : cleaned utility curve, sorted descending (from ``sorted_clean``).
    participation_ratio, perplexity : effective-K shape scalars.
    """
    out: dict = {}

    total = float(s.sum())
    csum = np.cumsum(s) if s.size else np.array([0.0])
    s_pad = np.concatenate([s, np.zeros(17)])
    gaps = s_pad[:15] - s_pad[1:16]

    def emit(name: str, raw_k: int):
        out[f"{name}_raw"] = int(raw_k)
        out[name] = _clip(raw_k)

    # 8.1 relative threshold
    k_rel_for = {}
    for a in REL_ALPHAS:
        k = _k_relative(s, a)
        k_rel_for[a] = k
        emit(f"k_rel_{_suf(a)}", k)

    # 8.2 cumulative mass
    k_mass_for = {}
    for t in MASS_TAUS:
        k = _k_mass(csum, total, t)
        k_mass_for[t] = k
        emit(f"k_mass_{_suf(t)}", k)

    # 8.3 effective-K
    pr = participation_ratio if math.isfinite(participation_ratio) else MIN_K
    px = perplexity if math.isfinite(perplexity) else MIN_K
    emit("k_eff_pr_round", int(round(pr)))
    emit("k_eff_pr_ceil", int(math.ceil(pr)))
    emit("k_eff_entropy_round", int(round(px)))

    # 8.4 elbow
    k_elbow10 = _k_elbow(gaps, 10)
    k_elbow15 = _k_elbow(gaps, 15)
    emit("k_elbow_10", k_elbow10)
    emit("k_elbow_15", k_elbow15)
    emit("k_elbow_norm_10", _k_elbow_norm(s, gaps, 10))
    emit("k_elbow_norm_15", _k_elbow_norm(s, gaps, 15))

    # 8.5 hybrid conservative: K = min(k_rel, k_mass), clipped components
    for a in HYBRID_ALPHAS:
        for t in HYBRID_TAUS:
            kr = _clip(k_rel_for[a])
            km = _clip(k_mass_for[t])
            emit(f"k_hybrid_min_a{_suf(a)}_t{_suf(t)}", min(kr, km))

    # 8.6 hybrid moderate (single representative pair a080/t065)
    kr = _clip(k_rel_for[0.80])
    km = _clip(k_mass_for[0.65])
    ke = _clip(k_elbow10)
    emit("k_hybrid_mean_a080_t065", int(round((kr + km) / 2.0)))
    emit("k_hybrid_median_a080_t065", int(np.median([kr, km, ke])))

    return out


# Column groups used by reporting / plotting -- single source of truth.
def heuristic_columns() -> list[str]:
    cols = [f"k_rel_{_suf(a)}" for a in REL_ALPHAS]
    cols += [f"k_mass_{_suf(t)}" for t in MASS_TAUS]
    cols += ["k_eff_pr_round", "k_eff_pr_ceil", "k_eff_entropy_round"]
    cols += ["k_elbow_10", "k_elbow_15", "k_elbow_norm_10", "k_elbow_norm_15"]
    cols += [f"k_hybrid_min_a{_suf(a)}_t{_suf(t)}"
             for a in HYBRID_ALPHAS for t in HYBRID_TAUS]
    cols += ["k_hybrid_mean_a080_t065", "k_hybrid_median_a080_t065"]
    return cols
