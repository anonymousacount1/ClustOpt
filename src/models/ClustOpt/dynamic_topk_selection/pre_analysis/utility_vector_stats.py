"""Per-utility-vector statistics for the Dynamic Top-K pre-analysis.

Pure-numpy, no pandas, so it stays cheap inside multiprocessing workers.

A "utility vector" ``u`` is the array of per-metric utility scores for a single
dataset *view* (record). All functions here are deterministic and side-effect
free. They never raise on degenerate input (all-NaN, all-zero, single metric);
instead they emit NaN / sentinel values so the caller can flag the record.

Conventions
-----------
* ``raw``      : utilities exactly as read from disk (may contain NaN/inf).
* ``cleaned``  : ``clip(nan_to_num(raw, nan=0), 0, 1)`` -- used for every
  mass / entropy / heuristic computation, per the task spec (section 7.2).
* ``s``        : ``cleaned`` sorted descending (the "sorted utility curve").

Gap / curve positions are 1-indexed in column names (gap_1 = s_1 - s_2).
"""

from __future__ import annotations

import math

import numpy as np

EPS = 1e-12

# Curve / gap positions we materialise as explicit columns.
N_GAPS = 15           # gap_1 .. gap_15  (needs s_1 .. s_16)
N_SORTED_KEEP = 15    # sorted_u_01 .. sorted_u_15 (for percentile curve plots)

# Threshold ladders (section 7.4).
REL_THRESH = (0.90, 0.85, 0.80, 0.75, 0.70)
ABS_THRESH = (0.90, 0.85, 0.80, 0.75, 0.70, 0.60)


def _pct_suffix(x: float) -> str:
    """0.90 -> '090', 0.5 -> '050'. Used to build stable column names."""
    return f"{int(round(x * 100)):03d}"


def clean_vector(raw: np.ndarray) -> np.ndarray:
    """NaN/inf -> 0, then clip to [0, 1]."""
    c = np.nan_to_num(np.asarray(raw, dtype=float), nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(c, 0.0, 1.0)


def compute_vector_stats(raw: np.ndarray) -> dict:
    """All per-vector descriptive / shape / curve / threshold statistics.

    Parameters
    ----------
    raw : 1-D array of utility values as read from disk.

    Returns
    -------
    dict of scalar features (section 7.1 - 7.4).
    """
    raw = np.asarray(raw, dtype=float)
    n = int(raw.size)
    finite = np.isfinite(raw)
    n_finite = int(finite.sum())
    n_nan = int(n - n_finite)

    out: dict = {
        "n_metrics": n,
        "n_finite": n_finite,
        "n_nan": n_nan,
        "all_nan": bool(n_finite == 0),
    }

    # ---- 7.1 basic stats (on finite values) ----------------------------------
    fin = raw[finite]
    if fin.size:
        q = np.percentile(fin, [5, 10, 25, 50, 75, 90, 95])
        out.update(
            min=float(fin.min()),
            max=float(fin.max()),
            mean=float(fin.mean()),
            median=float(np.median(fin)),
            std=float(fin.std(ddof=0)),
            p05=float(q[0]), p10=float(q[1]), p25=float(q[2]),
            p75=float(q[4]), p90=float(q[5]), p95=float(q[6]),
            iqr=float(q[4] - q[2]),
            range=float(fin.max() - fin.min()),
        )
    else:
        for k in ("min", "max", "mean", "median", "std", "p05", "p10",
                  "p25", "p75", "p90", "p95", "iqr", "range"):
            out[k] = float("nan")

    # ---- cleaned vector / sorted curve --------------------------------------
    c = clean_vector(raw)
    s = np.sort(c)[::-1]                      # descending
    total = float(c.sum())
    out["sum_utility"] = total
    out["max_clean"] = float(s[0]) if s.size else 0.0

    # sorted-curve samples for percentile plots (pad with NaN beyond length)
    for i in range(N_SORTED_KEEP):
        out[f"sorted_u_{i + 1:02d}"] = float(s[i]) if i < s.size else float("nan")

    # ---- 7.2 shape / concentration ------------------------------------------
    if total > EPS:
        p = c / total
        nz = p[p > EPS]
        entropy = float(-(nz * np.log(nz)).sum())
        out["entropy"] = entropy
        out["normalized_entropy"] = entropy / math.log(n) if n > 1 else 0.0
        out["perplexity_effective_k"] = float(math.exp(entropy))
        out["participation_ratio"] = float(1.0 / np.square(p).sum())
        csum = np.cumsum(s)
        for k in (1, 2, 3, 5, 10):
            out[f"top{k}_mass"] = float(csum[min(k, s.size) - 1] / total)
        out["tail_after_5_mass"] = float(1.0 - out["top5_mass"])
        out["tail_after_10_mass"] = float(1.0 - out["top10_mass"])
        s2 = s[1] if s.size > 1 else 0.0
        s3 = csum[2] if s.size >= 3 else csum[-1]
        s5 = csum[4] if s.size >= 5 else csum[-1]
        out["top1_over_sum"] = float(s[0] / total)
        out["top3_over_sum"] = float(s3 / total)
        out["top5_over_sum"] = float(s5 / total)
        out["top1_over_top2"] = float(s[0] / s2) if s2 > EPS else float("inf")
        out["top1_minus_top2"] = float(s[0] - s2)
    else:
        for k in ("entropy", "normalized_entropy", "perplexity_effective_k",
                  "participation_ratio", "top1_mass", "top2_mass", "top3_mass",
                  "top5_mass", "top10_mass", "tail_after_5_mass",
                  "tail_after_10_mass", "top1_over_sum", "top3_over_sum",
                  "top5_over_sum", "top1_over_top2", "top1_minus_top2"):
            out[k] = float("nan")

    # ---- 7.3 sorted curve / gaps --------------------------------------------
    # gap_i = s_i - s_{i+1}; pad sorted curve with zeros so positions exist.
    s_pad = np.concatenate([s, np.zeros(N_GAPS + 2)])
    gaps = s_pad[:N_GAPS] - s_pad[1:N_GAPS + 1]   # gaps[0] = gap_1
    for i in range(N_GAPS):
        out[f"gap_{i + 1}"] = float(gaps[i])

    def _largest_gap(upto: int):
        g = gaps[:upto]
        pos = int(np.argmax(g)) + 1           # 1-indexed position
        return pos, float(g[pos - 1])

    out["largest_gap_position_1_to_10"], out["largest_gap_value_1_to_10"] = _largest_gap(10)
    out["largest_gap_position_1_to_15"], out["largest_gap_value_1_to_15"] = _largest_gap(15)

    out["slope_top5"] = float((s_pad[0] - s_pad[4]) / 4.0)
    out["slope_top10"] = float((s_pad[0] - s_pad[9]) / 9.0)
    out["curve_auc_normalized"] = (
        float(s.sum() / (s.size * s[0])) if s.size and s[0] > EPS else float("nan")
    )

    # ---- 7.4 threshold counts -----------------------------------------------
    mx = out["max_clean"]
    for t in REL_THRESH:
        out[f"n_ge_rel_{_pct_suffix(t)}"] = int((c >= t * mx).sum()) if mx > EPS else 0
    for t in ABS_THRESH:
        out[f"n_ge_abs_{_pct_suffix(t)}"] = int((c >= t).sum())

    return out


# -- expose sorted curve to the heuristics layer without re-sorting -----------
def sorted_clean(raw: np.ndarray) -> np.ndarray:
    return np.sort(clean_vector(raw))[::-1]
