"""A block — partition / view-space statistics (30 features)."""
from __future__ import annotations

import numpy as np

from ..config import A_HIST_BINS_LIST, EPS
from ..utils.numeric_utils import (
    finite_values,
    mad_from_median,
    normalized_entropy,
    percentiles,
    safe_kurtosis,
    safe_skew,
    safe_zscores,
)


def _round_unique_ratio(X_view: np.ndarray) -> tuple[float, float]:
    if X_view.size == 0:
        return float("nan"), float("nan")
    arr = np.where(np.isfinite(X_view), X_view, np.nan)
    rounded = np.round(arr, 6)
    finite_rows = np.all(np.isfinite(rounded), axis=1)
    if not np.any(finite_rows):
        return float("nan"), float("nan")
    valid = rounded[finite_rows]
    n = valid.shape[0]
    unique = np.unique(valid, axis=0).shape[0]
    dup_ratio = float((n - unique) / n) if n > 0 else float("nan")
    uniq_ratio = float(unique / n) if n > 0 else float("nan")
    return dup_ratio, uniq_ratio


def compute_block_a(X_view: np.ndarray, view_dim: int) -> dict[str, float]:
    n_points = X_view.shape[0]
    flat = finite_values(X_view)
    total_values = X_view.size

    pcts = percentiles(flat, (1, 5, 25, 50, 75, 95, 99))

    if flat.size > 0:
        mean = float(flat.mean())
        std = float(flat.std(ddof=0))
        flat_min = float(flat.min())
        flat_max = float(flat.max())
        flat_median = float(np.median(flat))
        iqr = float(pcts[75] - pcts[25])
        rng = float(flat_max - flat_min)
        skew = safe_skew(flat)
        kurt = safe_kurtosis(flat)
    else:
        mean = std = flat_min = flat_max = flat_median = iqr = rng = float("nan")
        skew = kurt = float("nan")

    counts_32 = np.histogram(flat, bins=32)[0].astype(float) if flat.size > 0 else np.zeros(32)
    counts_64 = np.histogram(flat, bins=64)[0].astype(float) if flat.size > 0 else np.zeros(64)
    total32 = float(counts_32.sum())
    peak_bin_ratio_32 = float(counts_32.max() / total32) if total32 > 0 else float("nan")
    empty_bin_ratio_32 = float(np.sum(counts_32 == 0) / 32.0)
    entropy_32 = normalized_entropy(counts_32) if flat.size > 0 else float("nan")
    entropy_64 = normalized_entropy(counts_64) if flat.size > 0 else float("nan")

    if flat.size > 0:
        tail_low = float(np.mean(flat <= pcts[5]))
        tail_high = float(np.mean(flat >= pcts[95]))
        q25, q75 = pcts[25], pcts[75]
        iqr_span = q75 - q25
        lo = q25 - 1.5 * iqr_span
        hi = q75 + 1.5 * iqr_span
        outlier_iqr = float(np.mean((flat < lo) | (flat > hi)))
        z = safe_zscores(flat)
        outlier_z3 = float(np.mean(np.abs(z) > 3.0)) if z.size > 0 else float("nan")
        mean_abs_z = float(np.mean(np.abs(z))) if z.size > 0 else float("nan")
    else:
        tail_low = tail_high = outlier_iqr = outlier_z3 = mean_abs_z = float("nan")

    dup_ratio, uniq_ratio = _round_unique_ratio(X_view)

    return {
        "A_view_n_points_log1p": float(np.log1p(n_points)),
        "A_view_dim": float(view_dim),
        "A_view_finite_ratio": float(flat.size / total_values) if total_values > 0 else float("nan"),
        "A_view_duplicate_ratio_rounded_6": dup_ratio,
        "A_view_unique_ratio_rounded_6": uniq_ratio,
        "A_view_mean_flat": mean,
        "A_view_std_flat": std,
        "A_view_min_flat": flat_min,
        "A_view_max_flat": flat_max,
        "A_view_range_flat": rng,
        "A_view_median_flat": flat_median,
        "A_view_iqr_flat": iqr,
        "A_view_mad_flat": mad_from_median(flat),
        "A_view_q01_flat": float(pcts[1]),
        "A_view_q05_flat": float(pcts[5]),
        "A_view_q25_flat": float(pcts[25]),
        "A_view_q75_flat": float(pcts[75]),
        "A_view_q95_flat": float(pcts[95]),
        "A_view_q99_flat": float(pcts[99]),
        "A_view_skew_flat": skew,
        "A_view_kurtosis_flat": kurt,
        "A_view_entropy_hist_32": entropy_32,
        "A_view_entropy_hist_64": entropy_64,
        "A_view_peak_bin_ratio_32": peak_bin_ratio_32,
        "A_view_empty_bin_ratio_32": empty_bin_ratio_32,
        "A_view_tail_mass_low_5pct": tail_low,
        "A_view_tail_mass_high_95pct": tail_high,
        "A_view_outlier_ratio_iqr_1_5": outlier_iqr,
        "A_view_outlier_ratio_z_3": outlier_z3,
        "A_view_mean_abs_zscore": mean_abs_z,
    }
