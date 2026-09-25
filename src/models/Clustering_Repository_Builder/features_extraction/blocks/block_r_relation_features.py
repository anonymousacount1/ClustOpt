"""R block — relation features between original dimensions (25 features)."""
from __future__ import annotations

import numpy as np

from ..config import EPS
from ..utils.numeric_utils import (
    joint_histogram,
    linear_r2,
    mutual_information_hist,
    poly_r2,
    safe_corrcoef,
    safe_div,
    shannon_entropy_from_probs,
)
from ..view_builder import DatasetView


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    rx = np.argsort(np.argsort(x[mask])).astype(float)
    ry = np.argsort(np.argsort(y[mask])).astype(float)
    return safe_corrcoef(rx, ry)


def _approx_kendall(x: np.ndarray, y: np.ndarray, max_pairs: int = 50_000) -> float:
    """Approximate Kendall tau via random pair sampling."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    x = x[mask]
    y = y[mask]
    n = x.size
    rng = np.random.default_rng(0)
    pairs = min(max_pairs, n * (n - 1) // 2)
    if pairs <= 0:
        return np.nan
    i = rng.integers(0, n, size=pairs)
    j = rng.integers(0, n, size=pairs)
    keep = i != j
    i = i[keep]
    j = j[keep]
    dx = x[i] - x[j]
    dy = y[i] - y[j]
    sign = np.sign(dx) * np.sign(dy)
    sign = sign[sign != 0]
    if sign.size == 0:
        return 0.0
    return float(np.mean(sign))


def _conditional_variance(x: np.ndarray, y: np.ndarray, bins: int = 16) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < bins * 2:
        return np.nan
    x = x[mask]
    y = y[mask]
    edges = np.percentile(x, np.linspace(0, 100, bins + 1))
    edges = np.unique(edges)
    if edges.size < 2:
        return np.nan
    idx = np.digitize(x, edges[1:-1])
    variances = []
    for bin_id in np.unique(idx):
        ys = y[idx == bin_id]
        if ys.size >= 2:
            variances.append(float(ys.var(ddof=0)))
    if not variances:
        return np.nan
    return float(np.mean(variances))


def _monotonicity_score(x: np.ndarray, y: np.ndarray) -> float:
    """Sign-consistency score across binned medians: fraction of consecutive pairs
    with matching sign direction.
    """
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 20:
        return abs(_spearman(x, y))
    x = x[mask]
    y = y[mask]
    bins = 16
    edges = np.percentile(x, np.linspace(0, 100, bins + 1))
    edges = np.unique(edges)
    if edges.size < 3:
        return abs(_spearman(x, y))
    bin_idx = np.digitize(x, edges[1:-1])
    medians = []
    for b in range(edges.size - 1):
        ys = y[bin_idx == b]
        if ys.size > 0:
            medians.append(float(np.median(ys)))
    if len(medians) < 3:
        return np.nan
    diffs = np.diff(medians)
    diffs = diffs[diffs != 0]
    if diffs.size == 0:
        return 0.0
    signs = np.sign(diffs)
    dominant = max(float(np.mean(signs > 0)), float(np.mean(signs < 0)))
    return dominant


def compute_block_r(view: DatasetView) -> dict[str, float]:
    out: dict[str, float] = {key: np.nan for key in (
        "R_has_two_original_dims",
        "R_pearson_corr_xy", "R_abs_pearson_corr_xy",
        "R_spearman_corr_xy", "R_abs_spearman_corr_xy",
        "R_kendall_corr_xy_approx",
        "R_mutual_information_xy_hist_16", "R_mutual_information_xy_hist_32",
        "R_normalized_mi_xy_hist_16",
        "R_linear_r2_x_to_y", "R_linear_r2_y_to_x",
        "R_poly2_r2_x_to_y", "R_poly2_r2_y_to_x",
        "R_poly3_r2_x_to_y", "R_poly3_r2_y_to_x",
        "R_monotonicity_score_x_to_y", "R_monotonicity_score_y_to_x",
        "R_conditional_var_y_given_x_bins", "R_conditional_var_x_given_y_bins",
        "R_joint_entropy_xy_32", "R_entropy_x_32", "R_entropy_y_32",
        "R_entropy_ratio_x_y",
        "R_view_selected_axis_corr_with_other", "R_view_selected_axis_mi_with_other",
    )}

    X_orig = view.X_original
    has_two_dims = X_orig.ndim == 2 and X_orig.shape[1] >= 2
    out["R_has_two_original_dims"] = 1.0 if has_two_dims else 0.0
    if not has_two_dims:
        return out

    x = X_orig[:, 0]
    y = X_orig[:, 1]
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return out
    x = x[mask]
    y = y[mask]

    pearson = safe_corrcoef(x, y)
    spearman = _spearman(x, y)
    kendall = _approx_kendall(x, y)
    out["R_pearson_corr_xy"] = pearson
    out["R_abs_pearson_corr_xy"] = abs(pearson) if np.isfinite(pearson) else np.nan
    out["R_spearman_corr_xy"] = spearman
    out["R_abs_spearman_corr_xy"] = abs(spearman) if np.isfinite(spearman) else np.nan
    out["R_kendall_corr_xy_approx"] = kendall

    mi16 = mutual_information_hist(x, y, 16)
    mi32 = mutual_information_hist(x, y, 32)
    out["R_mutual_information_xy_hist_16"] = mi16
    out["R_mutual_information_xy_hist_32"] = mi32

    h16 = joint_histogram(x, y, 16)
    total = h16.sum()
    if total > 0:
        pxy = h16 / total
        px = pxy.sum(axis=1)
        py = pxy.sum(axis=0)
        hx = shannon_entropy_from_probs(px)
        hy = shannon_entropy_from_probs(py)
        denom = max(hx, hy)
        out["R_normalized_mi_xy_hist_16"] = float(mi16 / denom) if denom > EPS and np.isfinite(mi16) else np.nan

    out["R_linear_r2_x_to_y"] = linear_r2(x, y)
    out["R_linear_r2_y_to_x"] = linear_r2(y, x)
    out["R_poly2_r2_x_to_y"] = poly_r2(x, y, 2)
    out["R_poly2_r2_y_to_x"] = poly_r2(y, x, 2)
    out["R_poly3_r2_x_to_y"] = poly_r2(x, y, 3)
    out["R_poly3_r2_y_to_x"] = poly_r2(y, x, 3)

    out["R_monotonicity_score_x_to_y"] = _monotonicity_score(x, y)
    out["R_monotonicity_score_y_to_x"] = _monotonicity_score(y, x)

    out["R_conditional_var_y_given_x_bins"] = _conditional_variance(x, y)
    out["R_conditional_var_x_given_y_bins"] = _conditional_variance(y, x)

    h32 = joint_histogram(x, y, 32)
    total32 = h32.sum()
    if total32 > 0:
        pxy32 = h32 / total32
        out["R_joint_entropy_xy_32"] = shannon_entropy_from_probs(pxy32.ravel())
        out["R_entropy_x_32"] = shannon_entropy_from_probs(pxy32.sum(axis=1))
        out["R_entropy_y_32"] = shannon_entropy_from_probs(pxy32.sum(axis=0))
        out["R_entropy_ratio_x_y"] = safe_div(out["R_entropy_x_32"], out["R_entropy_y_32"])

    # View-specific axis-vs-other relation.
    if view.view_mode == "x_only":
        out["R_view_selected_axis_corr_with_other"] = pearson
        out["R_view_selected_axis_mi_with_other"] = mi16
    elif view.view_mode == "y_only":
        out["R_view_selected_axis_corr_with_other"] = pearson
        out["R_view_selected_axis_mi_with_other"] = mi16
    else:
        out["R_view_selected_axis_corr_with_other"] = np.nan
        out["R_view_selected_axis_mi_with_other"] = np.nan

    return out
