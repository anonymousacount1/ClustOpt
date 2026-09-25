"""B block — evaluation-space statistics (45 features)."""
from __future__ import annotations

import numpy as np

from ..config import B_DENSITY_GRID_BINS, EPS
from ..utils.numeric_utils import (
    cv,
    finite_values,
    gini,
    normalized_entropy,
    percentiles,
    safe_corrcoef,
    safe_div,
    safe_kurtosis,
    safe_skew,
)


def _axis_stats(values: np.ndarray, prefix: str, is_valid: bool) -> dict[str, float]:
    pcts = percentiles(values, (25, 75))
    if not is_valid or values.size == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_range": np.nan,
            f"{prefix}_iqr": np.nan,
            f"{prefix}_skew": np.nan,
            f"{prefix}_kurtosis": np.nan,
        }
    return {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_std": float(values.std(ddof=0)),
        f"{prefix}_min": float(values.min()),
        f"{prefix}_max": float(values.max()),
        f"{prefix}_range": float(values.max() - values.min()),
        f"{prefix}_iqr": float(pcts[75] - pcts[25]),
        f"{prefix}_skew": safe_skew(values),
        f"{prefix}_kurtosis": safe_kurtosis(values),
    }


def _density_grid_features(X_view: np.ndarray, view_dim: int, bins: int) -> dict[str, float]:
    n = X_view.shape[0]
    if n == 0:
        return {
            "occupied_ratio": np.nan,
            "entropy": np.nan,
            "max_cell_ratio": np.nan,
            "counts": np.zeros(0, dtype=float),
        }
    if view_dim == 1:
        col = X_view[:, 0]
        mask = np.isfinite(col)
        if mask.sum() < 2:
            return {"occupied_ratio": np.nan, "entropy": np.nan, "max_cell_ratio": np.nan, "counts": np.zeros(0)}
        counts, _ = np.histogram(col[mask], bins=bins)
    else:
        x = X_view[:, 0]
        y = X_view[:, 1]
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 2:
            return {"occupied_ratio": np.nan, "entropy": np.nan, "max_cell_ratio": np.nan, "counts": np.zeros(0)}
        counts, _, _ = np.histogram2d(x[mask], y[mask], bins=bins)
    counts = counts.astype(float).ravel()
    total_cells = counts.size
    total = counts.sum()
    occupied = int(np.sum(counts > 0))
    occupied_ratio = float(occupied / total_cells) if total_cells > 0 else np.nan
    entropy = normalized_entropy(counts) if total > 0 else np.nan
    max_cell_ratio = float(counts.max() / total) if total > 0 else np.nan
    return {
        "occupied_ratio": occupied_ratio,
        "entropy": entropy,
        "max_cell_ratio": max_cell_ratio,
        "counts": counts,
    }


def compute_block_b(X_view: np.ndarray, view_dim: int) -> dict[str, float]:
    out: dict[str, float] = {}

    axis0 = X_view[:, 0]
    axis0_finite = axis0[np.isfinite(axis0)]
    out.update(_axis_stats(axis0_finite, "B_axis0", is_valid=True))

    if view_dim >= 2 and X_view.shape[1] >= 2:
        axis1 = X_view[:, 1]
        axis1_finite = axis1[np.isfinite(axis1)]
        out.update(_axis_stats(axis1_finite, "B_axis1", is_valid=True))
    else:
        out.update(_axis_stats(np.empty(0), "B_axis1", is_valid=False))

    # bbox
    if view_dim >= 2 and X_view.shape[1] >= 2:
        x = X_view[:, 0]
        y = X_view[:, 1]
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() >= 2:
            xf = x[mask]
            yf = y[mask]
            width = float(xf.max() - xf.min())
            height = float(yf.max() - yf.min())
            area = width * height
            out["B_bbox_area"] = area if area > 0 else np.nan
            out["B_bbox_aspect_ratio"] = safe_div(width, height)
            out["B_bbox_density_points_per_area"] = safe_div(float(mask.sum()), area)
        else:
            out["B_bbox_area"] = np.nan
            out["B_bbox_aspect_ratio"] = np.nan
            out["B_bbox_density_points_per_area"] = np.nan
    else:
        out["B_bbox_area"] = np.nan
        out["B_bbox_aspect_ratio"] = np.nan
        out["B_bbox_density_points_per_area"] = np.nan

    # covariance and PCA
    valid_rows = np.all(np.isfinite(X_view), axis=1)
    X_valid = X_view[valid_rows]
    if X_valid.shape[0] >= 2:
        if view_dim >= 2 and X_valid.shape[1] >= 2:
            cov = np.cov(X_valid, rowvar=False, ddof=0)
            cov = np.atleast_2d(cov)
            eigvals = np.linalg.eigvalsh(cov)
            eigvals = np.sort(eigvals)[::-1]
            eig1, eig2 = float(eigvals[0]), float(eigvals[1])
            total_var = float(eigvals.sum())
            cov_det = float(np.linalg.det(cov))
            trace = float(np.trace(cov))
            cond = safe_div(abs(eig1), abs(eig2))
            corr = safe_corrcoef(X_valid[:, 0], X_valid[:, 1])
            out["B_cov_trace"] = trace
            out["B_cov_det"] = cov_det
            out["B_cov_condition_number"] = cond
            out["B_corr_axis0_axis1"] = corr
            out["B_abs_corr_axis0_axis1"] = abs(corr) if np.isfinite(corr) else np.nan
            out["B_pca_eigenvalue_1"] = eig1
            out["B_pca_eigenvalue_2"] = eig2
            out["B_pca_explained_ratio_1"] = safe_div(eig1, total_var)
            out["B_pca_explained_ratio_2"] = safe_div(eig2, total_var)
            out["B_pca_anisotropy_ratio"] = safe_div(eig1, eig2)
            centroid = X_valid.mean(axis=0)
            diffs = X_valid - centroid
            dists = np.sqrt(np.sum(diffs * diffs, axis=1))
            centroid_norm = float(np.linalg.norm(centroid))
        else:
            v = X_valid[:, 0]
            var = float(v.var(ddof=0))
            out["B_cov_trace"] = var
            out["B_cov_det"] = np.nan
            out["B_cov_condition_number"] = np.nan
            out["B_corr_axis0_axis1"] = np.nan
            out["B_abs_corr_axis0_axis1"] = np.nan
            out["B_pca_eigenvalue_1"] = var
            out["B_pca_eigenvalue_2"] = np.nan
            out["B_pca_explained_ratio_1"] = 1.0 if var > 0 else np.nan
            out["B_pca_explained_ratio_2"] = np.nan
            out["B_pca_anisotropy_ratio"] = np.nan
            centroid = X_valid.mean(axis=0)
            dists = np.abs(X_valid[:, 0] - centroid[0])
            centroid_norm = float(abs(centroid[0]))

        out["B_centroid_norm"] = centroid_norm
        out["B_mean_distance_to_centroid"] = float(dists.mean())
        out["B_std_distance_to_centroid"] = float(dists.std(ddof=0))
        dpct = np.percentile(dists, [25, 50, 75, 90])
        out["B_q25_distance_to_centroid"] = float(dpct[0])
        out["B_q50_distance_to_centroid"] = float(dpct[1])
        out["B_q75_distance_to_centroid"] = float(dpct[2])
        out["B_q90_distance_to_centroid"] = float(dpct[3])
    else:
        for key in (
            "B_cov_trace", "B_cov_det", "B_cov_condition_number",
            "B_corr_axis0_axis1", "B_abs_corr_axis0_axis1",
            "B_pca_eigenvalue_1", "B_pca_eigenvalue_2",
            "B_pca_explained_ratio_1", "B_pca_explained_ratio_2",
            "B_pca_anisotropy_ratio", "B_centroid_norm",
            "B_mean_distance_to_centroid", "B_std_distance_to_centroid",
            "B_q25_distance_to_centroid", "B_q50_distance_to_centroid",
            "B_q75_distance_to_centroid", "B_q90_distance_to_centroid",
        ):
            out[key] = np.nan

    # Density grids
    grid16 = _density_grid_features(X_view, view_dim, 16)
    grid32 = _density_grid_features(X_view, view_dim, 32)
    out["B_density_grid_16_occupied_ratio"] = grid16["occupied_ratio"]
    out["B_density_grid_16_entropy"] = grid16["entropy"]
    out["B_density_grid_16_max_cell_ratio"] = grid16["max_cell_ratio"]
    out["B_density_grid_32_occupied_ratio"] = grid32["occupied_ratio"]
    out["B_density_grid_32_entropy"] = grid32["entropy"]
    out["B_density_grid_32_max_cell_ratio"] = grid32["max_cell_ratio"]

    counts32 = grid32["counts"]
    if counts32.size > 0 and counts32.sum() > 0:
        nonzero = counts32[counts32 > 0]
        out["B_density_grid_cv"] = cv(nonzero)
        out["B_density_grid_gini"] = gini(counts32)
        out["B_density_grid_empty_cell_ratio"] = float(np.sum(counts32 == 0) / counts32.size)
    else:
        out["B_density_grid_cv"] = np.nan
        out["B_density_grid_gini"] = np.nan
        out["B_density_grid_empty_cell_ratio"] = np.nan

    return out
