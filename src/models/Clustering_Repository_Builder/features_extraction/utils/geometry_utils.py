"""Geometric helpers — convex hull, circle fit, sampling."""
from __future__ import annotations

import numpy as np


def convex_hull_2d_area_perimeter(points: np.ndarray) -> tuple[float, float]:
    """Return (area, perimeter) of the 2D convex hull. NaN, NaN if degenerate."""
    if points is None or points.ndim != 2 or points.shape[1] < 2 or points.shape[0] < 3:
        return float("nan"), float("nan")
    pts = points[:, :2]
    mask = np.all(np.isfinite(pts), axis=1)
    pts = pts[mask]
    if pts.shape[0] < 3:
        return float("nan"), float("nan")
    try:
        from scipy.spatial import ConvexHull

        hull = ConvexHull(pts)
        return float(hull.volume), float(hull.area)  # volume == area in 2D
    except Exception:
        return float("nan"), float("nan")


def circle_fit_residuals(points: np.ndarray) -> tuple[float, float]:
    """Algebraic circle fit (Kasa). Returns (mean_residual, std_residual).

    Residual = abs( sqrt((x-cx)^2 + (y-cy)^2) - r ).
    """
    if points is None or points.ndim != 2 or points.shape[0] < 3 or points.shape[1] < 2:
        return float("nan"), float("nan")
    pts = points[:, :2]
    mask = np.all(np.isfinite(pts), axis=1)
    pts = pts[mask]
    if pts.shape[0] < 3:
        return float("nan"), float("nan")
    x = pts[:, 0]
    y = pts[:, 1]
    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    b = x * x + y * y
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return float("nan"), float("nan")
    cx, cy, c = sol
    r2 = c + cx * cx + cy * cy
    if r2 <= 0:
        return float("nan"), float("nan")
    r = float(np.sqrt(r2))
    dists = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    residuals = np.abs(dists - r)
    return float(residuals.mean()), float(residuals.std(ddof=0))


def sample_indices(n: int, max_n: int, rng: np.random.Generator) -> np.ndarray:
    if n <= max_n:
        return np.arange(n)
    return rng.choice(n, size=max_n, replace=False)


def sampled_pairwise_distances(points: np.ndarray, max_pairs: int, rng: np.random.Generator) -> np.ndarray:
    n = points.shape[0]
    if n < 2:
        return np.empty(0)
    max_pairs = int(max_pairs)
    if max_pairs <= 0:
        return np.empty(0)
    # All pairs is fine for small n.
    if n * (n - 1) // 2 <= max_pairs:
        idx_i, idx_j = np.triu_indices(n, k=1)
    else:
        i = rng.integers(0, n, size=max_pairs * 2)
        j = rng.integers(0, n, size=max_pairs * 2)
        mask = i != j
        i = i[mask][:max_pairs]
        j = j[mask][:max_pairs]
        idx_i, idx_j = i, j
    diff = points[idx_i] - points[idx_j]
    return np.sqrt(np.sum(diff * diff, axis=1))
