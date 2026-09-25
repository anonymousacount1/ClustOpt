from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import numpy as np

from ._helpers import (
    _check_dims,
    _project_points_to_polyline,
    _project_points_to_segment_set,
    _rotation_matrix,
)


def _rank_progress(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return np.zeros((0,), dtype=float)
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.linspace(0.0, 1.0, num=len(x), endpoint=True)
    return ranks


def _estimate_progress(X: np.ndarray, comp_type: str, params_used: Dict[str, Any], i: int, j: int) -> np.ndarray:
    if len(X) == 0:
        return np.zeros((0,), dtype=float)
    if comp_type in {"circle", "ring_annulus"}:
        cx = float(params_used.get("cx", 0.0))
        cy = float(params_used.get("cy", 0.0))
        theta = np.arctan2(X[:, j] - cy, X[:, i] - cx)
        a0 = float(params_used.get("angle_low", -np.pi))
        a1 = float(params_used.get("angle_high", np.pi))
        span = max(1e-12, a1 - a0)
        t = (theta - a0) / span
        return np.mod(t, 1.0)

    if comp_type == "piecewise_polyline":
        verts = np.asarray(params_used.get("vertices", []), dtype=float)
        if len(verts) >= 2:
            _, t = _project_points_to_polyline(X[:, [i, j]], verts, closed=False)
            return t

    if comp_type == "polygon":
        verts = np.asarray(params_used.get("vertices", []), dtype=float)
        if len(verts) >= 3:
            _, t = _project_points_to_polyline(X[:, [i, j]], verts, closed=True)
            return t

    if comp_type == "grid_lattice":
        x = X[:, i]
        y = X[:, j]
        xmin, xmax = np.min(x), np.max(x)
        ymin, ymax = np.min(y), np.max(y)
        tx = (x - xmin) / max(1e-12, xmax - xmin)
        ty = (y - ymin) / max(1e-12, ymax - ymin)
        return 0.5 * np.clip(tx, 0.0, 1.0) + 0.5 * np.clip(ty, 0.0, 1.0)

    if comp_type == "branching_tree":
        verts = np.asarray(params_used.get("vertices", []), dtype=float)
        edges = np.asarray(params_used.get("edges", []), dtype=int)
        if len(verts) >= 2 and len(edges) >= 1:
            _, t = _project_points_to_segment_set(X[:, [i, j]], verts, edges)
            return t

    if comp_type == "meander_curve":
        verts = np.asarray(params_used.get("vertices", []), dtype=float)
        if len(verts) >= 2:
            _, t = _project_points_to_polyline(X[:, [i, j]], verts, closed=False)
            return t

    if "x_low" in params_used and "x_high" in params_used:
        xl = float(params_used["x_low"])
        xh = float(params_used["x_high"])
        span = max(1e-12, xh - xl)
        return np.clip((X[:, i] - xl) / span, 0.0, 1.0)

    x = X[:, i]
    xmin = np.min(x)
    xmax = np.max(x)
    if xmax - xmin <= 1e-12:
        return _rank_progress(x)
    return np.clip((x - xmin) / (xmax - xmin), 0.0, 1.0)


def _profile_values(t: np.ndarray, profile: str) -> np.ndarray:
    t = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
    profile = str(profile).lower()
    if profile == "constant":
        return np.ones_like(t)
    if profile == "linear_up":
        return 0.55 + 1.55 * t
    if profile == "linear_down":
        return 0.55 + 1.55 * (1.0 - t)
    if profile == "center_bulge":
        return 0.65 + 1.65 * np.exp(-((t - 0.5) ** 2) / 0.035)
    if profile == "bimodal":
        return 0.7 + 1.15 * (
            np.exp(-((t - 0.25) ** 2) / 0.02) + np.exp(-((t - 0.75) ** 2) / 0.02)
        )
    return np.ones_like(t)


def _sampling_weights(t: np.ndarray, profile: str, rng: np.random.Generator) -> np.ndarray:
    t = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
    profile = str(profile).lower()
    if profile == "uniform":
        w = np.ones_like(t)
    elif profile == "edge_heavy":
        w = 0.25 + 1.75 * np.abs(2.0 * t - 1.0)
    elif profile == "center_heavy":
        w = 0.25 + 1.75 * (1.0 - np.abs(2.0 * t - 1.0)) ** 2
    elif profile == "bursty":
        centers = rng.uniform(0.05, 0.95, size=3)
        widths = rng.uniform(0.035, 0.09, size=3)
        w = np.full_like(t, 0.08)
        for c, s in zip(centers, widths):
            w += 1.6 * np.exp(-((t - c) ** 2) / (2.0 * s * s))
    elif profile == "periodic_sparse":
        freq = int(rng.integers(3, 6))
        phase = rng.uniform(0.0, 2.0 * np.pi)
        osc = 0.5 * (1.0 + np.sin(2.0 * np.pi * freq * t + phase))
        w = 0.12 + 0.95 * osc
    else:
        w = np.ones_like(t)
    return np.maximum(w, 1e-8)


def _resample_with_weights(
    rng: np.random.Generator,
    X: np.ndarray,
    weights: np.ndarray,
    active_dims: Tuple[int, int],
    bounds_low: np.ndarray,
    bounds_high: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    n = len(X)
    if n == 0:
        return X, {"n_before": 0, "n_after": 0, "unique_fraction": 1.0}
    probs = weights / np.sum(weights)
    idx = rng.choice(n, size=n, replace=True, p=probs)
    X2 = X[idx].copy()
    i, j = active_dims
    span_i = max(1e-12, float(bounds_high[i] - bounds_low[i]))
    span_j = max(1e-12, float(bounds_high[j] - bounds_low[j]))
    X2[:, i] += rng.normal(0.0, 0.0025 * span_i, size=n)
    X2[:, j] += rng.normal(0.0, 0.0025 * span_j, size=n)
    np.clip(X2, bounds_low, bounds_high, out=X2)
    return X2, {
        "n_before": int(n),
        "n_after": int(n),
        "unique_fraction": float(len(np.unique(idx)) / max(1, n)),
    }


def _generic_centerline_from_bins(x: np.ndarray, y: np.ndarray, t: np.ndarray, nbins: int = 48) -> np.ndarray:
    if len(x) == 0:
        return np.zeros((0,), dtype=float)
    edges = np.linspace(0.0, 1.0, nbins + 1)
    mids = 0.5 * (edges[:-1] + edges[1:])
    vals = np.full(nbins, np.nan, dtype=float)
    ids = np.clip(np.digitize(t, edges) - 1, 0, nbins - 1)
    for b in range(nbins):
        mask = ids == b
        if np.any(mask):
            vals[b] = np.median(y[mask])
    if np.all(np.isnan(vals)):
        return np.full_like(y, np.median(y))
    valid = ~np.isnan(vals)
    vals[~valid] = np.interp(mids[~valid], mids[valid], vals[valid])
    return np.interp(t, mids, vals)


def _centerline_and_residual(
    X: np.ndarray,
    comp_type: str,
    params_used: Dict[str, Any],
    i: int,
    j: int,
    t: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    x = X[:, i]
    y = X[:, j]
    if comp_type == "piecewise_polyline":
        proj, _ = _project_points_to_polyline(np.column_stack([x, y]), np.asarray(params_used.get("vertices", []), dtype=float), closed=False)
        yc = proj[:, 1]
        return yc, y - yc
    if comp_type == "polygon":
        proj, _ = _project_points_to_polyline(np.column_stack([x, y]), np.asarray(params_used.get("vertices", []), dtype=float), closed=True)
        yc = proj[:, 1]
        return yc, y - yc
    if comp_type == "branching_tree":
        proj, _ = _project_points_to_segment_set(np.column_stack([x, y]), np.asarray(params_used.get("vertices", []), dtype=float), np.asarray(params_used.get("edges", []), dtype=int))
        yc = proj[:, 1]
        return yc, y - yc
    if comp_type == "meander_curve":
        proj, _ = _project_points_to_polyline(np.column_stack([x, y]), np.asarray(params_used.get("vertices", []), dtype=float), closed=False)
        yc = proj[:, 1]
        return yc, y - yc
    if comp_type == "band":
        m = float(params_used.get("slope", 0.0))
        b = float(params_used.get("intercept", 0.0))
        yc = m * x + b
        return yc, y - yc
    if comp_type == "parallel_bands":
        m = float(params_used.get("slope", 0.0))
        b0 = float(params_used.get("intercept0", 0.0))
        spacing = max(1e-12, float(params_used.get("spacing", 1.0)))
        n_bands = max(1, int(params_used.get("n_bands", 1)))
        base = y - m * x
        k = np.rint((base - b0) / spacing)
        k = np.clip(k, 0, n_bands - 1)
        yc = m * x + b0 + k * spacing
        return yc, y - yc
    if comp_type == "parabola":
        a = float(params_used.get("a", 1.0))
        x0 = float(params_used.get("x0", 0.0))
        b = float(params_used.get("b", 0.0))
        yc = a * (x - x0) ** 2 + b
        return yc, y - yc
    yc = _generic_centerline_from_bins(x, y, t)
    return yc, y - yc


def _apply_sampling_profile(
    rng: np.random.Generator,
    X: np.ndarray,
    profile: str,
    active_dims: Tuple[int, int],
    bounds_low: np.ndarray,
    bounds_high: np.ndarray,
    t: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if str(profile).lower() in {"", "uniform", "none", "native"}:
        return X, {"profile": "uniform", "applied": False}
    weights = _sampling_weights(t, profile, rng)
    X2, info = _resample_with_weights(rng, X, weights, active_dims, bounds_low, bounds_high)
    info.update({"profile": str(profile).lower(), "applied": True})
    return X2, info


def _apply_thickness_profile(
    rng: np.random.Generator,
    X: np.ndarray,
    comp_type: str,
    params_used: Dict[str, Any],
    active_dims: Tuple[int, int],
    bounds_low: np.ndarray,
    bounds_high: np.ndarray,
    t: np.ndarray,
    profile: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    profile = str(profile).lower()
    if profile in {"", "constant", "native", "none"}:
        return X, {"profile": "constant", "applied": False}
    i, j = active_dims
    X2 = X.copy()
    factors = _profile_values(t, profile)
    base_thickness = abs(float(params_used.get("thickness", 0.0)))
    span_y = max(1e-12, float(bounds_high[j] - bounds_low[j]))
    base_thickness = max(base_thickness, 0.004 * span_y)

    if comp_type in {"circle", "ring_annulus"}:
        cx = float(params_used.get("cx", 0.0))
        cy = float(params_used.get("cy", 0.0))
        dx = X[:, i] - cx
        dy = X[:, j] - cy
        r = np.sqrt(dx * dx + dy * dy)
        base_r = max(1e-12, float(params_used.get("r", np.median(r))))
        residual = r - base_r
        residual = residual * factors + rng.normal(0.0, base_thickness * 0.18, size=len(X)) * (factors - 1.0)
        r_new = np.maximum(1e-12, base_r + residual)
        theta = np.arctan2(dy, dx)
        X2[:, i] = cx + r_new * np.cos(theta)
        X2[:, j] = cy + r_new * np.sin(theta)
    else:
        yc, residual = _centerline_and_residual(X, comp_type, params_used, i, j, t)
        residual = residual * factors + rng.normal(0.0, base_thickness * 0.18, size=len(X)) * (factors - 1.0)
        X2[:, j] = yc + residual

    np.clip(X2, bounds_low, bounds_high, out=X2)
    return X2, {
        "profile": profile,
        "applied": True,
        "factor_min": float(np.min(factors)),
        "factor_max": float(np.max(factors)),
    }


def _apply_fragmentation_profile(
    rng: np.random.Generator,
    X: np.ndarray,
    profile: str,
    t: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    profile = str(profile).lower()
    if profile in {"", "none", "native"}:
        return X, {"profile": "none", "applied": False, "keep_fraction": 1.0}
    keep = np.ones(len(X), dtype=bool)
    if profile == "random_gaps":
        n_gaps = int(rng.integers(2, 5))
        for _ in range(n_gaps):
            c = rng.uniform(0.08, 0.92)
            w = rng.uniform(0.04, 0.12)
            keep &= np.abs(t - c) > w * 0.5
    elif profile == "regular_gaps":
        freq = int(rng.integers(3, 6))
        phase = rng.uniform(0.0, 1.0 / freq)
        gap_width = rng.uniform(0.06, 0.12)
        cyc = np.mod(t + phase, 1.0 / freq)
        keep &= cyc > gap_width / freq
    elif profile == "dropout_windows":
        n_windows = int(rng.integers(2, 4))
        for _ in range(n_windows):
            start = rng.uniform(0.0, 0.8)
            width = rng.uniform(0.08, 0.18)
            keep &= ~((t >= start) & (t <= min(1.0, start + width)))
    elif profile == "alternating_presence":
        stripes = int(rng.integers(6, 12))
        ids = np.floor(t * stripes).astype(int)
        keep &= (ids % 2) == 0
    else:
        return X, {"profile": profile, "applied": False, "keep_fraction": 1.0}

    min_keep = max(12, int(0.18 * len(X))) if len(X) > 0 else 0
    if np.sum(keep) < min_keep:
        order = np.argsort(np.abs(t - 0.5))
        keep[:] = False
        keep[order[:min_keep]] = True
    X2 = X[keep].copy()
    return X2, {
        "profile": profile,
        "applied": True,
        "keep_fraction": float(len(X2) / max(1, len(X))),
        "n_before": int(len(X)),
        "n_after": int(len(X2)),
    }


def _apply_roughness_profile(
    rng: np.random.Generator,
    X: np.ndarray,
    comp_type: str,
    params_used: Dict[str, Any],
    active_dims: Tuple[int, int],
    bounds_low: np.ndarray,
    bounds_high: np.ndarray,
    t: np.ndarray,
    profile: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    profile = str(profile).lower()
    if profile in {"", "smooth", "none", "native"}:
        return X, {"profile": "smooth", "applied": False}
    i, j = active_dims
    X2 = X.copy()
    base_thickness = abs(float(params_used.get("thickness", 0.0)))
    span_j = max(1e-12, float(bounds_high[j] - bounds_low[j]))
    amp = max(base_thickness * 0.8, 0.0035 * span_j)

    if profile == "gaussian_edge":
        edge = np.abs(2.0 * t - 1.0)
        delta = rng.normal(0.0, amp, size=len(X)) * (0.35 + 1.45 * edge)
    elif profile == "periodic_wobble":
        freq = int(rng.integers(3, 8))
        phase = rng.uniform(0.0, 2.0 * np.pi)
        delta = amp * np.sin(2.0 * np.pi * freq * t + phase)
    elif profile == "jagged":
        knots = int(rng.integers(8, 16))
        xk = np.linspace(0.0, 1.0, knots)
        yk = rng.normal(0.0, amp, size=knots)
        delta = np.interp(t, xk, yk)
    else:
        return X, {"profile": profile, "applied": False}

    if comp_type in {"circle", "ring_annulus"}:
        cx = float(params_used.get("cx", 0.0))
        cy = float(params_used.get("cy", 0.0))
        dx = X[:, i] - cx
        dy = X[:, j] - cy
        theta = np.arctan2(dy, dx)
        r = np.sqrt(dx * dx + dy * dy) + delta
        r = np.maximum(1e-12, r)
        X2[:, i] = cx + r * np.cos(theta)
        X2[:, j] = cy + r * np.sin(theta)
    else:
        X2[:, j] += delta
    np.clip(X2, bounds_low, bounds_high, out=X2)
    return X2, {"profile": profile, "applied": True, "amplitude": float(np.max(np.abs(delta)) if len(delta) else 0.0)}


def _apply_fill_mode(
    rng: np.random.Generator,
    X: np.ndarray,
    comp_type: str,
    params_used: Dict[str, Any],
    active_dims: Tuple[int, int],
    bounds_low: np.ndarray,
    bounds_high: np.ndarray,
    t: np.ndarray,
    fill_mode: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    fill_mode = str(fill_mode).lower()
    if fill_mode in {"", "native", "ribbon"}:
        return X, {"fill_mode": "native", "applied": False}
    i, j = active_dims
    X2 = X.copy()
    base_thickness = abs(float(params_used.get("thickness", 0.0)))
    span_j = max(1e-12, float(bounds_high[j] - bounds_low[j]))
    base_thickness = max(base_thickness, 0.004 * span_j)

    if comp_type in {"circle", "ring_annulus"}:
        cx = float(params_used.get("cx", 0.0))
        cy = float(params_used.get("cy", 0.0))
        base_r = max(1e-12, float(params_used.get("r", 1.0)))
        theta = np.arctan2(X[:, j] - cy, X[:, i] - cx)
        if fill_mode == "boundary_only":
            r_new = base_r + rng.normal(0.0, 0.22 * base_thickness, size=len(X))
        elif fill_mode == "shell":
            r_new = base_r + rng.uniform(-base_thickness, base_thickness, size=len(X))
        elif fill_mode == "filled":
            radial = np.sqrt(rng.uniform(0.0, 1.0, size=len(X)))
            r_new = base_r * radial
        else:
            return X, {"fill_mode": "native", "applied": False, "fallback": True}
        r_new = np.maximum(1e-12, r_new)
        X2[:, i] = cx + r_new * np.cos(theta)
        X2[:, j] = cy + r_new * np.sin(theta)
    else:
        yc, residual = _centerline_and_residual(X, comp_type, params_used, i, j, t)
        if fill_mode == "boundary_only":
            residual = rng.normal(0.0, 0.22 * base_thickness, size=len(X))
        elif fill_mode == "shell":
            residual = rng.uniform(-base_thickness, base_thickness, size=len(X))
        elif fill_mode == "filled":
            residual = rng.uniform(-2.4 * base_thickness, 2.4 * base_thickness, size=len(X))
        else:
            return X, {"fill_mode": "native", "applied": False, "fallback": True}
        X2[:, j] = yc + residual
    np.clip(X2, bounds_low, bounds_high, out=X2)
    return X2, {"fill_mode": fill_mode, "applied": True}


# =========================
# Stage 1-4 extended operators
# =========================

def _apply_symmetry_profile(rng, X, active_dims, bounds_low, bounds_high, profile):
    profile = str(profile).lower()
    if profile in {"", "none"} or len(X) == 0:
        return X, {"profile": "none", "applied": False}
    i, j = active_dims
    X2 = X.copy()
    cx = float(np.median(X[:, i])); cy = float(np.median(X[:, j]))
    sel = rng.random(len(X2)) < (0.45 if profile == "weak" else 0.7)
    if profile in {"weak", "axial_y"}:
        X2[sel, i] = 2.0 * cx - X2[sel, i]
    elif profile == "axial_x":
        X2[sel, j] = 2.0 * cy - X2[sel, j]
    elif profile == "central":
        X2[sel, i] = 2.0 * cx - X2[sel, i]
        X2[sel, j] = 2.0 * cy - X2[sel, j]
    elif profile == "rotational_2":
        X2[sel, i] = 2.0 * cx - X2[sel, i]
        X2[sel, j] = 2.0 * cy - X2[sel, j]
    elif profile == "rotational_4":
        pts = X2[sel][:, [i, j]] - np.array([[cx, cy]])
        k = rng.integers(0, 4, size=len(pts))
        out = np.zeros_like(pts)
        for kk in range(4):
            m = k == kk
            if np.any(m):
                th = kk * np.pi / 2.0
                out[m] = pts[m] @ _rotation_matrix(th).T
        out += np.array([[cx, cy]])
        X2[sel, i] = out[:, 0]; X2[sel, j] = out[:, 1]
    np.clip(X2, bounds_low, bounds_high, out=X2)
    return X2, {"profile": profile, "applied": True, "center": [cx, cy]}


def _apply_periodicity_profile(rng, X, active_dims, bounds_low, bounds_high, t, profile):
    profile = str(profile).lower()
    if profile in {"", "none"} or len(X) == 0:
        return X, {"profile": "none", "applied": False}
    i, j = active_dims
    X2 = X.copy()
    periods = int(rng.integers(4, 9))
    if profile == "uniform_periodic":
        centers = (np.floor(t * periods) + 0.5) / periods
        noise = rng.normal(0.0, 0.02 / periods, size=len(t))
        t2 = np.clip(centers + noise, 0.0, 1.0)
    elif profile == "jittered_periodic":
        centers = (np.floor(t * periods) + 0.5) / periods
        noise = rng.normal(0.0, 0.05 / periods, size=len(t))
        t2 = np.clip(centers + noise, 0.0, 1.0)
    elif profile == "locally_periodic":
        centers = (np.floor(t * periods) + 0.5) / periods
        keep_local = np.abs(t - 0.5) < 0.32
        t2 = t.copy(); t2[keep_local] = np.clip(centers[keep_local] + rng.normal(0.0, 0.03 / periods, size=np.sum(keep_local)), 0.0, 1.0)
    elif profile == "broken_periodic":
        centers = (np.floor(t * periods) + 0.5) / periods
        t2 = np.clip(centers + rng.normal(0.0, 0.03 / periods, size=len(t)), 0.0, 1.0)
        broken = rng.random(len(t2)) < 0.22
        t2[broken] = t[broken]
    else:
        return X, {"profile": profile, "applied": False}
    order = np.argsort(t)
    new_order = np.argsort(t2)
    X2[order, i] = X[new_order, i]
    X2[order, j] = X[new_order, j]
    np.clip(X2, bounds_low, bounds_high, out=X2)
    return X2, {"profile": profile, "applied": True, "periods": periods}


def _apply_junction_profile(rng, X, active_dims, bounds_low, bounds_high, profile):
    profile = str(profile).lower()
    if profile in {"", "none"} or len(X) == 0:
        return X, {"profile": "none", "applied": False}
    i, j = active_dims
    X2 = X.copy()
    ctr = np.array([np.median(X[:, i]), np.median(X[:, j])], dtype=float)
    pts = X[:, [i, j]]
    if profile in {"weak", "center_dense"}:
        low, high = ((0.75, 1.0) if profile == "weak" else (0.35, 0.65))
        alpha = rng.uniform(low, high, size=len(X))[:, None]
        pts2 = ctr[None, :] + alpha * (pts - ctr[None, :])
    elif profile == "strong":
        alpha = rng.uniform(0.35, 1.0, size=len(X))[:, None]
        pts2 = ctr[None, :] + alpha * (pts - ctr[None, :])
    elif profile == "branch_biased":
        theta = np.arctan2(pts[:, 1] - ctr[1], pts[:, 0] - ctr[0])
        bins = np.round((theta + np.pi) / (2.0 * np.pi / 3.0)).astype(int)
        target = rng.integers(np.min(bins), np.max(bins) + 1)
        mask = bins == target
        pts2 = pts.copy()
        pts2[mask] = ctr[None, :] + 1.18 * (pts[mask] - ctr[None, :])
    else:
        return X, {"profile": profile, "applied": False}
    X2[:, i] = pts2[:, 0]; X2[:, j] = pts2[:, 1]
    np.clip(X2, bounds_low, bounds_high, out=X2)
    return X2, {"profile": profile, "applied": True, "center": ctr.tolist()}


def _apply_curvature_profile(rng, X, active_dims, bounds_low, bounds_high, t, profile):
    profile = str(profile).lower()
    if profile in {"", "none"} or len(X) == 0:
        return X, {"profile": "none", "applied": False}
    i, j = active_dims
    X2 = X.copy()
    amp = 0.06 * max(1e-6, float(bounds_high[j] - bounds_low[j]))
    if profile == "flatten":
        yc = np.median(X[:, j])
        X2[:, j] = yc + 0.55 * (X[:, j] - yc)
    elif profile == "amplify":
        yc = np.median(X[:, j])
        X2[:, j] = yc + 1.35 * (X[:, j] - yc)
    elif profile == "sinusoidal_warp":
        X2[:, j] += amp * np.sin(2.0 * np.pi * (3.0 * t + rng.uniform(0.0, 1.0)))
    elif profile == "piecewise_kink":
        X2[:, j] += np.where(t < 0.5, -amp, amp) * np.minimum(1.0, np.abs(t - 0.5) * 4.0)
    else:
        return X, {"profile": profile, "applied": False}
    np.clip(X2, bounds_low, bounds_high, out=X2)
    return X2, {"profile": profile, "applied": True}


def _apply_occlusion_profile(rng, X, active_dims, profile):
    profile = str(profile).lower()
    if profile in {"", "none"} or len(X) == 0:
        return X, {"profile": "none", "applied": False, "keep_fraction": 1.0}
    i, j = active_dims
    keep = np.ones(len(X), dtype=bool)
    x = X[:, i]; y = X[:, j]
    xl, xh = np.percentile(x, [20, 80])
    yl, yh = np.percentile(y, [20, 80])
    if profile == "left_occluded":
        keep &= x > xl
    elif profile == "right_occluded":
        keep &= x < xh
    elif profile == "center_occluded":
        xm = np.median(x); ym = np.median(y)
        keep &= ~((np.abs(x - xm) < 0.18 * max(1e-9, x.max() - x.min())) & (np.abs(y - ym) < 0.18 * max(1e-9, y.max() - y.min())))
    elif profile == "random_patch":
        xm = rng.uniform(np.min(x), np.max(x)); ym = rng.uniform(np.min(y), np.max(y))
        keep &= ~((np.abs(x - xm) < 0.14 * max(1e-9, x.max() - x.min())) & (np.abs(y - ym) < 0.14 * max(1e-9, y.max() - y.min())))
    else:
        return X, {"profile": profile, "applied": False, "keep_fraction": 1.0}
    X2 = X[keep].copy()
    return X2, {"profile": profile, "applied": True, "keep_fraction": float(len(X2) / max(1, len(X)))}


def _apply_texture_profile(rng, X, active_dims, bounds_low, bounds_high, t, profile):
    profile = str(profile).lower()
    if profile in {"", "none", "smooth"} or len(X) == 0:
        return X, {"profile": "smooth", "applied": False}
    i, j = active_dims
    X2 = X.copy()
    sx = 0.01 * max(1e-6, float(bounds_high[i] - bounds_low[i]))
    sy = 0.01 * max(1e-6, float(bounds_high[j] - bounds_low[j]))
    if profile == "striped":
        X2[:, j] += sy * np.sin(2.0 * np.pi * (8.0 * t + rng.uniform(0.0, 1.0)))
    elif profile == "granular":
        X2[:, i] += rng.normal(0.0, 0.8 * sx, size=len(X2))
        X2[:, j] += rng.normal(0.0, 0.8 * sy, size=len(X2))
    elif profile == "speckled":
        idx = rng.choice(len(X2), size=max(1, len(X2) // 6), replace=False)
        X2[idx, i] += rng.normal(0.0, 2.0 * sx, size=len(idx))
        X2[idx, j] += rng.normal(0.0, 2.0 * sy, size=len(idx))
    elif profile == "oriented":
        th = rng.uniform(-np.pi, np.pi)
        u = np.cos(th) * X2[:, i] + np.sin(th) * X2[:, j]
        X2[:, j] += sy * np.sin(2.0 * np.pi * (u / max(1e-6, np.std(u))))
    elif profile == "multiscale":
        X2[:, j] += sy * (np.sin(2.0 * np.pi * 4.0 * t) + 0.5 * np.sin(2.0 * np.pi * 11.0 * t + 0.7))
    else:
        return X, {"profile": profile, "applied": False}
    np.clip(X2, bounds_low, bounds_high, out=X2)
    return X2, {"profile": profile, "applied": True}


def _apply_topology_profile(rng, X, active_dims, bounds_low, bounds_high, t, profile):
    profile = str(profile).lower()
    if profile in {"", "none", "connected"} or len(X) == 0:
        return X, {"profile": "connected", "applied": False}
    i, j = active_dims
    X2 = X.copy()
    if profile in {"disconnected", "fragmented"}:
        mid = np.abs(t - 0.5) > (0.12 if profile == "disconnected" else 0.2)
        X2 = X2[mid]
        return X2, {"profile": profile, "applied": True, "keep_fraction": float(len(X2) / max(1, len(X)))}
    if profile == "looped":
        ctr = np.array([np.median(X[:, i]), np.median(X[:, j])], dtype=float)
        extra_n = max(12, len(X) // 8)
        theta = np.linspace(0, 2 * np.pi, extra_n, endpoint=False)
        rr = 0.06 * max(float(bounds_high[i] - bounds_low[i]), float(bounds_high[j] - bounds_low[j]))
        loop = np.column_stack([ctr[0] + rr * np.cos(theta), ctr[1] + rr * np.sin(theta)])
        extra = np.zeros((extra_n, X.shape[1]), dtype=float)
        extra[:, i] = loop[:, 0]; extra[:, j] = loop[:, 1]
        X2 = np.vstack([X2, extra])
        np.clip(X2, bounds_low, bounds_high, out=X2)
        return X2, {"profile": profile, "applied": True, "added_points": int(extra_n)}
    if profile == "bridged":
        xlo = np.quantile(X[:, i], 0.15); xhi = np.quantile(X[:, i], 0.85)
        ymid = np.median(X[:, j])
        extra_n = max(16, len(X) // 7)
        extra = np.zeros((extra_n, X.shape[1]), dtype=float)
        extra[:, i] = np.linspace(xlo, xhi, extra_n)
        extra[:, j] = ymid + rng.normal(0.0, 0.01 * max(1e-6, float(bounds_high[j] - bounds_low[j])), size=extra_n)
        X2 = np.vstack([X2, extra])
        np.clip(X2, bounds_low, bounds_high, out=X2)
        return X2, {"profile": profile, "applied": True, "added_points": int(extra_n)}
    if profile == "tree_like":
        ctr = np.array([np.median(X[:, i]), np.median(X[:, j])], dtype=float)
        extra_n = max(18, len(X) // 6)
        angs = np.array([-0.8, -0.2, 0.4], dtype=float)
        picks = rng.integers(0, len(angs), size=extra_n)
        radii = rng.uniform(0.0, 0.18 * max(float(bounds_high[i] - bounds_low[i]), float(bounds_high[j] - bounds_low[j])), size=extra_n)
        extra = np.zeros((extra_n, X.shape[1]), dtype=float)
        extra[:, i] = ctr[0] + radii * np.cos(angs[picks])
        extra[:, j] = ctr[1] + radii * np.sin(angs[picks])
        X2 = np.vstack([X2, extra])
        np.clip(X2, bounds_low, bounds_high, out=X2)
        return X2, {"profile": profile, "applied": True, "added_points": int(extra_n)}
    return X, {"profile": profile, "applied": False}


def apply_component_operators(
    rng: np.random.Generator,
    X: np.ndarray,
    comp_type: str,
    params_used: Dict[str, Any],
    operators: Optional[Dict[str, Any]],
    bounds_low: np.ndarray,
    bounds_high: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    ops = dict(operators or {})
    if not ops or "dims" not in params_used:
        return X, {"requested": ops, "used": {}, "effects": {}}

    i, j = _check_dims(params_used["dims"], X.shape[1], f"operators({comp_type})")
    t = _estimate_progress(X, comp_type, params_used, i, j)

    used: Dict[str, Any] = {}
    effects: Dict[str, Any] = {}
    X2 = X

    fill_mode = str(ops.get("fill_mode", "native")).lower()
    X2, info = _apply_fill_mode(rng, X2, comp_type, params_used, (i, j), bounds_low, bounds_high, t, fill_mode)
    used["fill_mode"] = info.get("fill_mode", fill_mode)
    effects["fill_mode"] = info
    t = _estimate_progress(X2, comp_type, params_used, i, j) if len(X2) > 0 else np.zeros((0,), dtype=float)

    thickness_profile = str(ops.get("thickness_profile", "constant")).lower()
    X2, info = _apply_thickness_profile(rng, X2, comp_type, params_used, (i, j), bounds_low, bounds_high, t, thickness_profile)
    used["thickness_profile"] = info.get("profile", thickness_profile)
    effects["thickness_profile"] = info

    thickness_modulation_profile = str(ops.get("thickness_modulation_profile", "none")).lower()
    mod_map = {
        "monotone_increase": "linear_up",
        "monotone_decrease": "linear_down",
        "center_heavy": "center_bulge",
        "bimodal": "bimodal",
        "strong_bimodal": "bimodal",
    }
    if thickness_modulation_profile in mod_map:
        X2, info = _apply_thickness_profile(rng, X2, comp_type, params_used, (i, j), bounds_low, bounds_high, t, mod_map[thickness_modulation_profile])
        effects["thickness_modulation_profile"] = {**info, "requested_profile": thickness_modulation_profile}
        used["thickness_modulation_profile"] = thickness_modulation_profile
    else:
        effects["thickness_modulation_profile"] = {"profile": thickness_modulation_profile, "applied": False}
        used["thickness_modulation_profile"] = thickness_modulation_profile

    roughness_profile = str(ops.get("roughness_profile", "smooth")).lower()
    X2, info = _apply_roughness_profile(rng, X2, comp_type, params_used, (i, j), bounds_low, bounds_high, t, roughness_profile)
    used["roughness_profile"] = info.get("profile", roughness_profile)
    effects["roughness_profile"] = info

    curvature_profile = str(ops.get("curvature_profile", "none")).lower()
    X2, info = _apply_curvature_profile(rng, X2, (i, j), bounds_low, bounds_high, t, curvature_profile)
    used["curvature_profile"] = info.get("profile", curvature_profile)
    effects["curvature_profile"] = info

    symmetry_profile = str(ops.get("symmetry_profile", "none")).lower()
    X2, info = _apply_symmetry_profile(rng, X2, (i, j), bounds_low, bounds_high, symmetry_profile)
    used["symmetry_profile"] = info.get("profile", symmetry_profile)
    effects["symmetry_profile"] = info

    periodicity_profile = str(ops.get("periodicity_profile", "none")).lower()
    X2, info = _apply_periodicity_profile(rng, X2, (i, j), bounds_low, bounds_high, t, periodicity_profile)
    used["periodicity_profile"] = info.get("profile", periodicity_profile)
    effects["periodicity_profile"] = info

    junction_profile = str(ops.get("junction_profile", "none")).lower()
    X2, info = _apply_junction_profile(rng, X2, (i, j), bounds_low, bounds_high, junction_profile)
    used["junction_profile"] = info.get("profile", junction_profile)
    effects["junction_profile"] = info

    fragmentation_profile = str(ops.get("fragmentation_profile", "none")).lower()
    X2, info = _apply_fragmentation_profile(rng, X2, fragmentation_profile, t)
    used["fragmentation_profile"] = info.get("profile", fragmentation_profile)
    effects["fragmentation_profile"] = info
    t = _estimate_progress(X2, comp_type, params_used, i, j) if len(X2) > 0 else np.zeros((0,), dtype=float)

    occlusion_profile = str(ops.get("occlusion_profile", "none")).lower()
    X2, info = _apply_occlusion_profile(rng, X2, (i, j), occlusion_profile)
    used["occlusion_profile"] = info.get("profile", occlusion_profile)
    effects["occlusion_profile"] = info
    t = _estimate_progress(X2, comp_type, params_used, i, j) if len(X2) > 0 else np.zeros((0,), dtype=float)

    topology_profile = str(ops.get("topology_profile", "none")).lower()
    X2, info = _apply_topology_profile(rng, X2, (i, j), bounds_low, bounds_high, t, topology_profile)
    used["topology_profile"] = info.get("profile", topology_profile)
    effects["topology_profile"] = info
    t = _estimate_progress(X2, comp_type, params_used, i, j) if len(X2) > 0 else np.zeros((0,), dtype=float)

    sampling_profile = str(ops.get("sampling_profile", "uniform")).lower()
    X2, info = _apply_sampling_profile(rng, X2, sampling_profile, (i, j), bounds_low, bounds_high, t)
    used["sampling_profile"] = info.get("profile", sampling_profile)
    effects["sampling_profile"] = info

    texture_profile = str(ops.get("texture_profile", "none")).lower()
    X2, info = _apply_texture_profile(rng, X2, (i, j), bounds_low, bounds_high, t, texture_profile)
    used["texture_profile"] = info.get("profile", texture_profile)
    effects["texture_profile"] = info

    return X2, {"requested": ops, "used": used, "effects": effects}
