"""G block — geometry / structural features without clustering (60 features)."""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors

from ..config import (
    EPS,
    G_KNN_K_VALUES,
    G_LOCAL_DENSITY_K,
    G_MAX_PAIRWISE_PAIRS,
    G_MAX_POINTS_FOR_PAIRWISE,
)
from ..utils.geometry_utils import (
    circle_fit_residuals,
    convex_hull_2d_area_perimeter,
    sample_indices,
    sampled_pairwise_distances,
)
from ..utils.numeric_utils import (
    cv,
    finite_values,
    gini,
    linear_r2,
    normalized_entropy,
    poly_r2,
    safe_acf_peak,
    safe_div,
    safe_fft_peak_ratio,
)


def _knn_features(distances_to_k: np.ndarray, k: int) -> dict[str, float]:
    if distances_to_k.size == 0:
        return {
            f"G_knn{k}_dist_mean": np.nan,
            f"G_knn{k}_dist_std": np.nan,
            f"G_knn{k}_dist_cv": np.nan,
        }
    out = {
        f"G_knn{k}_dist_mean": float(distances_to_k.mean()),
        f"G_knn{k}_dist_std": float(distances_to_k.std(ddof=0)),
        f"G_knn{k}_dist_cv": cv(distances_to_k),
    }
    return out


def _approximate_mst_edge_lengths(points: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = points.shape[0]
    if n < 2:
        return np.empty(0)
    if n > 1500:
        idx = rng.choice(n, size=1500, replace=False)
        sub = points[idx]
    else:
        sub = points
    try:
        from scipy.sparse.csgraph import minimum_spanning_tree
        from scipy.spatial.distance import pdist, squareform

        d = squareform(pdist(sub))
        mst = minimum_spanning_tree(d).toarray()
        edges = mst[mst > 0]
        return edges
    except Exception:
        return np.empty(0)


def _radial_features(points2d: np.ndarray) -> dict[str, float]:
    if points2d.shape[0] < 2 or points2d.shape[1] < 2:
        return {
            "G_radial_distance_mean": np.nan,
            "G_radial_distance_std": np.nan,
            "G_radial_uniformity_cv": np.nan,
            "G_ring_likeness_radial_peak_score": np.nan,
        }
    centroid = points2d.mean(axis=0)
    diffs = points2d - centroid
    r = np.sqrt(np.sum(diffs * diffs, axis=1))
    mean = float(r.mean())
    std = float(r.std(ddof=0))
    counts, _ = np.histogram(r, bins=32)
    counts = counts.astype(float)
    peak = float(counts.max() / counts.sum()) if counts.sum() > 0 else np.nan
    return {
        "G_radial_distance_mean": mean,
        "G_radial_distance_std": std,
        "G_radial_uniformity_cv": safe_div(std, mean),
        "G_ring_likeness_radial_peak_score": peak,
    }


def _orientation_features(points2d: np.ndarray, knn_indices: np.ndarray | None) -> dict[str, float]:
    if points2d.shape[0] < 3 or points2d.shape[1] < 2:
        return {
            "G_curve_smoothness_knn_angle_mean": np.nan,
            "G_curve_smoothness_knn_angle_std": np.nan,
            "G_direction_entropy_knn": np.nan,
            "G_orientation_main_angle": np.nan,
            "G_orientation_concentration": np.nan,
        }
    cov = np.cov(points2d, rowvar=False, ddof=0)
    eigvals, eigvecs = np.linalg.eigh(cov)
    main_vec = eigvecs[:, -1]
    main_angle = float(np.arctan2(main_vec[1], main_vec[0]))

    if knn_indices is None or knn_indices.shape[1] < 2:
        return {
            "G_curve_smoothness_knn_angle_mean": np.nan,
            "G_curve_smoothness_knn_angle_std": np.nan,
            "G_direction_entropy_knn": np.nan,
            "G_orientation_main_angle": main_angle,
            "G_orientation_concentration": np.nan,
        }
    # Build local edge vectors using the first non-self neighbor.
    n = points2d.shape[0]
    nbr = knn_indices[:, 1]
    vecs = points2d[nbr] - points2d
    norms = np.linalg.norm(vecs, axis=1)
    valid = norms > EPS
    if valid.sum() < 3:
        return {
            "G_curve_smoothness_knn_angle_mean": np.nan,
            "G_curve_smoothness_knn_angle_std": np.nan,
            "G_direction_entropy_knn": np.nan,
            "G_orientation_main_angle": main_angle,
            "G_orientation_concentration": np.nan,
        }
    angles = np.arctan2(vecs[valid, 1], vecs[valid, 0])
    # collapse direction (vectors and their inverses are equivalent for orientation)
    angles_mod = np.mod(angles, np.pi)
    counts, _ = np.histogram(angles_mod, bins=18, range=(0.0, np.pi))
    counts = counts.astype(float)
    direction_entropy = normalized_entropy(counts)
    concentration = float(counts.max() / counts.sum()) if counts.sum() > 0 else np.nan

    # Smoothness: angle deltas along nearest-neighbor chain at each point (k=1 and k=2 neighbors).
    if knn_indices.shape[1] >= 3:
        nbr1 = knn_indices[:, 1]
        nbr2 = knn_indices[:, 2]
        v1 = points2d[nbr1] - points2d
        v2 = points2d[nbr2] - points2d
        n1 = np.linalg.norm(v1, axis=1)
        n2 = np.linalg.norm(v2, axis=1)
        ok = (n1 > EPS) & (n2 > EPS)
        if ok.sum() > 2:
            cosang = np.einsum("ij,ij->i", v1[ok], v2[ok]) / (n1[ok] * n2[ok])
            cosang = np.clip(cosang, -1.0, 1.0)
            ang = np.arccos(cosang)
            angle_mean = float(ang.mean())
            angle_std = float(ang.std(ddof=0))
        else:
            angle_mean = angle_std = np.nan
    else:
        angle_mean = angle_std = np.nan

    return {
        "G_curve_smoothness_knn_angle_mean": angle_mean,
        "G_curve_smoothness_knn_angle_std": angle_std,
        "G_direction_entropy_knn": direction_entropy,
        "G_orientation_main_angle": main_angle,
        "G_orientation_concentration": concentration,
    }


def _piecewise_linear_two_segment_gain(x: np.ndarray, y: np.ndarray) -> float:
    """SSE improvement of a best two-segment fit over a single linear fit, normalised to [0, 1]."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 10:
        return np.nan
    xs = x[mask]
    ys = y[mask]
    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]
    n = xs.size
    coeffs1 = np.polyfit(xs, ys, 1)
    sse_single = float(np.sum((ys - np.polyval(coeffs1, xs)) ** 2))
    if sse_single < EPS:
        return 0.0
    sample_breakpoints = np.linspace(int(n * 0.1), int(n * 0.9), 7).astype(int)
    sample_breakpoints = np.unique(sample_breakpoints[(sample_breakpoints > 1) & (sample_breakpoints < n - 1)])
    best = sse_single
    for bp in sample_breakpoints:
        try:
            c1 = np.polyfit(xs[:bp], ys[:bp], 1)
            c2 = np.polyfit(xs[bp:], ys[bp:], 1)
        except np.linalg.LinAlgError:
            continue
        sse = float(np.sum((ys[:bp] - np.polyval(c1, xs[:bp])) ** 2)) + float(
            np.sum((ys[bp:] - np.polyval(c2, xs[bp:])) ** 2)
        )
        best = min(best, sse)
    return float(1.0 - best / sse_single)


def _density_histogram(values: np.ndarray, bins: int = 64) -> np.ndarray:
    a = finite_values(values)
    if a.size < 4:
        return np.empty(0)
    counts, _ = np.histogram(a, bins=bins)
    return counts.astype(float)


def _multimodality_score(values: np.ndarray) -> float:
    """Peakiness score: count of bins above mean + std, normalized."""
    counts = _density_histogram(values, bins=64)
    if counts.size == 0:
        return np.nan
    if counts.sum() <= 0:
        return np.nan
    smooth = np.convolve(counts, np.ones(5) / 5.0, mode="same")
    threshold = smooth.mean() + smooth.std(ddof=0)
    peaks = int(np.sum(smooth > threshold))
    return float(peaks / counts.size)


def compute_block_g(X_view: np.ndarray, view_dim: int, rng: np.random.Generator) -> dict[str, float]:
    valid_rows = np.all(np.isfinite(X_view), axis=1)
    pts = X_view[valid_rows]
    n = pts.shape[0]

    out: dict[str, float] = {key: np.nan for key in (
        "G_knn1_dist_mean", "G_knn1_dist_std", "G_knn1_dist_cv",
        "G_knn1_dist_q10", "G_knn1_dist_q50", "G_knn1_dist_q90",
        "G_knn3_dist_mean", "G_knn3_dist_std", "G_knn3_dist_cv",
        "G_knn5_dist_mean", "G_knn5_dist_std", "G_knn5_dist_cv",
        "G_knn10_dist_mean", "G_knn10_dist_std", "G_knn10_dist_cv",
        "G_local_density_mean", "G_local_density_std", "G_local_density_cv",
        "G_local_density_entropy", "G_local_density_gini",
        "G_mst_edge_mean", "G_mst_edge_std", "G_mst_edge_cv",
        "G_mst_edge_q90", "G_mst_total_length_norm", "G_mst_long_edge_ratio",
        "G_pairwise_dist_mean_sampled", "G_pairwise_dist_std_sampled",
        "G_pairwise_dist_q10_sampled", "G_pairwise_dist_q50_sampled", "G_pairwise_dist_q90_sampled",
        "G_convex_hull_area", "G_convex_hull_perimeter",
        "G_convex_hull_area_ratio_to_bbox", "G_convex_hull_density",
        "G_alpha_shape_area_approx", "G_alpha_shape_compactness_approx",
        "G_linearity_pca_ratio", "G_planarity_or_2d_spread",
        "G_curve_smoothness_knn_angle_mean", "G_curve_smoothness_knn_angle_std",
        "G_direction_entropy_knn",
        "G_orientation_main_angle", "G_orientation_concentration",
        "G_radial_distance_mean", "G_radial_distance_std",
        "G_radial_uniformity_cv", "G_ring_likeness_radial_peak_score",
        "G_circle_fit_residual_mean", "G_circle_fit_residual_std",
        "G_parabola_fit_r2_axis0_to_axis1", "G_parabola_fit_r2_axis1_to_axis0",
        "G_linear_fit_r2_axis0_to_axis1", "G_linear_fit_r2_axis1_to_axis0",
        "G_piecewise_linear_two_segment_gain",
        "G_periodicity_acf_peak_1d_axis0", "G_periodicity_acf_peak_1d_axis1",
        "G_periodicity_fft_peak_axis0", "G_periodicity_fft_peak_axis1",
        "G_multimodality_projection_score",
    )}

    if n < 2:
        return out

    # Sample points for expensive ops.
    sample_idx = sample_indices(n, G_MAX_POINTS_FOR_PAIRWISE, rng)
    pts_sample = pts[sample_idx]
    n_sample = pts_sample.shape[0]

    # kNN distances (k up to 10 + 1 for self).
    max_k_needed = max(G_KNN_K_VALUES) + 1
    if n_sample >= max_k_needed:
        nn = NearestNeighbors(n_neighbors=max_k_needed, algorithm="auto").fit(pts_sample)
        knn_dists, knn_idx = nn.kneighbors(pts_sample)
        # column 0 is self.
        for k in G_KNN_K_VALUES:
            d_k = knn_dists[:, k]
            out.update(_knn_features(d_k, k))
        # k=1 percentiles
        d1 = knn_dists[:, 1]
        pcts = np.percentile(d1, [10, 50, 90])
        out["G_knn1_dist_q10"] = float(pcts[0])
        out["G_knn1_dist_q50"] = float(pcts[1])
        out["G_knn1_dist_q90"] = float(pcts[2])

        # Local density from k=10 distances.
        k_dens = min(G_LOCAL_DENSITY_K, max_k_needed - 1)
        dens = 1.0 / (knn_dists[:, k_dens] + EPS)
        out["G_local_density_mean"] = float(dens.mean())
        out["G_local_density_std"] = float(dens.std(ddof=0))
        out["G_local_density_cv"] = cv(dens)
        dens_counts, _ = np.histogram(dens, bins=32)
        out["G_local_density_entropy"] = normalized_entropy(dens_counts.astype(float))
        out["G_local_density_gini"] = gini(dens)
    else:
        knn_idx = None

    # MST (sampled).
    mst_edges = _approximate_mst_edge_lengths(pts_sample, rng)
    if mst_edges.size > 0:
        out["G_mst_edge_mean"] = float(mst_edges.mean())
        out["G_mst_edge_std"] = float(mst_edges.std(ddof=0))
        out["G_mst_edge_cv"] = cv(mst_edges)
        out["G_mst_edge_q90"] = float(np.percentile(mst_edges, 90))
        out["G_mst_total_length_norm"] = float(mst_edges.sum() / np.sqrt(max(n_sample, 1)))
        thr = float(np.percentile(mst_edges, 90))
        out["G_mst_long_edge_ratio"] = float(np.mean(mst_edges > thr))

    # Sampled pairwise.
    pair_d = sampled_pairwise_distances(pts_sample, G_MAX_PAIRWISE_PAIRS, rng)
    if pair_d.size > 0:
        out["G_pairwise_dist_mean_sampled"] = float(pair_d.mean())
        out["G_pairwise_dist_std_sampled"] = float(pair_d.std(ddof=0))
        pcts = np.percentile(pair_d, [10, 50, 90])
        out["G_pairwise_dist_q10_sampled"] = float(pcts[0])
        out["G_pairwise_dist_q50_sampled"] = float(pcts[1])
        out["G_pairwise_dist_q90_sampled"] = float(pcts[2])

    # Convex hull (2D only).
    if view_dim >= 2 and pts.shape[1] >= 2:
        area, perim = convex_hull_2d_area_perimeter(pts)
        out["G_convex_hull_area"] = area
        out["G_convex_hull_perimeter"] = perim
        x = pts[:, 0]
        y = pts[:, 1]
        bbox = (float(x.max() - x.min())) * (float(y.max() - y.min()))
        if np.isfinite(area) and bbox > 0:
            out["G_convex_hull_area_ratio_to_bbox"] = safe_div(area, bbox)
        if np.isfinite(area) and area > 0:
            out["G_convex_hull_density"] = float(n / area)
        # Alpha-shape approximation: shrink hull by removing far-from-centroid points
        # then re-hull. For first pass, fall back to hull-based compactness.
        if np.isfinite(area) and np.isfinite(perim) and perim > 0:
            out["G_alpha_shape_area_approx"] = area
            out["G_alpha_shape_compactness_approx"] = float(4.0 * np.pi * area / (perim * perim))

    # PCA-based linearity/planarity.
    if pts.shape[1] >= 2 and n >= 2:
        cov = np.cov(pts, rowvar=False, ddof=0)
        cov = np.atleast_2d(cov)
        eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
        if eigvals.size >= 2 and eigvals[0] > EPS:
            ratio = float(eigvals[0] / (eigvals[0] + eigvals[1] + EPS))
            out["G_linearity_pca_ratio"] = ratio
            out["G_planarity_or_2d_spread"] = float(eigvals[1] / (eigvals[0] + EPS))
        elif view_dim == 1 and eigvals.size >= 1:
            out["G_linearity_pca_ratio"] = 1.0
            out["G_planarity_or_2d_spread"] = np.nan

    # Orientation + smoothness from kNN.
    if view_dim >= 2 and pts_sample.shape[1] >= 2:
        out.update(_orientation_features(pts_sample, knn_idx))
        out.update(_radial_features(pts_sample))
        cmean, cstd = circle_fit_residuals(pts_sample)
        out["G_circle_fit_residual_mean"] = cmean
        out["G_circle_fit_residual_std"] = cstd
        x = pts[:, 0]
        y = pts[:, 1]
        out["G_parabola_fit_r2_axis0_to_axis1"] = poly_r2(x, y, 2)
        out["G_parabola_fit_r2_axis1_to_axis0"] = poly_r2(y, x, 2)
        out["G_linear_fit_r2_axis0_to_axis1"] = linear_r2(x, y)
        out["G_linear_fit_r2_axis1_to_axis0"] = linear_r2(y, x)
        out["G_piecewise_linear_two_segment_gain"] = _piecewise_linear_two_segment_gain(x, y)

        # Periodicity / FFT on 1D density histograms of each axis.
        counts_x = _density_histogram(x, bins=64)
        counts_y = _density_histogram(y, bins=64)
        out["G_periodicity_acf_peak_1d_axis0"] = safe_acf_peak(counts_x)
        out["G_periodicity_acf_peak_1d_axis1"] = safe_acf_peak(counts_y)
        out["G_periodicity_fft_peak_axis0"] = safe_fft_peak_ratio(counts_x)
        out["G_periodicity_fft_peak_axis1"] = safe_fft_peak_ratio(counts_y)
        out["G_multimodality_projection_score"] = max(
            _multimodality_score(x) or np.nan,
            _multimodality_score(y) or np.nan,
        )
    else:
        # 1D view — radial features collapse to absolute deviations from mean.
        col = pts[:, 0]
        mean_v = float(col.mean())
        r = np.abs(col - mean_v)
        out["G_radial_distance_mean"] = float(r.mean())
        out["G_radial_distance_std"] = float(r.std(ddof=0))
        out["G_radial_uniformity_cv"] = cv(r)
        counts_x = _density_histogram(col, bins=64)
        out["G_periodicity_acf_peak_1d_axis0"] = safe_acf_peak(counts_x)
        out["G_periodicity_fft_peak_axis0"] = safe_fft_peak_ratio(counts_x)
        out["G_multimodality_projection_score"] = _multimodality_score(col)

    return out
