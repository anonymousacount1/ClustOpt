"""V block — visual / raster features (60 features).

All inputs are global, unlabeled raster artifacts. cluster_id_raster.npy and
cluster_masks/ are intentionally excluded — they encode ground-truth structure.
"""
from __future__ import annotations

import numpy as np

from ..config import (
    EPS,
    V_BINARY_THRESHOLD,
    V_BOX_COUNTING_SCALES,
    V_ORIENTATION_BINS,
)
from ..dataset_loader import LoadedDataset
from ..utils.image_utils import binarize
from ..utils.numeric_utils import (
    cv,
    finite_values,
    normalized_entropy,
    safe_acf_peak,
    safe_div,
)


def _nan_record() -> dict[str, float]:
    keys = [
        "V_image_enabled", "V_image_height", "V_image_width",
        "V_foreground_fraction", "V_foreground_pixels_log1p",
        "V_grayscale_mean", "V_grayscale_std", "V_grayscale_max",
        "V_grayscale_entropy_32", "V_grayscale_nonzero_mean", "V_grayscale_nonzero_std",
        "V_binary_component_count", "V_binary_component_density",
        "V_largest_component_fraction", "V_small_component_fraction",
        "V_component_area_mean", "V_component_area_std", "V_component_area_cv",
        "V_hole_count_est", "V_euler_number_est",
        "V_edge_pixel_fraction", "V_edge_pixels_log1p",
        "V_edge_component_count", "V_edge_density_per_foreground",
        "V_skeleton_pixel_fraction", "V_skeleton_pixels_log1p",
        "V_skeleton_endpoint_count_est", "V_skeleton_junction_count_est",
        "V_skeleton_endpoint_per_pixel", "V_skeleton_junction_per_pixel",
        "V_distance_transform_mean", "V_distance_transform_std",
        "V_distance_transform_max", "V_distance_transform_q50", "V_distance_transform_q90",
        "V_local_thickness_mean", "V_local_thickness_std", "V_local_thickness_cv",
        "V_local_thickness_q50", "V_local_thickness_q90",
        "V_orientation_entropy", "V_orientation_dominant_bin_ratio",
        "V_orientation_second_bin_ratio", "V_orientation_anisotropy",
        "V_horizontal_projection_entropy", "V_vertical_projection_entropy",
        "V_horizontal_projection_peak_ratio", "V_vertical_projection_peak_ratio",
        "V_projection_periodicity_x_acf", "V_projection_periodicity_y_acf",
        "V_fft_energy_low_freq_ratio", "V_fft_energy_mid_freq_ratio",
        "V_fft_energy_high_freq_ratio", "V_fft_peak_to_background_ratio",
        "V_texture_lbp_entropy_approx", "V_texture_glcm_contrast_approx",
        "V_texture_glcm_homogeneity_approx", "V_texture_glcm_energy_approx",
        "V_fractal_box_counting_slope", "V_blob_log_response_peak_approx",
    ]
    out = {k: np.nan for k in keys}
    out["V_image_enabled"] = 0.0
    return out


def _connected_components(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (labels, sizes, count). Uses scipy.ndimage.label with 8-connectivity."""
    from scipy.ndimage import label

    structure = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=int)
    labels, count = label(mask.astype(bool), structure=structure)
    if count == 0:
        return labels, np.zeros(0, dtype=int), 0
    sizes = np.bincount(labels.ravel())[1:]
    return labels, sizes, int(count)


def _approximate_lbp_entropy(gray: np.ndarray) -> float:
    """Cheap LBP: each interior pixel against its 8 neighbors, then 256-bin entropy.
    Downsampled for speed.
    """
    img = gray
    if img.size == 0:
        return np.nan
    if img.shape[0] > 128 or img.shape[1] > 128:
        # nearest-neighbor downsample for speed.
        step_y = max(1, img.shape[0] // 128)
        step_x = max(1, img.shape[1] // 128)
        img = img[::step_y, ::step_x]
    h, w = img.shape
    if h < 3 or w < 3:
        return np.nan
    center = img[1:-1, 1:-1].astype(np.int32)
    code = np.zeros_like(center, dtype=np.uint8)
    neighbors = [
        img[:-2, :-2], img[:-2, 1:-1], img[:-2, 2:],
        img[1:-1, 2:],
        img[2:, 2:], img[2:, 1:-1], img[2:, :-2],
        img[1:-1, :-2],
    ]
    for bit, nb in enumerate(neighbors):
        code = code | ((nb >= center).astype(np.uint8) << bit)
    counts = np.bincount(code.ravel(), minlength=256).astype(float)
    return normalized_entropy(counts)


def _glcm_approx(gray: np.ndarray) -> tuple[float, float, float]:
    """Approximate GLCM features at offset (0,1) using 8 gray levels (downsampled)."""
    img = gray
    if img.size == 0:
        return np.nan, np.nan, np.nan
    if img.shape[0] > 96 or img.shape[1] > 96:
        step_y = max(1, img.shape[0] // 96)
        step_x = max(1, img.shape[1] // 96)
        img = img[::step_y, ::step_x]
    h, w = img.shape
    if h < 2 or w < 2:
        return np.nan, np.nan, np.nan
    # Quantize to 8 levels.
    q = np.floor(img.astype(float) / 32.0).clip(0, 7).astype(np.int32)
    a = q[:, :-1].ravel()
    b = q[:, 1:].ravel()
    glcm = np.zeros((8, 8), dtype=float)
    np.add.at(glcm, (a, b), 1.0)
    if glcm.sum() <= 0:
        return np.nan, np.nan, np.nan
    glcm /= glcm.sum()
    idx_i, idx_j = np.indices((8, 8))
    contrast = float(np.sum(((idx_i - idx_j) ** 2) * glcm))
    homogeneity = float(np.sum(glcm / (1.0 + np.abs(idx_i - idx_j))))
    energy = float(np.sum(glcm ** 2))
    return contrast, homogeneity, energy


def _box_counting_slope(mask: np.ndarray) -> float:
    if mask.size == 0 or mask.sum() == 0:
        return np.nan
    h, w = mask.shape
    side = min(h, w)
    if side < 4:
        return np.nan
    counts = []
    sizes = []
    for s in V_BOX_COUNTING_SCALES:
        if s >= side:
            continue
        nh = h // s
        nw = w // s
        if nh < 2 or nw < 2:
            continue
        sub = mask[: nh * s, : nw * s].reshape(nh, s, nw, s).any(axis=(1, 3))
        c = int(sub.sum())
        if c <= 0:
            continue
        counts.append(np.log(c))
        sizes.append(np.log(1.0 / s))
    if len(counts) < 2:
        return np.nan
    slope, _ = np.polyfit(sizes, counts, 1)
    return float(slope)


def _blob_log_approx(gray: np.ndarray) -> float:
    """Approximate LoG peak by Laplacian of a Gaussian-smoothed downsampled image."""
    from scipy.ndimage import gaussian_filter, laplace

    if gray is None or gray.size == 0:
        return np.nan
    img = gray.astype(float)
    if img.shape[0] > 128 or img.shape[1] > 128:
        step_y = max(1, img.shape[0] // 128)
        step_x = max(1, img.shape[1] // 128)
        img = img[::step_y, ::step_x]
    try:
        sm = gaussian_filter(img, sigma=2.0)
        lap = laplace(sm)
        return float(np.max(np.abs(lap)))
    except Exception:
        return np.nan


def _fft_band_energies(mask: np.ndarray) -> tuple[float, float, float, float]:
    """FFT-magnitude energy fractions in low/mid/high radial bands + peak/background ratio."""
    if mask.size == 0:
        return np.nan, np.nan, np.nan, np.nan
    img = mask.astype(float)
    if img.shape[0] > 128 or img.shape[1] > 128:
        step_y = max(1, img.shape[0] // 128)
        step_x = max(1, img.shape[1] // 128)
        img = img[::step_y, ::step_x]
    if img.sum() <= 0:
        return np.nan, np.nan, np.nan, np.nan
    spec = np.abs(np.fft.fftshift(np.fft.fft2(img - img.mean())))
    h, w = spec.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.indices(spec.shape)
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_max = min(cy, cx)
    if r_max <= 0:
        return np.nan, np.nan, np.nan, np.nan
    low = (radius <= r_max / 3.0)
    mid = (radius > r_max / 3.0) & (radius <= 2 * r_max / 3.0)
    high = (radius > 2 * r_max / 3.0) & (radius <= r_max)
    total = float(spec[low | mid | high].sum())
    if total < EPS:
        return np.nan, np.nan, np.nan, np.nan
    e_low = float(spec[low].sum() / total)
    e_mid = float(spec[mid].sum() / total)
    e_high = float(spec[high].sum() / total)
    nonzero = spec[low | mid | high]
    peak_ratio = safe_div(float(nonzero.max()), float(nonzero.mean()))
    return e_low, e_mid, e_high, peak_ratio


def _skeleton_endpoint_junction_counts(skel: np.ndarray) -> tuple[int, int]:
    """Count 8-neighborhood degree=1 endpoints and degree>=3 junctions."""
    if skel.size == 0:
        return 0, 0
    skel_bool = skel.astype(bool)
    kernel = np.ones((3, 3), dtype=int)
    from scipy.signal import convolve2d

    neighbor_count = convolve2d(skel_bool.astype(int), kernel, mode="same", boundary="fill", fillvalue=0)
    neighbor_count = neighbor_count - skel_bool.astype(int)  # exclude self
    endpoints = int(np.sum((skel_bool) & (neighbor_count == 1)))
    junctions = int(np.sum((skel_bool) & (neighbor_count >= 3)))
    return endpoints, junctions


def compute_block_v(loaded: LoadedDataset) -> dict[str, float]:
    out = _nan_record()
    images = loaded.rendering_images
    summary = loaded.render_summary or {}
    if not images:
        return out

    gray = images.get("grayscale.png")
    binary = images.get("binary_mask.png")
    edge = images.get("edge_map.png")
    skel = images.get("skeleton_mask.png")
    dist = images.get("distance_transform.png")
    thick = images.get("local_thickness_map.png")
    orient = images.get("gradient_orientation_map.png")

    if gray is None and binary is None:
        return out

    ref_img = gray if gray is not None else binary
    h, w = ref_img.shape
    total_pix = h * w
    out["V_image_enabled"] = 1.0
    out["V_image_height"] = float(h)
    out["V_image_width"] = float(w)

    # Foreground from binary mask or fallback to grayscale threshold.
    if binary is not None:
        mask_bool = binarize(binary, threshold=V_BINARY_THRESHOLD)
    else:
        mask_bool = binarize(gray, threshold=V_BINARY_THRESHOLD)
    fg_pixels = int(mask_bool.sum())
    out["V_foreground_fraction"] = float(fg_pixels / total_pix) if total_pix > 0 else np.nan
    out["V_foreground_pixels_log1p"] = float(np.log1p(fg_pixels))

    if gray is not None:
        gray_arr = gray.astype(float)
        out["V_grayscale_mean"] = float(gray_arr.mean())
        out["V_grayscale_std"] = float(gray_arr.std(ddof=0))
        out["V_grayscale_max"] = float(gray_arr.max())
        counts, _ = np.histogram(gray_arr.ravel(), bins=32, range=(0, 256))
        out["V_grayscale_entropy_32"] = normalized_entropy(counts.astype(float))
        nz = gray_arr[gray_arr > 0]
        if nz.size > 0:
            out["V_grayscale_nonzero_mean"] = float(nz.mean())
            out["V_grayscale_nonzero_std"] = float(nz.std(ddof=0))

    # Connected components on binary mask (foreground).
    if mask_bool.any():
        labels, sizes, count = _connected_components(mask_bool)
        out["V_binary_component_count"] = float(count)
        out["V_binary_component_density"] = safe_div(float(count), float(fg_pixels))
        if sizes.size > 0:
            largest = float(sizes.max())
            out["V_largest_component_fraction"] = safe_div(largest, float(fg_pixels))
            small_thr = max(4.0, 0.005 * fg_pixels)
            out["V_small_component_fraction"] = float(np.mean(sizes < small_thr))
            out["V_component_area_mean"] = float(sizes.mean())
            out["V_component_area_std"] = float(sizes.std(ddof=0))
            out["V_component_area_cv"] = cv(sizes.astype(float))
        # Holes: components of the inverted mask, excluding the background outside.
        try:
            inv_labels, inv_sizes, inv_count = _connected_components(~mask_bool)
            holes = max(0, inv_count - 1)  # subtract outer background
        except Exception:
            holes = 0
        out["V_hole_count_est"] = float(holes)
        out["V_euler_number_est"] = float(count - holes)
    else:
        out["V_binary_component_count"] = 0.0

    # Edge stats.
    if edge is not None:
        edge_bool = binarize(edge, threshold=V_BINARY_THRESHOLD)
        ep = int(edge_bool.sum())
        out["V_edge_pixel_fraction"] = safe_div(float(ep), float(total_pix))
        out["V_edge_pixels_log1p"] = float(np.log1p(ep))
        if ep > 0:
            _, _, ec = _connected_components(edge_bool)
            out["V_edge_component_count"] = float(ec)
        else:
            out["V_edge_component_count"] = 0.0
        out["V_edge_density_per_foreground"] = safe_div(float(ep), float(fg_pixels))

    # Skeleton stats.
    if skel is not None:
        skel_bool = binarize(skel, threshold=V_BINARY_THRESHOLD)
        sp = int(skel_bool.sum())
        out["V_skeleton_pixel_fraction"] = safe_div(float(sp), float(total_pix))
        out["V_skeleton_pixels_log1p"] = float(np.log1p(sp))
        if sp > 0:
            endpoints, junctions = _skeleton_endpoint_junction_counts(skel_bool)
            out["V_skeleton_endpoint_count_est"] = float(endpoints)
            out["V_skeleton_junction_count_est"] = float(junctions)
            out["V_skeleton_endpoint_per_pixel"] = safe_div(float(endpoints), float(sp))
            out["V_skeleton_junction_per_pixel"] = safe_div(float(junctions), float(sp))
        else:
            out["V_skeleton_endpoint_count_est"] = 0.0
            out["V_skeleton_junction_count_est"] = 0.0

    # Distance transform.
    if dist is not None:
        dt = dist.astype(float)
        # values may be uint8-scaled; we just compute stats over the foreground if available.
        if mask_bool.any():
            values = dt[mask_bool]
        else:
            values = dt.ravel()
        values = values[np.isfinite(values)]
        if values.size > 0:
            out["V_distance_transform_mean"] = float(values.mean())
            out["V_distance_transform_std"] = float(values.std(ddof=0))
            out["V_distance_transform_max"] = float(values.max())
            pcts = np.percentile(values, [50, 90])
            out["V_distance_transform_q50"] = float(pcts[0])
            out["V_distance_transform_q90"] = float(pcts[1])

    # Local thickness.
    if thick is not None:
        lt = thick.astype(float)
        if mask_bool.any():
            values = lt[mask_bool]
        else:
            values = lt.ravel()
        values = values[np.isfinite(values)]
        if values.size > 0:
            mean = float(values.mean())
            std = float(values.std(ddof=0))
            out["V_local_thickness_mean"] = mean
            out["V_local_thickness_std"] = std
            out["V_local_thickness_cv"] = safe_div(std, mean)
            pcts = np.percentile(values, [50, 90])
            out["V_local_thickness_q50"] = float(pcts[0])
            out["V_local_thickness_q90"] = float(pcts[1])

    # Orientation map.
    if orient is not None:
        ang_vals = orient.astype(float)
        if mask_bool.any():
            values = ang_vals[mask_bool]
        else:
            values = ang_vals.ravel()
        # Map 0..255 to [0, pi).
        values = (values.astype(float) / 256.0) * np.pi
        counts, _ = np.histogram(values, bins=V_ORIENTATION_BINS, range=(0.0, np.pi))
        counts = counts.astype(float)
        total = counts.sum()
        out["V_orientation_entropy"] = normalized_entropy(counts) if total > 0 else np.nan
        if total > 0:
            sorted_counts = np.sort(counts)[::-1]
            out["V_orientation_dominant_bin_ratio"] = float(sorted_counts[0] / total)
            out["V_orientation_second_bin_ratio"] = float(sorted_counts[1] / total) if sorted_counts.size > 1 else 0.0
            out["V_orientation_anisotropy"] = float((sorted_counts[0] - (sorted_counts[1] if sorted_counts.size > 1 else 0.0)) / total)

    # Projections (row and column sums of foreground mask).
    if mask_bool.any():
        row_sum = mask_bool.sum(axis=1).astype(float)
        col_sum = mask_bool.sum(axis=0).astype(float)
        total_fg = row_sum.sum()
        out["V_horizontal_projection_entropy"] = normalized_entropy(row_sum) if total_fg > 0 else np.nan
        out["V_vertical_projection_entropy"] = normalized_entropy(col_sum) if total_fg > 0 else np.nan
        out["V_horizontal_projection_peak_ratio"] = safe_div(float(row_sum.max()), float(total_fg))
        out["V_vertical_projection_peak_ratio"] = safe_div(float(col_sum.max()), float(total_fg))
        out["V_projection_periodicity_x_acf"] = safe_acf_peak(col_sum)
        out["V_projection_periodicity_y_acf"] = safe_acf_peak(row_sum)

    # FFT band energies + peak/background ratio (use grayscale if available else mask).
    fft_source = gray if gray is not None else mask_bool.astype(np.uint8) * 255
    e_low, e_mid, e_high, peak_ratio = _fft_band_energies(fft_source)
    out["V_fft_energy_low_freq_ratio"] = e_low
    out["V_fft_energy_mid_freq_ratio"] = e_mid
    out["V_fft_energy_high_freq_ratio"] = e_high
    out["V_fft_peak_to_background_ratio"] = peak_ratio

    # Texture approximations.
    if gray is not None:
        out["V_texture_lbp_entropy_approx"] = _approximate_lbp_entropy(gray)
        c, hom, en = _glcm_approx(gray)
        out["V_texture_glcm_contrast_approx"] = c
        out["V_texture_glcm_homogeneity_approx"] = hom
        out["V_texture_glcm_energy_approx"] = en
        out["V_blob_log_response_peak_approx"] = _blob_log_approx(gray)

    out["V_fractal_box_counting_slope"] = _box_counting_slope(mask_bool)

    # Use foreground_fraction from render_summary if available and binary mask was empty.
    if not np.isfinite(out["V_foreground_fraction"]) and isinstance(summary, dict):
        out["V_foreground_fraction"] = float(summary.get("foreground_fraction", np.nan))

    return out
