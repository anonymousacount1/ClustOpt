"""Authoritative 250-feature schema for the Clustering Repository Builder.

Every feature record produced by ``feature_extractor`` must contain ALL columns
listed here (audit + 250 model-input features) in the exact order defined by
``ALL_RECORD_COLUMNS``. Missing values must be ``np.nan`` — never silently
dropped.
"""
from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np


# ---------------------------------------------------------------------------
# Audit / metadata columns (NOT model input features)
# ---------------------------------------------------------------------------
AUDIT_COLUMNS: list[str] = [
    "dataset_id",
    "record_id",
    "view_mode",
    "view_dim",
    "source_dataset_dir",
    "family_id",
    "family_name",
    "subfamily_id",
    "difficulty",
    "cluster_count_from_generator",
    "dataset_seed",
    "family_seed",
    "n_points",
    "feature_names_original",
    "selected_features",
    "has_labels",
    "has_rendering",
    "has_structural_ground_truth",
    "extraction_status",
]

# Type kind per audit column. Drives the "no empty cells" defaults applied by
# `ensure_complete_feature_record`.
AUDIT_COLUMN_KINDS: dict[str, str] = {
    "dataset_id": "string",
    "record_id": "string",
    "view_mode": "string",
    "view_dim": "int",
    "source_dataset_dir": "string",
    "family_id": "string",
    "family_name": "string",
    "subfamily_id": "string",
    "difficulty": "string",
    "cluster_count_from_generator": "int",
    "dataset_seed": "int",
    "family_seed": "int",
    "n_points": "int",
    "feature_names_original": "list",
    "selected_features": "list",
    "has_labels": "bool",
    "has_rendering": "bool",
    "has_structural_ground_truth": "bool",
    "extraction_status": "string",
}

# Constants for the no-empty-cell fill policy.
FEATURE_FILL_VALUE: float = 0.0
AUDIT_STRING_DEFAULT: str = "unknown"
AUDIT_INT_DEFAULT: int = -1
AUDIT_BOOL_DEFAULT: bool = False


# ---------------------------------------------------------------------------
# A block — partition / view-space statistics (30)
# ---------------------------------------------------------------------------
FEATURE_COLUMNS_A: list[str] = [
    "A_view_n_points_log1p",
    "A_view_dim",
    "A_view_finite_ratio",
    "A_view_duplicate_ratio_rounded_6",
    "A_view_unique_ratio_rounded_6",
    "A_view_mean_flat",
    "A_view_std_flat",
    "A_view_min_flat",
    "A_view_max_flat",
    "A_view_range_flat",
    "A_view_median_flat",
    "A_view_iqr_flat",
    "A_view_mad_flat",
    "A_view_q01_flat",
    "A_view_q05_flat",
    "A_view_q25_flat",
    "A_view_q75_flat",
    "A_view_q95_flat",
    "A_view_q99_flat",
    "A_view_skew_flat",
    "A_view_kurtosis_flat",
    "A_view_entropy_hist_32",
    "A_view_entropy_hist_64",
    "A_view_peak_bin_ratio_32",
    "A_view_empty_bin_ratio_32",
    "A_view_tail_mass_low_5pct",
    "A_view_tail_mass_high_95pct",
    "A_view_outlier_ratio_iqr_1_5",
    "A_view_outlier_ratio_z_3",
    "A_view_mean_abs_zscore",
]


# ---------------------------------------------------------------------------
# B block — evaluation-space statistics (45)
# ---------------------------------------------------------------------------
FEATURE_COLUMNS_B: list[str] = [
    "B_axis0_mean",
    "B_axis0_std",
    "B_axis0_min",
    "B_axis0_max",
    "B_axis0_range",
    "B_axis0_iqr",
    "B_axis0_skew",
    "B_axis0_kurtosis",
    "B_axis1_mean",
    "B_axis1_std",
    "B_axis1_min",
    "B_axis1_max",
    "B_axis1_range",
    "B_axis1_iqr",
    "B_axis1_skew",
    "B_axis1_kurtosis",
    "B_bbox_area",
    "B_bbox_aspect_ratio",
    "B_bbox_density_points_per_area",
    "B_cov_trace",
    "B_cov_det",
    "B_cov_condition_number",
    "B_corr_axis0_axis1",
    "B_abs_corr_axis0_axis1",
    "B_pca_eigenvalue_1",
    "B_pca_eigenvalue_2",
    "B_pca_explained_ratio_1",
    "B_pca_explained_ratio_2",
    "B_pca_anisotropy_ratio",
    "B_centroid_norm",
    "B_mean_distance_to_centroid",
    "B_std_distance_to_centroid",
    "B_q25_distance_to_centroid",
    "B_q50_distance_to_centroid",
    "B_q75_distance_to_centroid",
    "B_q90_distance_to_centroid",
    "B_density_grid_16_occupied_ratio",
    "B_density_grid_16_entropy",
    "B_density_grid_16_max_cell_ratio",
    "B_density_grid_32_occupied_ratio",
    "B_density_grid_32_entropy",
    "B_density_grid_32_max_cell_ratio",
    "B_density_grid_cv",
    "B_density_grid_gini",
    "B_density_grid_empty_cell_ratio",
]


# ---------------------------------------------------------------------------
# G block — geometry / structural features without clustering (60)
# ---------------------------------------------------------------------------
FEATURE_COLUMNS_G: list[str] = [
    "G_knn1_dist_mean",
    "G_knn1_dist_std",
    "G_knn1_dist_cv",
    "G_knn1_dist_q10",
    "G_knn1_dist_q50",
    "G_knn1_dist_q90",
    "G_knn3_dist_mean",
    "G_knn3_dist_std",
    "G_knn3_dist_cv",
    "G_knn5_dist_mean",
    "G_knn5_dist_std",
    "G_knn5_dist_cv",
    "G_knn10_dist_mean",
    "G_knn10_dist_std",
    "G_knn10_dist_cv",
    "G_local_density_mean",
    "G_local_density_std",
    "G_local_density_cv",
    "G_local_density_entropy",
    "G_local_density_gini",
    "G_mst_edge_mean",
    "G_mst_edge_std",
    "G_mst_edge_cv",
    "G_mst_edge_q90",
    "G_mst_total_length_norm",
    "G_mst_long_edge_ratio",
    "G_pairwise_dist_mean_sampled",
    "G_pairwise_dist_std_sampled",
    "G_pairwise_dist_q10_sampled",
    "G_pairwise_dist_q50_sampled",
    "G_pairwise_dist_q90_sampled",
    "G_convex_hull_area",
    "G_convex_hull_perimeter",
    "G_convex_hull_area_ratio_to_bbox",
    "G_convex_hull_density",
    "G_alpha_shape_area_approx",
    "G_alpha_shape_compactness_approx",
    "G_linearity_pca_ratio",
    "G_planarity_or_2d_spread",
    "G_curve_smoothness_knn_angle_mean",
    "G_curve_smoothness_knn_angle_std",
    "G_direction_entropy_knn",
    "G_orientation_main_angle",
    "G_orientation_concentration",
    "G_radial_distance_mean",
    "G_radial_distance_std",
    "G_radial_uniformity_cv",
    "G_ring_likeness_radial_peak_score",
    "G_circle_fit_residual_mean",
    "G_circle_fit_residual_std",
    "G_parabola_fit_r2_axis0_to_axis1",
    "G_parabola_fit_r2_axis1_to_axis0",
    "G_linear_fit_r2_axis0_to_axis1",
    "G_linear_fit_r2_axis1_to_axis0",
    "G_piecewise_linear_two_segment_gain",
    "G_periodicity_acf_peak_1d_axis0",
    "G_periodicity_acf_peak_1d_axis1",
    "G_periodicity_fft_peak_axis0",
    "G_periodicity_fft_peak_axis1",
    "G_multimodality_projection_score",
]


# ---------------------------------------------------------------------------
# V block — visual / raster features (60)
# ---------------------------------------------------------------------------
FEATURE_COLUMNS_V: list[str] = [
    "V_image_enabled",
    "V_image_height",
    "V_image_width",
    "V_foreground_fraction",
    "V_foreground_pixels_log1p",
    "V_grayscale_mean",
    "V_grayscale_std",
    "V_grayscale_max",
    "V_grayscale_entropy_32",
    "V_grayscale_nonzero_mean",
    "V_grayscale_nonzero_std",
    "V_binary_component_count",
    "V_binary_component_density",
    "V_largest_component_fraction",
    "V_small_component_fraction",
    "V_component_area_mean",
    "V_component_area_std",
    "V_component_area_cv",
    "V_hole_count_est",
    "V_euler_number_est",
    "V_edge_pixel_fraction",
    "V_edge_pixels_log1p",
    "V_edge_component_count",
    "V_edge_density_per_foreground",
    "V_skeleton_pixel_fraction",
    "V_skeleton_pixels_log1p",
    "V_skeleton_endpoint_count_est",
    "V_skeleton_junction_count_est",
    "V_skeleton_endpoint_per_pixel",
    "V_skeleton_junction_per_pixel",
    "V_distance_transform_mean",
    "V_distance_transform_std",
    "V_distance_transform_max",
    "V_distance_transform_q50",
    "V_distance_transform_q90",
    "V_local_thickness_mean",
    "V_local_thickness_std",
    "V_local_thickness_cv",
    "V_local_thickness_q50",
    "V_local_thickness_q90",
    "V_orientation_entropy",
    "V_orientation_dominant_bin_ratio",
    "V_orientation_second_bin_ratio",
    "V_orientation_anisotropy",
    "V_horizontal_projection_entropy",
    "V_vertical_projection_entropy",
    "V_horizontal_projection_peak_ratio",
    "V_vertical_projection_peak_ratio",
    "V_projection_periodicity_x_acf",
    "V_projection_periodicity_y_acf",
    "V_fft_energy_low_freq_ratio",
    "V_fft_energy_mid_freq_ratio",
    "V_fft_energy_high_freq_ratio",
    "V_fft_peak_to_background_ratio",
    "V_texture_lbp_entropy_approx",
    "V_texture_glcm_contrast_approx",
    "V_texture_glcm_homogeneity_approx",
    "V_texture_glcm_energy_approx",
    "V_fractal_box_counting_slope",
    "V_blob_log_response_peak_approx",
]


# ---------------------------------------------------------------------------
# R block — relation features between original dimensions (25)
# ---------------------------------------------------------------------------
FEATURE_COLUMNS_R: list[str] = [
    "R_has_two_original_dims",
    "R_pearson_corr_xy",
    "R_abs_pearson_corr_xy",
    "R_spearman_corr_xy",
    "R_abs_spearman_corr_xy",
    "R_kendall_corr_xy_approx",
    "R_mutual_information_xy_hist_16",
    "R_mutual_information_xy_hist_32",
    "R_normalized_mi_xy_hist_16",
    "R_linear_r2_x_to_y",
    "R_linear_r2_y_to_x",
    "R_poly2_r2_x_to_y",
    "R_poly2_r2_y_to_x",
    "R_poly3_r2_x_to_y",
    "R_poly3_r2_y_to_x",
    "R_monotonicity_score_x_to_y",
    "R_monotonicity_score_y_to_x",
    "R_conditional_var_y_given_x_bins",
    "R_conditional_var_x_given_y_bins",
    "R_joint_entropy_xy_32",
    "R_entropy_x_32",
    "R_entropy_y_32",
    "R_entropy_ratio_x_y",
    "R_view_selected_axis_corr_with_other",
    "R_view_selected_axis_mi_with_other",
]


# ---------------------------------------------------------------------------
# L block — landmarking features (30, optional)
# ---------------------------------------------------------------------------
FEATURE_COLUMNS_L: list[str] = [
    "L_kmeans_k2_silhouette",
    "L_kmeans_k3_silhouette",
    "L_kmeans_k4_silhouette",
    "L_kmeans_k5_silhouette",
    "L_kmeans_best_k_by_silhouette",
    "L_kmeans_best_silhouette",
    "L_kmeans_inertia_slope_2_to_5",
    "L_gmm_k2_bic",
    "L_gmm_k3_bic",
    "L_gmm_k4_bic",
    "L_gmm_k5_bic",
    "L_gmm_best_k_by_bic",
    "L_gmm_bic_range_norm",
    "L_dbscan_default_n_clusters",
    "L_dbscan_default_noise_ratio",
    "L_dbscan_default_silhouette_if_valid",
    "L_dbscan_loose_n_clusters",
    "L_dbscan_loose_noise_ratio",
    "L_dbscan_strict_n_clusters",
    "L_dbscan_strict_noise_ratio",
    "L_hdbscan_light_n_clusters",
    "L_hdbscan_light_noise_ratio",
    "L_hdbscan_light_cluster_persistence_mean",
    "L_agglomerative_k2_silhouette",
    "L_agglomerative_k3_silhouette",
    "L_agglomerative_k4_silhouette",
    "L_agglomerative_k5_silhouette",
    "L_landmark_best_internal_score",
    "L_landmark_score_variance",
    "L_landmark_algorithm_disagreement",
]


# ---------------------------------------------------------------------------
# Aggregates
# ---------------------------------------------------------------------------
BLOCK_COLUMNS: dict[str, list[str]] = {
    "A": FEATURE_COLUMNS_A,
    "B": FEATURE_COLUMNS_B,
    "G": FEATURE_COLUMNS_G,
    "V": FEATURE_COLUMNS_V,
    "R": FEATURE_COLUMNS_R,
    "L": FEATURE_COLUMNS_L,
}

ALL_FEATURE_COLUMNS: list[str] = (
    FEATURE_COLUMNS_A
    + FEATURE_COLUMNS_B
    + FEATURE_COLUMNS_G
    + FEATURE_COLUMNS_V
    + FEATURE_COLUMNS_R
    + FEATURE_COLUMNS_L
)

ALL_RECORD_COLUMNS: list[str] = AUDIT_COLUMNS + ALL_FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# Schema invariants — fail loudly if anyone reorders or duplicates.
# ---------------------------------------------------------------------------
assert len(FEATURE_COLUMNS_A) == 30, "A block must have 30 features"
assert len(FEATURE_COLUMNS_B) == 45, "B block must have 45 features"
assert len(FEATURE_COLUMNS_G) == 60, "G block must have 60 features"
assert len(FEATURE_COLUMNS_V) == 60, "V block must have 60 features"
assert len(FEATURE_COLUMNS_R) == 25, "R block must have 25 features"
assert len(FEATURE_COLUMNS_L) == 30, "L block must have 30 features"
assert len(ALL_FEATURE_COLUMNS) == 250, "Total feature schema must be 250"
assert len(set(ALL_FEATURE_COLUMNS)) == 250, "Duplicate feature column detected"
assert len(set(ALL_RECORD_COLUMNS)) == len(ALL_RECORD_COLUMNS), (
    "Duplicate audit or feature column detected"
)


def _is_finite_number(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):  # bool is a subclass of int; we don't accept it as a numeric feature
        return False
    try:
        f = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(f))


def _coerce_audit_value(value: object, kind: str) -> object:
    """Coerce an audit value to a non-empty default based on its declared kind.

    - string: trimmed string, "unknown" if missing/empty/non-finite
    - int: integer, -1 if missing/non-coercible/non-finite
    - bool: bool, False if missing
    - list: list, [] if missing
    """
    if kind == "string":
        if value is None:
            return AUDIT_STRING_DEFAULT
        try:
            if isinstance(value, float) and not np.isfinite(value):
                return AUDIT_STRING_DEFAULT
        except (TypeError, ValueError):
            pass
        s = str(value).strip()
        return s if s else AUDIT_STRING_DEFAULT
    if kind == "int":
        if value is None or isinstance(value, bool):
            return AUDIT_INT_DEFAULT if value is None else int(bool(value))
        try:
            f = float(value)
        except (TypeError, ValueError):
            return AUDIT_INT_DEFAULT
        if not np.isfinite(f):
            return AUDIT_INT_DEFAULT
        return int(f)
    if kind == "bool":
        if value is None:
            return AUDIT_BOOL_DEFAULT
        if isinstance(value, bool):
            return value
        try:
            f = float(value)
            if not np.isfinite(f):
                return AUDIT_BOOL_DEFAULT
            return bool(f)
        except (TypeError, ValueError):
            pass
        if isinstance(value, str):
            s = value.strip().lower()
            if s in {"true", "1", "yes", "y"}:
                return True
            if s in {"false", "0", "no", "n", "", "unknown"}:
                return False
        return bool(value)
    if kind == "list":
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return list(value)
        try:
            if isinstance(value, float) and not np.isfinite(value):
                return []
        except (TypeError, ValueError):
            pass
        return [value]
    # Unknown kind: stringify safely.
    return AUDIT_STRING_DEFAULT if value is None else str(value)


def ensure_complete_feature_record(record: Mapping[str, object]) -> dict[str, object]:
    """Return a new dict containing every audit + feature column with no
    empty / NaN values.

    - Feature columns: anything non-finite (None, NaN, +/-Inf, non-numeric)
      becomes ``FEATURE_FILL_VALUE`` (0.0).
    - Audit columns: typed defaults — "unknown" for strings, -1 for ints,
      ``False`` for bools, ``[]`` for lists.

    Output order matches ``ALL_RECORD_COLUMNS`` so CSV columns stay stable.
    """
    out: dict[str, object] = {}
    for col in AUDIT_COLUMNS:
        kind = AUDIT_COLUMN_KINDS.get(col, "string")
        out[col] = _coerce_audit_value(record.get(col, None), kind)
    for col in ALL_FEATURE_COLUMNS:
        v = record.get(col, np.nan)
        out[col] = float(v) if _is_finite_number(v) else FEATURE_FILL_VALUE
    return out


def blocks_to_record(block_outputs: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    """Concatenate per-block dicts ``{A: {...}, B: {...}, ...}`` into one flat
    dict containing exactly the 250 feature columns. Unknown keys raise.
    Missing features become ``np.nan``.
    """
    flat: dict[str, float] = {}
    for block_name, cols in BLOCK_COLUMNS.items():
        block = block_outputs.get(block_name, {})
        for col in cols:
            flat[col] = float(block.get(col, np.nan)) if block.get(col, None) is not None else np.nan
    return flat


def schema_metadata() -> dict[str, object]:
    """Return the schema metadata block for `subfamily_feature_schema.json`."""
    return {
        "n_features": len(ALL_FEATURE_COLUMNS),
        "blocks": {name: len(cols) for name, cols in BLOCK_COLUMNS.items()},
        "audit_columns": list(AUDIT_COLUMNS),
        "feature_columns": list(ALL_FEATURE_COLUMNS),
        "record_columns": list(ALL_RECORD_COLUMNS),
    }


def iter_block_feature_names() -> Iterable[tuple[str, str]]:
    for block_name, cols in BLOCK_COLUMNS.items():
        for col in cols:
            yield block_name, col
