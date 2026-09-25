"""
head_groups.py
==============

Curated mapping from clustering-metric name to an output *head* of the
multi-head MLP.  The groups follow the six families suggested in the
implementation plan and were verified to cover exactly the 60 utility targets
of the unified dataset, each metric appearing in precisely one group.

The public entry point is :func:`build_head_mapping`, which takes the list of
target column names (e.g. ``utility__silhouette``) in their canonical order and
returns an ordered mapping ``head_name -> [target_indices...]`` plus a few
convenience structures.  Any metric that does not match a curated group is
routed to a safe ``other`` head so the model never silently drops a target.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List

# Metric name (suffix after the target prefix) -> head.  Order of the groups is
# the canonical head order; within the model, heads write their outputs back to
# the original target-column positions, so this ordering does not affect the
# output vector layout.
METRIC_HEAD_GROUPS: "OrderedDict[str, List[str]]" = OrderedDict(
    [
        (
            "classic_cvi",
            [
                "silhouette",
                "noise_aware_silhouette",
                "calinski_harabasz",
                "davies_bouldin",
                "s_dbw",
                "dbcv",
                "avg_within_cluster_dispersion",
                "avg_pca_isotropy",
                "min_centroid_distance",
                "min_intercluster_distance",
                "cluster_size_balance_entropy",
                "cluster_size_imbalance_entropy",
                "neighborhood_purity",
                "neighborhood_purity_multi_k",
            ],
        ),
        (
            "geometry_shape",
            [
                "convexity_ratio",
                "alpha_shape_compactness",
                "eccentricity_ratio",
                "circularity_compactness",
                "ellipse_fit_score",
                "rectangularity",
                "polygonality",
                "radial_uniformity",
                "convexity_solidity_image",
                "corner_sharpness",
            ],
        ),
        (
            "curves_lines_arcs",
            [
                "arc_circle_fit",
                "hough_circle_arc_strength",
                "hough_line_strength",
                "line_straightness",
                "parabolicity",
                "curvature_consistency",
                "piecewise_linearity",
                "turning_points_count",
                "ribbon_thickness",
                "parallel_bands",
                "ladderness",
                "bimodality_anti_thickness",
            ],
        ),
        (
            "image_morphology",
            [
                "blobness_log_dog",
                "connected_components_count",
                "contour_graph_connectivity",
                "corner_junction_density",
                "edge_coherence",
                "euler_holes_count",
                "morphological_band_count",
                "skeleton_connectivity",
                "thickness_uniformity",
            ],
        ),
        (
            "texture_frequency",
            [
                "fractal_dimension",
                "glcm_haralick",
                "gridness_2d",
                "gridness_fft_acf",
                "lbp_stationarity",
                "orientation_histogram_entropy",
                "periodicity_1d",
                "peak_to_background_hough",
                "ridge_strength",
            ],
        ),
        (
            "topology_connectivity_neighborhood",
            [
                "connectivity_components",
                "mst_smoothness",
                "hollow_score",
                "direction_entropy",
                "axial_symmetry",
                "symmetry_score",
            ],
        ),
    ]
)

OTHER_HEAD = "other"

# ---------------------------------------------------------------------------
# Named target groups (Stage-1 2x3 ablation).
#
# ``head14`` is the *established* CVI inventory: the 6 core CVIs plus the 8
# structural indices.  It is deliberately defined as an alias of the existing
# ``classic_cvi`` head group rather than as a second hand-maintained list, so
# there is exactly one canonical source for the Head-14 membership.  This is the
# same notion that paper_analysis/metric_registry.py calls
# ``is_classic_head_group`` and whose complement it calls ``is_new`` (New-46).
#
# NOTE: this is NOT ``CLASSIC_BASELINE_METRICS`` (silhouette,
# calinski_harabasz, davies_bouldin, dbcv) -- that is a separate 4-metric
# baseline-objective notion and the two must not be conflated.
CLASSIC_HEAD_GROUP = "classic_cvi"

TARGET_GROUP_ALIASES: Dict[str, str] = {
    "head14": CLASSIC_HEAD_GROUP,
    "classic_cvi": CLASSIC_HEAD_GROUP,
}


def resolve_target_group(group_name: str) -> List[str]:
    """Return the ordered metric names for a named target group.

    ``group_name`` may be a head-group name (e.g. ``classic_cvi``) or a
    documented alias (e.g. ``head14``).  Raises ``KeyError`` with the valid
    options if the name is unknown.
    """
    key = TARGET_GROUP_ALIASES.get(group_name, group_name)
    if key not in METRIC_HEAD_GROUPS:
        raise KeyError(
            f"Unknown target group '{group_name}'. Valid groups: "
            f"{sorted(METRIC_HEAD_GROUPS)}; aliases: {sorted(TARGET_GROUP_ALIASES)}."
        )
    return list(METRIC_HEAD_GROUPS[key])


def full60_metric_names() -> List[str]:
    """Every metric across all head groups, in canonical head-group order."""
    out: List[str] = []
    for metrics in METRIC_HEAD_GROUPS.values():
        out.extend(metrics)
    return out


def strip_prefix(target_columns: List[str], target_prefix: str) -> List[str]:
    """Return the bare metric names with ``target_prefix`` removed."""
    out = []
    for col in target_columns:
        out.append(col[len(target_prefix):] if col.startswith(target_prefix) else col)
    return out


def build_head_mapping(
    target_columns: List[str],
    target_prefix: str = "utility__",
) -> Dict[str, object]:
    """Build the head mapping for a given (ordered) list of target columns.

    Returns a dict with:
      - ``heads``: OrderedDict ``head_name -> [target_index, ...]`` (indices into
        ``target_columns``); only non-empty heads are included.
      - ``metric_to_head``: ``metric_name -> head_name``.
      - ``head_order``: list of head names in canonical order.
      - ``unmatched``: list of metric names routed to the ``other`` head.
    """
    metric_names = strip_prefix(target_columns, target_prefix)
    name_to_index = {name: i for i, name in enumerate(metric_names)}

    metric_to_group: Dict[str, str] = {}
    for group, metrics in METRIC_HEAD_GROUPS.items():
        for m in metrics:
            metric_to_group[m] = group

    heads: "OrderedDict[str, List[int]]" = OrderedDict()
    metric_to_head: Dict[str, str] = {}
    unmatched: List[str] = []

    for group in METRIC_HEAD_GROUPS:
        heads[group] = []

    for name in metric_names:
        group = metric_to_group.get(name)
        if group is None:
            group = OTHER_HEAD
            unmatched.append(name)
            heads.setdefault(OTHER_HEAD, [])
        heads[group].append(name_to_index[name])
        metric_to_head[name] = group

    # Drop empty heads (e.g. "other" when everything matched) while keeping order.
    heads = OrderedDict((h, idxs) for h, idxs in heads.items() if idxs)

    return {
        "heads": heads,
        "metric_to_head": metric_to_head,
        "head_order": list(heads.keys()),
        "unmatched": unmatched,
    }
