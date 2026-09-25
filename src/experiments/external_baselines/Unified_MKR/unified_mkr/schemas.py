"""Schema definitions for the Unified MKR artifacts.

Every on-disk table column is declared here so the storage layer, the runner,
and the validation layer agree on a single source of truth. The
``record_id`` / ``config_id`` formats are deterministic so resume and later
derivation can reconstruct identities without a side table.
"""
from __future__ import annotations

from typing import List

SCHEMA_VERSION = "1.0.0"

# View identifiers (ClustOpt loader/view-builder vocabulary). These are the
# canonical names used throughout the unified repository.
VIEW_TYPES: tuple[str, ...] = ("x_only", "y_only", "xy_2d")

# Number of clustering configurations evaluated per view (hard requirement).
N_CONFIGS_PER_VIEW = 32

# External (ground-truth) metrics stored for every valid evaluation. ARI/AMI/NMI/FMI
# are the design minimum; v_measure/homogeneity/completeness are added at no extra
# cost (compute_external_metrics returns them) so the full ML2DAC external CVI set
# (ARI, AMI, Homogeneity, V-measure, Completeness, Fowlkes-Mallows) is covered.
EXTERNAL_METRICS: tuple[str, ...] = (
    "ari",
    "ami",
    "nmi",
    "fmi",
    "v_measure",
    "homogeneity",
    "completeness",
)


def make_record_id(dataset_id: str, view_type: str) -> str:
    """Deterministic per-view record identifier, e.g. ``c2e_000801__xy_2d``."""
    return f"{dataset_id}__{view_type}"


def make_config_id(view_type: str, index: int) -> str:
    """Deterministic per-view configuration identifier, e.g. ``xy_2d__cfg_07``."""
    return f"{view_type}__cfg_{int(index):02d}"


# --- Table column orders (documentation + validation aid) ------------------

RECORDS_COLUMNS: List[str] = [
    "record_id",
    "dataset_id",
    "dataset_folder",
    "family",
    "subfamily",
    "split_id",
    "difficulty",
    "view_type",
    "n_samples",
    "n_features",
    "true_k",
    "data_path",
    "true_labels_path",
    "metadata_path",
    "created_at",
    "schema_version",
]

CONFIGURATIONS_COLUMNS: List[str] = [
    "config_id",
    "original_config_id",
    "view_type",
    "algorithm",
    "hyperparameters_json",
    "was_replaced",
    "replacement_reason",
    "source_json_path",
    "schema_version",
]

METRIC_REGISTRY_COLUMNS: List[str] = [
    "metric_id",
    "metric_name",
    "implementation_source",
    "direction",
    "used_by_ml2dac",
    "used_by_autoclust",
    "used_by_clustopt",
    "requires_ground_truth",
    "requires_distance_matrix",
    "requires_cluster_centers",
    "requires_noise_handling",
]

# Fixed (non-metric) columns of the evaluations table. Internal-metric columns
# are appended dynamically from the metric registry.
EVALUATIONS_FIXED_COLUMNS: List[str] = [
    "record_id",
    "config_id",
    "dataset_id",
    "view_type",
    "split_id",
    "algorithm",
    "hyperparameters_json",
    "valid_result",
    "failure_reason",
    "runtime_sec",
    "n_clusters_found",
    "predicted_k",
    "noise_fraction",
    "labels_path",
    "labels_hash",
    "ari",
    "ami",
    "nmi",
    "fmi",
    "v_measure",
    "homogeneity",
    "completeness",
]

METAFEATURES_FIXED_COLUMNS: List[str] = [
    "record_id",
    "dataset_id",
    "view_type",
    "split_id",
    "ml2dac_status",
    "ml2dac_runtime_sec",
    "ml2dac_error",
    "autoclust_status",
    "autoclust_runtime_sec",
    "autoclust_error",
]

FAILURES_COLUMNS: List[str] = [
    "record_id",
    "config_id",
    "dataset_id",
    "view_type",
    "split_id",
    "stage",
    "algorithm",
    "reason",
]
