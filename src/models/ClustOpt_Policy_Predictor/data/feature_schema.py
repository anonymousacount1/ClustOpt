"""Label-free meta-feature schema and column-role contract.

The Policy Predictor operates at DATASET x VIEW. Its base inputs are the
label-free meta-features the repository already extracts per view via
``features_extraction.feature_extractor.extract_features_for_view`` -- which
computes "the full 250-feature record for one view", with
``record_id = f"{dataset_id}__{view_mode}"``. Views are built by
``features_extraction.view_builder.build_views`` as ``x_only`` (dim 1),
``y_only`` (dim 1) and ``xy_2d`` (dim 2).

The canonical ordered column list is the one the accepted Full-60 utility MLP was
fitted on, so the Policy Predictor consumes exactly the same representation as the
utility predictors it selects between.

Column roles are declared here as *explicit semantic sets*, never by substring
matching. Stage 2A-2 recorded why: a substring scan for ``ari`` flags
``G_linearity_pca_ratio``, ``G_planarity_or_2d_spread`` and
``L_landmark_score_variance``, all of which are label-free geometry features.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Sequence

# --------------------------------------------------------------------------- #
# Canonical sources
# --------------------------------------------------------------------------- #
MLP_ARTIFACT_REL = ("results_analysis/mlp/"
                    "20260615_231715__metric_utility_mlp_split1_holdout")
FEATURE_COLUMNS_REL = f"{MLP_ARTIFACT_REL}/artifacts/feature_columns.json"
UNIFIED_DATASET_REL = ("results_analysis/clustering_repository/analyzed_data/"
                       "unified_training_dataset/unified_training_dataset.csv")
KNN_METRIC_SCHEMA_REL = ("results_analysis/metric_utility_knn/phase_e1/models/"
                         "utility_metric_schema.json")

VIEW_IDS: tuple[str, ...] = ("x_only", "y_only", "xy_2d")
VIEW_DIM: Dict[str, int] = {"x_only": 1, "y_only": 1, "xy_2d": 2}

# Feature blocks, from features_extraction/blocks/.
FEATURE_BLOCKS: Dict[str, str] = {
    "A": "partition statistics",
    "B": "evaluation statistics",
    "G": "geometry / structure",
    "L": "landmarking",
    "R": "relation features",
    "V": "visual raster",
}

# --------------------------------------------------------------------------- #
# Column roles
# --------------------------------------------------------------------------- #
ROLE_META = "FEATURE_ALLOWED_META"
ROLE_OOF_UTILITY = "FEATURE_ALLOWED_OOF_UTILITY"
ROLE_OPTIONAL = "FEATURE_ALLOWED_OPTIONAL"
ROLE_TARGET = "TARGET_ONLY"
ROLE_METADATA = "METADATA_ONLY"
ROLE_FORBIDDEN = "FORBIDDEN"

#: Identifiers and provenance. Never predictive inputs.
#: ``oof_heldout_split`` records which Stage-2A-1 fold produced this row's utility
#: features. It is provenance used by the leakage gate, not a dataset property.
METADATA_COLUMNS: tuple[str, ...] = (
    "dataset_id", "record_id", "split_id", "family", "subfamily", "view_id",
    "oof_heldout_split",
)

#: Semantically forbidden as features: ground-truth or oracle derived.
FORBIDDEN_COLUMNS: tuple[str, ...] = (
    "ari", "adjusted_rand_index", "nmi", "ami", "v_measure", "fowlkes_mallows",
    "true_k", "cluster_count", "cluster_count_from_generator", "labels", "y_true",
    "selected_ari", "best_view_ari", "view_policy_vbs", "oracle_best_ari",
    "winner_set", "n_exact_winners", "true_utility", "utility_vector",
)

#: Prefixes marking target/diagnostic columns.
TARGET_PREFIXES: tuple[str, ...] = (
    "policy_ari__", "policy_regret__", "target__",
)
TARGET_EXTRA: tuple[str, ...] = (
    "view_policy_vbs", "winner_set", "n_exact_winners",
    "top1_minus_top2_policy_score", "top1_minus_second_distinct_score",
    "n_within_0_001", "n_within_0_005", "n_within_0_01",
)

#: Optional derived features, kept out of the 120-vector.
OPTIONAL_FEATURE_COLUMNS: tuple[str, ...] = (
    "oof_mlp_dynamic_k", "oof_knn_dynamic_k",
    "oof_mlp_k_rel_raw", "oof_knn_k_rel_raw",
)


def load_meta_feature_columns(repo: Path) -> List[str]:
    """The canonical ordered 250 label-free meta-feature columns."""
    with open(_lp(repo / FEATURE_COLUMNS_REL), "r", encoding="utf-8") as fh:
        cols = json.load(fh)
    if not isinstance(cols, list) or not cols:
        raise ValueError("feature_columns.json is not a non-empty list")
    return list(cols)


def load_metric_names(repo: Path) -> List[str]:
    """The canonical 60-metric order shared by the MLP and KNN utility sources."""
    with open(_lp(repo / KNN_METRIC_SCHEMA_REL), "r", encoding="utf-8") as fh:
        return list(json.load(fh)["metric_names"])


def oof_feature_columns(metric_names: Sequence[str]) -> List[str]:
    """The 120 OOF utility feature columns, MLP block then KNN block."""
    return ([f"oof_mlp_utility__{m}" for m in metric_names]
            + [f"oof_knn_utility__{m}" for m in metric_names])


def block_of(column: str) -> str:
    return column.split("_", 1)[0] if column[:1] in FEATURE_BLOCKS else "?"


def schema_hash(columns: Sequence[str]) -> str:
    return hashlib.sha256(
        json.dumps(list(columns), separators=(",", ":")).encode()).hexdigest()


def classify_column(name: str, *, meta: Sequence[str],
                    oof: Sequence[str]) -> str:
    """Role of one column under the explicit semantic contract."""
    low = name.lower()
    if name in set(meta):
        return ROLE_META
    if name in set(oof):
        return ROLE_OOF_UTILITY
    if name in OPTIONAL_FEATURE_COLUMNS:
        return ROLE_OPTIONAL
    if name.startswith(TARGET_PREFIXES) or name in TARGET_EXTRA:
        return ROLE_TARGET
    if name in METADATA_COLUMNS:
        return ROLE_METADATA
    if low in FORBIDDEN_COLUMNS:
        return ROLE_FORBIDDEN
    return ROLE_FORBIDDEN          # deny by default


def _lp(p: Path) -> str:
    """Windows long-path-safe string form."""
    s = str(p)
    return s if s.startswith("\\\\?\\") else ("\\\\?\\" + s if len(s) > 240 else s)
