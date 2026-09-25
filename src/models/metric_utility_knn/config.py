"""
config.py
=========

Central configuration for the Phase-E1 KNN metric-utility predictor.

Everything the pipeline needs — data paths, the leakage-free split policy, the
96-configuration search space, the frozen Dynamic Top-K heuristic reference, and
the canonical output layout — is defined here so that no magic constants leak
into the engine modules.

Terminology (mandatory, see plan §5)
------------------------------------
* ``n_neighbors``          — number of nearest neighbours used by the KNN
                             utility predictor (``K_neighbors``).
* ``selected_metric_count``/``dynamic_k_metrics`` — number of clustering metrics
                             chosen by the Dynamic Top-K heuristic
                             (``K_metrics``).

A generic ``k`` is never used where the meaning is ambiguous.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

# --------------------------------------------------------------------------- #
# Repository / data locations
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(
    r"."
)

UNIFIED_CSV = (
    REPO_ROOT
    / "results_analysis" / "clustering_repository" / "analyzed_data"
    / "unified_training_dataset" / "unified_training_dataset.csv"
)

SPLIT_ASSIGNMENTS_CSV = (
    REPO_ROOT
    / "results_analysis" / "clustering_repository" / "analyzed_data"
    / "experiment_splits" / "dataset_split_assignments.csv"
)

# The existing MLP split-1 holdout artefact — the schema and reference metrics
# the KNN must align to.
MLP_REFERENCE_DIR = (
    REPO_ROOT / "results_analysis" / "mlp"
    / "20260615_231715__metric_utility_mlp_split1_holdout"
)

# Canonical (timestamp-free) results root for Phase E1.
RESULTS_ROOT = (
    REPO_ROOT / "results_analysis" / "metric_utility_knn" / "phase_e1"
)
CACHE_ROOT = RESULTS_ROOT / "cache"

# --------------------------------------------------------------------------- #
# Schema (mirrors the MLP DataConfig exactly — do NOT infer a new feature set)
# --------------------------------------------------------------------------- #
TARGET_PREFIX = "utility__"
N_ID_COLUMNS = 19
N_FEATURE_COLUMNS = 250
N_TARGET_COLUMNS = 60

DATASET_ID_COLUMN = "dataset_id"
RECORD_ID_COLUMN = "record_id"
VIEW_COLUMN = "view_mode"
FAMILY_COLUMN = "family_name"
SUBFAMILY_COLUMN = "subfamily_id"
DIFFICULTY_COLUMN = "difficulty"

# The 19 identifier / metadata columns (never used in the KNN distance).
ID_COLUMNS: List[str] = [
    "dataset_id", "record_id", "view_mode", "view_dim", "source_dataset_dir",
    "family_id", "family_name", "subfamily_id", "difficulty",
    "cluster_count_from_generator", "dataset_seed", "family_seed", "n_points",
    "feature_names_original", "selected_features", "has_labels",
    "has_rendering", "has_structural_ground_truth", "extraction_status",
]

# The three views; queries only ever search their own view's neighbours.
VIEWS: List[str] = ["x_only", "y_only", "xy_2d"]

# Reporting-only grouping columns (metadata, never a distance feature).
GROUP_COLUMNS: Dict[str, str] = {
    "view": VIEW_COLUMN,
    "family": FAMILY_COLUMN,
    "subfamily": SUBFAMILY_COLUMN,
    "difficulty": DIFFICULTY_COLUMN,
}

# --------------------------------------------------------------------------- #
# Leakage-free split policy (plan §7)
# --------------------------------------------------------------------------- #
LOCKED_HOLDOUT_SPLIT = 1                      # split 1 — touched exactly once
DEV_SPLITS: List[int] = list(range(2, 17))    # splits 2..16 — development only

# --------------------------------------------------------------------------- #
# 96-configuration search space (plan §12)
# --------------------------------------------------------------------------- #
N_NEIGHBORS_GRID: List[int] = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100]
SCALERS: List[str] = ["standard", "robust"]
DISTANCES: List[str] = ["euclidean", "cosine"]
WEIGHTINGS: List[str] = ["uniform", "inverse_distance"]

MAX_NEIGHBORS = 100                # neighbours queried once and reused for all n
EPS = 1e-12                        # inverse-distance epsilon
EXACT_MATCH_TOL = 1e-12           # distance <= tol -> treated as exact match

# --------------------------------------------------------------------------- #
# Dynamic Top-K heuristic reference (frozen, imported from ClustOpt)
# --------------------------------------------------------------------------- #
DYNAMIC_TOPK_HEURISTIC = "dynamic_topk_relsoft_a092_k5_ceil"

# Metric top-k / ndcg reporting lists (match the MLP evaluation config).
TOPK_LIST: List[int] = [3, 5, 10]
NDCG_LIST: List[int] = [1, 3, 5, 10]


@dataclass(frozen=True)
class KnnConfig:
    """A single KNN configuration (one point in the 96-config grid)."""

    n_neighbors: int
    scaler: str
    distance: str
    neighbor_weighting: str

    @property
    def config_id(self) -> str:
        return (
            f"n{self.n_neighbors:03d}"
            f"__{self.scaler}"
            f"__{self.distance}"
            f"__{self.neighbor_weighting}"
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "config_id": self.config_id,
            "n_neighbors": int(self.n_neighbors),
            "scaler": self.scaler,
            "distance": self.distance,
            "neighbor_weighting": self.neighbor_weighting,
        }


def all_configurations() -> List[KnnConfig]:
    """The full 12 x 2 x 2 x 2 = 96 configuration grid."""
    configs = [
        KnnConfig(n, s, d, w)
        for n, s, d, w in itertools.product(
            N_NEIGHBORS_GRID, SCALERS, DISTANCES, WEIGHTINGS
        )
    ]
    assert len(configs) == 96, f"expected 96 configs, got {len(configs)}"
    return configs


# Distinct (scaler, distance) neighbour-search combinations (4). A neighbour
# search is done once per (fold, view, scaler, distance) and reused for every
# n_neighbors and weighting.
def scaler_distance_combos() -> List[tuple]:
    return list(itertools.product(SCALERS, DISTANCES))


# --------------------------------------------------------------------------- #
# Composite-ranking metric groups (plan §19). Direction: True = higher better.
# --------------------------------------------------------------------------- #
RANKING_GROUPS: Dict[str, Dict[str, bool]] = {
    "value": {
        "mae": False, "rmse": False, "macro_mae": False, "macro_rmse": False,
        "r2": True, "macro_r2": True, "cosine_similarity_mean": True,
    },
    "global_rank": {
        "spearman_mean": True, "kendall_mean": True,
        "pairwise_ranking_accuracy": True, "ndcg@all": True,
        "rank_displacement_mean": False,
    },
    "head": {
        "top1_accuracy": True, "top3_overlap": True, "top5_overlap": True,
        "top10_overlap": True, "ndcg@3": True, "ndcg@5": True, "ndcg@10": True,
        "top1_rank_distance_mean": False, "top3_utility_mae": False,
        "top5_utility_mae": False, "top10_utility_mae": False,
    },
    "dynamic_k": {
        "dynamic_k_exact_accuracy": True, "dynamic_k_within_1_accuracy": True,
        "dynamic_k_mae": False, "dynamic_k_rmse": False,
    },
}


def result_dirs(root: Path | None = None) -> Dict[str, Path]:
    """Canonical output sub-directories (created on demand by the runner).

    ``root`` overrides the canonical results root (used only for smoke tests).
    """
    base = Path(root) if root is not None else RESULTS_ROOT
    return {
        "root": base,
        "reports": base / "reports",
        "tables": base / "tables",
        "predictions": base / "predictions",
        "models": base / "models",
        "plots": base / "plots",
        "validation": base / "validation",
        "cache": base / "cache",
    }
