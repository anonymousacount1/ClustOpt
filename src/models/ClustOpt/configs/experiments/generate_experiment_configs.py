"""Generate the experiment search-map configs for the metric-selection phase.

This script writes 30 configs:

  * 12 regressor configs: 4 Top-K folders x {x_only, y_only, xy_2d}
  * 18 baseline configs : 6 baseline folders x {x_only, y_only, xy_2d}

Each config carries ``search_space`` + ``search_algorithm`` + ``cvi``.

Search spaces are dimension-specific (x_only / y_only / xy_2d). The true
cluster count is never used: every K-based algorithm searches n_clusters /
n_components in [2, 3, 4, 5].

Run from the repo root::

    python -m models.ClustOpt.configs.experiments.generate_experiment_configs

The script is idempotent -- re-running it overwrites the configs with the same
content. It also imports the metric registry so the ``all_metrics_uniform``
baseline always covers exactly the currently supported metrics.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from models.ClustOpt.cluster_validity_indices.metrics.metrics_registry import (
    METRIC_REGISTRY,
)

HERE = Path(__file__).resolve().parent
REGRESSOR_ROOT = HERE / "regressor_metric_selection"
BASELINE_ROOT = HERE / "baselines"

VIEWS = ("x_only", "y_only", "xy_2d")

# Trained MLP run directory consumed by the regressor_dynamic configs.
# Points at the currently available trained run. Swap this out (and regenerate)
# to target a different trained regressor.
MODEL_RUN_DIR = "results_analysis/mlp/combined"


# ----------------------------------------------------------------- space helpers

def i(low: int, high: int) -> Dict[str, Any]:
    return {"type": "int", "low": low, "high": high}


def f(low: float, high: float) -> Dict[str, Any]:
    return {"type": "float", "low": low, "high": high}


def c(choices: List[Any]) -> Dict[str, Any]:
    return {"type": "categorical", "choices": choices}


K = i(2, 5)  # n_clusters / n_components search -> [2, 3, 4, 5]; no true-K leakage.


def _constrained(n: int, dim: str) -> Dict[str, Any]:
    """Constrained-HDBSCAN block with n_components fixed to ``n``."""
    return {
        "n_components": i(n, n),
        "min_cluster_size": c([25, 50]),
        "min_samples": c([3, 5, 10]),
        "persistence_thresh": c([0.0, 0.02, 0.05]),
    }


def build_search_space(dim: str) -> Dict[str, Any]:
    if dim == "x_only":
        space: Dict[str, Any] = {
            "kmeans": {"n_clusters": K, "init": c(["k-means++"]),
                       "n_init": i(10, 10), "random_state": i(42, 42)},
            "minibatch_kmeans": {"n_clusters": K, "init": c(["k-means++"]),
                                 "batch_size": c([100, 256]), "random_state": i(42, 42)},
            "gmm": {"n_components": K, "covariance_type": c(["full", "diag"]),
                    "random_state": i(42, 42)},
            "agglomerative": {"n_clusters": K,
                              "linkage": c(["ward", "average", "complete"])},
            "birch": {"n_clusters": K, "threshold": c([0.05, 0.1, 0.2, 0.35]),
                      "branching_factor": c([25, 50])},
            "dbscan": {"eps": c([0.03, 0.05, 0.08, 0.12, 0.2]),
                       "min_samples": c([3, 5, 10])},
            "hdbscan": {"min_cluster_size": c([10, 25, 50, 100]),
                        "min_samples": c([3, 5, 10])},
            "optics": {"min_samples": c([3, 5, 10]), "xi": c([0.03, 0.05, 0.1]),
                       "min_cluster_size": c([0.02, 0.05, 0.1])},
        }
        for n in (2, 3, 4, 5):
            space[f"hdbscan_constrained-{n}"] = _constrained(n, dim)
        return space

    if dim == "y_only":
        space = {
            "kmeans": {"n_clusters": K, "init": c(["k-means++"]),
                       "n_init": i(10, 10), "random_state": i(42, 42)},
            "minibatch_kmeans": {"n_clusters": K, "init": c(["k-means++"]),
                                 "batch_size": c([100, 256]), "random_state": i(42, 42)},
            "gmm": {"n_components": K, "covariance_type": c(["full", "diag"]),
                    "random_state": i(42, 42)},
            "agglomerative": {"n_clusters": K,
                              "linkage": c(["ward", "average", "complete"])},
            # Slightly wider density params than x_only.
            "birch": {"n_clusters": K, "threshold": c([0.1, 0.2, 0.35, 0.5]),
                      "branching_factor": c([25, 50])},
            "dbscan": {"eps": c([0.05, 0.1, 0.2, 0.35, 0.5]),
                       "min_samples": c([3, 5, 10])},
            "hdbscan": {"min_cluster_size": c([10, 25, 60, 120]),
                        "min_samples": c([3, 5, 10])},
            "optics": {"min_samples": c([3, 5, 10]), "xi": c([0.03, 0.05, 0.1]),
                       "min_cluster_size": c([0.02, 0.05, 0.1])},
        }
        for n in (2, 3, 4, 5):
            space[f"hdbscan_constrained-{n}"] = _constrained(n, dim)
        return space

    if dim == "xy_2d":
        space = {
            "kmeans": {"n_clusters": K, "init": c(["k-means++"]),
                       "n_init": i(10, 10), "random_state": i(42, 42)},
            "minibatch_kmeans": {"n_clusters": K, "init": c(["k-means++"]),
                                 "batch_size": c([100, 256]), "random_state": i(42, 42)},
            "gmm": {"n_components": K, "covariance_type": c(["full", "diag", "tied"]),
                    "random_state": i(42, 42)},
            "agglomerative": {"n_clusters": K,
                              "linkage": c(["ward", "average", "complete", "single"])},
            "birch": {"n_clusters": K, "threshold": c([0.1, 0.2, 0.35, 0.5, 0.8]),
                      "branching_factor": c([25, 50, 100])},
            "dbscan": {"eps": c([0.05, 0.1, 0.2, 0.3, 0.5, 0.8]),
                       "min_samples": c([3, 5, 10, 20])},
            "hdbscan": {"min_cluster_size": c([10, 25, 50, 100]),
                        "min_samples": c([3, 5, 10, 20])},
            "optics": {"min_samples": c([5, 10, 20]), "xi": c([0.03, 0.05, 0.1]),
                       "min_cluster_size": c([0.02, 0.05, 0.1])},
            "spectral": {"n_clusters": K,
                         "affinity": c(["nearest_neighbors", "rbf"]),
                         "n_neighbors": c([10, 20, 30]), "random_state": i(42, 42)},
        }
        for n in (2, 3, 4, 5):
            space[f"hdbscan_constrained-{n}"] = _constrained(n, dim)
        return space

    raise ValueError(f"Unknown dimension '{dim}'.")


# ------------------------------------------------------------- shared blocks

def optuna_block() -> Dict[str, Any]:
    # OptunaSearch accepts n_trials / timeout / patience. Direction is fixed to
    # 'maximize' and the sampler defaults to TPE in the current implementation,
    # so 'direction'/'sampler'/'seed' are intentionally omitted (see README).
    return {
        "type": "optuna",
        "params": {"n_trials": 50, "timeout": 150, "patience": 15},
    }


GENERIC_PARAMS = {"remove_noise": True, "noise_label": -1, "min_clusters": 2}


def regressor_cvi(top_k: int) -> Dict[str, Any]:
    return {
        "type": "regressor_dynamic",
        "params": {
            "model_run_dir": MODEL_RUN_DIR,
            "checkpoint_policy": "fold_ensemble",
            "top_k": top_k,
            "weighting": "softmax",
            "softmax_temperature": 0.5,
            "metric_name_prefix": "utility__",
            "feature_source": "features/features_records.csv",
            "view_aware": True,
            "fallback_on_error": True,
            "fallback_metrics": {"silhouette": 1.0},
        },
        "params_generic": dict(GENERIC_PARAMS),
    }


def generic_cvi(metrics: Dict[str, float]) -> Dict[str, Any]:
    return {"type": "generic", "metrics": metrics, "params": dict(GENERIC_PARAMS)}


def all_metrics_uniform() -> Dict[str, float]:
    return {name: 1.0 for name in METRIC_REGISTRY.keys()}


BASELINES: Dict[str, Dict[str, float]] = {
    "silhouette_single": {"silhouette": 1.0},
    "calinski_harabasz_single": {"calinski_harabasz": 1.0},
    "davies_bouldin_single": {"davies_bouldin": 1.0},
    "dbcv_single": {"dbcv": 1.0},
    "uniform_classic_cvi": {
        "silhouette": 0.25, "calinski_harabasz": 0.25,
        "davies_bouldin": 0.25, "dbcv": 0.25,
    },
    "all_metrics_uniform": all_metrics_uniform(),
}

REGRESSOR_FOLDERS = {
    "regressor_top1_softmax_t05": 1,
    "regressor_top3_softmax_t05": 3,
    "regressor_top5_softmax_t05": 5,
    "regressor_top10_softmax_t05": 10,
}


def write_config(path: Path, cvi: Dict[str, Any], dim: str) -> None:
    config = {
        "search_space": build_search_space(dim),
        "search_algorithm": optuna_block(),
        "cvi": cvi,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)


def main() -> None:
    written: List[Path] = []

    for folder, top_k in REGRESSOR_FOLDERS.items():
        cvi = regressor_cvi(top_k)
        for dim in VIEWS:
            p = REGRESSOR_ROOT / folder / f"{dim}.json"
            write_config(p, cvi, dim)
            written.append(p)

    for folder, metrics in BASELINES.items():
        cvi = generic_cvi(metrics)
        for dim in VIEWS:
            p = BASELINE_ROOT / folder / f"{dim}.json"
            write_config(p, cvi, dim)
            written.append(p)

    print(f"Wrote {len(written)} configs:")
    for p in written:
        print("  ", p.relative_to(HERE.parents[2]))


if __name__ == "__main__":
    main()
