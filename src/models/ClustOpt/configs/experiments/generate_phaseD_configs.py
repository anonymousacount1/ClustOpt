"""Generate the Phase-D ablation experiment configs (28 methods x 3 views = 84).

Phase-D is an *additive* ablation. It does not touch the existing
``regressor_metric_selection`` / ``baselines`` configs. All Phase-D configs live
under ``configs/experiments/phaseD/<method_name>/<view>.json`` and differ from
the existing configs in three controlled ways only:

* **early stopping disabled** -- the ``optuna`` block omits ``patience``
  (``OptunaSearch(patience=None)`` runs all requested trials);
* **50 trials requested** (already the existing value);
* new **weighting / utility-source / dynamic-K** combinations.

The clustering search space itself is reused verbatim from
``generate_experiment_configs.build_search_space`` (no new search spaces).

Each config also carries an extra top-level ``phaseD`` descriptor block. The
ClustOpt builder ignores unknown top-level keys; the experiment runner reads it
to emit Phase-D metadata without having to know method semantics.

Run from the repo root::

    python -m models.ClustOpt.configs.experiments.generate_phaseD_configs
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from models.ClustOpt.cluster_validity_indices.metrics.metrics_registry import (
    METRIC_REGISTRY,
)
from models.ClustOpt.cluster_validity_indices.dynamic_metric_selection.dynamic_k import (
    DYNAMIC_TOPK_POLICY,
)
from .generate_experiment_configs import build_search_space

HERE = Path(__file__).resolve().parent
PHASED_ROOT = HERE / "phaseD"
VIEWS = ("x_only", "y_only", "xy_2d")

PHASE_NAME = "ClustOpt_PhaseD_Ablation"
# §7: every regressor_* method loads this exact trained regressor.
REGRESSOR_PATH = "results_analysis/mlp/20260615_231715__metric_utility_mlp_split1_holdout"
REAL_UTILITY_SOURCE = "utility/<view_id>/utility_vector.json"
SOFTMAX_T = 0.5
GENERIC_PARAMS = {"remove_noise": True, "noise_label": -1, "min_clusters": 2}


def optuna_block_no_earlystop() -> Dict[str, Any]:
    """50 trials, early stopping DISABLED (no ``patience`` key)."""
    return {"type": "optuna", "params": {"n_trials": 50, "timeout": 150}}


# ---- weighting helpers -------------------------------------------------------
# "raw-normalized" weights == the existing ``normalized_positive`` scheme.
WEIGHTING_RAW = "normalized_positive"
WEIGHTING_SOFTMAX = "softmax"


def _regressor_cvi(*, top_k, weighting_scheme: str, k_source: str | None) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "model_run_dir": REGRESSOR_PATH,
        "checkpoint_policy": "fold_ensemble",
        "top_k": top_k,
        "weighting": weighting_scheme,
        "softmax_temperature": SOFTMAX_T,
        "metric_name_prefix": "utility__",
        "feature_source": "features/features_records.csv",
        "view_aware": True,
        "fallback_on_error": True,
        "fallback_metrics": {"silhouette": 1.0},
    }
    if k_source is not None:
        params["k_source"] = k_source
        params["real_utility_source"] = REAL_UTILITY_SOURCE
    return {"type": "regressor_dynamic", "params": params,
            "params_generic": dict(GENERIC_PARAMS)}


def _oracle_cvi(*, top_k, weighting_scheme: str) -> Dict[str, Any]:
    return {
        "type": "oracle_dynamic",
        "params": {
            "top_k": top_k,
            "weighting": weighting_scheme,
            "softmax_temperature": SOFTMAX_T,
            "metric_name_prefix": "utility__",
            "utility_source": REAL_UTILITY_SOURCE,
            "fallback_on_error": True,
            "fallback_metrics": {"silhouette": 1.0},
        },
        "params_generic": dict(GENERIC_PARAMS),
    }


def _generic_cvi(metrics: Dict[str, float]) -> Dict[str, Any]:
    return {"type": "generic", "metrics": metrics, "params": dict(GENERIC_PARAMS)}


FIXED_BASELINES: Dict[str, Dict[str, float]] = {
    "all_metrics_uniform": {name: 1.0 for name in METRIC_REGISTRY.keys()},
    "calinski_harabasz_single": {"calinski_harabasz": 1.0},
    "davies_bouldin_single": {"davies_bouldin": 1.0},
    "dbcv_single": {"dbcv": 1.0},
    "silhouette_single": {"silhouette": 1.0},
    "uniform_classic_cvi": {"silhouette": 0.25, "calinski_harabasz": 0.25,
                            "davies_bouldin": 0.25, "dbcv": 0.25},
}


def _weighting_mode(scheme: str) -> str:
    return "raw_normalized" if scheme == WEIGHTING_RAW else "softmax_t05"


def _phaseD_descriptor(*, method_name, utility_source, ranking_source,
                       dynamic_k_source, weighting_mode, top_k) -> Dict[str, Any]:
    return {
        "phase": PHASE_NAME,
        "method_name": method_name,
        "utility_source": utility_source,           # regressor | real | fixed
        "metric_ranking_source": ranking_source,    # regressor | real | fixed
        "dynamic_k_source": dynamic_k_source,       # predicted | real | none
        "weighting_mode": weighting_mode,           # raw_normalized | softmax_t05 | uniform | single
        "softmax_temperature": SOFTMAX_T if weighting_mode == "softmax_t05" else None,
        "top_k_config": top_k,
        "dynamic_topk_policy": DYNAMIC_TOPK_POLICY if top_k == "dynamic" else None,
        "regressor_path": REGRESSOR_PATH if utility_source == "regressor" else None,
        "real_utility_path": REAL_UTILITY_SOURCE if (
            utility_source == "real" or dynamic_k_source == "real") else None,
        "optuna_trials_requested": 50,
        "early_stopping_enabled": False,
    }


def _build_method_specs() -> List[Dict[str, Any]]:
    """Return the 28 Phase-D method descriptors (name, cvi, phaseD block)."""
    specs: List[Dict[str, Any]] = []

    def reg(name, top_k, scheme, k_source, dyn_k_src):
        specs.append({
            "name": name,
            "cvi": _regressor_cvi(top_k=top_k, weighting_scheme=scheme, k_source=k_source),
            "phaseD": _phaseD_descriptor(
                method_name=name, utility_source="regressor",
                ranking_source="regressor", dynamic_k_source=dyn_k_src,
                weighting_mode=_weighting_mode(scheme), top_k=top_k),
        })

    def real(name, top_k, scheme, dyn_k_src):
        specs.append({
            "name": name,
            "cvi": _oracle_cvi(top_k=top_k, weighting_scheme=scheme),
            "phaseD": _phaseD_descriptor(
                method_name=name, utility_source="real", ranking_source="real",
                dynamic_k_source=dyn_k_src, weighting_mode=_weighting_mode(scheme),
                top_k=top_k),
        })

    # 9.1 regressor utility, raw-normalized weights
    for k in (1, 3, 5, 10):
        reg(f"regressor_top{k}_raw", k, WEIGHTING_RAW, None, "none")
    reg("regressor_top_dynamic_realK_raw", "dynamic", WEIGHTING_RAW, "real", "real")
    reg("regressor_top_dynamic_predictK_raw", "dynamic", WEIGHTING_RAW, "predicted", "predicted")

    # 9.2 regressor utility, softmax weights
    for k in (1, 3, 5, 10):
        reg(f"regressor_top{k}_softmax_t05", k, WEIGHTING_SOFTMAX, None, "none")
    reg("regressor_top_dynamic_realK_softmax_t05", "dynamic", WEIGHTING_SOFTMAX, "real", "real")
    reg("regressor_top_dynamic_predictK_softmax_t05", "dynamic", WEIGHTING_SOFTMAX, "predicted", "predicted")

    # 9.3 real/oracle utility, raw-normalized weights
    for k in (1, 3, 5, 10):
        real(f"real_top{k}_raw", k, WEIGHTING_RAW, "none")
    real("real_top_dynamic_raw", "dynamic", WEIGHTING_RAW, "real")

    # 9.4 real/oracle utility, softmax weights
    for k in (1, 3, 5, 10):
        real(f"real_top{k}_softmax_t05", k, WEIGHTING_SOFTMAX, "none")
    real("real_top_dynamic_softmax_t05", "dynamic", WEIGHTING_SOFTMAX, "real")

    # 9.5 fixed baselines
    for name, metrics in FIXED_BASELINES.items():
        wmode = "uniform" if name in ("all_metrics_uniform", "uniform_classic_cvi") else "single"
        specs.append({
            "name": name,
            "cvi": _generic_cvi(metrics),
            "phaseD": _phaseD_descriptor(
                method_name=name, utility_source="fixed", ranking_source="fixed",
                dynamic_k_source="none", weighting_mode=wmode, top_k=None),
        })

    return specs


def write_config(path: Path, spec: Dict[str, Any], dim: str) -> None:
    config = {
        "search_space": build_search_space(dim),
        "search_algorithm": optuna_block_no_earlystop(),
        "cvi": spec["cvi"],
        "phaseD": spec["phaseD"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)


def phaseD_method_names() -> List[str]:
    return [s["name"] for s in _build_method_specs()]


def main() -> None:
    specs = _build_method_specs()
    assert len(specs) == 28, f"expected 28 Phase-D methods, got {len(specs)}"
    written: List[Path] = []
    for spec in specs:
        for dim in VIEWS:
            p = PHASED_ROOT / spec["name"] / f"{dim}.json"
            write_config(p, spec, dim)
            written.append(p)
    print(f"Wrote {len(written)} Phase-D configs ({len(specs)} methods x {len(VIEWS)} views):")
    for s in specs:
        print("  ", s["name"], "->", s["phaseD"]["utility_source"],
              s["phaseD"]["weighting_mode"], "k=", s["phaseD"]["top_k_config"])


if __name__ == "__main__":
    main()
