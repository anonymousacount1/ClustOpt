"""Generate the Phase-E2 KNN ablation configs (12 methods x 3 views = 36).

Phase E2 is an *additive* ablation. It does not touch the existing
``regressor_metric_selection`` / ``baselines`` / ``phaseD`` configs. All
Phase-E2 configs live under ``configs/experiments/phaseE2_knn/<method>/<view>.json``
and are byte-for-byte identical to the corresponding Phase-D *regressor* configs
except for the utility source:

* ``cvi.type`` = ``"knn_dynamic"`` (frozen Phase-E1 KNN) instead of
  ``"regressor_dynamic"`` (MLP), and
* ``cvi.params`` points at the frozen KNN model dir instead of the MLP run dir.

Everything downstream -- the clustering search space, the 50-trial no-early-stop
Optuna block, the Dynamic Top-K heuristic, raw/softmax weighting -- is reused
verbatim from the Phase-D generator so the *only* scientific change is
meta-features -> utility predictor.

Run from the repo root::

    python -m models.ClustOpt.configs.experiments.generate_phaseE2_knn_configs
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from models.ClustOpt.cluster_validity_indices.dynamic_metric_selection.dynamic_k import (
    DYNAMIC_TOPK_POLICY,
)
from .generate_experiment_configs import build_search_space
from .generate_phaseD_configs import (
    GENERIC_PARAMS,
    REAL_UTILITY_SOURCE,
    SOFTMAX_T,
    WEIGHTING_RAW,
    WEIGHTING_SOFTMAX,
    optuna_block_no_earlystop,
)

HERE = Path(__file__).resolve().parent
PHASE_E2_ROOT = HERE / "phaseE2_knn"
VIEWS = ("x_only", "y_only", "xy_2d")

PHASE_NAME = "ClustOpt_PhaseE2_KNN_Ablation"
# The frozen Phase-E1 KNN model directory (three view-specific models).
KNN_MODEL_DIR = "results_analysis/metric_utility_knn/phase_e1/models"


def _knn_cvi(*, top_k, weighting_scheme: str, k_source: str | None) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "knn_model_dir": KNN_MODEL_DIR,
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
    return {"type": "knn_dynamic", "params": params,
            "params_generic": dict(GENERIC_PARAMS)}


def _weighting_mode(scheme: str) -> str:
    return "raw_normalized" if scheme == WEIGHTING_RAW else "softmax_t05"


def _phaseE2_descriptor(*, method_name, dynamic_k_source, weighting_mode, top_k,
                        oracle_assisted: bool) -> Dict[str, Any]:
    """Reuse the ``phaseD`` descriptor key so the runner emits metadata unchanged.

    ``utility_source`` / ``metric_ranking_source`` are ``knn``; extra ``knn_*``
    provenance keys are additive (the runner copies a fixed subset and preserves
    the full block in ``config_snapshot.json``).
    """
    return {
        "phase": PHASE_NAME,
        "method_name": method_name,
        "method_group": "knn",
        "utility_source": "knn",
        "metric_ranking_source": "knn",
        "dynamic_k_source": dynamic_k_source,        # predicted | real | none
        "weighting_mode": weighting_mode,            # raw_normalized | softmax_t05
        "softmax_temperature": SOFTMAX_T if weighting_mode == "softmax_t05" else None,
        "top_k_config": top_k,
        "dynamic_topk_policy": DYNAMIC_TOPK_POLICY if top_k == "dynamic" else None,
        "utility_predictor_type": "knn",
        "knn_model_path": KNN_MODEL_DIR,
        "regressor_path": None,
        "real_utility_path": REAL_UTILITY_SOURCE if dynamic_k_source == "real" else None,
        "oracle_assisted": oracle_assisted,          # realK methods are non-deployable
        "optuna_trials_requested": 50,
        "early_stopping_enabled": False,
    }


def _build_method_specs() -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []

    def knn(name, top_k, scheme, k_source, dyn_k_src, oracle_assisted=False):
        specs.append({
            "name": name,
            "cvi": _knn_cvi(top_k=top_k, weighting_scheme=scheme, k_source=k_source),
            "phaseD": _phaseE2_descriptor(
                method_name=name, dynamic_k_source=dyn_k_src,
                weighting_mode=_weighting_mode(scheme), top_k=top_k,
                oracle_assisted=oracle_assisted),
        })

    # 6.1 fixed Top-K, raw-normalized weights
    for k in (1, 3, 5, 10):
        knn(f"knn_top{k}_raw", k, WEIGHTING_RAW, None, "none")
    # 6.2 fixed Top-K, softmax weights
    for k in (1, 3, 5, 10):
        knn(f"knn_top{k}_softmax_t05", k, WEIGHTING_SOFTMAX, None, "none")
    # 6.3 dynamic Top-K, raw-normalized weights
    knn("knn_top_dynamic_realK_raw", "dynamic", WEIGHTING_RAW, "real", "real",
        oracle_assisted=True)
    knn("knn_top_dynamic_predictK_raw", "dynamic", WEIGHTING_RAW, "predicted",
        "predicted")
    # 6.4 dynamic Top-K, softmax weights
    knn("knn_top_dynamic_realK_softmax_t05", "dynamic", WEIGHTING_SOFTMAX, "real",
        "real", oracle_assisted=True)
    knn("knn_top_dynamic_predictK_softmax_t05", "dynamic", WEIGHTING_SOFTMAX,
        "predicted", "predicted")

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


def phaseE2_method_names() -> List[str]:
    return [s["name"] for s in _build_method_specs()]


def main() -> None:
    specs = _build_method_specs()
    assert len(specs) == 12, f"expected 12 Phase-E2 KNN methods, got {len(specs)}"
    written: List[Path] = []
    for spec in specs:
        for dim in VIEWS:
            p = PHASE_E2_ROOT / spec["name"] / f"{dim}.json"
            write_config(p, spec, dim)
            written.append(p)
    print(f"Wrote {len(written)} Phase-E2 KNN configs ({len(specs)} methods x "
          f"{len(VIEWS)} views):")
    for s in specs:
        print("  ", s["name"], "->", s["phaseD"]["dynamic_k_source"],
              s["phaseD"]["weighting_mode"], "k=", s["phaseD"]["top_k_config"])


if __name__ == "__main__":
    main()
