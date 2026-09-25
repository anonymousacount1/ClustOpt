"""The in-domain AutoClust application phase for a single view (record)."""
from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np

from .models import ML2DAC_ONLY_CVIS, InDomainAutoClust


def _external_metrics(y_true: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import (
        adjusted_mutual_info_score, adjusted_rand_score,
        fowlkes_mallows_score, normalized_mutual_info_score,
    )
    return {
        "ari": float(adjusted_rand_score(y_true, labels)),
        "nmi": float(normalized_mutual_info_score(y_true, labels)),
        "ami": float(adjusted_mutual_info_score(y_true, labels)),
        "fmi": float(fowlkes_mallows_score(y_true, labels)),
    }


def _cvi_vector(engine, labels: np.ndarray, cvi_columns: List[str]) -> List[float]:
    """Compute the CVI vector (in cvi_columns order) for the ARI-predictor input.

    ClustOpt-registered metrics (incl. CH/DBI/SIL/DBCV and any structural metrics)
    are computed via the engine's ClustOpt evaluator; the 3 ML2DAC-only CVIs via
    the ML2DAC adapter. Undefined CVIs (degenerate partitions) stay NaN and are
    imputed downstream by the predictor pipeline.
    """
    clustopt_names = [c for c in cvi_columns if c in engine.clustopt_names]
    vals: Dict[str, float] = {}
    if clustopt_names:
        vals.update(engine._clustopt_raw(labels, clustopt_names))
    if any(c in ML2DAC_ONLY_CVIS for c in cvi_columns):
        vals.update(engine._ml2dac(labels))
    return [float(vals.get(c, float("nan"))) for c in cvi_columns]


def run_view(
    *,
    view,                      # SimpleNamespace: record_id, view_type, X_decision, X_full, y_true, true_k
    model: InDomainAutoClust,
    engine,                    # ViewMetricEngine for this view
    search_space,              # ClusteringSearchSpace (knows catalogue algorithms)
    n_evaluations: int,
    k_range,
    seed: int,
    constrain_space: Optional[dict] = None,  # ClustOpt grid for *_same_search_space
) -> Dict[str, object]:
    from experiments.external_baselines.Unified_MKR.unified_mkr.clustering import run_configuration
    from experiments.external_baselines.Unified_MKR.unified_mkr.configuration import ConfigSpec
    from experiments.external_baselines.ML2DAC.indomain.search import (
        sample_optimizer_configs, sample_optimizer_configs_constrained,
    )

    t0 = time.perf_counter()
    rid = view.record_id
    if not model.has_record(rid):
        return {"status": "failed", "failure_reason": f"no landmarking meta-features for {rid}",
                "record_id": rid, "view_type": view.view_type,
                "runtime_seconds": time.perf_counter() - t0}

    # 1) algorithm selection (k=1 NN over landmarking meta-features)
    feat = model.landmarking_features(rid)
    algorithm = model.predict_algorithm(feat)

    # 2) HPO: sample n_evaluations configs of the selected algorithm, K in k_range.
    # "same search space" variants draw candidate configs from the ClustOpt grid
    # (build_search_space) instead of ML2DAC's native ranges. The algorithm
    # selection above (k=1 NN meta-learning) is UNCHANGED — only the candidate
    # hyperparameter values differ. The selected algorithm is always in the
    # ClustOpt space (AutoClust's 7 selectable algorithms are a subset).
    if constrain_space is not None:
        opt_specs = sample_optimizer_configs_constrained(
            [algorithm], constrain_space, n_evaluations, k_range, seed)
    else:
        opt_specs = sample_optimizer_configs([algorithm], n_evaluations, k_range, seed)

    y_true = np.asarray(view.y_true)
    history: List[dict] = []
    best = None  # (predicted_ari, entry, labels)
    for cfg in opt_specs:
        spec = ConfigSpec(algorithm=cfg["algorithm"], params=dict(cfg["hyperparameters"]),
                          original_config_id="")
        res = run_configuration(search_space, spec, view.X_decision)
        entry = {
            "algorithm": cfg["algorithm"], "hyperparameters": cfg["hyperparameters"],
            "valid_result": bool(res.valid_result), "predicted_k": int(res.predicted_k),
            "n_clusters_found": int(res.n_clusters_found),
            "failure_reason": res.failure_reason or None,
            "predicted_ari": None, "ari": None, "nmi": None, "ami": None, "fmi": None,
        }
        if res.valid_result and res.labels is not None and res.predicted_k >= 1:
            cvi_vec = _cvi_vector(engine, res.labels, model.cvi_columns)
            pred_ari = model.predict_ari(cvi_vec)
            ext = _external_metrics(y_true, res.labels)
            entry.update({"predicted_ari": pred_ari, **ext})
            if best is None or pred_ari > best[0]:
                best = (pred_ari, entry, res.labels)
        history.append(entry)

    runtime = time.perf_counter() - t0
    if best is None:
        return {
            "status": "failed",
            "failure_reason": "no valid clustering among evaluated configs",
            "record_id": rid, "view_type": view.view_type,
            "selected_algorithm": algorithm, "n_evaluations": len(history),
            "config_history": history, "runtime_seconds": runtime,
        }

    pred_ari, inc, inc_labels = best
    return {
        "status": "success",
        "record_id": rid, "view_type": view.view_type,
        "selected_algorithm": algorithm,
        "n_evaluations": len(history),
        "incumbent_config": {"algorithm": inc["algorithm"], "hyperparameters": inc["hyperparameters"]},
        "algorithm": inc["algorithm"],
        "hyperparameters": inc["hyperparameters"],
        "predicted_k": inc["predicted_k"],
        "n_clusters_found": inc["n_clusters_found"],
        "incumbent_predicted_ari": pred_ari,
        "ari": inc["ari"], "nmi": inc["nmi"], "ami": inc["ami"], "fmi": inc["fmi"],
        "true_k": view.true_k,
        "config_history": history,
        "labels": np.asarray(inc_labels).astype(int),
        "runtime_seconds": runtime,
        "failure_reason": None,
    }
